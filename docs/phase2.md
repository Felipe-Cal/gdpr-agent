# Phase 2 — Agentic Workflows

Phase 2 transforms the static RAG pipeline from Phase 1 into a **reasoning agent** — a system that decides what to do next, calls tools, evaluates results, and can iterate before giving a final answer.

**Frameworks learned:** LangGraph, Google ADK, ReAct pattern, tool use, multi-turn conversation

---

## What changes from Phase 1

Phase 1 was a fixed pipeline:
```
question → retrieve → prompt → Gemini → answer
```

Phase 2 is a reasoning loop:
```
question → think → maybe search docs → think → maybe search web → think → answer
```

The agent can call tools multiple times, in any order, and decide when it has enough information to answer. This is qualitatively more powerful for complex legal analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LANGGRAPH REACT AGENT                                          │
│                                                                 │
│   START                                                         │
│     │                                                           │
│     ▼                                                           │
│  [call_model]  ◄────────────────────────────────────┐          │
│     │                                               │          │
│     ├── tool_calls? ──YES──► [run_tools] ───────────┘          │
│     │                                                           │
│     └── NO ──► END                                             │
│                                                                 │
│  State (messages list grows with each step):                    │
│   [SystemMessage, HumanMessage, AIMessage(tool_calls),          │
│    ToolMessage(result), AIMessage(tool_calls), ToolMessage,     │
│    AIMessage(final answer)]                                     │
└─────────────────────────────────────────────────────────────────┘

Available tools:
  ┌──────────────────────────────┐
  │ search_gdpr_documents        │ → BigQuery vector search (Phase 1 RAG)
  │ get_gdpr_article             │ → Fast static lookup by article number
  │ web_search                   │ → DuckDuckGo (recent news, enforcement)
  └──────────────────────────────┘
```

---

## Core concepts

### ReAct — Reason + Act

ReAct is the dominant pattern for LLM agents. The agent alternates between:
- **Reasoning** — thinking about what it knows and what it needs
- **Acting** — calling a tool to get more information

In LangGraph this maps directly to the graph: `call_model` is the "reason" node, `run_tools` is the "act" node. The loop continues until the model reasons that it has enough information to give a final answer (no more tool calls).

Without ReAct, the agent is forced to answer in one shot with whatever context it has. With ReAct, it can decompose a complex question into steps:

```
Q: "Does Company X (50 employees, doing health data analytics) need a DPO?"

Step 1: get_gdpr_article("37")
→ "DPO required when core activities involve large-scale processing of health data"

Step 2: search_gdpr_documents("what is large scale processing definition")
→ "EDPB guidelines: consider number of individuals, volume of data, geographic extent..."

Step 3: reason
→ "50 employees doing health analytics may or may not qualify as 'large scale' — need to apply the EDPB criteria"

Final answer: nuanced assessment with citations
```

---

### LangGraph — explicit graph structure

LangGraph models agent behaviour as a **directed graph** where:
- **Nodes** are Python functions (actions)
- **Edges** are transitions between nodes
- **State** is a typed dict that flows through the graph and accumulates data

**Why explicit graphs?**
- You can see and test every transition
- Conditional routing is clear code, not hidden framework magic
- State is inspectable at any point (great for debugging)
- Supports cycles (tool → model → tool → ...) — LCEL chains cannot loop

**The state in this agent:**
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` is a **reducer** — instead of replacing the message list on each update, it appends new messages. This is the mechanism that gives the agent memory: the entire reasoning trace (tool calls, results, intermediate responses) lives in this growing list.

**Multi-turn conversation via checkpointing:**

LangGraph persists state between invocations using a **checkpointer**. We use `MemorySaver` (in-memory), keyed by `thread_id`. The same `thread_id` = the same conversation history. A new `thread_id` = fresh start.

In production you would swap `MemorySaver` for a database-backed checkpointer (e.g. using PostgreSQL) so conversation history survives process restarts.

---

### Google ADK — declarative agent

Google ADK (Agent Development Kit) is Google's own agent framework. The same GDPR analyst is rebuilt in `adk_agent.py` to show the contrast:

| | LangGraph | Google ADK |
|---|---|---|
| Graph structure | Explicit — you define nodes, edges, routing | Implicit — framework handles the ReAct loop |
| State | Typed dict, fully inspectable | Managed by framework |
| Tool definition | `@tool` decorated functions | Plain Python functions with docstrings |
| Vertex AI integration | Via LangChain wrappers | Native |
| Boilerplate | More — but you understand every line | Less — but less visibility |
| Best for | Complex agents needing fine control | Rapid prototyping, standard ReAct patterns |

**When to choose which in an interview:**
> "For a client who needs a production agent with complex multi-step reasoning, human-in-the-loop checkpoints, and custom routing logic, I'd use LangGraph. For a team that needs to ship a functional agent quickly and doesn't need to customise the orchestration layer, ADK is faster to get working and has tighter Vertex AI integration."

---

### Tool design

Each tool is a Python function with a carefully written docstring. The docstring is literally what the LLM reads to decide when and how to call the tool — treat it like an API contract:

- **Name**: describes the action, not the implementation
- **Docstring first line**: when to use this tool (the model reads this)
- **Args section**: what each parameter means and example values

```python
@tool
def search_gdpr_documents(query: str) -> str:
    """Search the GDPR knowledge base for relevant legal text and guidance.

    Use this tool when you need to find specific provisions, definitions,
    or guidance from the ingested GDPR documents...

    Args:
        query: A natural language search query, e.g. 'conditions for valid consent'
    """
```

The three tools cover different information sources:
- `get_gdpr_article` — fast, deterministic, no API call needed (static dict)
- `search_gdpr_documents` — your ingested documents (BigQuery, semantic search)
- `web_search` — the live web (DuckDuckGo, for recent developments)

This layered approach teaches an important architecture pattern: **not everything needs to be in a vector store**. Frequently-accessed structured information (like article numbers) should be a fast static lookup, not a semantic search.

---

## File walk-through

### `phase2/tools.py`

Defines all three tools as LangChain `@tool` decorated functions plus plain Python versions for ADK. Key design decisions:

- **Lazy singleton** for the BigQuery vector store (`_get_vector_store()`): the store is only initialised on the first tool call, not at import time. Keeps startup fast.
- **`TOOLS` list**: a convenience export so `graph.py` and `adk_agent.py` don't need to import tools individually.
- **GDPR_ARTICLES dict**: 18 of the most commonly referenced articles with concise structured summaries. Fast and reliable — no embeddings needed.

### `phase2/graph.py`

The LangGraph implementation. Key sections:

**`AgentState`** — the single source of truth for the conversation. Everything the agent knows is in `messages`.

**`make_call_model_node(model)`** — a factory that returns the `call_model` node function. The model is built once (with tools bound) and closed over. System prompt is injected at the start of each invocation if not already present.

**`run_tools(state)`** — manually executes tool calls from the last AIMessage. Implemented explicitly (rather than using LangGraph's built-in `ToolNode`) so the mechanism is visible in the code.

**`should_continue(state)`** — the routing function. Checks whether the last message has `tool_calls`. This is the decision point that makes the agent loop.

**`build_graph()`** — assembles nodes, edges, and checkpointer into a compiled graph. The compiled graph is a callable.

**`ask()` and `stream_ask()`** — convenience functions that handle the `thread_id` config and parse the event stream.

### `phase2/adk_agent.py`

The ADK implementation. Intentionally more concise — the same behaviour in fewer lines because ADK handles the ReAct loop internally. Compare the two files side by side to understand the trade-off.

Note that tools are re-defined as plain functions (no `@tool` decorator) because ADK reads raw Python function signatures and docstrings.

### `phase2/main.py`

Enhanced CLI with:
- `--agent` flag to switch between LangGraph and ADK
- Tool visibility: prints which tools were called for each answer (educational)
- Persistent `session_id` for multi-turn conversation within a session
- `--new-session` flag to reset conversation history

---

## How to run it

### Install new dependencies

```bash
pip install -e ".[dev]"
```

This installs `langgraph`, `duckduckgo-search`, and `google-adk`.

### Run the LangGraph agent (default)

```bash
python -m phase2.main
```

### Run the ADK agent

```bash
python -m phase2.main --agent adk
```

### Single question

```bash
python -m phase2.main -q "Does a 50-person health data analytics company need a DPO?"
```

### Start a fresh conversation

```bash
python -m phase2.main --new-session
```

---

## What to expect

The agent will show which tools it calls before giving its answer:

```
Q: Does my company need a Data Protection Officer?

A:
  → calling tool: get_gdpr_article
  → calling tool: search_gdpr_documents

Based on Article 37 of the GDPR, a Data Protection Officer must be designated when...
[detailed answer with citations]

Tools used: get_gdpr_article, search_gdpr_documents
```

For a question about recent enforcement:
```
Q: What are the largest GDPR fines in 2024?

A:
  → calling tool: web_search

[answer citing recent enforcement actions with sources]
```

Multi-turn — the agent remembers context:
```
Q: What are the lawful bases for processing personal data?
A: [explains 6 lawful bases]

Q: Which one is most commonly misused?
A: [follows up on legitimate interests, referencing the previous answer context]
```

---

## Example questions for testing

Questions that require multiple tools:
```
Does a hospital with 200 staff processing patient records at scale need a DPO?
What are my obligations when a user requests erasure of their data under Article 17?
Is it legal to use legitimate interests as a basis for email marketing?
```

Questions that use web search:
```
What are the largest GDPR fines issued in the last year?
Has any DPA issued guidance on using AI for HR decisions?
What is the current status of EU-US data transfer adequacy?
```

Questions that test multi-turn memory:
```
[Turn 1] What is the right to data portability?
[Turn 2] How does it differ from the right of access?
[Turn 3] Give me a practical example for a SaaS company.
```

---

## What this teaches you

### What you can say in an interview

**"What's the difference between a LangChain chain and a LangGraph agent?"**

> "A LangChain LCEL chain is a DAG — data flows in one direction through fixed steps. It's great for simple pipelines where you always do the same thing in the same order. LangGraph adds cycles and conditional routing. In Phase 2 of my project, the model can call tools, evaluate the results, and call more tools if needed — that loop isn't possible in LCEL. The trade-off is that graphs are more complex to reason about, so I only use LangGraph when the task genuinely requires iterative reasoning."

**"How do you give an agent memory?"**

> "Two kinds of memory: within a turn and across turns. Within a turn, LangGraph state accumulates all messages — tool calls, results, intermediate reasoning — so the model sees the full context at each step. Across turns, LangGraph's checkpointing persists state by thread ID. In my implementation I use `MemorySaver` for development; in production I'd use a PostgreSQL-backed checkpointer so conversation history survives process restarts."

**"How do you design tools for an LLM agent?"**

> "The docstring is the tool's API contract — the model reads it to decide when and how to call the tool. I design each tool around a single, well-defined information source: a static article lookup for speed on known article numbers, BigQuery vector search for semantic retrieval over ingested docs, and DuckDuckGo for anything recent that might not be in the knowledge base. The key principle is that the agent should have the right tool for each information need, not one generic search."

**"When would you choose Google ADK over LangGraph?"**

> "ADK is faster to prototype with — you describe the agent declaratively and the framework handles the orchestration. LangGraph requires you to explicitly define nodes, edges, and routing logic, but that explicitness is valuable when you need custom control: human-in-the-loop interruptions, complex multi-agent coordination, or debugging specific reasoning paths. In regulated industries like finance or healthcare — which is exactly the GDPR context — I tend to prefer LangGraph's transparency for auditability."

---

## Bridge to Phase 3

Phase 2 gives the agent tools and reasoning. But we can't yet answer: **is it working well?**

- Which tool calls are slow?
- Is the agent retrieving the right chunks?
- How often does it give wrong answers?
- When a question gets a bad answer, why?

Phase 3 adds **observability and evaluation**:
- **LangFuse** — traces every LLM call, tool invocation, and token count
- **LangSmith** — LangChain/LangGraph native tracing, tight integration with the graph
- **Vertex AI Evaluation** — systematic scoring against a golden dataset of GDPR questions + ideal answers

Without Phase 3, you're flying blind. With it, you can debug why the agent failed on a specific question and measure whether a change improves or regresses quality.
