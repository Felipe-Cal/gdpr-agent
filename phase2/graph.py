"""
Phase 2 — LangGraph ReAct agent.

Key concept: instead of a fixed linear chain (Phase 1), this is a graph where
nodes are actions and edges are transitions. The agent decides at runtime what
to do next — retrieve more, search the web, or produce a final answer.

Graph structure:

    START
      │
      ▼
  [call_model] ──── has tool calls? ──── YES ──► [run_tools] ──┐
      ▲                                                         │
      │                                                         │
      └─────────────────────────────────────────────────────────┘
      │
     NO
      │
      ▼
     END

State: a list of messages that grows with each node execution. This is what
gives the agent memory — both within a multi-step reasoning trace AND across
multiple turns of a conversation (via LangGraph's checkpointing).

Why LangGraph over plain LCEL?
- LCEL chains are DAGs (directed acyclic graphs) — no loops, no branching
- LangGraph supports cycles (the model can call tools and then reason again)
- State is explicit and inspectable at every step
- Built-in support for streaming, interruption, and human-in-the-loop
"""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import settings
from phase2.tools import TOOLS

# ---------------------------------------------------------------------------
# System prompt
# The agent is now tool-aware — the prompt tells it to reason before acting
# and to use tools when it needs information it doesn't already have.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert GDPR legal analyst. You have access to three tools:

1. search_gdpr_documents — searches your knowledge base of GDPR regulation text and EDPB guidelines
2. get_gdpr_article — quickly retrieves the key provisions of a specific GDPR article by number
3. web_search — searches the web for recent enforcement actions, new guidelines, or news

How to reason:
- For specific article questions, start with get_gdpr_article for speed
- For broader questions or when you need the exact regulatory text with context, use search_gdpr_documents
- For recent developments (fines, new guidance, country-specific enforcement), use web_search
- You may call multiple tools if a question requires information from different sources
- Always cite your sources (article numbers, document names, URLs) in your final answer
- Distinguish between hard legal requirements and best-practice recommendations
- If information is insufficient, say so — do not speculate

You support multi-turn conversations. Use the conversation history to understand follow-up questions."""


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    The state that flows through the graph.

    `messages` uses the `add_messages` reducer — instead of replacing the list
    on each update, it appends new messages. This is how conversation history
    accumulates across node executions and across turns.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def build_model() -> ChatVertexAI:
    """Creates a Gemini instance with all tools bound to it.

    'Binding' tools means the model receives a description of each tool in its
    context and can emit a structured tool_call in its response instead of
    (or in addition to) regular text.
    """
    llm = ChatVertexAI(
        model_name=settings.gemini_model,
        project=settings.gcp_project_id,
        location=settings.llm_region,
        temperature=0,
        max_output_tokens=4096,
    )
    return llm.bind_tools(TOOLS)


def make_call_model_node(model):
    """Returns a node function that calls the LLM with the current message history."""

    def call_model(state: AgentState) -> dict:
        """
        Node 1: call the LLM.

        Prepends the system prompt if this is the start of a conversation,
        then invokes the model with the full message history. The model may
        respond with text (final answer) or with tool_calls (requesting tool use).
        """
        messages = state["messages"]

        # Inject system prompt at the start of each graph invocation
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = model.invoke(messages)
        return {"messages": [response]}

    return call_model


def run_tools(state: AgentState) -> dict:
    """
    Node 2: execute tool calls requested by the model.

    The last message in state is an AIMessage containing one or more
    tool_calls. This node executes each one and returns ToolMessages
    with the results, which are added to the state for the next model call.

    LangGraph's ToolNode handles this automatically — we implement it
    manually here so the code is transparent and educational.
    """
    from langchain_core.messages import ToolMessage

    last_message: AIMessage = state["messages"][-1]
    tool_map = {t.name: t for t in TOOLS}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name not in tool_map:
            result = f"Unknown tool: {tool_name}"
        else:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Tool '{tool_name}' failed: {e}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
        )

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# Conditional edge — the routing logic
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> Literal["run_tools", "__end__"]:
    """
    Decides what happens after the model responds.

    If the last message has tool_calls → route to run_tools.
    If no tool_calls → the model produced a final answer → END.

    This is the core of the ReAct loop: Reason (model call) → Act (tools) → repeat.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    """
    Compiles the LangGraph agent graph.

    The compiled graph is a callable that accepts state and returns updated state.
    With a checkpointer attached, it also persists state between invocations
    under the same thread_id — enabling multi-turn conversation.
    """
    model = build_model()
    call_model = make_call_model_node(model)

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("run_tools", run_tools)

    # Entry point
    workflow.add_edge(START, "call_model")

    # After model responds: either call tools or finish
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {"run_tools": "run_tools", "__end__": END},
    )

    # After tools run: always go back to the model to reason about the results
    workflow.add_edge("run_tools", "call_model")

    # MemorySaver persists state in memory (per thread_id).
    # In production you would use a database-backed checkpointer (e.g. Postgres).
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# Module-level graph instance — built once and reused
graph = build_graph()


def ask(question: str, thread_id: str = "default") -> tuple[str, list[str]]:
    """
    Ask the agent a question and return the answer plus tool names used.

    Args:
        question:  The user's question.
        thread_id: Conversation ID. Same thread_id = same conversation history.
                   Use a new thread_id to start a fresh conversation.

    Returns:
        (answer_text, list_of_tool_names_called)
    """
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    # Find the final AI response (last AIMessage without tool_calls)
    answer = ""
    tools_used = []

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools_used.extend(tc["name"] for tc in msg.tool_calls)
            else:
                answer = msg.content

    return answer, tools_used


def stream_ask(question: str, thread_id: str = "default"):
    """
    Streaming version of ask(). Yields (chunk, event_type) tuples.

    event_type is one of: 'token', 'tool_start', 'tool_end', 'done'
    """
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}

    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "call_model":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage) and not (
                        hasattr(msg, "tool_calls") and msg.tool_calls
                    ):
                        yield msg.content, "answer"
            elif node_name == "run_tools":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    yield msg.name, "tool_used"
