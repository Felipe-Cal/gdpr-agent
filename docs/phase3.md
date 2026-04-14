# Phase 3 — Observability & Evaluation

Phase 3 adds crucial engineering infrastructure to the GDPR Legal Analyst: **observability** (seeing what the agent is doing) and **evaluation** (measuring how well it is doing).

**Frameworks learned:** LangSmith (tracing), LangFuse (self-hostable tracing), Vertex AI Rapid Evaluation (scoring), Golden Datasets, LLM-as-judge

---

## What changes from Phase 2

Phase 2 built a powerful reasoning agent, but it was a "black box." You couldn't easily see:
1. Why did the agent choose a specific tool?
2. What chunk did it actually retrieve from BigQuery?
3. How many tokens did it use?
4. Is the answer actually correct according to the law?

Phase 3 adds layers to answer these questions without changing a single line of the Phase 2 agent logic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  OBSERVABILITY (Runtime)                                    │
│                                                             │
│  User Query ──► [traced_agent] ──► LangGraph Agent          │
│                       │                   │                 │
│                       ▼                   ▼                 │
│                LangSmith Traces    Tool Calls / Reasoning   │
│                (cloud-hosted)      (transparent & logged)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  EVALUATION (Offline)                                       │
│                                                             │
│  [eval_dataset] ──► [eval_runner] ──► [Vertex AI EvalTask]  │
│  (Golden Questions)      │                    │             │
│                          ▼                    ▼             │
│                   Agent Answers        Gemini as Judge      │
│                   (Predictions)        (1-5 Scores)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Core concepts

### Observability: LangSmith and LangFuse

Observability means being able to reconstruct the internal state of a system from its external outputs. For LLM agents, this is done via **tracing**.

A **Trace** is a record of the entire execution path. Inside a trace are **Spans**:
- `call_model` span: what prompt was sent, what raw JSON tool calls came back.
- `run_tools` span: what arguments were passed to the tool, what raw text was returned.

**How it works:** We use LangChain's `callbacks` system. We don't modify the agent — we pass callback handlers into the agent's `invoke` or `stream` call. Both LangSmith and LangFuse implement the same LangChain callback interface, so either (or both) can be enabled simultaneously via `.env`.

**LangSmith** — LangChain's hosted tracing platform. Zero setup, best-in-class LangGraph integration (you can visualise the graph execution step by step). Uses the EU endpoint (`eu.api.smith.langchain.com`) to keep query data in Europe.

**LangFuse** — open-source alternative. The critical difference: it can be **self-hosted on GCP Cloud Run**, keeping all trace data inside your EU infrastructure perimeter. This is the GDPR-safe choice for production systems that process real personal data — no data leaves your GCP project. In Phase 6 we will deploy LangFuse on Cloud Run and switch `LANGFUSE_HOST` to that URL.

| | LangSmith | LangFuse |
|---|---|---|
| Setup | Hosted SaaS, zero infra | Self-hostable on Cloud Run |
| Data residency | LangChain's servers (EU endpoint available) | Your GCP project — fully controlled |
| LangGraph integration | Native, best visualisation | Good, via callback handler |
| Cost | Free tier, then paid | Free to self-host (infra costs only) |
| Best for | Development, debugging | Production with personal data (GDPR) |

---

### GCP-Native Observability: Cloud Trace & OpenTelemetry

While LangSmith and Langfuse are "GenAI-native" tools, GCP has its own enterprise-grade observability stack that you would use for system-wide monitoring.

**Google Cloud Trace** is GCP's distributed tracing system.
- **How it compares:** Cloud Trace is designed for latency analysis and microservice orchestration. It shows you "Spans" for every network call.
- **OpenTelemetry:** To use Cloud Trace with GenAI, you would use the OpenTelemetry standard. Modern libraries like `google-adk` (which we used in Phase 2) have first-class support for exporting traces directly to the GCP Console.
- **When to use it:** You would use Cloud Trace when your AI agent is part of a larger microservice architecture and you need to see how the LLM latency affects the overall user request. In the GCP Console, these appear under **Monitoring > Trace**.

**Cloud Logging:** Every Vertex AI call can also be logged to Cloud Logging, where you can set up **Log-based Metrics** to alert you if your model starts failing or returning empty responses.

---

### Evaluation: LLM-as-a-Judge

Traditional software is tested with assertions (`assert output == expected`). LLM outputs are non-deterministic and semantic, making code-based assertions impossible.

**LLM-as-a-Judge** uses a second, highly capable model (the "judge") to evaluate the output of the first model (the "student").

**The metrics:**
- **Groundedness**: Does the answer only use information found in the retrieved context? (Prevents hallucinations)
- **Relevance**: Does the answer actually address the specific question asked?
- **Coherence**: Is the answer logically structured and easy to read?
- **Safety**: Does the answer contain harmful or restricted content?

---

### Vertex AI Evaluation (Rapid Eval)

Vertex AI provides a managed service for running these evaluations at scale. Instead of writing your own judge prompts, you use `vertexai.evaluation.EvalTask`.

**Why it matters for GCP:** It integrates directly with your Vertex AI experiments. You can track how your scores change as you modify your prompt or chunking strategy. To keep costs low, we use `gemini-1.5-flash` as the evaluator model—it is significantly cheaper than `1.5-pro` while remaining highly capable of scoring legal logic.

---

## File walk-through

### `phase3/callbacks.py`
The "observability provider." It checks if `LANGCHAIN_API_KEY` is present and returns the appropriate tracer. This ensures the app doesn't crash if tracing is disabled.

### `phase3/eval_dataset.py`
The **Golden Dataset**. A list of 10-15 "perfect" questions and answers. This is the source of truth. If you change the agent and your scores on this dataset go down, you have a regression.

### `phase3/eval_runner.py`
The logic that connects the agent to the Vertex AI Evaluation service. It:
1. Loops through the golden dataset.
2. Gets predictions from the agent.
3. Submits the (Question, Prediction, Reference) triplets to Vertex AI.
4. Prints a formatted results table using **Rich**.

### `phase3/traced_agent.py`
An **additive wrapper**. It imports the Phase 2 `graph` and wraps it in a new function that automatically includes the tracing callbacks. This is a best-practice pattern: keep the core logic pure and add "cross-cutting concerns" (like logging/tracing) as wrappers.

### `phase3/main.py`
The Phase 3 entry point. Use this to run the agent with tracing enabled, or use the `--eval` flag to trigger the Vertex AI evaluation suite.

---

## How to run it

### 1. Set up LangSmith (Tracing)
1. Sign up at [smith.langchain.com](https://smith.langchain.com).
2. Create an API Key.
3. Add to your `.env`:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-api-key
   LANGCHAIN_PROJECT=gdpr-agent
   ```

### 2. Install dependencies
```bash
pip install -e ".[dev]"
```

### 3. Run the traced agent
```bash
python -m phase3.main
```
Ask a few questions, then check your LangSmith dashboard to see the full reasoning traces.

### 4. Run the Evaluation Suite
```bash
python -m phase3.main --eval
```
*Note: This will call your GCP project and incur a small cost (cents) for the Gemini Flash judge calls.*

---

## What to expect (Evaluation Output)

When you run the eval, you'll see a progress bar followed by a summary table:

```
Starting Vertex AI Evaluation
Metrics: groundedness, coherence, relevance, safety
Evaluator Model: gemini-1.5-flash

Evaluation Metrics (Avg Scores)
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric       ┃ Score ┃ Explanation ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━┩
│ groundedness │ 4.80  │             │
│ coherence    │ 4.90  │             │
│ relevance    │ 4.75  │             │
│ safety       │ 0.00  │             │
└──────────────┴───────┴─────────────┘
```

---

## What this teaches you

### What you can say in an interview

**"How do you know if your agent is any good?"**
> "You can't rely on 'vibe checks.' I implemented an evaluation pipeline using Vertex AI Evaluation. I created a golden dataset of GDPR questions and used `gemini-1.5-flash` as an LLM-judge to score the agent on groundedness and relevance. This creates a quantitative baseline: if I change my chunking strategy in Phase 1, I can immediately see if it improves or degrades the final answer quality."

**"How do you debug an agent that's making wrong tool calls?"**
> "I implemented tracing using LangSmith. By injecting a callback handler into the LangGraph execution, I get a full visual trace of every step. I can see exactly what text the model received, the raw JSON of the tool calls it emitted, and the results returned by the tools. This lets me pinpoint exactly where the reasoning chain broke down—whether it was a bad retrieval, a misunderstood prompt, or a tool failure. For a 100% GCP-native approach, I could also export these same traces to **Google Cloud Trace** using OpenTelemetry."

**"Why is observability important for GDPR compliance?"**
> "Under GDPR Article 31 (Cooperation with the supervisory authority) and Article 35 (DPIA), you need to be able to explain how your AI system makes decisions. Tracing provides an audit trail of the agent's reasoning. By self-hosting these traces (using a tool like Langfuse on EU infrastructure), you maintain observability while ensuring that sensitive query data remains within the legal perimeter."

---

## Bridge to Phase 4

Phase 3 gave us the tools to prove the agent works. But the agent is still "cloud-native"—it depends entirely on Google's paid APIs for every token.

In an enterprise or high-volume scenario, you might want more control over cost and data residence. **Phase 4** moves us toward **Self-hosted Serving**:
- Setting up **vLLM** on GKE (Google Kubernetes Engine).
- Using **quantized models** (GPTQ/AWQ) to run large models on cheaper hardware.
- Connecting our LangGraph agent to our own private model endpoint.
