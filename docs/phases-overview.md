# Phases Overview — GDPR Legal Analyst Agent

This document maps the full six-phase learning arc: what each phase builds, what it teaches, and how it connects to the next. Read this as a curriculum guide, not just a feature roadmap.

---

## The learning arc

```
Phase 1: RAG Foundation
    "I can retrieve relevant GDPR text and answer questions"
        ↓
Phase 2: Agentic Workflows
    "I can reason across multiple steps and remember context"
        ↓
Phase 3: Observability & Eval
    "I can measure whether my answers are actually correct"
        ↓
Phase 4: Self-hosted Serving
    "I can run models without paying per-token to Google"
        ↓
Phase 5: Fine-tuning
    "I can train a model that understands GDPR deeply, not just retrieves it"
        ↓
Phase 6: Production MLOps
    "I can operate this in a regulated enterprise environment"
```

Each phase teaches a distinct skill set. They also build on each other: you cannot meaningfully evaluate an agent (Phase 3) without a working agent to evaluate (Phase 2), and you cannot fine-tune a model on legal text (Phase 5) without understanding what "good" answers look like (Phase 3).

---

## Phase 1 — RAG Foundation (✅ Built)

**What it adds:** End-to-end document Q&A. Load GDPR PDFs, embed them, store in BigQuery, retrieve on query, answer with Gemini.

**What it teaches:**
- BigQuery Vector Search (`VECTOR_SEARCH()` SQL function, serverless vs. persistent endpoint trade-offs)
- Vertex AI Embeddings — `text-embedding-004`, 768 dimensions, multilingual
- LangChain LCEL — the Runnable abstraction, pipe composition, `RunnableParallel`, streaming
- Gemini 1.5 Pro via Vertex AI — regional endpoints, temperature, system prompts
- Document chunking strategy for legal text (overlapping chunks, separator hierarchy)
- Data sovereignty fundamentals — why `europe-west4` matters for GDPR

**Job description bullet it maps to:** Hands-on experience with Vertex AI, LangChain, Gemini, and BigQuery. Ability to design and explain a RAG pipeline end-to-end.

**See:** [`docs/phase1.md`](phase1.md) for the deep-dive.

---

## Phase 2 — Agentic Workflows

**What it adds:** The agent can now hold a conversation (multi-turn), decompose complex questions into sub-questions, use tools (web search, citation lookup), and reflect on its own answers before committing to them.

**What it teaches:**

**LangGraph** is the central framework. Where LCEL gives you a linear pipeline, LangGraph gives you a directed graph where nodes are actions and edges are state transitions. State is explicitly typed and carried between nodes, enabling multi-turn conversation.

Key LangGraph concepts:
- `StateGraph` — define a graph with typed state
- `add_node()` / `add_edge()` / `add_conditional_edges()` — graph structure
- `MemorySaver` — persist conversation state across turns (using a `thread_id`)
- Compiling the graph and calling it with `.invoke()` or `.stream()`

**Google Agent Development Kit (ADK)** is Google's opinionated framework for building agents that use Gemini as the reasoning core. It handles tool registration, function calling, and the agent loop. Key concepts:
- `LlmAgent` — an agent with a model, tools, and instructions
- `@tool` decorator — expose a Python function as a callable tool for the agent
- `Runner` — manages agent execution and session state
- Integration with Vertex AI for production deployment

**ReAct pattern (Reason + Act):** The agent alternates between "thinking" (reasoning about what to do next) and "acting" (calling a tool or the retriever). This is implemented as a LangGraph node that generates a thought step before each retrieval, improving relevance. After generating an answer, a reflection node checks whether the answer is grounded and complete — if not, it loops back to retrieve more.

**What changes in the architecture:**

```
Phase 1:  question → retrieve → answer  (one shot)

Phase 2:  question
              ↓
          [plan node]  "I need to find: (1) the lawful basis, (2) the DPO requirement"
              ↓
          [retrieve node]  → BigQuery, get chunks for sub-question 1
              ↓
          [reason node]  → partial answer
              ↓
          [retrieve node]  → BigQuery, get chunks for sub-question 2
              ↓
          [reflect node]  → "Is this answer complete and grounded?"
              ↓  (if yes)
          [answer node]  → final response with citations
```

**Job description bullet it maps to:** LangGraph and agentic frameworks, multi-step reasoning, tool use, Google ADK.

---

## Phase 3 — Observability & Evaluation

**What it adds:** Every query is traced. You can see which chunks were retrieved, how long each step took, what the model received and generated. You can run automated evaluation suites to score answer quality.

**What it teaches:**

**LangFuse** is an open-source LLM observability platform. You self-host it on Cloud Run (no vendor lock-in, no data leaving your infrastructure). It integrates with LangChain via a callback handler — add one line and every chain execution is traced.

Key LangFuse concepts:
- Traces, spans, and generations — the three-level hierarchy of an LLM call
- Scores — human or automated quality ratings attached to traces
- Datasets — collections of (input, expected output) pairs for evaluation runs
- Prompt management — versioned prompts with A/B testing

**LangSmith** is LangChain's hosted observability platform. It has tighter LangChain integration (no callback handler needed — set `LANGCHAIN_TRACING_V2=true`) and is useful for debugging chain execution during development. For production you would use LangFuse (self-hosted, GDPR-compliant) rather than sending data to LangSmith's US servers.

**Vertex AI Evaluation** is GCP's evaluation framework. It uses a second LLM as a judge — you define evaluation criteria (faithfulness, groundedness, relevance, coherence) and it scores your agent's responses against them. Key workflow:
- Define an evaluation dataset (question, context, expected answer)
- Run `EvalTask` with your criteria
- Get back numerical scores and explanations
- Compare evaluation results across phases to measure improvement

**Why evaluation is non-negotiable:** An agent that gives confident-sounding wrong answers is worse than no agent. Evaluation creates a feedback loop: change the chunking strategy → run eval → see if scores improve → commit the change. Without this loop, you are guessing. With it, you are engineering.

**What gets added to the architecture:**
- LangFuse callback injected into the LCEL chain
- Cloud Run deployment for LangFuse server (Terraform or Cloud Run YAML)
- Evaluation dataset of GDPR Q&A pairs with reference answers
- CI step that runs evaluation and fails the build if scores drop

**Job description bullet it maps to:** LLM evaluation, observability, MLOps quality gates, understanding of faithfulness / groundedness metrics.

---

## Phase 4 — Self-hosted Serving

**What it adds:** The agent can use a model running on your own GKE cluster instead of the Vertex AI Gemini API. You control latency, cost, and data residency completely.

**What it teaches:**

**vLLM** is the dominant open-source LLM inference engine. It achieves 2–24x higher throughput than naive inference through:
- **PagedAttention:** manages the KV cache (the memory structure that stores previous tokens) in pages, like virtual memory in an OS. This eliminates memory fragmentation and allows much larger batch sizes.
- **Continuous batching:** rather than waiting for a full batch to fill before starting inference, vLLM starts processing requests as they arrive and adds new requests to the batch mid-flight.
- OpenAI-compatible REST API — drop-in replacement for any code using the OpenAI SDK.

**Model quantization** reduces model size (and therefore GPU memory requirements) by representing weights in lower precision:
- **GPTQ (Post-Training Quantization):** quantizes weights to 4-bit or 8-bit after training. Fast to apply, good quality, widely supported.
- **AWQ (Activation-aware Weight Quantization):** identifies which weights are most important based on activation patterns and protects them during quantization. Better quality than GPTQ at the same bit width, especially for instruction-following tasks.

Quantization trade-offs to understand: a 7B model in 4-bit GPTQ runs on a single A100-40GB where the full 16-bit model would require two. The quality loss is small for most tasks but measurable on complex reasoning.

**GKE deployment:** Running vLLM on GKE teaches:
- GPU node pools (selecting the right machine type — L4, A100, H100 — for the model size)
- Kubernetes resource requests/limits for GPU resources (`nvidia.com/gpu: 1`)
- Readiness probes for model loading (a 7B model takes 30–60 seconds to load)
- Horizontal Pod Autoscaler based on request queue depth

**Cost benchmarking:** Phase 4 includes a benchmark script that runs the same queries through Gemini API (Phase 1/2/3 approach) and through the self-hosted vLLM endpoint and compares:
- Cost per query (API pricing vs. GPU node-hour / queries-per-hour)
- Latency (p50, p95, p99)
- Quality (using the evaluation framework from Phase 3)

This benchmark gives you real data for the "build vs. buy" conversation that comes up in every enterprise AI architecture discussion.

**Job description bullet it maps to:** vLLM, GKE, quantization (GPTQ/AWQ), cost optimization, GPU infrastructure.

---

## Phase 5 — Fine-tuning

**What it adds:** A version of the agent where the base model has been fine-tuned on GDPR legal text and Q&A pairs, making it more accurate on GDPR questions without needing as much context in the prompt.

**What it teaches:**

**LoRA (Low-Rank Adaptation)** is the standard technique for efficient fine-tuning of large models. Instead of updating all of the model's parameters (which requires as much memory as the model itself), LoRA injects small trainable matrices into each transformer layer and trains only those. A 7B model has ~7 billion parameters; a LoRA adapter for it might have 50–100 million — 70x fewer.

Key LoRA hyperparameters to understand:
- `r` (rank) — the inner dimension of the low-rank matrices. Higher = more capacity = more parameters. Typical: 8–64.
- `alpha` — scaling factor for the LoRA updates. Often set equal to `r`.
- `target_modules` — which layers to apply LoRA to. Typically the attention projections (`q_proj`, `v_proj`).
- `dropout` — regularization for the LoRA layers.

**QLoRA** extends LoRA by quantizing the base model to 4-bit (using NF4 quantization) before training. This makes fine-tuning a 7B model feasible on a single consumer GPU or a smaller cloud GPU — important for cost.

**PyTorch/XLA** is the PyTorch backend for Cloud TPUs. XLA (Accelerated Linear Algebra) is the compiler that Google uses for both TPUs and its own training infrastructure. Writing PyTorch/XLA code means marking `torch.no_grad()` / `xm.optimizer_step()` calls correctly and understanding that XLA traces the computation graph and compiles it — the first step is slow (compilation), subsequent steps are fast (executing the compiled program).

**JAX** is Google's array computation library built on XLA. It is functional (no in-place mutation), supports `jit` (just-in-time compilation), `vmap` (automatic vectorization), and `pmap` (multi-device parallelism). JAX is used inside Google for most large-scale model training and is increasingly the framework of choice for new research models (Gemma is trained with JAX/Flax). Understanding the functional paradigm and the compilation model is essential for working with TPUs at scale.

**Cloud TPUs:** Google's custom AI accelerators. TPU v4 and v5 pods are what Gemini was trained on. For fine-tuning, you would use a TPU v3-8 or v4-8 node via Vertex AI Training (a managed job that provisions and tears down the hardware for you). Key concepts: TPU topology (number of chips), high-bandwidth memory (HBM), the XLA compilation pipeline.

**Vertex AI Training:** The managed training service. You define a `CustomTrainingJob` or `CustomContainerTrainingJob`, specify machine type and accelerator, and Vertex AI handles provisioning, distributed coordination, and artifact storage. Output model checkpoints go to Cloud Storage.

**What to evaluate:** After fine-tuning, you re-run the Phase 3 evaluation suite on the fine-tuned model and compare scores against the base model + RAG approach. This gives you quantitative evidence for when fine-tuning adds value over RAG alone.

**Job description bullet it maps to:** LoRA, fine-tuning, PyTorch, JAX, Cloud TPU, Vertex AI Training.

---

## Phase 6 — Production MLOps

**What it adds:** The agent runs in a production-grade, enterprise-compliant architecture: versioned pipelines, autoscaling, encrypted data, and a security perimeter.

**What it teaches:**

**Vertex AI Pipelines** is GCP's managed ML pipeline orchestrator, built on Kubeflow Pipelines. You define pipelines as Python functions decorated with `@component` and connected with `>>`. The pipeline is compiled to YAML and submitted to Vertex AI, which runs each component as a containerised step on managed compute. Key concepts:
- Components — isolated Python functions with typed inputs/outputs, each packaged as a Docker container
- Artifacts — structured outputs (Model, Dataset, Metrics) with lineage tracking
- Parameters — scalar inputs to a pipeline run
- Pipeline runs and experiments — versioned executions with metrics attached
- Scheduling — run the pipeline on a cron schedule (e.g., nightly re-ingestion + evaluation)

For the GDPR agent, the pipeline would cover: ingest new documents → evaluate quality → conditionally promote to production retriever → notify.

**GKE autoscaling:** Phase 6 moves the serving layer to GKE (if not already there from Phase 4) and adds:
- **Horizontal Pod Autoscaler (HPA)** — scale the number of pods based on CPU/memory or custom metrics (request rate, queue depth).
- **Cluster Autoscaler** — scale the number of nodes in a node pool based on pending pods. When HPA adds pods that cannot be scheduled, Cluster Autoscaler provisions new nodes; when nodes are underutilised, it drains and removes them.
- **Vertical Pod Autoscaler (VPA)** — automatically adjust CPU/memory requests for pods based on historical usage. Less commonly needed but useful for right-sizing GPU workloads.

**VPC Service Controls (VPC-SC):** A GCP security feature that creates a perimeter around your GCP resources. Once a perimeter is established, API calls that try to access resources inside the perimeter from outside it (e.g., from a VM in a different project, or from the internet) are blocked — even if the caller has the right IAM permissions. This is the control that prevents an insider threat from exfiltrating data by calling `bq export` from a personal GCP project.

For GDPR compliance, VPC-SC addresses GDPR Article 32 (security of processing) — you are implementing a technical measure to prevent unauthorised access or transfer.

**CMEK (Customer-Managed Encryption Keys):** By default, GCP encrypts data at rest using Google-managed keys. With CMEK, you create a key in Cloud KMS, hand it to the GCP service (BigQuery, Vertex AI, Cloud Storage), and GCP uses it to encrypt your data. You can revoke the key at any time, which makes the data inaccessible — this is the "right to erasure" control at the infrastructure level. For regulated industries (finance, healthcare, public sector) CMEK is often a contractual requirement.

**Job description bullet it maps to:** Vertex AI Pipelines, GKE, VPC Service Controls, CMEK, production MLOps, regulatory compliance.

---

## How the phases connect

Each phase is additive. You are building the same agent, not rebuilding it.

```
Phase 1: Basic RAG pipeline (LCEL chain)
  │
  ├─ Phase 2: Wrap the chain in a LangGraph agent with memory and tools
  │             The LCEL chain becomes one node in the graph
  │
  ├─ Phase 3: Inject observability callbacks into the LangGraph agent
  │             The graph nodes emit traces; evaluation runs against the agent
  │
  ├─ Phase 4: Swap the Gemini endpoint for a self-hosted vLLM endpoint
  │             One config change in config.py (or a new model provider class)
  │
  ├─ Phase 5: Replace the base model with a fine-tuned checkpoint in vLLM
  │             The serving infrastructure is unchanged; the model artifact changes
  │
  └─ Phase 6: Wrap everything in a Vertex AI Pipeline
                The agent components become pipeline stages with versioned artifacts
```

By the end of Phase 6, you will have touched every major component of the GCP AI/ML stack — and you will have built each one into a system that actually works, which is the kind of evidence that makes the difference in a senior-level interview.
