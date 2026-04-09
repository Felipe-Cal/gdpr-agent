# GDPR Legal Analyst Agent

A hands-on GCP AI/ML learning project that builds a real GDPR legal analyst agent incrementally — one phase at a time, one framework at a time.

## Why this project exists

This is not a toy demo. It is a structured learning vehicle designed to cover the full GCP AI/ML stack that a senior ML architect is expected to know: from basic RAG pipelines, through agentic orchestration and observability, all the way to TPU fine-tuning and production MLOps.

Each phase adds a new capability to the agent while teaching a specific cluster of frameworks. By the end you will have built — and be able to explain in depth — every major piece of the Vertex AI / LangChain / GKE ecosystem.

The GDPR domain is intentional. Privacy and data sovereignty are first-class concerns in every enterprise AI project, and being able to ground a discussion of ML architecture in real regulatory constraints is exactly the kind of depth that separates senior candidates from mid-level ones.

---

## Phase roadmap

| # | Feature added | Frameworks learned | Status |
|---|--------------|-------------------|--------|
| 1 | RAG Foundation | BigQuery Vector Search, `text-embedding-004`, LangChain LCEL, Gemini 2.5 Flash Lite via Vertex AI | ✅ Built |
| 2 | Agentic Workflows | LangGraph, Google Agent Development Kit (ADK), ReAct / reflection patterns | 🔜 Coming |
| 3 | Observability & Eval | LangFuse (on Cloud Run), LangSmith, Vertex AI Evaluation | 🔜 Coming |
| 4 | Self-hosted Serving | vLLM on GKE, quantization (GPTQ/AWQ), cost benchmarking | 🔜 Coming |
| 5 | Fine-tuning | LoRA, PyTorch/XLA, JAX on Cloud TPU, Vertex AI Training | 🔜 Coming |
| 6 | Production MLOps | Vertex AI Pipelines, GKE autoscaling, VPC-SC / CMEK for GDPR | 🔜 Coming |

See [`docs/phases-overview.md`](docs/phases-overview.md) for the detailed forward-looking plan.

---

## Project structure

```
gdpr-agent/
├── config.py                    # Centralised config via Pydantic-settings + .env
├── pyproject.toml               # All Python dependencies (PEP 517, hatchling)
├── .env.example                 # Template — copy to .env and fill in your values
├── .env                         # Your actual config (gitignored)
│
├── setup/
│   └── create_dataset.py        # One-time: creates the BigQuery dataset
│
├── phase1/
│   ├── ingest.py                # Load PDFs → chunk → embed → store in BigQuery
│   ├── chain.py                 # LCEL RAG chain: retriever | prompt | Gemini
│   └── main.py                  # Interactive CLI to query the agent
│
├── docs/
│   ├── phase1.md                # Deep-dive on Phase 1 architecture and concepts
│   └── phases-overview.md       # Forward-looking plan for all 6 phases
│
└── data/
    └── gdpr_docs/               # Drop your GDPR PDFs here before ingesting
```

### File-by-file

**`config.py`** — Uses `pydantic-settings` to load all configuration from `.env`. Every setting has a type and a default; the entire codebase imports `settings` from here. This is the right pattern for GCP projects: keep secrets and project IDs out of code, validated at startup.

**`setup/create_dataset.py`** — One-time script. Creates the BigQuery dataset in `europe-west4` (Netherlands) so that all data stays inside the EU — a direct GDPR compliance requirement. You run this once; the `document_chunks` table is created automatically on first ingest.

**`phase1/ingest.py`** — The document pipeline. Loads PDFs and text files, splits them into overlapping ~1000-character chunks, calls the Vertex AI embedding API (`text-embedding-004`) to convert each chunk to a 768-dimensional vector, and stores everything in BigQuery. Run this whenever you add new documents.

**`phase1/chain.py`** — The RAG chain. Uses LangChain LCEL (pipe syntax) to wire together: BigQuery vector retriever → prompt template → Gemini 2.5 Flash Lite → output parser. This is the core "brain" of the agent.

**`phase1/main.py`** — A Rich-powered interactive CLI. Accepts a single `--question` flag or drops into an interactive question loop with streaming output.

---

## Quick start

### 1. GCP project setup

```bash
# Create project
gcloud projects create gdpr-agent-project

# Link billing (required for Vertex AI)
# Do this in the Cloud Console: console.cloud.google.com/billing

# Enable APIs
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com \
  --project=gdpr-agent-project

# Authenticate (Application Default Credentials — what all GCP SDKs use)
gcloud auth application-default login
```

### 2. Python environment

```bash
# Requires Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env and set GCP_PROJECT_ID=gdpr-agent-project
# Everything else has sensible defaults
```

`.env` values:

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `GCP_PROJECT_ID` | (required) | Your GCP project |
| `GCP_REGION` | `europe-west4` | Region for BigQuery and embeddings |
| `LLM_REGION` | `europe-west4` | Region for Gemini calls (can differ from GCP_REGION) |
| `BQ_DATASET` | `gdpr_agent` | BigQuery dataset name |
| `BQ_TABLE` | `document_chunks` | BigQuery table for embeddings |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | The generative model (cheapest current Gemini) |
| `EMBEDDING_MODEL` | `text-embedding-004` | The embedding model |

### 4. BigQuery dataset

```bash
python -m setup.create_dataset
```

### 5. Add documents and ingest

Drop GDPR PDFs into `data/gdpr_docs/`. Good starting documents:
- [GDPR full text (EUR-Lex)](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679)
- [EDPB guidelines](https://edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en)
- [ICO guidance](https://ico.org.uk/for-organisations/)

```bash
python -m phase1.ingest
```

### 6. Query the agent

```bash
# Interactive mode
python -m phase1.main

# Single question
python -m phase1.main --question "What are the lawful bases for processing personal data?"
```

---

## Cost model

Everything in Phase 1 runs as close to $0 as possible. There are no persistent VMs or endpoints.

| Service | Pricing | Expected cost |
|---------|---------|---------------|
| BigQuery storage | First 10 GB/month free | $0 |
| BigQuery queries | First 1 TB/month free | $0 |
| Vertex AI Embeddings (`text-embedding-004`) | ~$0.00002 per 1,000 characters | Cents (one-time ingestion) |
| Gemini 2.5 Flash Lite | ~$0.075 per 1M input tokens | Cents per session |
| Cloud Storage / networking | Negligible at this scale | ~$0 |

The key architectural choice that keeps costs near zero: BigQuery Vector Search uses `VECTOR_SEARCH()` as a SQL function — there is no persistent endpoint process running and charging you per hour. Vertex AI Vector Search (the alternative) runs a dedicated index server that costs ~$65+/month even when idle. For a learning project, BigQuery is the right choice.

---

## What you will be able to talk about in an interview

This section maps each phase to specific competencies that come up in senior GCP AI/ML architect interviews.

### Phase 1 — RAG & the Vertex AI / LangChain baseline

- **Vertex AI Embeddings and generative models**: You have hands-on experience calling `text-embedding-004` and Gemini 2.5 Flash Lite, know the API surface, and understand token/character pricing. You can also explain the model family trade-offs (Flash Lite vs. Flash vs. Pro) in terms of cost and capability.
- **BigQuery as a vector store**: You can explain `VECTOR_SEARCH()`, the trade-off between BigQuery and Vertex AI Vector Search (serverless vs. persistent endpoint, latency vs. cost), and when you would choose each.
- **LangChain LCEL**: You understand the Runnable abstraction, pipe composition, `RunnableParallel`, and why this pattern makes chains easy to test and extend. You can explain how LCEL chains map to LangGraph nodes in Phase 2.
- **RAG architecture**: You can whiteboard the full pipeline — document loading, chunking strategy (why overlapping chunks matter for legal text), embedding, retrieval, prompt construction, generation — and articulate the trade-offs at each step.
- **Data sovereignty and GDPR compliance**: You pinned everything to `europe-west4` and can explain why — GDPR Article 44 restricts transfers of personal data outside the EEA, and running AI workloads in-region is the baseline control.

### Phase 2 — Agentic frameworks (LangGraph, Google ADK)

- Stateful multi-step agents, tool use, ReAct reasoning loops, and how LangGraph's graph model differs from simple LCEL chains.

### Phase 3 — Observability and evaluation

- Distributed tracing for LLM calls, evaluation metrics (faithfulness, relevance, groundedness), and how to use Vertex AI Evaluation for automated quality scoring.

### Phase 4 — Self-hosted serving

- vLLM architecture, continuous batching, PagedAttention, quantization trade-offs (GPTQ vs. AWQ), and how to deploy on GKE with proper autoscaling.

### Phase 5 — Fine-tuning

- LoRA / QLoRA, PyTorch/XLA on TPUs, JAX fundamentals, Vertex AI Training jobs, and how to evaluate whether fine-tuning or RAG is the right solution for a given problem.

### Phase 6 — Production MLOps

- Vertex AI Pipelines, GKE node pool autoscaling, VPC Service Controls for data exfiltration protection, and CMEK (Customer-Managed Encryption Keys) — all directly relevant to regulated-industry AI deployments.
