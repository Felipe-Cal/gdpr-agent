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
| 2 | Agentic Workflows | LangGraph, Google Agent Development Kit (ADK), ReAct / reflection patterns | ✅ Built |
| 3 | Observability & Eval | LangFuse (on Cloud Run), LangSmith, Vertex AI Evaluation | ✅ Built |
| 4 | Self-hosted Serving | vLLM on GKE, quantization (GPTQ/AWQ), cost benchmarking | ✅ Built |
| 5 | Fine-tuning | LoRA, PyTorch/XLA, JAX on Cloud TPU, Vertex AI Training | ✅ Built |
| 6 | Production MLOps | Vertex AI Pipelines, GKE autoscaling, VPC-SC / CMEK for GDPR | ✅ Built |

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
├── phase2/
│   ├── tools.py                 # Tool definitions: search_gdpr_docs, web_search, get_gdpr_article
│   ├── graph.py                 # LangGraph ReAct agent with explicit nodes, edges, state
│   ├── adk_agent.py             # Same agent rebuilt with Google ADK (for comparison)
│   └── main.py                  # CLI supporting both LangGraph and ADK backends
│
├── phase3/
│   ├── callbacks.py             # LangSmith + LangFuse tracing backends
│   ├── eval_dataset.py          # Golden dataset: 10 GDPR Q&A pairs
│   ├── eval_runner.py           # Vertex AI EvalTask (BYOR mode, custom groundedness metric)
│   ├── traced_agent.py          # Wrapper adding tracing callbacks to the Phase 2 agent
│   └── main.py                  # CLI: traced agent + --eval flag
│
├── phase4/
│   ├── serving.py               # LangGraph agent using vLLM endpoint (ChatOpenAI adapter)
│   ├── benchmarks.py            # Cost comparison: Gemini API vs GKE/vLLM
│   ├── main.py                  # CLI: --endpoint, --benchmark flags
│   └── k8s/
│       ├── namespace.yaml       # Kubernetes namespace
│       ├── deployment.yaml      # vLLM pod with GPU nodeSelector + AWQ flags
│       ├── service.yaml         # LoadBalancer service
│       └── hpa.yaml             # Horizontal Pod Autoscaler
│
├── phase6/
│   ├── pipeline.py              # KFP pipeline: ingest → eval → quality gate → fine-tune
│   ├── submit.py                # Compile + submit pipeline to Vertex AI (with scheduling)
│   ├── kms.py                   # CMEK setup: Cloud KMS keyring, key, BigQuery patch
│   ├── vpc_sc.sh                # VPC Service Controls perimeter setup (gcloud)
│   ├── main.py                  # CLI: compile, submit, kms-setup, kms-patch-bq
│   └── k8s/
│       ├── vpa.yaml             # VerticalPodAutoscaler (Off mode — recommend only)
│       └── pdb.yaml             # PodDisruptionBudget (minAvailable=1 for safe drain)
│
├── docs/
│   ├── phase1.md                # Deep-dive on Phase 1 architecture and concepts
│   ├── phase2.md                # Deep-dive on Phase 2: LangGraph, ADK, tools, ReAct
│   ├── phase3.md                # Deep-dive on Phase 3: tracing, eval, LLM-as-judge
│   ├── phase4.md                # Deep-dive on Phase 4: vLLM, AWQ, GKE, cost benchmarking
│   ├── phase5.md                # Deep-dive on Phase 5: LoRA, JAX/TPU, adapter merging
│   ├── phase6.md                # Deep-dive on Phase 6: pipelines, CMEK, VPC-SC, autoscaling
│   └── phases-overview.md       # Forward-looking plan for all 6 phases
│
└── data/
    └── gdpr_docs/               # Drop your GDPR PDFs here before ingesting
```

### File-by-file

**`config.py`** — Uses `pydantic-settings` to load all configuration from `.env`. Every setting has a type and a default; the entire codebase imports `settings` from here. This is the right pattern for GCP projects: keep secrets and project IDs out of code, validated at startup.

**`setup/create_dataset.py`** — One-time script. Creates the BigQuery dataset in `europe-west4` (Netherlands) so that all data stays inside the EU — a direct GDPR compliance requirement. You run this once; the `document_chunks` table is created automatically on first ingest.

**`phase1/ingest.py`** — The document pipeline. Loads PDFs and text files, splits them into overlapping ~1000-character chunks, calls the Vertex AI embedding API (`text-embedding-004`) to convert each chunk to a 768-dimensional vector, and stores everything in BigQuery. Run this whenever you add new documents.

**`phase1/chain.py`** — The RAG chain. Uses LangChain LCEL (pipe syntax) to wire together: BigQuery vector retriever → prompt template → Gemini 2.0 Flash Lite → output parser. This is the core "brain" of the agent.

**`phase1/main.py`** — A Rich-powered interactive CLI. Accepts a single `--question` flag or drops into an interactive question loop with streaming output.

---

## Setup from scratch

### 1. GCP project

```bash
# Create project and link billing in Cloud Console: console.cloud.google.com/billing
gcloud projects create gdpr-agent-project

# Enable APIs needed across all phases
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  cloudkms.googleapis.com \
  container.googleapis.com \
  storage.googleapis.com \
  --project=gdpr-agent-project

# Authenticate (Application Default Credentials — what all GCP SDKs use)
gcloud auth application-default login --project=gdpr-agent-project
```

### 2. Python environment

```bash
# Requires Python 3.11+
python -m venv .venv
source .venv/bin/activate

# Core dependencies (Phases 1–4, 6 inference)
pip install -e ".[dev]"

# Phase 5 local training (LoRA on CPU/MPS — no GPU required)
pip install -e ".[train]"

# Phase 6 pipeline compilation
pip install -e ".[pipelines]"
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env — only GCP_PROJECT_ID is required, everything else has defaults
```

Key variables:

| Variable | Default | What it controls |
|---|---|---|
| `GCP_PROJECT_ID` | **(required)** | Your GCP project |
| `GCP_REGION` | `europe-west4` | BigQuery + embeddings (EU data sovereignty) |
| `LLM_REGION` | `europe-west1` | Gemini — wider model availability, still EU |
| `GEMINI_MODEL` | `gemini-2.0-flash-lite` | Generative model (cheapest Gemini) |
| `EMBEDDING_MODEL` | `text-embedding-004` | 768-dim embedding model |
| `FINETUNE_GCS_BUCKET` | — | GCS bucket for Phase 5/6 artifacts |

### 4. One-time GCP setup

```bash
# Create the BigQuery dataset (europe-west4 — EU data sovereignty)
python -m setup.create_dataset
```

### 5. Ingest GDPR documents

Drop PDFs into `data/gdpr_docs/`. Good starting documents:
- GDPR full text (EUR-Lex PDF)
- EDPB guidelines
- ICO guidance

```bash
python -m phase1.ingest
```

---

## Running each phase

### Phase 1 — RAG (free)

```bash
# Interactive Q&A
python -m phase1.main

# Single question
python -m phase1.main --question "What are the lawful bases for processing personal data?"
```

### Phase 2 — Agentic (free)

```bash
# LangGraph ReAct agent (default)
python -m phase2.main

# Google ADK agent
python -m phase2.main --backend adk
```

### Phase 3 — Evaluation (cents per run)

```bash
# Run the full evaluation suite (LLM-as-judge via Vertex AI)
python -m phase3.main --eval

# Traced agent (enable LANGSMITH_TRACING=true or LANGFUSE_TRACING=true in .env)
python -m phase3.main
```

### Phase 4 — Self-hosted serving (GKE required)

> **⚠️ Billable.** GKE cluster + T4 GPU node: ~$0.35–$0.50/hr. Deploy only when needed.

```bash
# After deploying vLLM to GKE (see docs/phase4.md):
python -m phase4.main --endpoint http://<LoadBalancer-IP>

# Cost benchmark: Gemini API vs self-hosted vLLM
python -m phase4.main --benchmark
```

### Phase 5 — Fine-tuning

```bash
# Generate training dataset (free, ~100 GDPR Q&A examples)
python -m phase5.main dataset

# Train locally on CPU or Apple MPS (~2–15 min, free)
python -m phase5.train_local

# Merge adapter into base model for zero-overhead inference
python -m phase5.merge_adapter

# Submit to Vertex AI Training — T4 GPU
# ⚠️  Cost: ~$0.40 (requires GPU quota increase — see docs/phase5.md)
python -m phase5.main train --bucket your-gcs-bucket

# Submit to Cloud TPU v4-8
# ⚠️  Cost: ~$1.00
python -m phase5.tpu_job --bucket your-gcs-bucket
```

### Phase 6 — Production MLOps

```bash
# Compile the Vertex AI Pipeline to YAML (free, no GCP calls)
python -m phase6.pipeline
# → phase6/gdpr_pipeline.yaml

# Set up CMEK encryption (~$0.06/month)
python -m phase6.kms show-commands   # preview gcloud commands
python -m phase6.kms setup           # create keyring + key
python -m phase6.kms patch-bq        # enable CMEK on BigQuery dataset

# Apply GKE autoscaling manifests (requires Phase 4 cluster)
kubectl apply -f phase6/k8s/vpa.yaml
kubectl apply -f phase6/k8s/pdb.yaml

# Submit pipeline to Vertex AI
# ⚠️  Cost: ~$0.05 per run (+ ~$0.40 if fine-tuning triggered)
python -m phase6.submit --bucket your-gcs-bucket

# Add nightly schedule (2am UTC)
python -m phase6.submit --bucket your-gcs-bucket --schedule
```

---

## Cost summary

| Phase | What runs | Estimated cost |
|---|---|---|
| 1 — RAG | BigQuery + Vertex AI Embeddings + Gemini | ~$0 (free tier covers it) |
| 2 — Agentic | Same as Phase 1 | ~$0 |
| 3 — Eval | Gemini LLM-as-judge (10 questions) | ~$0.01 per run |
| 4 — Serving | GKE cluster + T4 GPU node | ~$0.35–$0.50/hr while running |
| 5 — Fine-tuning (local) | CPU / Apple MPS | $0 |
| 5 — Fine-tuning (Vertex AI) | T4 GPU, ~45 min | ~$0.40 |
| 5 — Fine-tuning (TPU v4-8) | Cloud TPU, ~20 min | ~$1.00 |
| 6 — Pipeline run | Vertex AI managed compute | ~$0.05/run |
| 6 — CMEK key | Cloud KMS | ~$0.06/month |

**Teardown commands** — run these when you're done to stop billing:

```bash
# Phase 4: delete GKE cluster
gcloud container clusters delete gdpr-serving --zone=europe-west4-b --project=YOUR_PROJECT

# Phase 6: delete pipeline schedule (stops nightly runs)
gcloud ai pipeline-jobs list-schedules --region=europe-west4 --project=YOUR_PROJECT
gcloud ai pipeline-jobs delete-schedule --region=europe-west4 --project=YOUR_PROJECT <id>

# Phase 6: delete CMEK key (cryptographic erasure of all encrypted data)
gcloud kms keys versions destroy 1 \
  --key=gdpr-data-key --keyring=gdpr-agent-keyring \
  --location=europe-west4 --project=YOUR_PROJECT
```

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
- Built the same agent in both LangGraph (explicit graph) and Google ADK (declarative) — can articulate trade-offs between frameworks.
- Tool design principles: layered information sources (static lookup → vector search → live web), docstring as API contract.

### Phase 3 — Observability and evaluation

- Distributed tracing for LLM calls, evaluation metrics (faithfulness, relevance, groundedness), and how to use Vertex AI Evaluation for automated quality scoring.

### Phase 4 — Self-hosted serving

- vLLM architecture, continuous batching, PagedAttention, quantization trade-offs (GPTQ vs. AWQ), and how to deploy on GKE with proper autoscaling.

### Phase 5 — Fine-tuning

- LoRA / QLoRA, PyTorch/XLA on TPUs, JAX fundamentals, Vertex AI Training jobs, and how to evaluate whether fine-tuning or RAG is the right solution for a given problem.

### Phase 6 — Production MLOps

- **Vertex AI Pipelines / KFP**: authoring `@component` / `@pipeline` decorated functions, artifact lineage in Vertex ML Metadata, pipeline caching, `dsl.Condition` for runtime branching, nightly scheduling.
- **GKE autoscaling trio**: HPA (pod count), VPA (resource right-sizing), Cluster Autoscaler (node provisioning), PodDisruptionBudget (safe drain for GPU pods). Can explain when each triggers and how they interact.
- **VPC Service Controls**: service perimeter, access levels, ingress rules for Vertex AI Pipelines SA, and why VPC-SC addresses a different threat model than IAM alone (stolen credentials vs. unauthorized identity).
- **CMEK and cryptographic erasure**: Cloud KMS key hierarchy, rotation, and how CMEK implements GDPR Article 17 (right to erasure) at the infrastructure level without per-row deletion tracking.
