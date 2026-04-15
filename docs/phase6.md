# Phase 6 — Production MLOps

Phase 6 wraps everything built in Phases 1–5 into a production-grade, enterprise-compliant architecture: versioned ML pipelines, autoscaling, CMEK encryption, and a VPC security perimeter. These are the controls that turn a working prototype into something a regulated enterprise can actually deploy.

**Frameworks learned:** Vertex AI Pipelines (KFP), GKE autoscaling (HPA / VPA / Cluster Autoscaler / PDB), Cloud KMS + CMEK, VPC Service Controls

---

## Why production MLOps matters for a regulated domain

A GDPR-compliant AI system isn't just about data location and access control. It's about **accountability** — GDPR Article 5(2) requires you to demonstrate compliance, not just achieve it. That means:

- Every document batch that was ingested and when (audit trail)
- Which model version answered which query (lineage)
- Evidence that quality was checked before deployment (evaluation gate)
- Proof that data cannot leave your perimeter (VPC-SC)
- Cryptographic erasure capability (CMEK)

Vertex AI Pipelines and Artifact Lineage provide the audit trail. VPC-SC and CMEK provide the security controls. Together they address the "demonstrate compliance" requirement, not just the "be compliant" one.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Vertex AI Pipeline (nightly schedule, 2am UTC)                          │
│                                                                          │
│  [ingest_documents]                                                      │
│    GCS PDFs → chunk → embed → BigQuery (CMEK-encrypted)                 │
│         │                                                                │
│         ▼                                                                │
│  [run_evaluation]                                                        │
│    Golden dataset → LLM-as-judge → groundedness / coherence metrics     │
│         │                                                                │
│         ▼                                                                │
│  [quality_gate]  ── FAIL → pipeline halts, alert fires ──►  ✗           │
│         │                                                                │
│         ▼ (pass)                                                         │
│  [scores degraded?]  ── YES → [trigger_finetuning] (LoRA, Phase 5)      │
│         │              NO  → done                                        │
│         ▼                                                                │
│  Artifact lineage stored in Vertex ML Metadata                           │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  GKE Serving (Phase 4 infrastructure + Phase 6 additions)                │
│                                                                          │
│  HPA  ─── scales pod count (1→3) based on CPU utilisation               │
│  VPA  ─── recommends right-sized CPU/memory requests (Off mode)         │
│  PDB  ─── guarantees ≥1 pod available during node drain                 │
│  Cluster Autoscaler ─── adds/removes GPU nodes based on pending pods    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Security Controls                                                       │
│                                                                          │
│  VPC-SC  ─── service perimeter: blocks exfiltration even with valid IAM │
│  CMEK    ─── KMS-managed key: cryptographic erasure on demand           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core concepts

### Vertex AI Pipelines and KFP

Vertex AI Pipelines is the managed Kubeflow Pipelines (KFP) service. You write the pipeline in Python using the KFP SDK, compile it to YAML, and submit it. Vertex AI runs each step as an isolated Kubernetes pod.

**`@dsl.component`** converts a Python function into a pipeline step. KFP:
1. Packages the function's code + `packages_to_install` into a container image
2. Pushes the image to Artifact Registry
3. Creates a Kubernetes pod spec for the step
4. Streams logs to Cloud Logging

The key constraint: a `@component` function body cannot reference anything outside it — no imports from the outer module, no closures. Every import happens inside the function. This is because the function runs in a different container from the one that compiled it.

**`@dsl.pipeline`** assembles components into a DAG. Data flows between steps via typed `Input` / `Output` ports:
- **Scalars** (str, float, int, bool): serialised through KFP's metadata store
- **Artifacts** (Dataset, Model, Metrics): stored as GCS URIs tracked in Vertex ML Metadata

**`dsl.Condition`** wraps a subgraph with a runtime predicate. Unlike Python `if` statements (evaluated at compile time), `dsl.Condition` is evaluated by the pipeline executor after upstream steps complete.

**Pipeline caching**: if a component's inputs haven't changed since the last run, Vertex AI reuses the cached output. This makes nightly re-runs cheap — only the components affected by new documents actually execute.

**Artifact lineage**: every pipeline run creates a lineage graph in Vertex ML Metadata. You can trace: which GCS file → which BigQuery rows → which eval result → which model version. This is exactly the audit trail Article 5(2) requires.

### GKE autoscaling trio

Three independent autoscaling mechanisms work together for the vLLM serving deployment:

| Controller | What it scales | When it acts |
|---|---|---|
| **HPA** | Number of pods | When CPU/memory exceeds threshold |
| **VPA** | CPU/memory *per pod* | Based on historical usage (Off mode: recommend only) |
| **Cluster Autoscaler** | Number of nodes | When pods can't be scheduled (scale up) or nodes are idle (scale down) |

**PodDisruptionBudget (PDB)** is the safety mechanism: it tells Kubernetes "never allow all pods of this deployment to be evicted simultaneously." Without it, a node drain (triggered by Cluster Autoscaler scale-down or a GKE node upgrade) can evict the single vLLM pod and cause 30–60 seconds of downtime while the replacement loads the model.

**Cascade: how scale-up works end-to-end:**
1. Traffic increases → HPA detects CPU > 70% → creates a new pod
2. No node has available GPU capacity → pod stays `Pending`
3. Cluster Autoscaler sees `Pending` pods → provisions a new GPU node
4. NVIDIA device plugin registers GPU → pod is scheduled and starts loading

**Cascade: how scale-down works safely:**
1. Traffic drops → HPA removes a pod
2. A node becomes underutilised → Cluster Autoscaler wants to drain it
3. PDB check: is `minAvailable=1` satisfied? Only if another pod is Running and Ready
4. For a single-replica deployment: PDB blocks the drain → node stays until the next scheduled maintenance window

### VPC Service Controls

VPC-SC creates a **service perimeter** — a logical boundary around your GCP project. API calls crossing the boundary are rejected at the Google API layer, before they reach IAM or any application code.

**IAM vs VPC-SC:**
- IAM: "user X is allowed to call `bigquery.tables.get`"
- VPC-SC: "calls to BigQuery from outside the perimeter are rejected, regardless of IAM"

They are complementary. IAM alone fails if credentials are stolen — the attacker calls the API from their own laptop with the stolen key. VPC-SC blocks that call at the network boundary.

**The perimeter for this project protects:**
- `bigquery.googleapis.com` — document embeddings and chunks
- `aiplatform.googleapis.com` — Vertex AI Training and Pipelines
- `storage.googleapis.com` — fine-tuning artifacts and pipeline runs
- `cloudkms.googleapis.com` — encryption keys

**Ingress rules** grant specific service accounts the right to cross the perimeter. The Vertex AI Pipelines service account needs ingress so pipeline pods can access BigQuery and GCS. Without the ingress rule, the pipeline would fail with a VPC-SC violation.

**GDPR Article 32 relevance:** VPC-SC is a "technical measure" that prevents unauthorised access or transfer. EDPB guidelines on Article 32 explicitly cite network-level controls as qualifying technical measures. Implementing VPC-SC demonstrates that you have operationalised Article 32, not just acknowledged it.

### CMEK — Customer-Managed Encryption Keys

By default, GCP uses Google-managed encryption keys (GMEK). With CMEK, you create a key in Cloud KMS and hand its resource name to each GCP service. GCP encrypts your data with your key — you own the key lifecycle.

**Key hierarchy:**
```
KMS Key Ring (europe-west4)
  └─ Crypto Key (gdpr-data-key, AES-256-GCM)
       ├─ Version 1 (current primary — encrypts new data)
       └─ Version 2 (rotated — still needed to decrypt old data)
```

**90-day rotation**: Cloud KMS auto-creates a new key version every 90 days and sets it as the primary. Old versions remain active for decryption. You can manually destroy a version — all data encrypted with it becomes permanently unreadable.

**GDPR Article 17 — cryptographic erasure:**
When a data subject requests erasure, deleting database rows is straightforward but not sufficient — the data may persist in BigQuery snapshots, GCS backups, or the embedding cache. CMEK provides a practical solution: destroy the KMS key version → all data encrypted under it becomes inaccessible. This is recognised as equivalent to deletion in most regulatory and legal frameworks.

**What CMEK protects and doesn't protect:**
- ✅ Data at rest in BigQuery, GCS, Vertex AI artifacts
- ✅ Access audit trail via Cloud KMS audit logs (who decrypted what, when)
- ✅ Revocation control — you can cut off access to all data instantly
- ❌ Data in transit (covered by TLS, not KMS)
- ❌ Data in memory during processing (covered by Confidential Computing)

---

## File walk-through

### `phase6/pipeline.py`
The core artifact. Defines four KFP components and assembles them into a pipeline DAG.

- **`ingest_documents`**: downloads PDFs from GCS, chunks them, calls the Vertex AI Embeddings API, upserts into BigQuery. Outputs a `Metrics` artifact counting ingested chunks — visible in the pipeline UI without reading logs.
- **`run_evaluation`**: runs the Phase 3 evaluation suite (same golden dataset, same custom `PointwiseMetric`) inside the pipeline container. Emits `groundedness_mean`, `coherence_mean`, `fluency_mean` as metric artifacts — Vertex AI plots these as trend lines across pipeline runs.
- **`quality_gate`**: pure Python, no GCP calls. Fails the pipeline with a `RuntimeError` if scores fall below thresholds. Downstream steps don't run if this raises.
- **`trigger_finetuning`**: submits the Phase 5 `CustomTrainingJob` inside a `dsl.Condition` block — only executes when `groundedness_mean < finetune_trigger_threshold`. The `finetune_model` Output[Model] artifact records the GCS URI of the adapter in the lineage graph.

Compile locally (free): `python -m phase6.pipeline` → produces `phase6/gdpr_pipeline.yaml`.

### `phase6/submit.py`
Compiles the pipeline and submits it to Vertex AI. Supports `--schedule` to create a nightly cron schedule (2am UTC). Includes the exact teardown command to delete the schedule.

### `phase6/kms.py`
Three subcommands:
- `setup`: creates the KMS keyring and key programmatically via the Cloud KMS SDK
- `patch-bq`: sets the CMEK key as the default encryption for the BigQuery dataset
- `show-commands`: prints the equivalent `gcloud` commands (useful when you don't want to run Python)

### `phase6/vpc_sc.sh`
Shell script documenting the full VPC-SC setup: create access level, create perimeter, add ingress rule for the Vertex AI Pipelines service account. Includes the teardown command. This is configuration-as-code for a security control — checking it into git means the perimeter config is auditable and reproducible.

### `phase6/k8s/vpa.yaml`
`VerticalPodAutoscaler` in `updateMode: "Off"` — recommends CPU/memory right-sizing without evicting pods. GPU workloads can't tolerate surprise evictions (30–60s reload time), so VPA is advisory only. Check recommendations with `kubectl describe vpa vllm-mistral`.

### `phase6/k8s/pdb.yaml`
`PodDisruptionBudget` with `minAvailable: 1`. Prevents Cluster Autoscaler and node upgrade operations from evicting the last vLLM pod. The safety mechanism that makes autoscaling safe for a latency-sensitive GPU endpoint.

---

## How to run

### Step 1 — Install the KFP SDK

```bash
pip install -e ".[pipelines]"
```

### Step 2 — Compile the pipeline (free)

```bash
python -m phase6.pipeline
# → phase6/gdpr_pipeline.yaml
```

Open the YAML to see the compiled component specs, input/output schemas, and DAG edges.

### Step 3 — Set up CMEK (optional, ~$0.06/month)

```bash
# See all commands first
python -m phase6.kms show-commands

# Create keyring + key
python -m phase6.kms setup

# Grant service accounts access (run the printed gcloud commands)

# Patch BigQuery dataset
python -m phase6.kms patch-bq
```

### Step 4 — Apply autoscaling manifests

```bash
# VPA (requires VPA addon enabled on the cluster)
kubectl apply -f phase6/k8s/vpa.yaml

# PDB
kubectl apply -f phase6/k8s/pdb.yaml

# Check VPA recommendations after traffic runs for a while
kubectl describe vpa vllm-mistral -n gdpr-serving
```

### Step 5 — Submit the pipeline

> **⚠️ This will start billing. Estimated cost: ~$0.05 per run (ingest + eval). Fine-tuning adds ~$0.40 if triggered.**

```bash
python -m phase6.submit --bucket your-gcs-bucket

# With nightly schedule:
python -m phase6.submit --bucket your-gcs-bucket --schedule
```

Teardown (stop nightly billing):
```bash
# List schedules
gcloud ai pipeline-jobs list-schedules --region=europe-west4 --project=YOUR_PROJECT

# Delete schedule
gcloud ai pipeline-jobs delete-schedule --region=europe-west4 \
  --project=YOUR_PROJECT <schedule-id>
```

---

## What to say in an interview

**"What is Vertex AI Pipelines and why use it over a cron job?"**
> "Vertex AI Pipelines is the managed Kubeflow Pipelines service. The key advantage over a cron job is artifact lineage — every run creates a provenance graph in Vertex ML Metadata: which GCS files were ingested, which BigQuery rows they produced, which eval scores resulted, which model was deployed. That lineage is what GDPR Article 5(2) requires when you need to demonstrate compliance, not just claim it. Cron jobs produce logs; Vertex AI Pipelines produces a queryable lineage graph. The second advantage is the quality gate as a first-class pipeline primitive — if the gate fails, no downstream work runs, and the failure is visible in the console with an alert attached."

**"Explain HPA, VPA, and Cluster Autoscaler — when does each trigger?"**
> "They operate at three different layers. HPA scales the number of pods in a deployment based on metrics — typically CPU or a custom metric like request queue depth. VPA scales the resource requests of individual pods based on historical usage — it can't change a running pod (on GPU workloads I run it in Off mode to get recommendations without automatic eviction). Cluster Autoscaler operates at the node level: when HPA tries to add a pod but no node has capacity, Cluster Autoscaler provisions one; when nodes are underutilised, it drains and removes them. The PodDisruptionBudget is what makes scale-down safe — it tells Kubernetes never to evict pods below `minAvailable`, so a GPU serving pod with a 60-second model reload time can't be disrupted by routine cluster maintenance."

**"What is VPC Service Controls and how does it differ from IAM?"**
> "IAM is identity-based: it controls what an authenticated principal can do. VPC-SC is network-based: it creates a perimeter around your GCP resources and rejects API calls that cross the boundary, regardless of IAM. The practical difference is the stolen-credentials scenario. IAM alone fails if a service account key is compromised — the attacker calls BigQuery from their laptop with the stolen key and exfiltrates the table. VPC-SC blocks that call because it originates outside the perimeter. For GDPR Article 32, EDPB guidance recognises network-level controls as qualifying technical measures. Together IAM + VPC-SC give you defence in depth: even if credentials are stolen, data can't leave the perimeter."

**"Explain CMEK and how it relates to GDPR's right to erasure."**
> "CMEK means your GCP services encrypt data using a key you own in Cloud KMS, rather than a Google-managed key. The GDPR relevance is Article 17 — the right to erasure. Deleting rows from BigQuery is straightforward, but data may persist in snapshots, GCS backups, or exported datasets. CMEK provides cryptographic erasure: destroy the KMS key version and all data encrypted under it becomes permanently inaccessible, even if the bits remain on disk. Most data protection authorities and legal frameworks accept this as equivalent to deletion. It's also faster — revoking a key takes seconds; hunting down every copy of data in backups can take days."

---

## What you built across all 6 phases

```
Phase 1  RAG pipeline          BigQuery vector search, LCEL, text-embedding-004, Gemini
Phase 2  Agentic workflows     LangGraph, Google ADK, ReAct, tool use, multi-turn memory
Phase 3  Observability + eval  LangFuse, LangSmith, Vertex AI EvalTask, LLM-as-judge
Phase 4  Self-hosted serving   vLLM, GKE, quantization (AWQ), cost benchmarking
Phase 5  Fine-tuning           LoRA/QLoRA, TRL, JAX/Flax, pmap, Vertex AI Training
Phase 6  Production MLOps      Vertex AI Pipelines, KFP, CMEK, VPC-SC, GKE autoscaling
```

Each phase is a layer in the same system. The final architecture is an agent that retrieves GDPR text from BigQuery (Phase 1), reasons across multiple steps with tool use (Phase 2), traces every call with observable telemetry (Phase 3), serves responses from a self-hosted model (Phase 4) that has been domain-adapted via fine-tuning (Phase 5), all orchestrated by a versioned pipeline with enterprise security controls (Phase 6).
