# Phase 4 — Self-hosted Model Serving

Phase 4 replaces the managed Gemini API with a **self-hosted open-source model** served by **vLLM on GKE**. The agent logic, tools, and prompts are unchanged — only the model backend is swapped.

**Frameworks learned:** vLLM, AWQ quantization, GKE GPU node pools, Kubernetes (Deployment / Service / HPA), LangChain OpenAI adapter, cost benchmarking

---

## Why self-host at all?

Three reasons come up in every enterprise AI architecture conversation:

| Reason | Managed API | Self-hosted |
|---|---|---|
| **Cost at scale** | Pay per token — expensive above ~50k queries/month | Pay per hour — fixed infra cost amortised over volume |
| **Data sovereignty** | Query text leaves your infrastructure (even with EU endpoints) | Query text never leaves your GCP project — fully GDPR-controlled |
| **Model control** | You get what the vendor ships | Fine-tune, quantize, or swap models without API changes |

The trade-off: self-hosted has higher operational complexity and requires a minimum traffic volume to justify the infrastructure cost. The `phase4/benchmarks.py` module calculates the exact break-even point for this project.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  GKE Cluster — europe-west4-a                                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  GPU Node Pool (n1-standard-4 + NVIDIA T4)              │     │
│  │                                                         │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │  vLLM Pod (vllm/vllm-openai:v0.6.3)             │   │     │
│  │  │                                                 │   │     │
│  │  │  Model: Mistral-7B-Instruct-v0.2-AWQ            │   │     │
│  │  │  Quantization: AWQ (4GB VRAM vs 14GB full)      │   │     │
│  │  │  API: /v1/chat/completions (OpenAI-compatible)  │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                          │                                       │
│              LoadBalancer Service (port 80)                      │
└──────────────────────────┼───────────────────────────────────────┘
                           │
                    External IP
                           │
┌──────────────────────────┼───────────────────────────────────────┐
│  LangGraph Agent (phase4/serving.py)                            │
│                                                                  │
│  ChatOpenAI(base_url="http://<IP>/v1")                          │
│       ↕ same tools, same graph, same prompts as Phase 2         │
└──────────────────────────────────────────────────────────────────┘
```

The key insight: **vLLM exposes an OpenAI-compatible API**. LangChain's `ChatOpenAI` accepts a `base_url` parameter pointing at any OpenAI-compatible server. The swap from `ChatVertexAI` to `ChatOpenAI` is literally one line — the rest of the agent is identical.

---

## Core concepts

### vLLM

vLLM is the de-facto standard inference server for open-source LLMs. Two innovations make it significantly faster than naive HuggingFace serving:

**PagedAttention**: LLM inference requires a KV (key-value) cache that grows with sequence length. Naive implementations allocate this cache contiguously in GPU memory, wasting space on fragmentation. PagedAttention manages the KV cache in non-contiguous pages (like OS virtual memory), increasing GPU memory utilisation and enabling larger batch sizes.

**Continuous batching**: Traditional batching waits until a full batch is assembled before running inference. This wastes GPU cycles when requests arrive at different times. Continuous batching inserts new requests into the running batch mid-generation — GPU utilisation goes from ~20% to ~80-90%.

**OpenAI-compatible API**: vLLM serves `/v1/chat/completions`, `/v1/completions`, and `/v1/models` — the same endpoints as the OpenAI API. Any client that works with OpenAI works with vLLM without modification.

### AWQ Quantization

Full-precision (FP16) Mistral-7B requires ~14GB of VRAM. An NVIDIA T4 has 16GB — close, but leaves almost no headroom for the KV cache.

**AWQ (Activation-aware Weight Quantization)** reduces weights to 4-bit integers while preserving accuracy:
- Identifies the ~1% of "salient" weights (those that activation values are most sensitive to) and keeps them at higher precision
- Quantizes the remaining 99% to INT4
- Result: Mistral-7B-AWQ requires ~4GB VRAM — 3.5x reduction, with less than 1% quality degradation on benchmarks

The pre-quantized `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` checkpoint is available on HuggingFace. vLLM detects the AWQ format from the checkpoint config and handles dequantization during inference.

**AWQ vs GPTQ:**

| | GPTQ | AWQ |
|---|---|---|
| Method | Layer-wise quantization with second-order information | Activation-aware salient weight selection |
| Speed | Slower inference (simulates quantization in FP16) | Faster — custom CUDA kernels for INT4 |
| Quality | Good | Slightly better on instruction following |
| vLLM support | ✅ | ✅ (preferred) |

For new deployments, prefer AWQ. GPTQ is more widely available for older models.

### GKE GPU Node Pools

GKE separates compute into node pools. A GPU node pool:
- Uses `n1-standard-*` machine types (required for GPU attachment)
- Attaches one or more NVIDIA GPUs per node
- Is tainted with `nvidia.com/gpu:NoSchedule` — non-GPU pods are automatically excluded

The NVIDIA device plugin (auto-installed by GKE) exposes GPUs as a schedulable resource (`nvidia.com/gpu: 1`). Kubernetes then tracks GPU availability across nodes just like CPU and memory.

**Cost optimisation levers** (for interviews):
- **Spot/preemptible GPU nodes**: 60-70% discount, accept ~5% preemption rate — fine for batch workloads
- **Committed use discounts**: 1yr/3yr commitment for 37-55% off
- **Scale to zero with KEDA**: use Kubernetes Event-driven Autoscaling to scale the GPU node pool to 0 when no requests arrive, then scale back up when traffic resumes (adds ~2 min cold start)

---

## File walk-through

### `phase4/serving.py`
The LangGraph agent wired to the vLLM backend. The graph structure is identical to `phase2/graph.py` — only `build_model()` changes. The `ChatOpenAI` constructor takes `base_url` pointing at the vLLM server and `api_key="none"` (vLLM doesn't require auth by default in a private cluster).

### `phase4/benchmarks.py`
Cost comparison between Gemini 2.0 Flash Lite (managed API) and Mistral-7B-AWQ on GKE. Calculates the break-even query volume — the point above which self-hosting is cheaper. Run standalone:
```bash
python -m phase4.benchmarks
```

### `phase4/main.py`
CLI entry point. Accepts `--endpoint` to override the vLLM URL (useful when testing locally before pointing at GKE) and `--benchmark` to print the cost table without connecting to any model.

### `phase4/k8s/`
Production Kubernetes manifests:
- `namespace.yaml` — isolates the serving workload
- `deployment.yaml` — the vLLM pod with GPU nodeSelector, AWQ flags, readiness/liveness probes, and shared memory for PyTorch
- `service.yaml` — LoadBalancer exposing the endpoint externally
- `hpa.yaml` — Horizontal Pod Autoscaler scaling on CPU utilisation

---

## Deployment

> ⚠️ The steps below start billing. See cost estimates before each step.

### Prerequisites
```bash
# Enable required APIs (one-time, free)
gcloud services enable container.googleapis.com \
  --project=gdpr-agent-project
```

### Step 1 — Create the GKE cluster

> **⚠️ Billing starts here. GKE control plane: ~$0.10/hr.**

```bash
gcloud container clusters create gdpr-serving \
  --zone=europe-west4-a \
  --num-nodes=1 \
  --machine-type=e2-medium \
  --no-enable-autoupgrade \
  --project=gdpr-agent-project
```

### Step 2 — Add GPU node pool

> **⚠️ Billing for GPU node: ~$0.54/hr (n1-standard-4 + T4). Total: ~$0.64/hr.**

```bash
gcloud container node-pools create gpu-pool \
  --cluster=gdpr-serving \
  --zone=europe-west4-a \
  --num-nodes=1 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --no-enable-autoupgrade \
  --project=gdpr-agent-project
```

### Step 3 — Configure kubectl

```bash
gcloud container clusters get-credentials gdpr-serving \
  --zone=europe-west4-a \
  --project=gdpr-agent-project
```

### Step 4 — Install NVIDIA device plugin

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Step 5 — Deploy vLLM

```bash
kubectl apply -f phase4/k8s/namespace.yaml
kubectl apply -f phase4/k8s/deployment.yaml
kubectl apply -f phase4/k8s/service.yaml
kubectl apply -f phase4/k8s/hpa.yaml
```

Wait for the pod to be ready (~3-4 min for model download):
```bash
kubectl rollout status deployment/vllm-mistral -n gdpr-serving
```

### Step 6 — Get the endpoint IP

```bash
kubectl get svc vllm-mistral -n gdpr-serving
# Copy the EXTERNAL-IP
```

### Step 7 — Update .env and run

```bash
# In .env:
VLLM_ENDPOINT=http://<EXTERNAL-IP>

python -m phase4.main
```

---

## 🛑 Teardown — stop all billing

Run this when done. **Do not forget — the cluster charges ~$0.64/hr while running.**

```bash
gcloud container clusters delete gdpr-serving \
  --zone=europe-west4-a \
  --project=gdpr-agent-project
```

Verify in the GCP Console: [Kubernetes Engine > Clusters](https://console.cloud.google.com/kubernetes/list)

---

## Cost benchmark output

```
                 Cost per Query — Gemini API vs GKE/vLLM
╭──────────────────────┬───────────────┬───────────────────┬──────────────────────╮
│ Traffic Level        │ Queries/Month │ Gemini Flash Lite │ GKE + Mistral-7B-AWQ │
├──────────────────────┼───────────────┼───────────────────┼──────────────────────┤
│ Low (dev/test)       │         1,000 │ $0.12             │ $468                 │
│ Medium (small team)  │        10,000 │ $1.20             │ $468                 │
│ High (production)    │       100,000 │ $12.00            │ $468                 │
│ Very high            │     1,000,000 │ $120.00           │ $468                 │
╰──────────────────────┴───────────────┴───────────────────┴──────────────────────╯

Break-even: ~3,900 queries/month
Below that: Gemini API wins on cost.
Above that: GKE self-hosting saves money.
```

---

## What to say in an interview

**"When would you self-host a model instead of using the managed API?"**
> "It comes down to three levers: cost at scale, data sovereignty, and model control. For a GDPR-regulated system, data sovereignty is often non-negotiable — query text must stay within your EU infrastructure perimeter. I ran the numbers: for this GDPR agent, the break-even with Gemini Flash Lite is roughly 4,000 queries per month. Below that, just use the managed API. Above it, a single T4 node on GKE pays for itself. The operational overhead is real, but for a production system that processes legal queries about real users, keeping all text inside your GCP project is worth it."

**"Why vLLM over just loading a model in Python?"**
> "Two reasons: PagedAttention and continuous batching. Naively loading a model gives you sequential, one-at-a-time inference. vLLM manages the KV cache like OS virtual memory — non-contiguous pages instead of one giant contiguous allocation — so GPU memory is used much more efficiently. Continuous batching inserts new requests mid-generation instead of waiting for a full batch, pushing GPU utilisation from ~20% to ~80%. In production, this means you can serve the same traffic with fewer GPU nodes."

**"What's the difference between GPTQ and AWQ?"**
> "Both are post-training quantization methods that reduce model weights to 4-bit integers. GPTQ uses second-order gradient information (Hessian) to minimise reconstruction error layer by layer. AWQ identifies the ~1% of weights that activations are most sensitive to and preserves those at higher precision while quantizing the rest. In practice, AWQ is slightly better on instruction-following tasks and has faster inference because of custom CUDA INT4 kernels. I used AWQ for Mistral-7B here — it fits in ~4GB VRAM vs ~14GB for FP16, which means a T4 instead of an A10G or A100."

---

## Bridge to Phase 5

Phase 4 proved we can run open-source models on our own infrastructure. But we're still using a general-purpose model (Mistral) fine-tuned on generic instruction data — not GDPR legal text.

**Phase 5** adds fine-tuning:
- **LoRA / QLoRA**: parameter-efficient fine-tuning — adapt the model to GDPR legal language without full retraining
- **Vertex AI Training**: managed training jobs on GCP
- **Cloud TPU + JAX**: for larger fine-tuning runs, TPUs offer 2-3x better price-performance than GPUs
