# Phase 5 — Fine-tuning

Phase 5 moves from using a general-purpose model to training a **GDPR-specialised** one. Instead of prompting Mistral or Gemini to answer legal questions, we fine-tune a base model on GDPR Q&A data so it internalises the regulation.

**Frameworks learned:** LoRA / QLoRA, HuggingFace PEFT + TRL, bitsandbytes quantization, Vertex AI Custom Training Jobs, JAX/Flax on Cloud TPU v4, adapter merging

---

## Why fine-tune at all?

RAG (Phase 1) and prompting (Phases 2-4) are powerful, but have limits:

| Approach | Strength | Weakness |
|---|---|---|
| RAG | Up-to-date, traceable sources | Retrieval can fail; answer quality depends on chunk quality |
| Prompting | Zero training cost | General model may not know GDPR nuances; hallucination risk |
| Fine-tuning | Model "knows" the domain deeply | Training cost; can't update knowledge without retraining |

The right answer in production is often **RAG + fine-tuning**: fine-tune for tone, format, and domain vocabulary; use RAG for factual grounding and recency.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Training Pipeline (Vertex AI)                                 │
│                                                                │
│  phase5/dataset.py   →   GCS (gdpr_finetune.jsonl)            │
│       ↓                                                        │
│  Vertex AI Training Job                                        │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Base model: google/gemma-2-2b-it (loaded in 4-bit)   │   │
│  │  QLoRA: train only adapter layers (r=16, ~8M params)  │   │
│  │  Output: LoRA adapter checkpoint → GCS                │   │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  Serving Pipeline (vLLM, Phase 4 infrastructure)              │
│                                                                │
│  vLLM --model gemma-2-2b-it --lora-modules gdpr=<GCS-path>   │
│       ↓                                                        │
│  LangGraph agent (phase4/serving.py) uses fine-tuned model    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  Evaluation (Phase 3 infrastructure)                          │
│                                                                │
│  phase5/evaluate.py: base model vs fine-tuned, side by side   │
│  Metrics: groundedness, coherence, fluency (Vertex AI Eval)   │
└────────────────────────────────────────────────────────────────┘
```

---

## Core concepts

### LoRA (Low-Rank Adaptation)

Full fine-tuning updates every weight in the model (billions of parameters). LoRA freezes the base model and inserts small trainable matrices into each attention layer.

For a weight matrix W ∈ ℝ^{d×d}, LoRA adds:
```
W' = W + BA   where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}, r << d
```

With r=16 and d=2048 (Gemma-2B hidden size), each LoRA pair has 2 × 2048 × 16 = 65,536 parameters vs 2048² = 4,194,304 for the full weight. We apply this to q_proj, k_proj, v_proj, o_proj — total trainable parameters: ~8M out of 2.5B (0.33%).

**Why this works:** The weight updates needed for domain adaptation lie in a low-rank subspace. Most of the model's "knowledge" is already in the base weights — you only need to nudge the attention patterns.

### QLoRA (Quantized LoRA)

QLoRA combines LoRA with 4-bit quantization:

1. Load the base model in NF4 (Normal Float 4-bit) — reduces Gemma-2B from ~5GB to ~1.5GB VRAM
2. Use double quantization (quantize the quantization constants) — saves another ~0.4 bits/param
3. Train the LoRA adapters in BFloat16 — full precision for the small trainable portion

Result: fine-tune a 2B model on a single T4 GPU (16GB VRAM) that would otherwise require ~16GB just to load in FP16.

**NF4 vs INT4**: NF4 (Normal Float 4) maps values to a distribution optimised for normally distributed weights (which LLM weights are). It loses less information than uniform INT4 quantization.

### Vertex AI Custom Training Jobs

Vertex AI Training manages the full lifecycle:
- Provisions a VM with the requested GPU
- Pulls your container image (or uses a pre-built one)
- Mounts GCS buckets as local paths via GCS FUSE
- Streams logs to Cloud Logging
- Tears down the VM when the job completes

You pay only for job runtime — no idle cost. For our dataset (~100 examples, 3 epochs on a T4), the job runs in ~45 minutes at ~$0.54/hr ≈ **$0.40 total**.

### JAX on Cloud TPU

For larger fine-tuning runs (7B+ models, 100k+ examples), TPUs offer better price-performance than GPUs:

| Hardware | FLOPS | Memory | $/hr (GCP) | Best for |
|---|---|---|---|---|
| NVIDIA T4 | 65 TFLOPS (FP16) | 16GB | $0.35 | Small models, prototyping |
| NVIDIA A100 | 312 TFLOPS (BF16) | 40/80GB | $2.93 | Mid-size models |
| TPU v4 (8 chips) | 1,120 TFLOPS | 32GB/chip | $3.22 | Large models, JAX workloads |

**JAX** is Google's array computation library — think NumPy with JIT compilation and first-class TPU support. Key primitives:
- `jax.jit`: JIT-compile a Python function to XLA
- `jax.vmap`: auto-vectorise over a batch dimension
- `jax.pmap`: parallelise across multiple devices (TPU chips)

HuggingFace models can be converted to JAX/Flax with `from_pretrained(..., from_pt=True)`. The training loop then uses `jax.pmap` to shard the model across TPU chips automatically.

For this project we use PyTorch + Vertex AI Training for fine-tuning (simpler tooling, wider ecosystem). In a production scenario with a 70B model, you'd switch to JAX + TPU v4 pods.

### PyTorch/XLA vs JAX on TPU

Both compile to XLA (Google's accelerated linear algebra compiler), but the path is different:

| | PyTorch/XLA | JAX |
|---|---|---|
| **How it works** | Translates PyTorch ops to XLA at runtime — `torch.device("xla")` | Designed for XLA from the ground up — `jax.jit` compiles directly |
| **Graph breaks** | Python control flow (`if`, `for`) breaks the XLA graph; frequent recompilation | JAX traces through Python to build a static graph; `jax.lax.cond` for conditional ops |
| **Multi-device** | `xm.optimizer_step()` + `MpDeviceLoader` — imperative | `jax.pmap` — declarative, one line to shard across N chips |
| **State model** | Stateful (`.backward()`, `.step()`) | Pure functional — `train_step(state, batch) → new_state` |
| **Ecosystem** | Huge (all HF models, any PyTorch code) | Growing (HF Flax models, `optimum-tpu`) |
| **When to use** | Existing PyTorch codebase, < 7B params, single-host | New model, TPU pod (> 8 chips), research, JAX-native models |

The key friction with PyTorch/XLA is **graph breaks**: any Python-side `if` or `print` in the training loop forces XLA to materialise results back to CPU, breaking the compiled computation. JAX avoids this by tracing — you write Python, JAX captures the computation graph once, then the compiled kernel runs entirely on the TPU.

---

## Dataset format

We use the **Alpaca instruction format**:

```json
{"instruction": "What are the lawful bases for processing under GDPR?", "input": "", "output": "Under GDPR Article 6(1), the six lawful bases are..."}
```

The training script formats this as:
```
### Instruction:
What are the lawful bases for processing under GDPR?

### Response:
Under GDPR Article 6(1), the six lawful bases are...
```

The training script formats both instruction and response into a single `text` field. `SFTTrainer` applies a causal language model objective over the full sequence — the model sees the instruction as context and learns to predict the response tokens that follow.

---

## File walk-through

### `phase5/dataset.py`
Generates ~100 synthetic training examples from the 10 golden eval pairs. Each pair is paraphrased into 8-10 variations using instruction templates ("Explain X under GDPR", "What does GDPR say about X?", etc.). Run standalone — no GCP calls, no cost.

### `phase5/train_local.py`
Runs LoRA fine-tuning on any machine — no CUDA required. Auto-detects Apple MPS (M1/M2/M3) for ~3-4× speedup over CPU. Uses `Qwen/Qwen2.5-0.5B-Instruct` (0.5B params, ~1GB RAM), skips 4-bit quantization (bitsandbytes is CUDA-only), and trains adapters in FP32. TRL 1.x API: `SFTConfig` (not `TrainingArguments`) with `max_length` and `dataset_text_field` fields; `processing_class=tokenizer` (not `tokenizer=`). Actual result: 540K trainable params, 2m24s on MPS, eval_loss 2.571→2.474.

### `phase5/train.py`
The GPU training script for Vertex AI. Uses QLoRA (4-bit NF4 base + BFloat16 adapters) via HuggingFace PEFT + TRL's `SFTTrainer`. Reads config from environment variables (injected by Vertex AI). Target hardware: NVIDIA T4 (16GB VRAM) — requires GPU quota on the GCP project.

### `phase5/vertex_job.py`
Submits `train.py` as a Vertex AI Custom Training Job. Uses the pre-built `pytorch-gpu.2-1` container — no Dockerfile needed. Uploads the dataset to GCS and passes its URI to the job via environment variables.

### `phase5/jax_train.py`
Full JAX/Flax training loop targeting TPU v4-8. Teaches the JAX programming model:
- **Functional state**: `train_step` is a pure function — no mutation, returns a new `TrainState`.
- **`jax.value_and_grad`**: single call computes both the loss and its gradient (forward+backward).
- **`jax.pmap`**: wraps `train_step` to shard it across all TPU chips; each chip gets a slice of the batch and runs independently.
- **`lax.pmean`**: collective operation — sums gradients across all chips and divides by device count (equivalent to `DistributedDataParallel` allreduce in PyTorch).
- **`shard` / `replicate` / `unreplicate`**: Flax utilities to reshape data into `(num_devices, per_device_batch, seq_len)` and broadcast/collapse model state.

The script can be smoke-tested on CPU with `JAX_PLATFORM_NAME=cpu python -m phase5.jax_train` (slow but validates the logic without TPU hardware).

### `phase5/tpu_job.py`
Submits `jax_train.py` to Vertex AI Training on a Cloud TPU v4-8 (8 chips, `machine_type="cloud-tpu"`, `accelerator_type="TPU_V4"`, `accelerator_count=8`). Uses the Google-maintained JAX TPU container (`us-docker.pkg.dev/vertex-ai/training/jax-tpu.0-4.py310:latest`) which has `libtpu` pre-installed. No Dockerfile needed. ⚠️ Cost: ~$3.22/hr (expected ~$1.00 total for this dataset).

### `phase5/merge_adapter.py`
Collapses the LoRA adapter back into the base model weights using `model.merge_and_unload()`. This computes `W' = W + (B @ A) × (α/r)` for every adapted layer, producing a standalone model with zero inference overhead. Use when deploying via vLLM without `--enable-lora`, or pushing to HuggingFace Hub. The merged checkpoint is full-size (~1GB for 0.5B); the adapter (5MB) is no longer needed.

### `phase5/evaluate.py`
Runs both the base model and the fine-tuned model through the Phase 3 golden eval dataset and compares scores side by side. Uses the same Vertex AI EvalTask + custom groundedness metric from Phase 3.

---

## How to run

### Step 1 — Generate the dataset (free)

```bash
python -m phase5.main dataset
```

Output: `data/gdpr_finetune.jsonl` (~100 examples)

### Step 2 — Preview the dataset

```bash
head -3 data/gdpr_finetune.jsonl | python -m json.tool
```

### Step 3 — Submit the training job

> **⚠️ This will start billing. Estimated cost: ~$0.40 (T4, ~45 min runtime).**

```bash
python -m phase5.main train --bucket your-gcs-bucket-name
```

### Step 4 — Serve the fine-tuned adapter with vLLM

After training completes, download the adapter from GCS and serve with vLLM's `--lora-modules` flag:

```bash
# In k8s/deployment.yaml, add to args:
- "--enable-lora"
- "--lora-modules"
- "gdpr-lora=gs://your-bucket/phase5/adapter"
```

### Step 5 — Evaluate

```bash
python -m phase5.main evaluate \
  http://<base-model-ip> \
  http://<finetuned-model-ip>
```

---

## What to say in an interview

**"When would you fine-tune vs. just use RAG?"**
> "It depends on what problem you're solving. RAG is better when you need up-to-date information or clear source traceability — which is important for GDPR compliance. Fine-tuning is better when you want the model to adopt a specific tone, format, or domain vocabulary consistently. For a legal assistant, I'd use both: fine-tune the model on legal writing style and terminology so it produces correctly-formatted citations automatically, and keep RAG for factual grounding so answers are traceable to specific regulation text."

**"Why QLoRA over full fine-tuning?"**
> "Compute and memory. Full fine-tuning Gemma-2B needs ~16GB VRAM just for the weights in FP16, plus optimizer states (Adam stores two moments per parameter — another 2x memory). QLoRA quantizes the frozen base to 4-bit (~1.5GB), then trains only the LoRA adapter layers (~8M parameters vs 2.5B). You can run it on a single T4 that costs $0.35/hr instead of needing an A100 at $2.93/hr. The quality gap is less than 1% on standard benchmarks."

**"What's the difference between LoRA and full fine-tuning architecturally?"**
> "Full fine-tuning updates the entire weight matrix W. LoRA decomposes the update as ΔW = BA where B and A are much smaller matrices — rank r vs full rank d. For Gemma-2B with hidden size 2048 and r=16, each LoRA pair is 65K parameters vs 4M for the full layer — a 64x reduction. The insight from the LoRA paper is that the weight updates needed for domain adaptation lie in a low-dimensional subspace, so the low-rank approximation loses almost nothing in practice."

**"Why JAX over PyTorch/XLA for large-scale TPU training?"**
> "Both compile to XLA under the hood, but JAX is designed for it from the ground up. PyTorch/XLA translates PyTorch ops at runtime — any Python control flow in the training loop causes a graph break, materialising results back to CPU and forcing a recompilation. JAX traces through Python once to build a static computation graph, so the compiled kernel runs entirely on the TPU with no Python in the hot path. For multi-chip training, `jax.pmap` shards the training step across all chips in one line, compared to PyTorch/XLA's `MpDeviceLoader` + `xm.optimizer_step()` boilerplate. For a 70B model on a v4 pod (512 chips), you can't afford those Python round-trips."

---

## Bridge to Phase 6

Phase 5 gave us a fine-tuned model. Phase 6 makes it production-ready:
- **Vertex AI Pipelines**: orchestrate ingest → fine-tune → eval → deploy as a reproducible DAG
- **GKE autoscaling**: scale the vLLM deployment to zero when idle, up to N replicas under load
- **VPC Service Controls**: prevent data exfiltration — no query can leave your GCP project's perimeter
- **CMEK**: Customer-Managed Encryption Keys for data at rest — required by some regulated industries
