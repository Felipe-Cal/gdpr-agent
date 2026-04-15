"""
Phase 5 — JAX/Flax fine-tuning for Cloud TPU.

This script targets Vertex AI Training with a TPU v4-8 accelerator.
It cannot run on a Mac or a standard GPU — it requires the JAX TPU backend
(or jax[cpu] for unit-testing the logic without hardware).

Why JAX over PyTorch for TPU training?
─────────────────────────────────────
PyTorch/XLA exists but has friction: it translates PyTorch ops to XLA at
runtime, which can produce slow compilation and unexpected graph breaks.

JAX is designed for XLA from the ground up:
  - jax.jit:  ahead-of-time compile a Python function to an XLA computation.
              The compiled kernel runs entirely on the TPU — no Python in the loop.
  - jax.pmap: replicate a computation across N devices (TPU chips), each
              operating on a shard of the batch. For a v4-8 (8 chips), you get
              8x throughput with one line of code.
  - jax.grad: pure functional autodiff — no .backward(), no side effects.

The programming model:
─────────────────────
In PyTorch, training is stateful:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

In JAX, training is functional — the train step is a pure function:
    new_state = train_step(state, batch)  # returns a new state, no mutation

This makes it trivial to JIT-compile and shard:
    pmap_train_step = jax.pmap(train_step, axis_name="batch")

Install:
    pip install jax[tpu] flax optax transformers[flax]
    # On a Vertex AI TPU VM this is pre-installed in the managed container.

Run locally (CPU simulation, slow but useful for testing logic):
    JAX_PLATFORM_NAME=cpu python -m phase5.jax_train
"""

import json
from functools import partial
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Conditional imports — JAX/Flax are not in the default requirements.
# The training logic is fully defined even if the imports fail, so the
# module can be imported and inspected without the TPU stack installed.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    import optax
    import flax.linen as nn
    from flax.training import train_state
    from flax.training.common_utils import shard, shard_prng_key
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

MODEL_ID   = "Qwen/Qwen2.5-0.5B"   # base (non-instruct) for Flax compatibility
DATASET    = "data/gdpr_finetune.jsonl"
OUTPUT_DIR = "data/gdpr-flax-checkpoint"
EPOCHS     = 2
BATCH_SIZE = 8     # per device — with 8 TPU chips: 64 effective
LR         = 2e-5
MAX_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------
class TrainState(train_state.TrainState):
    """
    Extends Flax's TrainState with a dropout RNG key.

    In JAX, randomness is explicit — every stochastic operation needs a key.
    We store it in the training state so pmap can replicate it across devices.
    """
    dropout_rng: Any


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_tokenize(tokenizer) -> dict:
    """Returns tokenized train/eval splits as numpy arrays."""
    records = []
    with open(DATASET) as f:
        for line in f:
            ex = json.loads(line)
            records.append(
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n{ex['output']}"
            )

    encoded = tokenizer(
        records,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",   # JAX works with numpy arrays, not torch tensors
    )

    n = len(records)
    split = int(n * 0.9)
    return (
        {k: v[:split] for k, v in encoded.items()},   # train
        {k: v[split:] for k, v in encoded.items()},   # eval
    )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def cross_entropy_loss(logits, labels, attention_mask):
    """
    Causal LM loss — predict the next token at each position.

    In JAX, loss functions are pure functions (no side effects).
    jax.grad(loss_fn) returns a function that computes the gradient.

    We mask padding tokens so they don't contribute to the loss.
    """
    # Shift: predict token[i+1] from token[i]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask   = attention_mask[:, 1:]

    # Cross-entropy over vocabulary
    # jax.nn.log_softmax is numerically stable
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    token_loss = -jnp.take_along_axis(
        log_probs,
        shift_labels[..., None],
        axis=-1,
    ).squeeze(-1)

    # Average over non-padding tokens
    masked_loss = token_loss * shift_mask
    return masked_loss.sum() / (shift_mask.sum() + 1e-9)


# ---------------------------------------------------------------------------
# Train step — the core of JAX training
# ---------------------------------------------------------------------------
def train_step(state: TrainState, batch: dict) -> tuple[TrainState, dict]:
    """
    A single gradient update step.

    This function is:
    1. Pure (no side effects — returns a new state)
    2. JIT-compiled via @jax.jit or jax.pmap
    3. Differentiable via jax.value_and_grad

    jax.value_and_grad(fn)(params) returns (fn(params), grad(fn)(params))
    in a single forward+backward pass — equivalent to loss.backward() in PyTorch.
    """
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        outputs = state.apply_fn(
            **batch,
            params=params,
            dropout_rng=dropout_rng,
            train=True,
        )
        return cross_entropy_loss(
            outputs.logits,
            batch["input_ids"],
            batch["attention_mask"],
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # When using pmap, average gradients across all TPU chips.
    # lax.pmean is a collective operation — it sums gradients across the
    # "batch" axis (all devices) and divides by device count.
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")

    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    return new_state, {"loss": loss}


def eval_step(state: TrainState, batch: dict) -> dict:
    """Evaluation step — no gradient computation, no dropout."""
    outputs = state.apply_fn(
        **batch,
        params=state.params,
        train=False,
    )
    loss = cross_entropy_loss(
        outputs.logits,
        batch["input_ids"],
        batch["attention_mask"],
    )
    loss = jax.lax.pmean(loss, axis_name="batch")
    return {"loss": loss}


# ---------------------------------------------------------------------------
# pmap wrappers
# ---------------------------------------------------------------------------
# jax.pmap compiles train_step for N devices (TPU chips).
# axis_name="batch" names the device axis so pmean knows which axis to reduce.
# Each device receives a shard of the batch and runs train_step independently,
# then gradients are averaged via pmean before the parameter update.
pmap_train_step = jax.pmap(train_step, axis_name="batch")
pmap_eval_step  = jax.pmap(eval_step,  axis_name="batch")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run():
    if not HAS_JAX:
        raise ImportError(
            "JAX stack not installed. Run:\n"
            "  pip install 'jax[tpu]' flax optax 'transformers[flax]'"
        )

    num_devices = jax.device_count()
    print(f"JAX devices: {num_devices} × {jax.devices()[0].device_kind}")
    # On a TPU v4-8: "8 × TPU v4"
    # On CPU (for testing): "1 × TFRT_CPU_0"

    # -----------------------------------------------------------------------
    # 1. Model + tokenizer (Flax checkpoint)
    # -----------------------------------------------------------------------
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = FlaxAutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=jnp.bfloat16,   # bfloat16 is the native TPU dtype — full throughput
        # On TPU v4, bfloat16 matmuls are hardware-accelerated.
        # On GPU you'd use float16; on CPU use float32.
    )

    # -----------------------------------------------------------------------
    # 2. Optimizer (optax — JAX's optimizer library)
    # -----------------------------------------------------------------------
    # optax.adamw is the standard choice. It returns a GradientTransformation —
    # a pair of (init_fn, update_fn) that are pure functions.
    # warmup_cosine_decay_schedule: linear warmup then cosine decay of LR.
    total_steps = (EPOCHS * 63) // (BATCH_SIZE * num_devices)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LR,
        warmup_steps=max(1, total_steps // 10),
        decay_steps=total_steps,
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)

    # -----------------------------------------------------------------------
    # 3. Training state
    # -----------------------------------------------------------------------
    rng = jax.random.PRNGKey(42)
    rng, dropout_rng = jax.random.split(rng)

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

    # Replicate state across all devices.
    # flax.training.common_utils.replicate copies the state to each TPU chip.
    # Each chip will have its own copy of the parameters, updated in sync
    # via gradient averaging (pmean in train_step).
    from flax.training.common_utils import replicate
    state = replicate(state)

    # -----------------------------------------------------------------------
    # 4. Data
    # -----------------------------------------------------------------------
    print("Tokenizing dataset...")
    train_data, eval_data = load_and_tokenize(tokenizer)

    def make_batches(data: dict, batch_size: int):
        """
        Yields batches shaped (num_devices, per_device_batch, seq_len).

        pmap expects the first axis to be the device axis.
        shard() from flax reshapes (total_batch, seq) → (devices, per_device, seq).
        """
        n = len(data["input_ids"])
        total_batch = batch_size * num_devices
        for i in range(0, n - total_batch + 1, total_batch):
            batch = {k: v[i:i + total_batch] for k, v in data.items()}
            yield shard(batch)  # (num_devices, batch_size, seq_len)

    # -----------------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------------
    print("Training...")
    for epoch in range(EPOCHS):
        train_losses = []
        for batch in make_batches(train_data, BATCH_SIZE):
            state, metrics = pmap_train_step(state, batch)
            # metrics["loss"] has shape (num_devices,) — take device 0
            train_losses.append(float(metrics["loss"][0]))

        eval_losses = []
        for batch in make_batches(eval_data, max(1, BATCH_SIZE // 4)):
            metrics = pmap_eval_step(state, batch)
            eval_losses.append(float(metrics["loss"][0]))

        avg_train = sum(train_losses) / max(len(train_losses), 1)
        avg_eval  = sum(eval_losses)  / max(len(eval_losses),  1)
        print(f"Epoch {epoch + 1}/{EPOCHS} — train_loss: {avg_train:.4f}  eval_loss: {avg_eval:.4f}")

    # -----------------------------------------------------------------------
    # 6. Save checkpoint
    # -----------------------------------------------------------------------
    # Unreplicate: collapse the (num_devices, ...) state back to a single copy
    from flax.training.common_utils import unreplicate
    final_state = unreplicate(state)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, params=final_state.params)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
