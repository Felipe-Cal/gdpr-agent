"""
Phase 5 — LoRA fine-tuning script.

This script is designed to run inside a Vertex AI Training job (custom container),
but can also be run locally on a machine with a GPU.

Architecture:
    Base model:  google/gemma-2-2b-it  (2B params, fits on a single T4)
    Method:      QLoRA — quantize the base model to 4-bit (bitsandbytes NF4),
                 then train only the LoRA adapter layers (r=16, ~8M params).
    Framework:   HuggingFace PEFT + Transformers + TRL (SFTTrainer)

Why QLoRA over full fine-tuning?
    Full fine-tuning Gemma-2B requires ~16GB VRAM (FP16) + optimizer states.
    QLoRA quantizes the frozen base model to 4-bit (~4GB) and only trains the
    small LoRA adapters (~8M params), fitting comfortably on a single T4 (16GB).
    Quality degradation vs full fine-tuning: <1% on standard benchmarks.

Why LoRA specifically?
    LoRA (Low-Rank Adaptation) inserts trainable rank-decomposition matrices
    into each attention layer.  Instead of updating all W ∈ ℝ^{d×d} weights,
    it trains A ∈ ℝ^{d×r} and B ∈ ℝ^{r×d} where r << d.
    Parameters trained: r=16, α=32, applied to q_proj and v_proj in each layer.

Environment variables (set by Vertex AI Training or locally in .env):
    MODEL_ID:        HuggingFace model ID (default: google/gemma-2-2b-it)
    DATASET_PATH:    Path to the JSONL training file
    OUTPUT_DIR:      Where to write the adapter checkpoint
    GCS_OUTPUT_URI:  GCS path to upload the checkpoint after training
    EPOCHS:          Number of training epochs (default: 3)
    BATCH_SIZE:      Per-device batch size (default: 4)
"""

import os
import json
from pathlib import Path

# These imports are only available in the training environment.
# Running `pip install -e ".[train]"` installs them.
try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    HAS_TRAIN_DEPS = True
except ImportError:
    HAS_TRAIN_DEPS = False


# ---------------------------------------------------------------------------
# Configuration — read from environment (Vertex AI injects these)
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET_PATH = os.getenv("DATASET_PATH", "data/gdpr_finetune.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/gdpr-lora-adapter")
GCS_OUTPUT_URI = os.getenv("GCS_OUTPUT_URI", "")   # e.g. gs://my-bucket/phase5/adapter
EPOCHS = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "1024"))


def load_dataset_from_jsonl(path: str) -> "Dataset":
    """
    Loads the Alpaca-format JSONL file into a HuggingFace Dataset.
    Supports local paths and GCS URIs (gs://...).
    """
    # If running in Vertex AI and path is a GCS URI, download it first
    if path.startswith("gs://"):
        import subprocess
        local_path = "/tmp/gdpr_finetune.jsonl"
        subprocess.run(["gsutil", "cp", path, local_path], check=True)
        path = local_path

    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_alpaca_prompt(example: dict) -> dict:
    """
    Converts an Alpaca record into a single text string for SFTTrainer.

    Format:
        ### Instruction:
        <instruction>

        ### Response:
        <output>

    The model learns to generate everything after '### Response:'.
    """
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}


def run_training():
    """Main training loop."""
    if not HAS_TRAIN_DEPS:
        raise ImportError(
            "Training dependencies not installed. Run: pip install -e '.[train]'"
        )

    print(f"Model:      {MODEL_ID}")
    print(f"Dataset:    {DATASET_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Epochs:     {EPOCHS}")
    print(f"LoRA r:     {LORA_R}, alpha: {LORA_ALPHA}")

    # -----------------------------------------------------------------------
    # 1. Quantization config (QLoRA: 4-bit NF4)
    # -----------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 is better than int4 for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # nested quantization saves ~0.4 bits/param
    )

    # -----------------------------------------------------------------------
    # 2. Load base model in 4-bit
    # -----------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",                  # automatically places layers on GPU
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)  # enables gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"       # required for causal LMs

    # -----------------------------------------------------------------------
    # 3. LoRA config
    # target_modules: which attention projections to adapt.
    # Gemma uses q_proj, k_proj, v_proj, o_proj — we target all four for
    # better quality at modest parameter cost.
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Typical output: trainable params: 8,388,608 || all params: 2,514,534,400 || 0.33%

    # -----------------------------------------------------------------------
    # 4. Dataset
    # -----------------------------------------------------------------------
    dataset = load_dataset_from_jsonl(DATASET_PATH)
    dataset = dataset.map(format_alpaca_prompt)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # -----------------------------------------------------------------------
    # 5. Training arguments
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,      # effective batch = BATCH_SIZE * 4
        warmup_ratio=0.03,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",                   # disable wandb/tensorboard in Vertex
        dataloader_pin_memory=False,
    )

    # -----------------------------------------------------------------------
    # 6. Trainer
    # DataCollatorForCompletionOnlyLM masks the instruction tokens in the loss —
    # the model only learns to predict the ### Response: portion, not the prompt.
    # This is critical: without masking, the model wastes capacity memorising
    # the fixed instruction format instead of learning the legal answers.
    # -----------------------------------------------------------------------
    response_template = "### Response:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=MAX_SEQ_LEN,
    )

    # -----------------------------------------------------------------------
    # 7. Train
    # -----------------------------------------------------------------------
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")

    # -----------------------------------------------------------------------
    # 8. Upload to GCS (Vertex AI expects outputs in GCS_OUTPUT_URI)
    # -----------------------------------------------------------------------
    if GCS_OUTPUT_URI:
        import subprocess
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", OUTPUT_DIR, GCS_OUTPUT_URI],
            check=True,
        )
        print(f"Adapter uploaded to {GCS_OUTPUT_URI}")


if __name__ == "__main__":
    run_training()
