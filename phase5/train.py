"""
Phase 5 — QLoRA fine-tuning script for Vertex AI (GPU).

Designed to run inside a Vertex AI Training custom container job, but also
works locally on any CUDA GPU.

Architecture:
    Base model:  Qwen/Qwen2.5-1.5B-Instruct  (1.5B params, fits on a T4)
    Method:      QLoRA — quantize the base model to 4-bit (bitsandbytes NF4),
                 then train only the LoRA adapter layers (r=16, ~6M params).
    Framework:   HuggingFace PEFT + TRL 1.x (SFTConfig + SFTTrainer)

Why QLoRA over full fine-tuning?
    Full fine-tuning a 1.5B model requires ~12GB VRAM in FP16 + optimizer states.
    QLoRA quantizes the frozen base to 4-bit (~1.5GB) and only trains the
    small LoRA adapters (~6M params), fitting on a single T4 (16GB).
    Quality degradation vs full fine-tuning: <1% on standard benchmarks.

TRL 1.x API changes vs 0.8:
    - SFTConfig replaces TrainingArguments for SFT jobs
    - max_length and dataset_text_field belong in SFTConfig, not SFTTrainer
    - evaluation_strategy → eval_strategy
    - SFTTrainer(processing_class=tokenizer) replaces tokenizer= kwarg
    - DataCollatorForCompletionOnlyLM was removed in TRL 1.0

Environment variables (set by Vertex AI Training or locally):
    MODEL_ID:        HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)
    DATASET_PATH:    Path to the JSONL training file (local or gs://)
    OUTPUT_DIR:      Where to write the adapter checkpoint
    GCS_OUTPUT_URI:  GCS path to upload the checkpoint after training
    EPOCHS:          Number of training epochs (default: 3)
    BATCH_SIZE:      Per-device batch size (default: 4)
"""

import os
import json
from pathlib import Path

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    HAS_TRAIN_DEPS = True
except ImportError:
    HAS_TRAIN_DEPS = False


# ---------------------------------------------------------------------------
# Configuration — read from environment (Vertex AI injects these)
# ---------------------------------------------------------------------------
MODEL_ID      = os.getenv("MODEL_ID",       "Qwen/Qwen2.5-1.5B-Instruct")
DATASET_PATH  = os.getenv("DATASET_PATH",   "data/gdpr_finetune.jsonl")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR",     "/tmp/gdpr-lora-adapter")
GCS_OUTPUT_URI = os.getenv("GCS_OUTPUT_URI", "")   # e.g. gs://my-bucket/phase5/adapter
EPOCHS        = int(os.getenv("EPOCHS",     "3"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "4"))
LORA_R        = int(os.getenv("LORA_R",     "16"))
LORA_ALPHA    = int(os.getenv("LORA_ALPHA", "32"))
MAX_SEQ_LEN   = int(os.getenv("MAX_SEQ_LEN", "1024"))


def load_dataset_from_jsonl(path: str) -> "Dataset":
    """Loads the Alpaca-format JSONL file. Supports local paths and gs:// URIs."""
    if path.startswith("gs://"):
        import subprocess
        local_path = "/tmp/gdpr_finetune.jsonl"
        subprocess.run(["gsutil", "cp", path, local_path], check=True)
        path = local_path

    records = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            records.append({
                "text": (
                    f"### Instruction:\n{ex['instruction']}\n\n"
                    f"### Response:\n{ex['output']}"
                )
            })
    return Dataset.from_list(records)


def run_training():
    """Main QLoRA training loop for Vertex AI (GPU)."""
    if not HAS_TRAIN_DEPS:
        raise ImportError(
            "Training dependencies not installed. Run: pip install -e '.[train]'"
        )

    print(f"Model:      {MODEL_ID}")
    print(f"Dataset:    {DATASET_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Epochs:     {EPOCHS}  LoRA r={LORA_R}  alpha={LORA_ALPHA}")

    # -----------------------------------------------------------------------
    # 1. Quantization config (QLoRA: 4-bit NF4)
    # -----------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 preserves more information than int4
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
    # prepare_model_for_kbit_training: enables gradient checkpointing and
    # casts layer norms to float32 so gradients don't underflow in 4-bit.
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"       # required for causal LM training

    # -----------------------------------------------------------------------
    # 3. LoRA config
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        # Qwen uses q_proj, k_proj, v_proj, o_proj — targeting all four
        # attention projections gives better quality than q_proj + v_proj only.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Typical: trainable params: 6,291,456 || all params: 1,549,000,000 || 0.41%

    # -----------------------------------------------------------------------
    # 4. Dataset
    # -----------------------------------------------------------------------
    raw = load_dataset_from_jsonl(DATASET_PATH)
    dataset = raw.train_test_split(test_size=0.1, seed=42)

    # -----------------------------------------------------------------------
    # 5. SFTConfig (TRL 1.x — replaces TrainingArguments for SFT jobs)
    #
    # Key TRL 1.x changes from 0.8:
    #   - SFTConfig: inherits from TrainingArguments + SFT-specific fields
    #   - max_length / dataset_text_field: go here, not in SFTTrainer
    #   - eval_strategy: replaces deprecated evaluation_strategy
    #   - DataCollatorForCompletionOnlyLM: removed — SFT loss is applied to
    #     the full sequence by default (instruction + response)
    # -----------------------------------------------------------------------
    training_args = SFTConfig(
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
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",                   # disable wandb/tensorboard in Vertex
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
    )

    # -----------------------------------------------------------------------
    # 6. Trainer (TRL 1.x: processing_class= replaces tokenizer=)
    # -----------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
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
