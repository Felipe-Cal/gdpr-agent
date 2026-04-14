"""
Phase 5 — Local LoRA fine-tuning (CPU / Apple MPS).

Compatible with TRL 1.x (SFTConfig replaces TrainingArguments for SFT).

Runs on any machine without a CUDA GPU:
  - Uses Qwen/Qwen2.5-0.5B-Instruct (0.5B params, ~1GB RAM)
  - Skips 4-bit quantization (bitsandbytes requires CUDA)
  - Uses LoRA in FP32 — slower but universally compatible
  - Auto-detects Apple MPS (M1/M2/M3) for ~3-4x speedup over CPU

Expected runtime:
  - Apple M-series (MPS): ~10-15 min
  - Intel CPU:            ~30-45 min

Run:
    python -m phase5.train_local
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from rich.console import Console

console = Console()

MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET    = "data/gdpr_finetune.jsonl"
OUTPUT_DIR = "data/gdpr-lora-adapter"
EPOCHS     = 2
BATCH_SIZE = 1
GRAD_ACCUM = 8
LORA_R     = 8
LORA_ALPHA = 16
MAX_SEQ_LEN = 512


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset() -> Dataset:
    records = []
    with open(DATASET) as f:
        for line in f:
            ex = json.loads(line)
            # Format as a single text field for SFTTrainer
            records.append({
                "text": (
                    f"### Instruction:\n{ex['instruction']}\n\n"
                    f"### Response:\n{ex['output']}"
                )
            })
    return Dataset.from_list(records)


def run():
    device = _pick_device()
    console.print("[bold cyan]Phase 5 — Local LoRA Fine-tuning[/bold cyan]")
    console.print(f"[dim]Model:  {MODEL_ID}[/dim]")
    console.print(f"[dim]Device: {device.upper()}[/dim]")
    console.print(f"[dim]Output: {OUTPUT_DIR}[/dim]\n")

    # -----------------------------------------------------------------------
    # 1. Load model in FP32 (no bitsandbytes on CPU/MPS)
    # -----------------------------------------------------------------------
    console.print("[cyan]Loading model...[/cyan]")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -----------------------------------------------------------------------
    # 2. LoRA config
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # 3. Dataset
    # -----------------------------------------------------------------------
    console.print("[cyan]Loading dataset...[/cyan]")
    dataset = _load_dataset().train_test_split(test_size=0.1, seed=42)

    # -----------------------------------------------------------------------
    # 4. SFTConfig (TRL 1.x replaces TrainingArguments for SFT jobs)
    # -----------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        warmup_steps=10,           # replaces deprecated warmup_ratio
    )

    # -----------------------------------------------------------------------
    # 5. Trainer
    # -----------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    # -----------------------------------------------------------------------
    # 6. Train
    # -----------------------------------------------------------------------
    console.print("[cyan]Training...[/cyan]")
    trainer.train()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    console.print(f"\n[green]✓ Adapter saved to {OUTPUT_DIR}[/green]")


if __name__ == "__main__":
    run()
