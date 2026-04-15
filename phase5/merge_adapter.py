"""
Phase 5 — Merge LoRA adapter into the base model.

After local training, the adapter is stored as a small set of matrices (BA pairs)
in `data/gdpr-lora-adapter/`. At inference time you have two options:

  A. Load base + adapter separately (PEFT dynamic loading):
       model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
       model = PeftModel.from_pretrained(model, ADAPTER_DIR)
     Pro: adapter is swappable at runtime (useful for vLLM --lora-modules)
     Con: slight overhead on the first forward pass (adapter application)

  B. Merge adapter into base weights (this script):
       merged = model.merge_and_unload()
     Pro: zero inference overhead — LoRA matrices collapsed back into W
     Con: merged checkpoint is full-size (~1GB for 0.5B), adapter is no longer
          separable, can't hot-swap at runtime

Use this script when you want to deploy the fine-tuned model as a standalone
model (e.g., push to HuggingFace Hub, load with vLLM without --enable-lora).

Run:
    python -m phase5.merge_adapter
    python -m phase5.merge_adapter --adapter-dir path/to/adapter --output-dir path/to/merged
"""

import typer
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rich.console import Console

console = Console()

app = typer.Typer()

BASE_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "data/gdpr-lora-adapter"
OUTPUT_DIR  = "data/gdpr-merged-model"


@app.command()
def main(
    base_model: str  = typer.Option(BASE_MODEL,  help="HuggingFace model ID or local path"),
    adapter_dir: str = typer.Option(ADAPTER_DIR, help="Path to the LoRA adapter checkpoint"),
    output_dir: str  = typer.Option(OUTPUT_DIR,  help="Where to write the merged model"),
):
    """
    Merge a LoRA adapter into the base model weights.

    merge_and_unload() computes W' = W + BA for every adapted layer, then
    removes the PEFT wrapper. The result is a plain HuggingFace model that
    can be loaded with AutoModelForCausalLM.from_pretrained — no PEFT needed.
    """
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        console.print(f"[red]Adapter not found at {adapter_dir}. Run phase5.train_local first.[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading base model:[/cyan] {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,   # merge in full precision to preserve accuracy
        device_map="cpu",            # merge on CPU — avoids GPU memory pressure
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    console.print(f"[cyan]Loading adapter:[/cyan] {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    # merge_and_unload:
    #   1. For every LoRA layer: W_merged = W_frozen + (lora_B @ lora_A) * (alpha / r)
    #   2. Replaces the LoRA Linear with a standard nn.Linear using W_merged
    #   3. Returns a plain model with no PEFT overhead
    console.print("[cyan]Merging adapter weights into base model...[/cyan]")
    merged_model = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    console.print(f"\n[green]Merged model saved to {output_dir}[/green]")
    console.print(
        "[dim]Load with: AutoModelForCausalLM.from_pretrained"
        f"('{output_dir}') — no PEFT required[/dim]"
    )


if __name__ == "__main__":
    app()
