"""
Phase 5 — CLI entry point.

Commands:
    python -m phase5.main dataset          # generate training data locally (free)
    python -m phase5.main train            # submit Vertex AI Training job (⚠️ ~$0.40)
    python -m phase5.main evaluate         # compare base vs fine-tuned models
"""

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def dataset(
    output: str = typer.Option("data/gdpr_finetune.jsonl", help="Output JSONL path"),
):
    """Generate the synthetic GDPR fine-tuning dataset (free, no GCP calls)."""
    from phase5.dataset import build_dataset
    examples = build_dataset(output)
    console.print(f"[green]✓[/green] {len(examples)} training examples → {output}")

    # Preview
    console.print("\n[dim]First example:[/dim]")
    import json
    with open(output) as f:
        first = json.loads(f.readline())
    console.print(f"[cyan]Instruction:[/cyan] {first['instruction']}")
    console.print(f"[cyan]Output:[/cyan] {first['output'][:200]}...")


@app.command()
def train(
    bucket: str = typer.Option(..., help="GCS bucket name (without gs://) for data + output"),
    epochs: int = typer.Option(3, help="Training epochs"),
    async_: bool = typer.Option(False, "--async", help="Don't wait for job completion"),
):
    """Submit a LoRA fine-tuning job to Vertex AI Training. ⚠️ Costs ~$0.40."""
    console.print("[yellow]⚠️  This will submit a Vertex AI Training job.[/yellow]")
    console.print("[yellow]   Estimated cost: ~$0.40 (n1-standard-8 + T4, ~45 min)[/yellow]")
    confirm = typer.confirm("Proceed?")
    if not confirm:
        raise typer.Exit()

    from phase5.vertex_job import submit_training_job, upload_dataset_to_gcs
    from phase5.dataset import build_dataset

    console.print("Generating dataset...")
    build_dataset("data/gdpr_finetune.jsonl")

    dataset_gcs = f"gs://{bucket}/phase5/data/gdpr_finetune.jsonl"
    output_gcs = f"gs://{bucket}/phase5/adapter"
    upload_dataset_to_gcs("data/gdpr_finetune.jsonl", dataset_gcs)

    console.print(f"Submitting job... output → {output_gcs}")
    submit_training_job(dataset_gcs, output_gcs, epochs=epochs, sync=not async_)
    console.print(f"[green]✓[/green] Done. Adapter at {output_gcs}")


@app.command()
def evaluate(
    base_url: str = typer.Argument(..., help="Base model vLLM endpoint URL"),
    finetuned_url: str = typer.Argument(..., help="Fine-tuned model vLLM endpoint URL"),
):
    """Compare base vs fine-tuned model on the GDPR golden eval dataset."""
    from phase5.evaluate import main as run_eval
    run_eval(base_url=base_url, finetuned_url=finetuned_url)


if __name__ == "__main__":
    app()
