"""
Phase 5 — Fine-tuned model evaluation.

Compares the base model against the fine-tuned (LoRA-adapted) model using
the same Vertex AI evaluation pipeline from Phase 3.

The fine-tuned adapter is served via vLLM (Phase 4 infrastructure) using
vLLM's --lora-modules flag, which loads the adapter on top of the base model
without merging weights — allowing A/B testing at runtime.

Usage:
    # Compare base vs fine-tuned (both must be served by vLLM)
    python -m phase5.evaluate \
        --base-url http://<IP-base>/v1 \
        --finetuned-url http://<IP-finetuned>/v1
"""

import pandas as pd
import typer
import vertexai
from rich.console import Console
from rich.table import Table

from config import settings
from phase3.eval_dataset import EVAL_DATASET
from phase3.eval_runner import groundedness_metric

console = Console()
app = typer.Typer()


def _collect_responses(endpoint_url: str, model_name: str) -> list[dict]:
    """Runs the eval dataset through a vLLM endpoint and collects responses."""
    from phase4.serving import ask_vllm

    results = []
    for entry in EVAL_DATASET:
        question = entry["question"]
        reference = entry["reference"]
        answer, _ = ask_vllm(question, endpoint_url=endpoint_url)
        results.append({
            "prompt": question,
            "response": answer,
            "reference": reference,
            "context": reference,   # use reference as context proxy for offline eval
            "model": model_name,
        })
        console.print(f"[dim]  [{model_name}] {question[:60]}...[/dim]")
    return results


def _run_eval(results: list[dict]) -> dict:
    """Runs Vertex AI evaluation on a set of results. Returns summary metrics."""
    from vertexai.evaluation import EvalTask

    eval_task = EvalTask(
        dataset=pd.DataFrame(results),
        metrics=[groundedness_metric, "coherence", "fluency", "safety"],
        experiment="gdpr-agent-eval",
    )
    result = eval_task.evaluate(
        experiment_run_name=f"phase5-{results[0]['model']}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
    )
    return result.summary_metrics


@app.command()
def main(
    base_url: str = typer.Option(
        None, "--base-url", help="Base model vLLM endpoint (e.g. http://34.x.x.x)"
    ),
    finetuned_url: str = typer.Option(
        None, "--finetuned-url", help="Fine-tuned model vLLM endpoint"
    ),
):
    """Compare base vs fine-tuned model on the GDPR golden dataset."""
    if not base_url or not finetuned_url:
        console.print("[red]Both --base-url and --finetuned-url are required.[/red]")
        raise typer.Exit(1)

    vertexai.init(project=settings.gcp_project_id, location=settings.llm_region)

    console.print("[bold cyan]Phase 5 — Fine-tuning Evaluation[/bold cyan]")
    console.print("[dim]Collecting base model responses...[/dim]")
    base_results = _collect_responses(base_url, "base")

    console.print("[dim]Collecting fine-tuned model responses...[/dim]")
    ft_results = _collect_responses(finetuned_url, "finetuned")

    console.print("[dim]Running Vertex AI evaluation...[/dim]")
    base_metrics = _run_eval(base_results)
    ft_metrics = _run_eval(ft_results)

    # -----------------------------------------------------------------------
    # Side-by-side comparison table
    # -----------------------------------------------------------------------
    table = Table(title="Base vs Fine-tuned Model — Avg Scores")
    table.add_column("Metric", style="cyan")
    table.add_column("Base Model", justify="right")
    table.add_column("Fine-tuned (LoRA)", justify="right")
    table.add_column("Delta", justify="right")

    all_metrics = set(base_metrics) | set(ft_metrics)
    for metric in sorted(all_metrics):
        base_val = base_metrics.get(metric, float("nan"))
        ft_val = ft_metrics.get(metric, float("nan"))
        try:
            delta = ft_val - base_val
            delta_str = f"[green]+{delta:.2f}[/green]" if delta > 0 else f"[red]{delta:.2f}[/red]"
            table.add_row(metric, f"{base_val:.2f}", f"{ft_val:.2f}", delta_str)
        except TypeError:
            table.add_row(metric, str(base_val), str(ft_val), "—")

    console.print(table)


if __name__ == "__main__":
    app()
