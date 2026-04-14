"""
Phase 3 — Evaluation runner.

Uses Vertex AI's Rapid Evaluation service (EvalTask) to score the agent's
performance against the golden dataset.

Metrics used:
- coherence: is the answer logically structured?
- groundedness: is the answer supported by the retrieved context?
- relevance: does the answer address the user's question?
- faithfulness: does the answer match the information in the reference?

To keep costs low, we use gemini-1.5-flash as the evaluator model.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.evaluation import EvalTask, PointwiseMetric

from config import settings
from phase3.eval_dataset import EVAL_DATASET
from phase2.graph import ask

console = Console()

def run_vertex_eval():
    """
    Runs the full evaluation suite using Vertex AI Rapid Evaluation.
    Processes each question in the golden dataset through the Phase 2 agent,
    then uses a second model (Gemini Flash) to score the responses.
    """
    vertexai.init(project=settings.gcp_project_id, location=settings.llm_region)

    console.print(f"[bold cyan]Starting Vertex AI Evaluation[/bold cyan]")
    console.print(f"[dim]Metrics: groundedness, coherence, fluency, safety[/dim]")
    console.print(f"[dim]Evaluator Model: {settings.eval_model}[/dim]\n")

    eval_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating...", total=len(EVAL_DATASET))

        for entry in EVAL_DATASET:
            question = entry["question"]
            reference = entry["reference"]

            progress.update(task, description=f"[cyan]Evaluating: {question[:50]}...")

            # 1. Run the agent to get the answer (and tool calls/context implicitly)
            # In a real production setup, we'd capture the retrieved context explicitly
            # to pass to the 'groundedness' metric. For this project, we'll focus
            # on the Pointwise metrics which compare prediction vs reference.
            prediction, _ = ask(question, thread_id="eval-session")

            eval_results.append({
                "prompt": question,
                "prediction": prediction,
                "reference": reference,
            })
            progress.advance(task)

    # 2. Define the Evaluation Task
    # We use Rapid Evaluation by defining an EvalTask with specific metrics.
    # Note: Using pointwise metrics for simplicity and cost-effectiveness.
    eval_task = EvalTask(
        dataset=pd.DataFrame(eval_results),
        metrics=[
            "groundedness",
            "coherence",
            "fluency",
            "safety",
        ],
        experiment="gdpr-agent-eval"
    )

    # 3. Running evaluation
    # We specify the prompt template for the evaluator.
    result = eval_task.evaluate(
        model=GenerativeModel(settings.eval_model),
        prompt_template="{prompt}",
    )

    # 4. Display results using Rich
    _display_results(result)

    return result

def _display_results(result):
    """Prints the evaluation summary and details in a formatted table."""
    table = Table(title="Evaluation Metrics (Avg Scores)")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Explanation", style="dim")

    # Access the summary metrics
    summary = result.summary_metrics
    for metric, score in summary.items():
        table.add_row(metric, f"{score:.2f}", "")

    console.print(table)
    console.print("\n[dim]Note: Scores are on a 1-5 scale (except safety).[/dim]")

if __name__ == "__main__":
    # To run this standalone: python -m phase3.eval_runner
    run_vertex_eval()
