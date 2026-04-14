"""
Phase 3 — Evaluation runner.

Uses Vertex AI's Rapid Evaluation service (EvalTask) to score the agent's
performance against the golden dataset.

Metrics used:
- groundedness: is the answer supported by the retrieved context? (custom PointwiseMetric)
- coherence:    is the answer logically structured?
- fluency:      is the answer grammatically correct and readable?
- safety:       does the answer avoid harmful content?

Why a custom PointwiseMetric for groundedness?
  The built-in string "groundedness" uses a computation-based scorer that returns 0
  in BYOR mode (Bring Your Own Response) when no inference model is specified.
  A PointwiseMetric with an explicit LLM-as-judge prompt template lets the Vertex AI
  Evaluation Service use its default judge (Gemini) to semantically compare the
  agent's response against the retrieved context — which is the correct approach.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import vertexai
from vertexai.evaluation import EvalTask, PointwiseMetric

from config import settings
from phase3.eval_dataset import EVAL_DATASET
from phase2.graph import ask

console = Console()

# ---------------------------------------------------------------------------
# Custom groundedness metric
# ---------------------------------------------------------------------------
# The built-in "groundedness" string metric uses a computation-based scorer
# that does not work reliably in BYOR mode.  We define an explicit LLM-as-judge
# prompt that references the {context} column so Vertex AI's evaluation service
# can compare the response against the retrieved GDPR text semantically.
# ---------------------------------------------------------------------------
_GROUNDEDNESS_TEMPLATE = """\
You are an expert evaluator assessing a GDPR legal assistant's response.

RETRIEVED CONTEXT (the text the assistant retrieved from its knowledge base):
{context}

USER QUESTION:
{prompt}

ASSISTANT RESPONSE:
{response}

Task: Rate how well the ASSISTANT RESPONSE is GROUNDED in the RETRIEVED CONTEXT.
A grounded response only makes factual claims that are directly supported by the context above.
It is acceptable to paraphrase or summarise, but the claims must be traceable back to the context.

Scoring rubric:
5 - Excellent: every claim in the response is supported by the context
4 - Good: nearly all claims are supported; minor additions are reasonable inferences
3 - Adequate: most claims are supported; some notable unsupported additions
2 - Poor: significant claims are not supported by the context
1 - Failing: the response contradicts the context or largely ignores it

Respond with a single integer (1, 2, 3, 4, or 5). Do not add any explanation."""

groundedness_metric = PointwiseMetric(
    metric="groundedness",
    metric_prompt_template=_GROUNDEDNESS_TEMPLATE,
)


def run_vertex_eval():
    """
    Runs the full evaluation suite using Vertex AI Rapid Evaluation.
    Processes each question in the golden dataset through the Phase 2 agent,
    then uses LLM-as-judge scoring to evaluate the responses.
    """
    vertexai.init(project=settings.gcp_project_id, location=settings.llm_region)

    console.print("[bold cyan]Starting Vertex AI Evaluation[/bold cyan]")
    console.print("[dim]Metrics: groundedness (custom), coherence, fluency, safety[/dim]\n")

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

            # Run the agent — returns (answer, tools_used, context).
            # context = raw text from search_gdpr_documents or get_gdpr_article.
            # Passing it as "context" lets the groundedness judge verify that the
            # answer is faithful to what was actually retrieved.
            prediction, tools_used, context = ask(question, thread_id=f"eval-{question[:20]}")

            console.print(f"[dim]  tools={tools_used}  context_len={len(context)}[/dim]")

            eval_results.append({
                "prompt": question,
                "response": prediction,
                "reference": reference,
                "context": context,
            })
            progress.advance(task)

    # Build the EvalTask with our custom groundedness metric alongside the
    # built-in string metrics for coherence, fluency, and safety.
    eval_task = EvalTask(
        dataset=pd.DataFrame(eval_results),
        metrics=[
            groundedness_metric,  # custom PointwiseMetric — uses explicit LLM judge
            "coherence",
            "fluency",
            "safety",
        ],
        experiment="gdpr-agent-eval",
    )

    # BYOR mode: no model= parameter — responses are already in the dataset.
    # Vertex AI's evaluation service uses its default judge model (Gemini) to
    # score each metric using the templates defined above.
    result = eval_task.evaluate(
        experiment_run_name=f"gdpr-eval-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}",
    )

    _display_results(result)
    return result


def _display_results(result):
    """Prints the evaluation summary in a formatted table."""
    table = Table(title="Evaluation Metrics (Avg Scores)")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")

    summary = result.summary_metrics
    for metric, score in summary.items():
        try:
            table.add_row(metric, f"{score:.2f}")
        except (TypeError, ValueError):
            table.add_row(metric, str(score))

    console.print(table)
    console.print("\n[dim]Note: groundedness/coherence/fluency are 1–5. safety is 0/1.[/dim]")


if __name__ == "__main__":
    # To run this standalone: python -m phase3.eval_runner
    run_vertex_eval()
