"""
Phase 4 — Cost benchmarking: Gemini API vs self-hosted vLLM on GKE.

This module calculates and compares the per-query cost of:
  1. Gemini 2.0 Flash Lite (managed API, pay-per-token)
  2. Mistral-7B-AWQ on a GKE T4 GPU node (self-hosted, pay-per-hour)

Run:
    python -m phase4.benchmarks

The key insight: self-hosted is cheaper per query only above a break-even
request volume.  Below that volume, the managed API wins on cost.
This is a classic build-vs-buy trade-off and a common interview question.
"""

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Pricing constants (as of early 2025, europe-west4)
# ---------------------------------------------------------------------------

# Gemini 2.0 Flash Lite — cheapest managed Gemini model
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
GEMINI_FLASH_LITE_INPUT_PER_1M = 0.075   # USD per 1M input tokens
GEMINI_FLASH_LITE_OUTPUT_PER_1M = 0.30   # USD per 1M output tokens

# GKE infrastructure (europe-west4, on-demand pricing)
# Source: https://cloud.google.com/compute/vm-instance-pricing
#         https://cloud.google.com/compute/gpus-pricing
GKE_CONTROL_PLANE_PER_HR = 0.10          # Standard GKE cluster management fee
GKE_N1_STANDARD_4_PER_HR = 0.1900       # n1-standard-4 VM (4 vCPU, 15GB RAM)
GKE_T4_GPU_PER_HR = 0.3500              # NVIDIA T4 GPU
GKE_TOTAL_PER_HR = (
    GKE_CONTROL_PLANE_PER_HR
    + GKE_N1_STANDARD_4_PER_HR
    + GKE_T4_GPU_PER_HR
)

# Typical GDPR agent query profile
AVG_INPUT_TOKENS = 800    # system prompt + question + retrieved context
AVG_OUTPUT_TOKENS = 400   # typical legal explanation

# vLLM throughput on T4 with Mistral-7B-AWQ (conservative estimate)
# Real benchmarks: ~40-80 tokens/sec for 7B AWQ on T4
VLLM_TOKENS_PER_SEC = 50   # tokens/sec (output)
VLLM_SECS_PER_QUERY = AVG_OUTPUT_TOKENS / VLLM_TOKENS_PER_SEC  # ~8 seconds
VLLM_QUERIES_PER_HOUR = 3600 / VLLM_SECS_PER_QUERY              # ~450 queries/hr


def gemini_cost_per_query() -> float:
    """Cost of one query to Gemini 2.0 Flash Lite."""
    input_cost = (AVG_INPUT_TOKENS / 1_000_000) * GEMINI_FLASH_LITE_INPUT_PER_1M
    output_cost = (AVG_OUTPUT_TOKENS / 1_000_000) * GEMINI_FLASH_LITE_OUTPUT_PER_1M
    return input_cost + output_cost


def gke_cost_per_query(queries_per_hour: float = VLLM_QUERIES_PER_HOUR) -> float:
    """
    Cost of one query when running vLLM on GKE.

    This is the hourly infrastructure cost divided by the number of queries
    served in that hour. High throughput = lower per-query cost.
    """
    return GKE_TOTAL_PER_HR / queries_per_hour


def break_even_queries_per_month() -> float:
    """
    Monthly query volume at which GKE becomes cheaper than Gemini API.

    Below this volume: pay the managed API, don't run a cluster.
    Above this volume: self-host to save money.
    """
    # GKE monthly cost (24/7 operation)
    monthly_gke = GKE_TOTAL_PER_HR * 24 * 30

    # Gemini cost per query
    gemini_per_q = gemini_cost_per_query()

    # Break-even: monthly_gke = break_even_queries * gemini_per_q
    return monthly_gke / gemini_per_q


def run_benchmark():
    """Prints the full cost comparison."""
    g_cost = gemini_cost_per_query()
    gke_cost = gke_cost_per_query()
    be = break_even_queries_per_month()

    # -----------------------------------------------------------------------
    # Table 1: Per-query cost at different traffic levels
    # -----------------------------------------------------------------------
    t1 = Table(title="Cost per Query — Gemini API vs GKE/vLLM", box=box.ROUNDED)
    t1.add_column("Traffic Level", style="cyan")
    t1.add_column("Queries/Month", justify="right")
    t1.add_column("Gemini Flash Lite", justify="right", style="yellow")
    t1.add_column("GKE + Mistral-7B-AWQ", justify="right", style="green")
    t1.add_column("Cheaper", justify="center")

    traffic_levels = [
        ("Low (dev/test)", 1_000),
        ("Medium (small team)", 10_000),
        ("High (production)", 100_000),
        ("Very high", 1_000_000),
    ]

    for label, monthly_queries in traffic_levels:
        total_gemini = g_cost * monthly_queries
        # GKE cost is fixed per month regardless of query volume (24/7 cluster)
        total_gke = GKE_TOTAL_PER_HR * 24 * 30
        cheaper = "Gemini" if total_gemini < total_gke else "GKE"
        t1.add_row(
            label,
            f"{monthly_queries:,}",
            f"${total_gemini:.2f}",
            f"${total_gke:.2f}",
            f"{'[yellow]Gemini[/yellow]' if cheaper == 'Gemini' else '[green]GKE[/green]'}",
        )

    console.print(t1)

    # -----------------------------------------------------------------------
    # Table 2: Infrastructure breakdown
    # -----------------------------------------------------------------------
    t2 = Table(title="GKE Infrastructure Cost Breakdown", box=box.ROUNDED)
    t2.add_column("Component", style="cyan")
    t2.add_column("$/hr", justify="right")
    t2.add_column("$/month (24/7)", justify="right")

    t2.add_row("GKE control plane", f"${GKE_CONTROL_PLANE_PER_HR:.2f}", f"${GKE_CONTROL_PLANE_PER_HR*24*30:.0f}")
    t2.add_row("n1-standard-4 VM", f"${GKE_N1_STANDARD_4_PER_HR:.2f}", f"${GKE_N1_STANDARD_4_PER_HR*24*30:.0f}")
    t2.add_row("NVIDIA T4 GPU", f"${GKE_T4_GPU_PER_HR:.2f}", f"${GKE_T4_GPU_PER_HR*24*30:.0f}")
    t2.add_row("[bold]Total[/bold]", f"[bold]${GKE_TOTAL_PER_HR:.2f}[/bold]", f"[bold]${GKE_TOTAL_PER_HR*24*30:.0f}[/bold]")

    console.print(t2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    console.print(f"\n[bold]Break-even point:[/bold] [yellow]{be:,.0f} queries/month[/yellow]")
    console.print(f"Below that: Gemini API (${g_cost*1000:.3f} per 1,000 queries)")
    console.print(f"Above that: GKE saves money\n")
    console.print(
        "[dim]Assumptions: 800 input tokens, 400 output tokens per query. "
        f"vLLM at {VLLM_TOKENS_PER_SEC} tok/sec on T4. "
        "Prices: europe-west4, on-demand, early 2025.[/dim]"
    )
    console.print(
        "\n[dim]Note: committed-use discounts (1yr/3yr) reduce GKE costs by 37-55%. "
        "Spot GPU nodes reduce costs by ~60-70% at the cost of preemptibility.[/dim]"
    )


if __name__ == "__main__":
    run_benchmark()
