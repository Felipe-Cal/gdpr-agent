"""
Phase 4 — CLI entry point.

Usage:
    # Run the agent against the self-hosted vLLM endpoint
    python -m phase4.main

    # Specify a custom endpoint (e.g., GKE LoadBalancer IP)
    python -m phase4.main --endpoint http://34.90.x.x

    # Show the cost benchmark only (no GKE needed)
    python -m phase4.main --benchmark
"""

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        help="vLLM endpoint URL. Defaults to VLLM_ENDPOINT in .env (http://localhost:8000).",
    ),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help="Print cost comparison table and exit. No GKE connection needed.",
    ),
):
    if benchmark:
        from phase4.benchmarks import run_benchmark
        run_benchmark()
        return

    from phase4.serving import ask_vllm
    from config import settings

    url = endpoint or settings.vllm_endpoint

    console.print(Panel(
        f"[bold cyan]GDPR Agent — Self-hosted (vLLM)[/bold cyan]\n"
        f"[dim]Endpoint: {url}[/dim]\n"
        f"[dim]Model:    {settings.vllm_model}[/dim]\n\n"
        "Type [bold]exit[/bold] to quit.",
        title="Phase 4",
    ))

    while True:
        try:
            question = console.input("\n[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        with console.status("[cyan]Thinking...[/cyan]"):
            try:
                answer, tools_used = ask_vllm(question, endpoint_url=url)
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                console.print(
                    "[dim]Make sure the vLLM server is running and reachable at "
                    f"{url}[/dim]"
                )
                continue

        if tools_used:
            console.print(f"[dim]Tools used: {', '.join(tools_used)}[/dim]")
        console.print(f"\n[bold blue]Agent:[/bold blue] {answer}")

    console.print("\n[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    app()
