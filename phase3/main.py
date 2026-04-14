"""
Phase 3 — Interactive CLI for the traced GDPR analyst.

Key additions from Phase 2:
  - Tracing: optionally sends traces to LangSmith
  - Evaluation: run automated Vertex AI evaluation suites
  - Trace metrics: displays trace status if enabled

Usage:
    python -m phase3.main                          # Interactive with tracing
    python -m phase3.main --eval                   # Run Vertex AI evaluation
    python -m phase3.main -q "What is Article 5?"  # Single traced question
"""

import uuid
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from phase3.traced_agent import traced_ask, traced_stream_ask
from phase3.eval_runner import run_vertex_eval
from phase3.callbacks import active_backends
from config import settings

console = Console()
app = typer.Typer()

@app.command()
def main(
    question: str = typer.Option(None, "--question", "-q", help="Ask a single question and exit"),
    run_eval: bool = typer.Option(False, "--eval", "-e", help="Run the Vertex AI evaluation suite"),
    new_session: bool = typer.Option(False, "--new-session", "-n", help="Start a fresh conversation"),
) -> None:
    """Interactive GDPR Legal Analyst — Phase 3: Observability & Evaluation."""

    if run_eval:
        run_vertex_eval()
        return

    backends = active_backends()
    if backends:
        tracing_status = f"[green]ENABLED ({', '.join(backends)})[/green]"
    else:
        tracing_status = "[yellow]DISABLED — set LANGSMITH_TRACING=true or LANGFUSE_TRACING=true[/yellow]"

    console.print(Panel.fit(
        f"[bold cyan]GDPR Legal Analyst[/bold cyan]\n"
        f"[dim]Phase 3 — Observability & Evaluation[/dim]\n"
        f"[dim]Tracing: {tracing_status}[/dim]",
        border_style="cyan",
    ))

    session_id = str(uuid.uuid4()) if new_session else "phase3-default"

    if question:
        _ask_and_print(question, session_id)
        return

    console.print("[dim]Multi-turn conversation enabled. Traces will be sent to LangSmith if configured.[/dim]\n")

    while True:
        try:
            user_input = typer.prompt("\nYour question (or 'exit')")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ("exit", "quit", "q"):
            break

        if not user_input.strip():
            continue

        _ask_and_print(user_input, session_id)

def _ask_and_print(question: str, session_id: str) -> None:
    console.print(f"\n[bold]Q:[/bold] {question}\n")
    console.print("[bold cyan]A:[/bold cyan]")

    tools_used = []
    
    # Use the traced version of stream_ask
    for content, event_type in traced_stream_ask(question, thread_id=session_id):
        if event_type == "tool_used":
            tools_used.append(content)
            console.print(f"\n  [dim yellow]→ calling tool:[/dim yellow] [yellow]{content}[/yellow]")
        elif event_type == "answer":
            console.print(content, end="")
    
    console.print("\n")

    if tools_used:
        tool_text = Text()
        tool_text.append("Tools used: ", style="dim")
        tool_text.append(", ".join(tools_used), style="dim yellow")
        console.print(tool_text)

    backends = active_backends()
    if backends:
        console.print(f"[dim]Trace sent to: {', '.join(backends)}[/dim]")

    console.print()

if __name__ == "__main__":
    app()
