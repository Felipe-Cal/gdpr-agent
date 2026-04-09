"""
Phase 1 — Interactive GDPR analyst CLI.

Usage:
    python -m phase1.main
    python -m phase1.main --question "What are the requirements for a valid consent?"
"""

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from phase1.chain import get_chain

console = Console()
app = typer.Typer()

EXAMPLE_QUESTIONS = [
    "What are the lawful bases for processing personal data under GDPR?",
    "What must a privacy notice include?",
    "When is a Data Protection Impact Assessment (DPIA) required?",
    "What are the rights of data subjects under GDPR?",
    "What constitutes a personal data breach and what are the notification obligations?",
]


@app.command()
def main(
    question: str = typer.Option(None, "--question", "-q", help="Ask a single question and exit"),
    stream: bool = typer.Option(True, help="Stream the response token by token"),
) -> None:
    """Interactive GDPR Legal Analyst powered by Vertex AI + Gemini."""

    console.print(Panel.fit(
        "[bold cyan]GDPR Legal Analyst[/bold cyan]\n"
        "[dim]Phase 1 — RAG with Vertex AI Vector Search + Gemini[/dim]",
        border_style="cyan",
    ))

    console.print("[dim]Initialising Vertex AI connection...[/dim]")
    chain = get_chain()
    console.print("[green]Ready.[/green]\n")

    if question:
        _ask(chain, question, stream)
        return

    # Interactive loop
    console.print("[dim]Example questions to try:[/dim]")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        console.print(f"  [dim]{i}.[/dim] {q}")
    console.print()

    while True:
        try:
            question = typer.prompt("\nYour question (or 'exit')")
        except (KeyboardInterrupt, EOFError):
            break

        if question.strip().lower() in ("exit", "quit", "q"):
            break

        if not question.strip():
            continue

        _ask(chain, question, stream)


def _ask(chain, question: str, stream: bool) -> None:
    console.print(f"\n[bold]Q:[/bold] {question}\n")
    console.print("[bold cyan]A:[/bold cyan]")

    if stream:
        answer_parts = []
        for chunk in chain.stream(question):
            console.print(chunk, end="")
            answer_parts.append(chunk)
        console.print("\n")
    else:
        answer = chain.invoke(question)
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    app()
