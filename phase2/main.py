"""
Phase 2 — Interactive CLI for the agentic GDPR analyst.

Supports two agent backends:
  --agent langgraph  (default) — LangGraph ReAct graph with explicit state
  --agent adk        — Google ADK declarative agent

Key differences from Phase 1 main.py:
  - Multi-turn conversation: questions build on previous ones within a session
  - Tool visibility: shows which tools the agent called for each answer
  - Agent selection: compare LangGraph vs ADK side by side

Usage:
    python -m phase2.main                          # LangGraph, interactive
    python -m phase2.main --agent adk              # ADK, interactive
    python -m phase2.main -q "What is Article 6?"  # single question
"""

import uuid

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from phase2.graph import ask, stream_ask

console = Console()
app = typer.Typer()

EXAMPLE_QUESTIONS = [
    "What are the lawful bases for processing personal data under GDPR?",
    "When is a DPIA required and what must it contain?",
    "What are the conditions for valid consent?",
    "Does my company need a Data Protection Officer?",
    "What must I do within 72 hours of a data breach?",
    "What is privacy by design and how do I implement it?",
]


@app.command()
def main(
    question: str = typer.Option(None, "--question", "-q", help="Ask a single question and exit"),
    agent: str = typer.Option("langgraph", "--agent", "-a", help="Agent backend: 'langgraph' or 'adk'"),
    new_session: bool = typer.Option(False, "--new-session", "-n", help="Start a fresh conversation (new session ID)"),
) -> None:
    """Interactive GDPR Legal Analyst — Phase 2: Agentic with tools."""

    agent = agent.lower()
    if agent not in ("langgraph", "adk"):
        console.print("[red]--agent must be 'langgraph' or 'adk'[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]GDPR Legal Analyst[/bold cyan]\n"
        f"[dim]Phase 2 — Agentic ({agent}) · Tools: search_gdpr_documents, web_search, get_gdpr_article[/dim]",
        border_style="cyan",
    ))

    # Each session has a stable ID so conversation history is preserved across questions.
    # Pass --new-session to reset.
    session_id = str(uuid.uuid4()) if new_session else "phase2-default"

    if question:
        _ask_and_print(question, agent, session_id)
        return

    console.print("[dim]Multi-turn conversation enabled — follow-up questions use previous context.[/dim]")
    console.print("[dim]Use --new-session to reset conversation history.[/dim]\n")
    console.print("[dim]Example questions:[/dim]")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        console.print(f"  [dim]{i}.[/dim] {q}")
    console.print()

    while True:
        try:
            user_input = typer.prompt("\nYour question (or 'exit')")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ("exit", "quit", "q"):
            break

        if not user_input.strip():
            continue

        _ask_and_print(user_input, agent, session_id)


def _ask_and_print(question: str, agent: str, session_id: str) -> None:
    console.print(f"\n[bold]Q:[/bold] {question}\n")

    if agent == "langgraph":
        _langgraph_ask(question, session_id)
    else:
        _adk_ask(question, session_id)


def _langgraph_ask(question: str, session_id: str) -> None:
    """Stream the LangGraph agent response, showing tool calls as they happen."""
    tools_used = []
    answer_parts = []

    console.print("[bold cyan]A:[/bold cyan]")

    for content, event_type in stream_ask(question, thread_id=session_id):
        if event_type == "tool_used":
            tools_used.append(content)
            console.print(f"\n  [dim yellow]→ calling tool:[/dim yellow] [yellow]{content}[/yellow]")
        elif event_type == "answer":
            console.print(content, end="")
            answer_parts.append(content)

    console.print("\n")

    if tools_used:
        tool_text = Text()
        tool_text.append("Tools used: ", style="dim")
        tool_text.append(", ".join(tools_used), style="dim yellow")
        console.print(tool_text)

    console.print()


def _adk_ask(question: str, session_id: str) -> None:
    """Run the ADK agent and print the response."""
    console.print("[dim]Running ADK agent...[/dim]")

    try:
        from phase2.adk_agent import ask_adk
        answer = ask_adk(question, session_id=session_id)
        console.print("[bold cyan]A:[/bold cyan]")
        console.print(Markdown(answer))
        console.print()
    except ImportError as e:
        console.print(f"[red]ADK not available: {e}[/red]")
        console.print("[dim]Install with: pip install google-adk[/dim]")


if __name__ == "__main__":
    app()
