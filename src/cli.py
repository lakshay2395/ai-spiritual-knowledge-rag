import os
import sys
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.orchestrator import RAGOrchestrator
from src.retrieval.hybrid_retriever import HybridRetriever

console = Console()


def display_welcome():
    welcome_text = """
# 🙏 AI Spiritual Knowledge Engine
Search for wisdom across the Bhagavad Gita and the Holy Bible.
    """
    console.print(Markdown(welcome_text))
    console.print(
        "[italic yellow]Type 'exit' or 'quit' to leave the session.[/italic yellow]\n"
    )


def display_response(query: str, response: dict):
    # Display Answer
    console.print(
        Panel(
            Markdown(response["answer"]),
            title=f"[bold green]Answer to: {query}[/bold green]",
            border_style="green",
        )
    )

    # Display Sources
    if response.get("sources"):
        table = Table(
            title="Source Cards", show_header=True, header_style="bold magenta"
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Citation", style="bold cyan")
        table.add_column("Snippet")

        for i, source in enumerate(response["sources"], 1):
            text = source["text"]
            if len(text) > 200:
                text = text[:197] + "..."
            table.add_row(str(i), source["citation"], text)

        console.print(table)
    else:
        console.print("[red]No sources found for this answer.[/red]")

    console.print("\n")


def process_query(orchestrator, query, religion, top_k):
    with console.status("[bold blue]Searching for wisdom...[/bold blue]"):
        try:
            response = orchestrator.generate_answer(
                query, religion=religion, top_k=top_k
            )
            display_response(query, response)
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")


@click.group()
def cli():
    """CLI for the AI Spiritual Knowledge Engine."""
    pass


@cli.command()
@click.option("--port", default=6006, help="Port to run Phoenix on.")
def phoenix(port: int):
    """Launches the Arize Phoenix observability UI locally."""
    try:
        import phoenix as px

        console.print(
            f"[bold green]Starting Arize Phoenix on port {port}...[/bold green]"
        )
        session = px.launch_app(port=port)
        console.print(f"[bold blue]Phoenix UI available at: {session.url}[/bold blue]")
        console.print("[yellow]Press Ctrl+C to stop the server.[/yellow]")

        # Keep the process alive
        import time

        while True:
            time.sleep(1)
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] arize-phoenix not installed. Run 'pip install arize-phoenix'."
        )
    except Exception as e:
        console.print(f"[bold red]Failed to start Phoenix:[/bold red] {e}")


@cli.command()
@click.option(
    "--religion",
    type=click.Choice(["bible", "bhagavad_gita"], case_sensitive=False),
    help="Filter results by a specific text (bible or bhagavad_gita).",
)
@click.option("--top-k", default=3, help="Number of source documents to retrieve.")
@click.option("--query", help="Run a single query and exit.")
def ask(religion: Optional[str], top_k: int, query: Optional[str]):
    """Interact with the AI Spiritual Knowledge Engine."""

    # Initialize
    with console.status("[bold blue]Initializing Engine...[/bold blue]"):
        try:
            retriever = HybridRetriever()
            orchestrator = RAGOrchestrator(retriever=retriever)
        except Exception as e:
            console.print(f"[bold red]Error initializing engine:[/bold red] {e}")
            return

    if query:
        # Single query mode
        process_query(orchestrator, query, religion, top_k)
    else:
        # Interactive mode
        display_welcome()
        while True:
            user_input = console.input("[bold blue]Query > [/bold blue]").strip()

            if user_input.lower() in ["exit", "quit"]:
                console.print(
                    "[bold yellow]Farewell. May you find peace.[/bold yellow]"
                )
                break

            if not user_input:
                continue

            process_query(orchestrator, user_input, religion, top_k)


if __name__ == "__main__":
    cli()
