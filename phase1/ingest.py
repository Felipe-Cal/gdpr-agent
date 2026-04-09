"""
Document ingestion pipeline for Phase 1.

Loads GDPR PDFs/text files → chunks them → embeds with Vertex AI →
stores in BigQuery for vector search.

Usage:
    python -m phase1.ingest
    python -m phase1.ingest --docs-dir data/gdpr_docs

Good GDPR documents to start with (all publicly available):
- GDPR full text (PDF): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679
- EDPB guidelines: https://edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en
- ICO guidance: https://ico.org.uk/for-organisations/
"""

from pathlib import Path

import typer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_community import BigQueryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from rich.console import Console
from rich.progress import track

from config import settings

console = Console()
app = typer.Typer()


def load_documents(docs_dir: Path) -> list[Document]:
    """
    Loads all PDFs and .txt files from docs_dir.

    Each document gets metadata with its source path and document type,
    which we use for citations in answers.
    """
    documents: list[Document] = []

    pdf_files = list(docs_dir.glob("**/*.pdf"))
    txt_files = list(docs_dir.glob("**/*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        console.print(f"[yellow]No PDF or .txt files found in {docs_dir}[/yellow]")
        console.print("Add GDPR documents there and re-run.")
        raise typer.Exit(1)

    console.print(f"Found [bold]{len(all_files)}[/bold] documents.")

    for file_path in track(all_files, description="Loading documents..."):
        loader = PyPDFLoader(str(file_path)) if file_path.suffix == ".pdf" else TextLoader(str(file_path))
        docs = loader.load()

        # Enrich metadata — used for citations in answers
        for doc in docs:
            doc.metadata["source_file"] = file_path.name
            doc.metadata["doc_type"] = "gdpr_regulation" if "gdpr" in file_path.name.lower() else "guidance"

        documents.extend(docs)

    console.print(f"Loaded [bold]{len(documents)}[/bold] pages/sections.")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into overlapping chunks suitable for embedding.

    Why RecursiveCharacterTextSplitter for legal text:
    - Tries to split on paragraph breaks first (\\n\\n), then sentences
    - Overlap preserves context across chunk boundaries (important for
      legal text where definitions span multiple sentences)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    console.print(
        f"Split into [bold]{len(chunks)}[/bold] chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})."
    )
    return chunks


def get_vector_store(embeddings: VertexAIEmbeddings) -> BigQueryVectorStore:
    """
    Returns a LangChain BigQueryVectorSearch store.

    BigQueryVectorSearch:
    - Stores document text + embeddings in a BigQuery table
    - Uses VECTOR_SEARCH() SQL function for similarity queries — serverless,
      no persistent endpoint running 24/7
    - Table is created automatically on first add_documents() call
    """
    return BigQueryVectorStore(
        project_id=settings.gcp_project_id,
        dataset_name=settings.bq_dataset,
        table_name=settings.bq_table,
        location=settings.gcp_region,
        embedding=embeddings,
    )


@app.command()
def ingest(
    docs_dir: Path = typer.Option(Path("data/gdpr_docs"), help="Directory containing GDPR documents"),
) -> None:
    """Load, chunk, embed and store GDPR documents in BigQuery."""

    # Step 1 — Load
    documents = load_documents(docs_dir)

    # Step 2 — Chunk
    chunks = chunk_documents(documents)

    # Step 3 — Embed + store
    # text-embedding-004: Google's best embedding model (768 dims, multilingual)
    console.print("[bold cyan]Initialising Vertex AI embeddings...[/bold cyan]")
    embeddings = VertexAIEmbeddings(
        model_name=settings.embedding_model,
        project=settings.gcp_project_id,
    )

    console.print("[bold cyan]Connecting to BigQuery vector store...[/bold cyan]")
    vector_store = get_vector_store(embeddings)

    console.print(f"[bold cyan]Upserting {len(chunks)} chunks into BigQuery...[/bold cyan]")
    
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch = (i // batch_size) + 1
        console.print(f"Upserting batch {current_batch}/{total_batches} ({len(batch)} chunks)...")
        vector_store.add_documents(batch)

    console.print("[bold green]Ingestion complete.[/bold green]")
    console.print("You can now query the agent: python -m phase1.main")


if __name__ == "__main__":
    app()
