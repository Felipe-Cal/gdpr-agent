"""
One-time setup script: creates a BigQuery dataset for vector search.

Run this ONCE before ingesting documents:
    python -m setup.create_dataset

Why BigQuery Vector Search instead of Vertex AI Vector Search?
- No persistent endpoint = no hourly cost when idle
- Serverless: pay only per query (first 1 TB/month is free)
- Same GCP ecosystem, same learning value for architecture discussions
- Perfect for dev/learning projects — production would use Vertex AI Vector Search
  for lower latency at high query volume

What this creates:
- A BigQuery dataset (gdpr_agent) in europe-west4
- The document_chunks table is created automatically by the ingest step
"""

from google.cloud import bigquery
from rich.console import Console

from config import settings

console = Console()


def create_dataset() -> None:
    client = bigquery.Client(project=settings.gcp_project_id)

    dataset_id = f"{settings.gcp_project_id}.{settings.bq_dataset}"
    dataset = bigquery.Dataset(dataset_id)

    # Keep data in Europe — data sovereignty for GDPR compliance
    dataset.location = settings.gcp_region
    dataset.description = "GDPR Legal Analyst Agent — document chunks with embeddings"

    try:
        dataset = client.create_dataset(dataset, timeout=30)
        console.print(f"[green]Dataset created:[/green] {dataset_id} (location: {settings.gcp_region})")
    except Exception as e:
        if "Already Exists" in str(e):
            console.print(f"[yellow]Dataset already exists:[/yellow] {dataset_id}")
        else:
            raise

    console.print("\n[bold green]Setup complete.[/bold green]")
    console.print("Next step: python -m phase1.ingest")


if __name__ == "__main__":
    create_dataset()
