"""
One-time setup script: creates a Vertex AI Vector Search index + endpoint.

Run this ONCE before ingesting documents:
    python -m setup.create_index

After it completes, copy the printed IDs into your .env file:
    VECTOR_SEARCH_INDEX_ID=...
    VECTOR_SEARCH_ENDPOINT_ID=...

What this creates:
- A Vector Search Index (streaming update mode, so we can add docs incrementally)
- An Index Endpoint (the deployed, queryable version of the index)
"""

import time

from google.cloud import aiplatform
from rich.console import Console

from config import settings

console = Console()


def create_index() -> aiplatform.MatchingEngineIndex:
    """
    Creates a Vertex AI Vector Search index.

    Key decisions:
    - streaming update: lets us add documents one by one via ingest.py
    - dimensions=768: matches text-embedding-004 output size
    - distance_measure: DOT_PRODUCT_DISTANCE works well with normalized embeddings
    """
    console.print("[bold cyan]Creating Vector Search index...[/bold cyan]")

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="gdpr-agent-index",
        contents_delta_uri=f"gs://{settings.gcp_bucket}/vector-search/",
        dimensions=768,  # text-embedding-004 output dimension
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        index_update_method="STREAM_UPDATE",  # allows incremental upserts
        description="GDPR legal documents index for RAG",
    )

    console.print(f"[green]Index created:[/green] {index.resource_name}")
    return index


def create_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """Creates a public index endpoint (no VPC needed for dev)."""
    console.print("[bold cyan]Creating index endpoint...[/bold cyan]")

    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="gdpr-agent-endpoint",
        public_endpoint_enabled=True,
        description="Endpoint for GDPR agent RAG queries",
    )

    console.print(f"[green]Endpoint created:[/green] {endpoint.resource_name}")
    return endpoint


def deploy_index(
    index: aiplatform.MatchingEngineIndex,
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
) -> None:
    """Deploys the index to the endpoint. Takes ~30 minutes on first deploy."""
    console.print("[bold cyan]Deploying index to endpoint (this takes ~30 min)...[/bold cyan]")

    endpoint.deploy_index(
        index=index,
        deployed_index_id="gdpr_agent_deployed",
        display_name="gdpr-agent-deployed",
        # machine_type and min_replica_count kept minimal for dev cost
        machine_type="e2-standard-2",  # smallest available — fine for dev/learning
        min_replica_count=1,
        max_replica_count=1,
    )

    console.print("[green]Index deployed.[/green]")


def main() -> None:
    aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)

    index = create_index()
    endpoint = create_endpoint()

    # Index needs a moment before deployment
    console.print("Waiting 60s before deploying...")
    time.sleep(60)

    deploy_index(index, endpoint)

    # Extract short IDs from full resource names
    # resource_name format: projects/.../indexes/INDEX_ID
    index_id = index.resource_name.split("/")[-1]
    endpoint_id = endpoint.resource_name.split("/")[-1]

    console.print("\n[bold green]Setup complete! Add these to your .env:[/bold green]")
    console.print(f"VECTOR_SEARCH_INDEX_ID={index_id}")
    console.print(f"VECTOR_SEARCH_ENDPOINT_ID={endpoint_id}")


if __name__ == "__main__":
    main()
