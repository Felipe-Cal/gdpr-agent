"""
Phase 6 — Submit the compiled pipeline to Vertex AI Pipelines.

Vertex AI Pipelines is the managed KFP service. It takes a compiled pipeline
YAML, provisions pods for each component (one Kubernetes pod per step), tracks
lineage, and stores run metadata in a managed metadata store.

Pricing model:
──────────────
  - Pipeline orchestration: free (you pay only for the compute each component uses)
  - Each component runs on a Vertex AI managed VM:
      default machine: n1-standard-4 (~$0.19/hr, billed per second)
      The ingest + eval components take ~5-10 min total ≈ <$0.05 per run
  - Fine-tuning component (if triggered): ~$0.40 (T4 GPU, 45 min)
  - Scheduled runs (e.g. nightly): ~$0.05/night baseline

Scheduling:
───────────
  Vertex AI Pipelines supports cron scheduling natively:
    aiplatform.PipelineJob.schedule(cron_expression="0 2 * * *")  # 2am UTC daily
  This means new GDPR documents ingested to GCS each night are automatically
  embedded, evaluated, and (if quality drops) trigger a fine-tuning job —
  with zero manual intervention.

Run:
    python -m phase6.submit --bucket your-gcs-bucket [--schedule]
"""

import typer
import vertexai
from google.cloud import aiplatform
from phase6.pipeline import gdpr_agent_pipeline
from kfp import compiler
from pathlib import Path

from config import settings

app = typer.Typer()

PIPELINE_YAML = "phase6/gdpr_pipeline.yaml"


def _compile_pipeline() -> str:
    """Compile the pipeline to YAML if not already compiled."""
    compiler.Compiler().compile(gdpr_agent_pipeline, PIPELINE_YAML)
    typer.echo(f"Pipeline compiled → {PIPELINE_YAML}")
    return PIPELINE_YAML


@app.command()
def main(
    bucket: str = typer.Option(..., help="GCS bucket for pipeline artifacts"),
    gcs_pdf_uri: str = typer.Option(
        None, help="GCS URI for GDPR PDFs (default: gs://bucket/gdpr-docs/)"
    ),
    schedule: bool = typer.Option(False, help="Also create a nightly cron schedule"),
    dry_run: bool = typer.Option(False, help="Compile only, do not submit"),
):
    """
    Compile and submit the GDPR agent lifecycle pipeline to Vertex AI.

    ⚠️  Cost: ~$0.05 per run (ingest + eval only).
         Fine-tuning step adds ~$0.40 if triggered (requires T4 GPU quota).
    """
    _compile_pipeline()

    if dry_run:
        typer.echo("Dry-run: pipeline compiled but not submitted.")
        return

    typer.confirm(
        "⚠️  Submit pipeline to Vertex AI? Estimated cost: ~$0.05 (no fine-tuning).",
        abort=True,
    )

    vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
    aiplatform.init(
        project=settings.gcp_project_id,
        location=settings.gcp_region,
        staging_bucket=f"gs://{bucket}",
    )

    pdf_uri = gcs_pdf_uri or f"gs://{bucket}/gdpr-docs/"

    job = aiplatform.PipelineJob(
        display_name="gdpr-agent-lifecycle",
        template_path=PIPELINE_YAML,
        pipeline_root=f"gs://{bucket}/pipeline-runs",
        parameter_values={
            "gcp_project_id": settings.gcp_project_id,
            "gcp_region": settings.gcp_region,
            "gcs_pdf_uri": pdf_uri,
            "bq_dataset": settings.bq_dataset,
            "bq_table": settings.bq_table,
            "gemini_model": settings.gemini_model,
            "embedding_model": settings.embedding_model,
            "gcs_bucket": bucket,
        },
        enable_caching=True,   # skip re-running components whose inputs haven't changed
    )

    job.submit()
    typer.echo(f"Pipeline submitted. Track at:")
    typer.echo(f"  https://console.cloud.google.com/vertex-ai/pipelines/runs?project={settings.gcp_project_id}")

    if schedule:
        typer.echo("\nCreating nightly schedule (2am UTC)...")
        job.create_schedule(
            display_name="gdpr-agent-nightly",
            cron="0 2 * * *",
            max_concurrent_run_count=1,
            max_run_count=None,   # run indefinitely
        )
        typer.echo("Schedule created. Pipeline will run nightly at 2am UTC.")
        typer.echo("\n⚠️  Teardown: delete the schedule to stop billing.")
        typer.echo(f"  gcloud ai pipeline-jobs delete-schedule --region={settings.gcp_region} \\")
        typer.echo(f"    --project={settings.gcp_project_id} <schedule-id>")


if __name__ == "__main__":
    app()
