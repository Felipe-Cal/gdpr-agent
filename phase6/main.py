"""
Phase 6 — CLI entry point.

Commands:
  compile   Compile the pipeline to YAML (free, no GCP calls)
  submit    Submit the compiled pipeline to Vertex AI
  kms       CMEK key management
"""

import typer

app = typer.Typer(help="Phase 6 — Production MLOps for the GDPR agent.")


@app.command()
def compile():
    """Compile the Vertex AI Pipeline to YAML (free — no GCP calls)."""
    from kfp import compiler as kfp_compiler
    from phase6.pipeline import gdpr_agent_pipeline

    output = "phase6/gdpr_pipeline.yaml"
    kfp_compiler.Compiler().compile(gdpr_agent_pipeline, output)
    typer.echo(f"Pipeline compiled → {output}")


@app.command()
def submit(
    bucket: str = typer.Option(..., help="GCS bucket for pipeline artifacts"),
    schedule: bool = typer.Option(False, help="Create nightly cron schedule"),
):
    """Submit the pipeline to Vertex AI Pipelines."""
    from phase6.submit import main as _submit
    _submit(bucket=bucket, schedule=schedule, gcs_pdf_uri=None, dry_run=False)


@app.command()
def kms_setup():
    """Create the Cloud KMS keyring and key for CMEK."""
    from phase6.kms import setup
    setup()


@app.command()
def kms_patch_bq():
    """Configure BigQuery dataset to use the CMEK key."""
    from phase6.kms import patch_bigquery
    patch_bigquery()


if __name__ == "__main__":
    app()
