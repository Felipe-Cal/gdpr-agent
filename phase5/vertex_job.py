"""
Phase 5 — Vertex AI Training job launcher.

Submits the LoRA fine-tuning script (phase5/train.py) as a managed
Vertex AI Custom Training Job using a pre-built PyTorch container.

Why Vertex AI Training vs running locally?
  - Managed: GCP provisions and tears down the GPU VM automatically.
    You pay only for the time the job runs (typically 30-60 min for this dataset).
  - Reproducible: job config is code, not a manual SSH session.
  - Integration: outputs go to GCS, can feed directly into Vertex AI Model Registry.
  - Scalability: trivially switch from 1xT4 to 8xA100 by changing machine_type.

Cost for this job:
  - n1-standard-8 + 1x T4: ~$0.54/hr
  - Expected runtime: ~45 min for 3 epochs on ~100 examples
  - Estimated total: ~$0.40

Run:
    python -m phase5.vertex_job
"""

import vertexai
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomTrainingJob

from config import settings


def submit_training_job(
    gcs_dataset_uri: str,
    gcs_output_uri: str,
    epochs: int = 3,
    sync: bool = True,
) -> aiplatform.jobs.CustomTrainingJob:
    """
    Submits a Vertex AI Custom Training Job to fine-tune Gemma-2-2B with LoRA.

    Args:
        gcs_dataset_uri:  GCS path to the training JSONL file.
                          e.g. gs://gdpr-agent-bucket/data/gdpr_finetune.jsonl
        gcs_output_uri:   GCS path where the adapter checkpoint will be saved.
                          e.g. gs://gdpr-agent-bucket/phase5/adapter
        epochs:           Number of training epochs.
        sync:             If True, block until the job completes.
                          If False, return immediately (job runs in background).

    Returns:
        The Vertex AI CustomTrainingJob object.
    """
    vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)

    job = CustomTrainingJob(
        display_name="gdpr-lora-finetune",
        # Pre-built PyTorch 2.1 container with CUDA 12.1 — no Dockerfile needed.
        # For JAX/TPU training you would use the pre-built JAX container instead.
        script_path="phase5/train.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=[
            "peft>=0.10",
            "trl>=0.8",
            "bitsandbytes>=0.43",
            "datasets>=2.18",
            "accelerate>=0.28",
        ],
        model_serving_container_image_uri=None,  # we serve via vLLM (Phase 4), not Vertex
    )

    model = job.run(
        args=[],
        environment_variables={
            "MODEL_ID": "google/gemma-2-2b-it",
            "DATASET_PATH": "/gcs/dataset/gdpr_finetune.jsonl",  # GCS FUSE mount
            "OUTPUT_DIR": "/gcs/output/adapter",
            "GCS_OUTPUT_URI": gcs_output_uri,
            "EPOCHS": str(epochs),
            "BATCH_SIZE": "4",
            "LORA_R": "16",
            "LORA_ALPHA": "32",
        },
        # n1-standard-8 + T4: enough VRAM for Gemma-2B QLoRA; cheaper than A10G
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        # GCS FUSE mounts: Vertex mounts GCS buckets as local paths (/gcs/...)
        # so the training script can read/write as if they were local filesystems.
        base_output_dir=gcs_output_uri,
        sync=sync,
    )

    return job


def upload_dataset_to_gcs(local_path: str, gcs_uri: str):
    """Uploads the local training JSONL to GCS before submitting the job."""
    from google.cloud import storage
    import os

    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    client = storage.Client(project=settings.gcp_project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → {gcs_uri}")


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        bucket: str = typer.Option(..., help="GCS bucket name (without gs://)"),
        epochs: int = typer.Option(3, help="Training epochs"),
        async_: bool = typer.Option(False, "--async", help="Don't wait for completion"),
    ):
        from phase5.dataset import build_dataset

        # 1. Generate dataset locally
        print("Generating training dataset...")
        examples = build_dataset("data/gdpr_finetune.jsonl")
        print(f"  {len(examples)} examples generated.")

        # 2. Upload to GCS
        dataset_gcs = f"gs://{bucket}/phase5/data/gdpr_finetune.jsonl"
        output_gcs = f"gs://{bucket}/phase5/adapter"
        upload_dataset_to_gcs("data/gdpr_finetune.jsonl", dataset_gcs)

        # 3. Submit training job
        # ⚠️ This will start billing (~$0.54/hr for T4, ~45 min = ~$0.40 total)
        print(f"\n⚠️  Submitting Vertex AI Training job (estimated cost: ~$0.40)")
        print(f"   Dataset: {dataset_gcs}")
        print(f"   Output:  {output_gcs}\n")
        job = submit_training_job(
            gcs_dataset_uri=dataset_gcs,
            gcs_output_uri=output_gcs,
            epochs=epochs,
            sync=not async_,
        )
        print(f"\nJob complete. Adapter at: {output_gcs}")

    app()
