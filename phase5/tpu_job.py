"""
Phase 5 — Vertex AI Training job for JAX/TPU fine-tuning.

Submits phase5/jax_train.py to a Vertex AI Custom Training Job
targeting a Cloud TPU v4-8 accelerator (8 chips, 32GB HBM per chip).

TPU v4 vs GPU for fine-tuning:
──────────────────────────────
| Accelerator      | TFLOPS (bf16) | Memory      | $/hr (GCP)  |
|------------------|---------------|-------------|-------------|
| NVIDIA T4        |    65 TFLOPS  | 16 GB GDDR6 |  $0.35      |
| NVIDIA A100 40GB |   312 TFLOPS  | 40 GB HBM2  |  $2.93      |
| TPU v4-8 (pod)   |  1,120 TFLOPS | 32 GB × 8   |  $3.22      |

For Qwen-0.5B on 100 examples: T4 is cheaper ($0.15 total).
For Mistral-7B or Llama-70B on millions of tokens: TPU v4 is 3-5x
better price-performance because XLA compilation eliminates Python overhead
and pmap scales linearly across all 8 chips.

When to use TPU:
  - Model > 7B parameters
  - Training data > 100k tokens
  - You need reproducible, compiled kernels (regulatory environments)
  - You're training a JAX/Flax model (no PyTorch/XLA translation overhead)

Run:
    python -m phase5.tpu_job --bucket gdpr-agent-project-data
"""

import typer
import vertexai
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomTrainingJob

from config import settings

app = typer.Typer()


def submit_tpu_training_job(
    gcs_dataset_uri: str,
    gcs_output_uri: str,
    epochs: int = 2,
    sync: bool = False,
) -> CustomTrainingJob:
    """
    Submits jax_train.py to Vertex AI Training on a Cloud TPU v4-8.

    Key differences from the GPU job (vertex_job.py):
      - container_uri: uses the pre-built JAX TPU container instead of PyTorch
      - tpu_type: "TPU_V4" with tpu_topology="2x2x2" (= 8 chips = v4-8)
      - requirements: jax[tpu] instead of torch + bitsandbytes
      - No quantization needed: bfloat16 is TPU's native dtype, full precision
        at hardware speed

    TPU topology notation:
      "2x2x2" means a 3D torus of 2×2×2 = 8 TPU v4 chips.
      "2x2x4" = 16 chips (v4-16), "4x4x4" = 64 chips (v4-64), etc.
      Each chip has 2 TensorCores and 32GB HBM.
    """
    vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
    aiplatform.init(
        project=settings.gcp_project_id,
        location=settings.gcp_region,
        staging_bucket=f"gs://{settings.finetune_gcs_bucket}",
    )

    job = CustomTrainingJob(
        display_name="gdpr-jax-tpu-finetune",
        script_path="phase5/jax_train.py",
        # Pre-built JAX container with libtpu pre-installed.
        # This is the Google-maintained image for TPU training on Vertex AI.
        container_uri="us-docker.pkg.dev/vertex-ai/training/jax-tpu.0-4.py310:latest",
        requirements=[
            "flax>=0.8",
            "optax>=0.2",
            "transformers[flax]>=4.40",
            "datasets>=2.18",
        ],
        model_serving_container_image_uri=None,
    )

    model = job.run(
        args=[],
        environment_variables={
            "MODEL_ID": "Qwen/Qwen2.5-0.5B",
            "DATASET_PATH": gcs_dataset_uri,
            "OUTPUT_DIR": "/tmp/gdpr-flax-checkpoint",
            "GCS_OUTPUT_URI": gcs_output_uri,
            "EPOCHS": str(epochs),
            "BATCH_SIZE": "8",          # per device; ×8 chips = 64 effective
        },
        # TPU v4-8: 8 chips in a 2x2x2 topology
        # No machine_type needed — TPU pods are their own resource class
        machine_type="cloud-tpu",
        accelerator_type="TPU_V4",
        accelerator_count=8,
        base_output_dir=gcs_output_uri,
        sync=sync,
    )

    return job


@app.command()
def main(
    bucket: str = typer.Option(..., help="GCS bucket name"),
    epochs: int = typer.Option(2, help="Training epochs"),
):
    """
    Submit a JAX/TPU fine-tuning job to Vertex AI.

    ⚠️  Cost: Cloud TPU v4-8 at ~$3.22/hr.
        Expected runtime: ~15-20 min for this dataset = ~$1.00 total.
        GPU (T4) would be ~$0.15 for the same job but 3x slower for large models.
    """
    from phase5.vertex_job import upload_dataset_to_gcs
    from phase5.dataset import build_dataset

    typer.echo("⚠️  TPU v4-8 costs ~$3.22/hr (~$1.00 total for this job)")
    typer.confirm("Proceed?", abort=True)

    build_dataset("data/gdpr_finetune.jsonl")
    dataset_gcs = f"gs://{bucket}/phase5/data/gdpr_finetune.jsonl"
    output_gcs  = f"gs://{bucket}/phase5/flax-checkpoint"
    upload_dataset_to_gcs("data/gdpr_finetune.jsonl", dataset_gcs)

    typer.echo(f"Submitting JAX/TPU training job...")
    job = submit_tpu_training_job(dataset_gcs, output_gcs, epochs=epochs)
    typer.echo(f"Job submitted. Output → {output_gcs}")


if __name__ == "__main__":
    app()
