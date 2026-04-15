"""
Phase 6 — CMEK (Customer-Managed Encryption Keys) setup.

By default, GCP encrypts data at rest using Google-managed keys (GMEK). With
CMEK, you provide a key from Cloud KMS. GCP uses it to encrypt your data, but
you own and control the key lifecycle.

Why CMEK matters for GDPR:
───────────────────────────
  GDPR Article 17 — Right to erasure ("right to be forgotten"):
    When a data subject requests erasure, you must delete all their data.
    In a BigQuery table this is straightforward (DELETE WHERE user_id = X),
    but in vector stores, embedding caches, and backup snapshots it's harder.

    CMEK provides a cryptographic shortcut: rotate/revoke the KMS key and all
    data encrypted with it becomes inaccessible — effectively erased from a
    practical standpoint, even if bits remain on disk. This is accepted as
    equivalent to deletion in most regulatory frameworks.

  GDPR Article 32 — Security of processing:
    "Appropriate technical measures" must protect personal data. Key management
    (rotation, access audit trails, HSM-backed keys) is explicitly cited in
    EDPB guidance as a qualifying technical measure.

  Regulated industries (finance, healthcare, public sector):
    Many organisations require CMEK as a contractual baseline — they will not
    use a cloud service that holds data under vendor-managed keys.

Key hierarchy in Cloud KMS:
────────────────────────────
  Key Ring   — logical container, bound to a region and project
    └─ Key   — a named cryptographic key with a rotation schedule
         └─ Key Version — the actual key material (can be enabled/disabled/destroyed)

  We create:  gdpr-agent-keyring (europe-west4)
                └─ gdpr-data-key  (AES-256-GCM, 90-day rotation)

⚠️  Cost: ~$0.06/month per key version + $0.03 per 10,000 API operations.
    For this project: essentially free.

Run:
    python -m phase6.kms setup
    python -m phase6.kms show-commands
"""

import typer
from rich.console import Console
from rich.syntax import Syntax

from config import settings

app = typer.Typer()
console = Console()

KEYRING_NAME = "gdpr-agent-keyring"
KEY_NAME = "gdpr-data-key"
KEY_PURPOSE = "encryption"
ROTATION_PERIOD = "7776000s"  # 90 days in seconds


@app.command()
def setup():
    """
    Create the KMS keyring and key, then configure BigQuery and GCS to use it.

    ⚠️  Cost: ~$0.06/month per key version.
    """
    from google.cloud import kms

    project = settings.gcp_project_id
    region  = settings.gcp_region

    kms_client = kms.KeyManagementServiceClient()

    # --- Keyring ---
    parent = f"projects/{project}/locations/{region}"
    keyring_name = f"{parent}/keyRings/{KEYRING_NAME}"

    try:
        kms_client.create_key_ring(
            request={"parent": parent, "key_ring_id": KEYRING_NAME, "key_ring": {}}
        )
        console.print(f"[green]Created keyring:[/green] {keyring_name}")
    except Exception as e:
        if "already exists" in str(e):
            console.print(f"[dim]Keyring already exists: {keyring_name}[/dim]")
        else:
            raise

    # --- Key with auto-rotation ---
    try:
        kms_client.create_crypto_key(
            request={
                "parent": keyring_name,
                "crypto_key_id": KEY_NAME,
                "crypto_key": {
                    "purpose": kms.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT,
                    # Rotation period: GCP auto-creates a new key version and
                    # sets it as primary. Old versions remain for decryption
                    # until explicitly destroyed.
                    "rotation_period": {"seconds": int(ROTATION_PERIOD.rstrip("s"))},
                },
            }
        )
        console.print(f"[green]Created key:[/green] {keyring_name}/cryptoKeys/{KEY_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            console.print(f"[dim]Key already exists[/dim]")
        else:
            raise

    key_resource = f"{keyring_name}/cryptoKeys/{KEY_NAME}"
    console.print(f"\n[bold]Key resource name:[/bold]\n  {key_resource}")

    # --- Grant GCP services access to the key ---
    # BigQuery and GCS service accounts must have roles/cloudkms.cryptoKeyEncrypterDecrypter
    # on the key. The service accounts follow a predictable format.
    console.print("\n[cyan]Grant service accounts access to the key:[/cyan]")
    sa_bq  = f"bq-{project}@bigquery-encryption.iam.gserviceaccount.com"
    sa_gcs = f"service-{_get_project_number(project)}@gs-project-accounts.iam.gserviceaccount.com"

    for sa in [sa_bq, sa_gcs]:
        console.print(f"  gcloud kms keys add-iam-policy-binding {KEY_NAME} \\")
        console.print(f"    --keyring={KEYRING_NAME} --location={region} \\")
        console.print(f"    --member=serviceAccount:{sa} \\")
        console.print(f"    --role=roles/cloudkms.cryptoKeyEncrypterDecrypter \\")
        console.print(f"    --project={project}")

    console.print("\n[yellow]Then configure BigQuery dataset to use CMEK:[/yellow]")
    console.print(f"  See: python -m phase6.kms patch-bq")


@app.command(name="patch-bq")
def patch_bigquery():
    """
    Patch the BigQuery dataset to use the CMEK key for default table encryption.

    ⚠️  This changes the default encryption for NEW tables created in the dataset.
        Existing tables retain their current encryption and must be recreated.
    """
    from google.cloud import bigquery

    project = settings.gcp_project_id
    region  = settings.gcp_region
    key_resource = (
        f"projects/{project}/locations/{region}"
        f"/keyRings/{KEYRING_NAME}/cryptoKeys/{KEY_NAME}"
    )

    bq_client = bigquery.Client(project=project)
    dataset_ref = bigquery.DatasetReference(project, settings.bq_dataset)
    dataset = bq_client.get_dataset(dataset_ref)

    dataset.default_encryption_configuration = bigquery.EncryptionConfiguration(
        kms_key_name=key_resource
    )
    bq_client.update_dataset(dataset, ["default_encryption_configuration"])

    console.print(f"[green]BigQuery dataset `{settings.bq_dataset}` now uses CMEK.[/green]")
    console.print(f"[dim]Key: {key_resource}[/dim]")
    console.print(
        "\n[yellow]Note:[/yellow] Existing tables are unaffected. "
        "Re-run ingest (phase1.ingest) to re-create the document_chunks table with CMEK."
    )


@app.command(name="show-commands")
def show_commands():
    """Print the gcloud commands to set up CMEK without running them."""
    project = settings.gcp_project_id
    region  = settings.gcp_region

    cmds = f"""\
# 1. Create keyring and key
gcloud kms keyrings create {KEYRING_NAME} \\
  --location={region} --project={project}

gcloud kms keys create {KEY_NAME} \\
  --keyring={KEYRING_NAME} --location={region} \\
  --purpose=encryption \\
  --rotation-period=90d \\
  --project={project}

# 2. Grant BigQuery service account access
BQ_SA=$(gcloud projects get-iam-policy {project} \\
  --flatten="bindings[].members" \\
  --filter="bindings.role:roles/bigquery.encryptionUser" \\
  --format="value(bindings.members)")

gcloud kms keys add-iam-policy-binding {KEY_NAME} \\
  --keyring={KEYRING_NAME} --location={region} \\
  --member=$BQ_SA \\
  --role=roles/cloudkms.cryptoKeyEncrypterDecrypter \\
  --project={project}

# 3. Set default encryption on BigQuery dataset
bq update --default_kms_key \\
  projects/{project}/locations/{region}/keyRings/{KEYRING_NAME}/cryptoKeys/{KEY_NAME} \\
  {project}:{settings.bq_dataset}

# 4. Set default encryption on GCS bucket (fine-tuning artifacts)
gsutil kms authorize -p {project} -k \\
  projects/{project}/locations/{region}/keyRings/{KEYRING_NAME}/cryptoKeys/{KEY_NAME}

gsutil defkms set -k \\
  projects/{project}/locations/{region}/keyRings/{KEYRING_NAME}/cryptoKeys/{KEY_NAME} \\
  gs://your-finetune-bucket

# 5. To "erase" all CMEK-encrypted data (cryptographic deletion):
gcloud kms keys versions destroy 1 \\
  --key={KEY_NAME} --keyring={KEYRING_NAME} \\
  --location={region} --project={project}
# → All data encrypted with this key version becomes permanently inaccessible.
"""

    console.print(Syntax(cmds, "bash", theme="monokai", line_numbers=False))


def _get_project_number(project_id: str) -> str:
    """Resolve project ID to project number (needed for GCS service account format)."""
    from google.cloud import resourcemanager_v3
    client = resourcemanager_v3.ProjectsClient()
    project = client.get_project(name=f"projects/{project_id}")
    return project.name.split("/")[-1]


if __name__ == "__main__":
    app()
