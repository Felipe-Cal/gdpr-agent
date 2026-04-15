#!/usr/bin/env bash
# Phase 6 — VPC Service Controls setup
#
# VPC-SC creates a security perimeter around your GCP project.
# Inside the perimeter: your Vertex AI, BigQuery, GCS, and KMS resources.
# Outside the perimeter: everything else — the internet, other GCP projects,
#   even other projects in the same organisation.
#
# Why this matters for GDPR Article 32 (security of processing):
# ──────────────────────────────────────────────────────────────
#   IAM controls WHO can act on your resources (identity-based).
#   VPC-SC controls WHERE requests can come FROM (network-based).
#
#   Without VPC-SC, a malicious insider with BigQuery read access could:
#     1. Create a personal GCP project
#     2. Call `bq export` from that project using a stolen service account key
#     3. Exfiltrate the entire document_chunks table to their personal GCS bucket
#
#   VPC-SC blocks step 2: the `bq export` call originates outside the perimeter
#   and is rejected, even with valid credentials.
#
#   This is why regulated industries (finance, healthcare, public sector) require
#   VPC-SC as a baseline control alongside CMEK — it prevents data exfiltration
#   from a compromised credential.
#
# Perimeter architecture for this project:
# ─────────────────────────────────────────
#   Service perimeter: gdpr-agent-perimeter
#   Protected resources:
#     - BigQuery (bigquery.googleapis.com)
#     - Vertex AI (aiplatform.googleapis.com)
#     - Cloud Storage (storage.googleapis.com)
#     - Cloud KMS (cloudkms.googleapis.com)
#   Access levels:
#     - CORP_NETWORK: allow from corporate IP ranges (or Cloud VPN)
#     - CI_SA: allow from the CI/CD service account (for pipeline runs)
#
# ⚠️  WARNING: VPC-SC is org-level policy. It affects ALL users and services
#     in the project. Misconfiguration can break Vertex AI Pipelines, Cloud Build,
#     and other services. Test in a non-production project first.
#
# Prerequisites:
#   gcloud auth login (as org admin)
#   gcloud components install alpha
#   export PROJECT_ID=your-project-id
#   export ORG_ID=your-org-id  (gcloud organizations list)
#   export POLICY_NAME=your-access-policy-name  (gcloud access-context-manager policies list)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
ORG_ID="${ORG_ID:?Set ORG_ID}"
POLICY_NAME="${POLICY_NAME:?Set POLICY_NAME}"
REGION="europe-west4"
PERIMETER_NAME="gdpr_agent_perimeter"
ACCESS_LEVEL_NAME="gdpr_corp_network"
# Your corporate egress IP range — replace with actual range
CORP_IP_RANGE="${CORP_IP_RANGE:-203.0.113.0/24}"

# ── Step 1: Create an access level (who/where can bypass the perimeter) ──────
echo "Creating access level: ${ACCESS_LEVEL_NAME}"
gcloud access-context-manager levels create "${ACCESS_LEVEL_NAME}" \
  --policy="${POLICY_NAME}" \
  --title="GDPR Agent Corporate Network" \
  --basic-level-spec=<(cat <<YAML
conditions:
  - ipSubnetworks:
      - "${CORP_IP_RANGE}"
YAML
)

# ── Step 2: Create the service perimeter ─────────────────────────────────────
echo "Creating service perimeter: ${PERIMETER_NAME}"
gcloud access-context-manager perimeters create "${PERIMETER_NAME}" \
  --policy="${POLICY_NAME}" \
  --title="GDPR Agent Data Perimeter" \
  --resources="projects/${PROJECT_ID}" \
  --restricted-services=\
"bigquery.googleapis.com,\
aiplatform.googleapis.com,\
storage.googleapis.com,\
cloudkms.googleapis.com" \
  --access-levels="${ACCESS_LEVEL_NAME}" \
  --enable-vpc-accessible-services \
  --vpc-allowed-services=\
"bigquery.googleapis.com,\
aiplatform.googleapis.com,\
storage.googleapis.com,\
cloudkms.googleapis.com"

echo "Perimeter created. It may take up to 30 seconds to propagate."

# ── Step 3: Add ingress rule for Vertex AI Pipelines service account ──────────
# Vertex AI Pipelines runs pipeline components using a Google-managed SA.
# That SA must be explicitly granted ingress, or the pipeline pods can't
# call BigQuery / GCS from inside the perimeter.
#
# The Vertex AI Pipelines SA format:
#   service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com
#
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
VERTEX_SA="service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"

echo "Adding ingress rule for Vertex AI Pipelines SA: ${VERTEX_SA}"
# Ingress rules are defined in JSON and applied via update
cat > /tmp/ingress_rule.json <<JSON
[
  {
    "ingressFrom": {
      "identities": ["serviceAccount:${VERTEX_SA}"],
      "identityType": "ANY_IDENTITY"
    },
    "ingressTo": {
      "resources": ["*"],
      "operations": [
        {"serviceName": "bigquery.googleapis.com", "methodSelectors": [{"method": "*"}]},
        {"serviceName": "storage.googleapis.com", "methodSelectors": [{"method": "*"}]},
        {"serviceName": "aiplatform.googleapis.com", "methodSelectors": [{"method": "*"}]}
      ]
    }
  }
]
JSON

gcloud access-context-manager perimeters update "${PERIMETER_NAME}" \
  --policy="${POLICY_NAME}" \
  --set-ingress-policies=/tmp/ingress_rule.json

echo ""
echo "VPC-SC setup complete."
echo "Perimeter: ${PERIMETER_NAME}"
echo "Protected services: BigQuery, Vertex AI, GCS, KMS"
echo ""
echo "To delete the perimeter (teardown):"
echo "  gcloud access-context-manager perimeters delete ${PERIMETER_NAME} --policy=${POLICY_NAME}"
