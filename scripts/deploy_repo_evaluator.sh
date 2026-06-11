#!/usr/bin/env bash
# Build and deploy the Cloud Run repo-evaluator job image.
# Usage (from repo root):
#   ./scripts/deploy_repo_evaluator.sh

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-exaai-sdk}"
REGION="${GCP_REGION:-asia-south1}"
JOB_NAME="${CLOUD_RUN_SANDBOX_JOB_NAME:-repo-evaluator}"
IMAGE="${REPO_EVALUATOR_IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/sandbox-images/repo-evaluator:latest}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Building evaluator image: ${IMAGE}"
gcloud builds submit \
  --project "${PROJECT_ID}" \
  --config agent/sandbox/evaluator/cloudbuild.yaml \
  --substitutions "_IMAGE=${IMAGE}" \
  .

echo "Updating Cloud Run job: ${JOB_NAME} (${REGION})"
gcloud run jobs update "${JOB_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}"

echo "Done. Current job image:"
gcloud run jobs describe "${JOB_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format "value(spec.template.spec.template.spec.containers[0].image)"
