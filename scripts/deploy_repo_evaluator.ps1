# Build and deploy the Cloud Run repo-evaluator job image.
# Usage (from repo root):
#   .\scripts\deploy_repo_evaluator.ps1
# Optional env overrides:
#   $env:GCP_PROJECT_ID = "exaai-sdk"
#   $env:GCP_REGION = "asia-south1"
#   $env:REPO_EVALUATOR_IMAGE = "asia-south1-docker.pkg.dev/exaai-sdk/sandbox-images/repo-evaluator:latest"

$ErrorActionPreference = "Stop"

$ProjectId = if ($env:GCP_PROJECT_ID) { $env:GCP_PROJECT_ID } else { "exaai-sdk" }
$Region = if ($env:GCP_REGION) { $env:GCP_REGION } else { "asia-south1" }
$JobName = if ($env:CLOUD_RUN_SANDBOX_JOB_NAME) { $env:CLOUD_RUN_SANDBOX_JOB_NAME } else { "repo-evaluator" }
$Image = if ($env:REPO_EVALUATOR_IMAGE) {
    $env:REPO_EVALUATOR_IMAGE
} else {
    "${Region}-docker.pkg.dev/${ProjectId}/sandbox-images/repo-evaluator:latest"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot

Write-Host "Building evaluator image: $Image"
gcloud builds submit `
    --project $ProjectId `
    --config agent/sandbox/evaluator/cloudbuild.yaml `
    --substitutions "_IMAGE=$Image" `
    .

Write-Host "Updating Cloud Run job: $JobName ($Region)"
gcloud run jobs update $JobName `
    --project $ProjectId `
    --region $Region `
    --image $Image

Write-Host "Done. Current job image:"
gcloud run jobs describe $JobName `
    --project $ProjectId `
    --region $Region `
    --format "value(spec.template.spec.template.spec.containers[0].image)"

Pop-Location
