#!/bin/bash

# Cloud Scheduler Setup Script
# Creates a daily scheduled job to trigger the trace extraction service

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-product-research-460317}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="trace-extraction-service"
JOB_NAME="trace-extraction-daily"
SCHEDULE="${SCHEDULE:-0 2 * * *}"  # Default: 2 AM UTC daily
TIMEZONE="${TIMEZONE:-UTC}"

echo "=========================================="
echo "Setting up Cloud Scheduler"
echo "=========================================="
echo "Project ID: ${PROJECT_ID}"
echo "Job Name: ${JOB_NAME}"
echo "Schedule: ${SCHEDULE} (${TIMEZONE})"
echo "=========================================="

# Set the project
gcloud config set project ${PROJECT_ID}

# Get the Cloud Run service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

if [ -z "$SERVICE_URL" ]; then
    echo "Error: Could not find Cloud Run service '${SERVICE_NAME}'"
    echo "Please deploy the service first using: ./deploy_to_cloud_run.sh"
    exit 1
fi

echo "Service URL: ${SERVICE_URL}"

# Create a service account for the scheduler if it doesn't exist
SA_NAME="trace-extraction-scheduler"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Creating service account for scheduler..."
if ! gcloud iam service-accounts describe ${SA_EMAIL} &> /dev/null; then
    gcloud iam service-accounts create ${SA_NAME} \
        --display-name "Trace Extraction Scheduler Service Account"
    echo "Service account created: ${SA_EMAIL}"
else
    echo "Service account already exists: ${SA_EMAIL}"
fi

# Grant the service account permission to invoke Cloud Run
echo "Granting Cloud Run Invoker role..."
gcloud run services add-iam-policy-binding ${SERVICE_NAME} \
    --region ${REGION} \
    --member "serviceAccount:${SA_EMAIL}" \
    --role "roles/run.invoker"

# Delete existing job if it exists
if gcloud scheduler jobs describe ${JOB_NAME} --location ${REGION} &> /dev/null; then
    echo "Deleting existing scheduler job..."
    gcloud scheduler jobs delete ${JOB_NAME} \
        --location ${REGION} \
        --quiet
fi

# Create the Cloud Scheduler job
echo "Creating Cloud Scheduler job..."
gcloud scheduler jobs create http ${JOB_NAME} \
    --location ${REGION} \
    --schedule "${SCHEDULE}" \
    --time-zone "${TIMEZONE}" \
    --uri "${SERVICE_URL}" \
    --http-method GET \
    --oidc-service-account-email ${SA_EMAIL} \
    --oidc-token-audience ${SERVICE_URL} \
    --description "Daily trace extraction from Vertex AI Agent Engine"

echo "=========================================="
echo "Cloud Scheduler Setup Complete!"
echo "=========================================="
echo "Job Name: ${JOB_NAME}"
echo "Schedule: ${SCHEDULE} (${TIMEZONE})"
echo "Target: ${SERVICE_URL}"
echo ""
echo "The job will run automatically according to the schedule."
echo ""
echo "Useful commands:"
echo "  # Run the job manually:"
echo "  gcloud scheduler jobs run ${JOB_NAME} --location ${REGION}"
echo ""
echo "  # View job details:"
echo "  gcloud scheduler jobs describe ${JOB_NAME} --location ${REGION}"
echo ""
echo "  # View job execution history:"
echo "  gcloud scheduler jobs logs ${JOB_NAME} --location ${REGION}"
echo ""
echo "  # Update the schedule:"
echo "  gcloud scheduler jobs update http ${JOB_NAME} --location ${REGION} --schedule 'NEW_SCHEDULE'"
echo "=========================================="
