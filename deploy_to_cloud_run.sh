#!/bin/bash

# Trace Extraction Service - Cloud Run Deployment Script
# This script deploys the trace extraction service to Google Cloud Run

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-product-research-460317}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="trace-extraction-service"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
AGENT_ENGINE_ID="${AGENT_ENGINE_ID:-3122210601628073984}"
GCS_BUCKET="${GCS_BUCKET:-evaluation-research}"
BIFROST_URL="${BIFROST_URL:-https://bifrost-service-qa-845835740344.us-central1.run.app/webhook/bifrost_service}"

echo "=========================================="
echo "Deploying Trace Extraction Service"
echo "=========================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service Name: ${SERVICE_NAME}"
echo "Image: ${IMAGE_NAME}"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set the project
echo "Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    cloudscheduler.googleapis.com \
    cloudtrace.googleapis.com \
    logging.googleapis.com \
    storage.googleapis.com

# Build the container image
echo "Building container image..."
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --timeout=20m

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 1 \
    --min-instances 0 \
    --no-allow-unauthenticated \
    --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${GCS_BUCKET},AGENT_ENGINE_ID=${AGENT_ENGINE_ID},BIFROST_URL=${BIFROST_URL}"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Next steps:"
echo "1. Set up Cloud Scheduler to trigger the service daily:"
echo "   ./setup_scheduler.sh"
echo ""
echo "2. Test the service manually:"
echo "   gcloud run services invoke ${SERVICE_NAME} --region ${REGION}"
echo "=========================================="
