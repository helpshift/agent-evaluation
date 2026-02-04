# Trace Extraction Service - Cloud Deployment Guide

This guide explains how to deploy the trace extraction service to Google Cloud Run with daily scheduled execution.

## Overview

The trace extraction service:
- Runs on **Google Cloud Run** as a containerized HTTP service
- Triggered **daily** by **Cloud Scheduler**
- Processes traces **incrementally** (only new traces since last run)
- Stores state in **Google Cloud Storage**
- Outputs results to **GCS** in JSONL format

## Architecture

```
Cloud Scheduler (Daily 2 AM UTC)
    ↓ (HTTP GET)
Cloud Run Service
    ↓
extract_traces_v7.py (Incremental Mode)
    ↓
- Cloud Trace API (fetch new traces)
- Cloud Logging API (fetch metadata)
- Cloud Storage (read/write state & data)
```

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **gcloud CLI** installed and configured
3. Required **permissions**:
   - Cloud Run Admin
   - Cloud Scheduler Admin
   - Cloud Build Editor
   - Service Account Admin
   - Storage Admin
   - Trace Viewer
   - Logging Viewer

## Environment Variables

Set these before deployment:

```bash
export GCP_PROJECT_ID="product-research-460317"
export GCP_REGION="us-central1"
export GCS_BUCKET="evaluation-research"
export AGENT_ENGINE_ID="3122210601628073984"
export BIFROST_URL="https://bifrost-service-qa-845835740344.us-central1.run.app/webhook/bifrost_service"
```

## Deployment Steps

### Step 1: Deploy to Cloud Run

Make the deployment script executable and run it:

```bash
chmod +x deploy_to_cloud_run.sh
./deploy_to_cloud_run.sh
```

This script will:
1. Enable required Google Cloud APIs
2. Build the Docker container using Cloud Build
3. Deploy the container to Cloud Run with:
   - 2 GB memory
   - 2 CPUs
   - 1 hour timeout
   - Auto-scaling (0-1 instances)
4. Configure environment variables

### Step 2: Set Up Daily Scheduler

Make the scheduler script executable and run it:

```bash
chmod +x setup_scheduler.sh
./setup_scheduler.sh
```

This script will:
1. Create a service account for the scheduler
2. Grant necessary permissions
3. Create a Cloud Scheduler job that runs daily at 2 AM UTC
4. Configure OIDC authentication

### Step 3: Verify Deployment

Check the Cloud Run service:

```bash
gcloud run services describe trace-extraction-service \
    --region us-central1 \
    --platform managed
```

Check the Cloud Scheduler job:

```bash
gcloud scheduler jobs describe trace-extraction-daily \
    --location us-central1
```

## Configuration

### Changing the Schedule

The default schedule is `0 2 * * *` (2 AM UTC daily). To change it:

```bash
# Export custom schedule (cron format)
export SCHEDULE="0 */6 * * *"  # Every 6 hours
export TIMEZONE="America/New_York"

# Re-run the scheduler setup
./setup_scheduler.sh
```

Common schedules:
- Daily at midnight UTC: `0 0 * * *`
- Every 6 hours: `0 */6 * * *`
- Hourly: `0 * * * *`
- Weekly on Mondays at 3 AM: `0 3 * * 1`

### Modifying Resource Limits

Edit the `deploy_to_cloud_run.sh` script and change:

```bash
--memory 2Gi \        # Increase if needed
--cpu 2 \             # Increase for faster processing
--timeout 3600 \      # Max 1 hour
--max-instances 1 \   # Increase for parallel processing
```

Then redeploy:

```bash
./deploy_to_cloud_run.sh
```

## How Incremental Processing Works

### First Run
1. Fetches all traces from the last 7 days
2. Processes and uploads to GCS as `incremental_traces.jsonl`
3. Saves state to `gs://evaluation-research/state/last_processed.json`:
   ```json
   {
     "last_processed_time": "2026-01-28T10:30:45+00:00",
     "last_run_time": "2026-01-28T10:35:12+00:00",
     "traces_processed": 150
   }
   ```

### Subsequent Runs
1. Reads state file to get `last_processed_time`
2. Fetches only traces newer than that timestamp
3. Appends new traces to existing GCS file (avoiding duplicates)
4. Updates state with new `last_processed_time`

### State File Location

The state is stored in GCS at:
```
gs://evaluation-research/state/last_processed.json
```

You can customize this with the `STATE_FILE_GCS` environment variable.

## Manual Execution

### Trigger the Job Manually

Run the scheduled job immediately:

```bash
gcloud scheduler jobs run trace-extraction-daily --location us-central1
```

### Invoke Cloud Run Service Directly

```bash
gcloud run services invoke trace-extraction-service \
    --region us-central1
```

### Run Locally (for Testing)

```bash
# Set environment variables
export GCP_PROJECT_ID="product-research-460317"
export GCS_BUCKET="evaluation-research"
export AGENT_ENGINE_ID="3122210601628073984"

# Run in incremental mode
python scripts/extract_traces_v7.py --incremental --limit 100
```

## Monitoring

### View Logs

Cloud Run logs:

```bash
gcloud logging read \
    "resource.type=cloud_run_revision AND resource.labels.service_name=trace-extraction-service" \
    --limit 50 \
    --format json
```

Cloud Scheduler logs:

```bash
gcloud scheduler jobs logs trace-extraction-daily --location us-central1
```

### View in Cloud Console

1. **Cloud Run Logs**: https://console.cloud.google.com/run
2. **Cloud Scheduler**: https://console.cloud.google.com/cloudscheduler
3. **Cloud Build History**: https://console.cloud.google.com/cloud-build/builds

### Check State File

```bash
gsutil cat gs://evaluation-research/state/last_processed.json
```

### Check Output Data

```bash
# List all traces
gsutil ls gs://evaluation-research/*.jsonl

# Download the latest
gsutil cp gs://evaluation-research/incremental_traces.jsonl .

# View summary
cat incremental_traces.jsonl | wc -l
```

## Output Files

### Trace Data
- **Location**: `gs://evaluation-research/incremental_traces.jsonl`
- **Format**: JSONL (one JSON object per line)
- **Fields**: See extract_traces_v7.py documentation

### Content Mapping
- **Location**: `gcs_data/content_mapping.json` (local during processing)
- **Purpose**: Stores instruction and FAQ metadata for easy lookup
- **Structure**:
  ```json
  {
    "instructions": {
      "instruction_123": {
        "instruction_id": "instruction_123",
        "domain": "example.com",
        "instruction_version": "v2",
        "usecase_id": "use_case_456"
      }
    },
    "faqs": {
      "faq_789": {
        "faq_id": "faq_789",
        "faq_slug": "2355-how-to-reset-password",
        "title": "How to reset password",
        "domain": "example.com"
      }
    }
  }
  ```

Note: Only metadata (identifiers) is stored in content_mapping. Full instruction text and FAQ body content remain in the JSONL trace files.

### State File
- **Location**: `gs://evaluation-research/state/last_processed.json`
- **Purpose**: Tracks last processed timestamp for incremental updates
- **Structure**:
  ```json
  {
    "last_processed_time": "2026-01-28T10:30:45+00:00",
    "last_run_time": "2026-01-28T10:35:12+00:00",
    "traces_processed": 150
  }
  ```

## Troubleshooting

### Issue: "Permission denied" errors

**Solution**: Ensure the Cloud Run service account has necessary permissions:

```bash
# Get the service account email
SA_EMAIL=$(gcloud run services describe trace-extraction-service \
    --region us-central1 \
    --format 'value(spec.template.spec.serviceAccountName)')

# Grant permissions
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
    --member "serviceAccount:${SA_EMAIL}" \
    --role "roles/cloudtrace.user"

gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
    --member "serviceAccount:${SA_EMAIL}" \
    --role "roles/logging.viewer"

gsutil iam ch serviceAccount:${SA_EMAIL}:objectAdmin \
    gs://${GCS_BUCKET}
```

### Issue: Job times out after 1 hour

**Solution**: Either increase timeout (max 1 hour for Cloud Run) or reduce the limit:

```bash
# Increase limit in server.py
# Change: "--limit", "1000"
# To: "--limit", "500"

# Then redeploy
./deploy_to_cloud_run.sh
```

### Issue: Memory errors during processing

**Solution**: Increase memory allocation:

```bash
# Edit deploy_to_cloud_run.sh
# Change: --memory 2Gi
# To: --memory 4Gi

# Redeploy
./deploy_to_cloud_run.sh
```

### Issue: Scheduler job fails with authentication errors

**Solution**: Recreate the service account and permissions:

```bash
# Delete and recreate the scheduler
gcloud scheduler jobs delete trace-extraction-daily --location us-central1 --quiet

# Re-run setup
./setup_scheduler.sh
```

### Issue: No traces found

**Solution**: Check filters and date range:

1. Verify `AGENT_ENGINE_ID` is correct
2. Verify `BIFROST_URL` matches your environment
3. Check if there are traces in the time range:
   ```bash
   # View traces in Cloud Console
   # https://console.cloud.google.com/traces/list
   ```

### Issue: State file is corrupted

**Solution**: Reset the state file:

```bash
# Backup current state
gsutil cp gs://evaluation-research/state/last_processed.json \
    ./last_processed_backup.json

# Delete state to start fresh
gsutil rm gs://evaluation-research/state/last_processed.json

# Next run will start from the past 7 days
```

## Cost Optimization

### Current Costs (Estimated)

- **Cloud Run**: ~$0.50/month (1 invocation/day, minimal runtime)
- **Cloud Scheduler**: $0.10/month (1 job)
- **Cloud Storage**: ~$0.50/month (assuming 10 GB data)
- **Cloud Trace API**: Free tier (first 2.5M traces/month)
- **Cloud Logging API**: Free tier (first 50 GB/month)

**Total**: ~$1-2/month

### Reducing Costs

1. **Reduce frequency**: Change from daily to weekly
2. **Use spot instances**: Not available for Cloud Run, but minimal cost impact
3. **Clean up old data**: Set GCS lifecycle policies

```bash
# Set 90-day retention policy
gsutil lifecycle set - gs://${GCS_BUCKET} <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["incremental_traces"]
        }
      }
    ]
  }
}
EOF
```

## Cleanup

To remove all deployed resources:

```bash
# Delete Cloud Scheduler job
gcloud scheduler jobs delete trace-extraction-daily \
    --location us-central1 \
    --quiet

# Delete Cloud Run service
gcloud run services delete trace-extraction-service \
    --region us-central1 \
    --quiet

# Delete service account
SA_EMAIL="trace-extraction-scheduler@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts delete ${SA_EMAIL} --quiet

# (Optional) Delete GCS data
# gsutil rm -r gs://${GCS_BUCKET}/incremental_traces.jsonl
# gsutil rm -r gs://${GCS_BUCKET}/state/
```

## Support

For issues or questions:
1. Check logs in Cloud Console
2. Review this documentation
3. Contact the development team
