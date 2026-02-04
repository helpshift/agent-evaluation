
# ============================================
# AI PLATFORM ROLES
# ============================================
echo "Restoring AI Platform roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:product-research-460317@appspot.gserviceaccount.com" --role="roles/aiplatform.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform-re.iam.gserviceaccount.com" --role="roles/aiplatform.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-vertex-nb.iam.gserviceaccount.com" --role="roles/aiplatform.colabServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform-vm.iam.gserviceaccount.com" --role="roles/aiplatform.notebookServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-vertex-rag.iam.gserviceaccount.com" --role="roles/aiplatform.ragServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-vertex-eval.iam.gserviceaccount.com" --role="roles/aiplatform.rapidevalServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform-re.iam.gserviceaccount.com" --role="roles/aiplatform.reasoningEngineServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform.iam.gserviceaccount.com" --role="roles/aiplatform.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/aiplatform.user" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/aiplatform.user" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-app@product-research-460317.iam.gserviceaccount.com" --role="roles/aiplatform.user" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/aiplatform.user" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform-re.iam.gserviceaccount.com" --role="roles/aiplatform.user" --quiet

# ============================================
# ARTIFACT REGISTRY
# ============================================
echo "Restoring Artifact Registry roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-artifactregistry.iam.gserviceaccount.com" --role="roles/artifactregistry.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/artifactregistry.writer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/artifactregistry.writer" --quiet

# ============================================
# BIGQUERY
# ============================================
echo "Restoring BigQuery roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/bigquery.dataEditor" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/bigquery.dataViewer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/bigquery.jobUser" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/bigquery.user" --quiet

# ============================================
# CLOUD BUILD
# ============================================
echo "Restoring Cloud Build roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344@cloudbuild.gserviceaccount.com" --role="roles/cloudbuild.builds.builder" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/cloudbuild.builds.builder" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-cloudbuild.iam.gserviceaccount.com" --role="roles/cloudbuild.serviceAgent" --quiet

# ============================================
# CLOUD FUNCTIONS
# ============================================
echo "Restoring Cloud Functions roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcf-admin-robot.iam.gserviceaccount.com" --role="roles/cloudfunctions.serviceAgent" --quiet

# ============================================
# CLOUD SCHEDULER
# ============================================
echo "Restoring Cloud Scheduler roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-cloudscheduler.iam.gserviceaccount.com" --role="roles/cloudscheduler.serviceAgent" --quiet

# ============================================
# CLOUD TRACE
# ============================================
echo "Restoring Cloud Trace roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/cloudtrace.agent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-app@product-research-460317.iam.gserviceaccount.com" --role="roles/cloudtrace.agent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/cloudtrace.agent" --quiet

# ============================================
# COMPUTE ENGINE
# ============================================
echo "Restoring Compute Engine roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform-re.iam.gserviceaccount.com" --role="roles/compute.networkAdmin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform.iam.gserviceaccount.com" --role="roles/compute.networkAdmin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform.iam.gserviceaccount.com" --role="roles/compute.networkUser" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform.iam.gserviceaccount.com" --role="roles/compute.networkViewer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="user:pnq2.c02fq7rdmd6m@helpshift.com" --role="roles/compute.osAdminLogin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="user:pnq2.c02g924smd6m@helpshift.com" --role="roles/compute.osAdminLogin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@compute-system.iam.gserviceaccount.com" --role="roles/compute.serviceAgent" --quiet

# ============================================
# CONTAINER / GKE
# ============================================
echo "Restoring Container/GKE roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@container-engine-robot.iam.gserviceaccount.com" --role="roles/container.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@container-analysis.iam.gserviceaccount.com" --role="roles/containeranalysis.ServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@containerregistry.iam.gserviceaccount.com" --role="roles/containerregistry.ServiceAgent" --quiet

# ============================================
# DATA SERVICES
# ============================================
echo "Restoring Data services roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-dataform.iam.gserviceaccount.com" --role="roles/dataform.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-dataplex.iam.gserviceaccount.com" --role="roles/dataplex.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@dataproc-accounts.iam.gserviceaccount.com" --role="roles/dataproc.serviceAgent" --quiet

# ============================================
# DISCOVERY ENGINE
# ============================================
echo "Restoring Discovery Engine roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:setu-discovery-sa@product-research-460317.iam.gserviceaccount.com" --role="roles/discoveryengine.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-app@product-research-460317.iam.gserviceaccount.com" --role="roles/discoveryengine.editor" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/discoveryengine.editor" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:setu-discovery-sa@product-research-460317.iam.gserviceaccount.com" --role="roles/discoveryengine.editor" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-discoveryengine.iam.gserviceaccount.com" --role="roles/discoveryengine.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:setu-discovery-sa@product-research-460317.iam.gserviceaccount.com" --role="roles/discoveryengine.user" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:setu-discovery-sa@product-research-460317.iam.gserviceaccount.com" --role="roles/discoveryengine.viewer" --quiet

# ============================================
# DLP
# ============================================
echo "Restoring DLP roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:pii-service-product-rsearch@product-research-460317.iam.gserviceaccount.com" --role="roles/dlp.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@dlp-api.iam.gserviceaccount.com" --role="roles/dlp.serviceAgent" --quiet

# ============================================
# EVENTARC
# ============================================
echo "Restoring Eventarc roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-eventarc.iam.gserviceaccount.com" --role="roles/eventarc.serviceAgent" --quiet

# ============================================
# FIREBASE
# ============================================
echo "Restoring Firebase roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-firebase.iam.gserviceaccount.com" --role="roles/firebase.managementServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:firebase-adminsdk-fbsvc@product-research-460317.iam.gserviceaccount.com" --role="roles/firebase.sdkAdminServiceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:firebase-adminsdk-fbsvc@product-research-460317.iam.gserviceaccount.com" --role="roles/firebaseauth.admin" --quiet

# ============================================
# IAM
# ============================================
echo "Restoring IAM roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:firebase-adminsdk-fbsvc@product-research-460317.iam.gserviceaccount.com" --role="roles/iam.serviceAccountTokenCreator" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/iam.serviceAccountUser" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/iam.serviceAccountUser" --quiet

# ============================================
# IAP
# ============================================
echo "Restoring IAP roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="user:pnq2.c02fq7rdmd6m@helpshift.com" --role="roles/iap.tunnelResourceAccessor" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="user:pnq2.c02g924smd6m@helpshift.com" --role="roles/iap.tunnelResourceAccessor" --quiet

# ============================================
# LOGGING
# ============================================
echo "Restoring Logging roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/logging.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/logging.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/logging.logWriter" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-app@product-research-460317.iam.gserviceaccount.com" --role="roles/logging.logWriter" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/logging.logWriter" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-logging.iam.gserviceaccount.com" --role="roles/logging.serviceAgent" --quiet

# ============================================
# ML ENGINE
# ============================================
echo "Restoring ML Engine roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:product-research-460317@appspot.gserviceaccount.com" --role="roles/ml.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/ml.developer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/ml.developer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="user:priti.chaudhary@helpshift.com" --role="roles/ml.developer" --quiet

# ============================================
# MODEL ARMOR
# ============================================
echo "Restoring Model Armor roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="user:priti.chaudhary@helpshift.com" --role="roles/modelarmor.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="user:priti.chaudhary@helpshift.com" --role="roles/modelarmor.floorSettingsAdmin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-modelarmor.iam.gserviceaccount.com" --role="roles/modelarmor.serviceAgent" --quiet

# ============================================
# MONITORING
# ============================================
echo "Restoring Monitoring roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/monitoring.alertViewer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/monitoring.metricWriter" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-monitoring-notification.iam.gserviceaccount.com" --role="roles/monitoring.notificationServiceAgent" --quiet

# ============================================
# PUB/SUB
# ============================================
echo "Restoring Pub/Sub roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/pubsub.publisher" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-pubsub.iam.gserviceaccount.com" --role="roles/pubsub.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/pubsub.subscriber" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/pubsub.viewer" --quiet

# ============================================
# REDIS
# ============================================
echo "Restoring Redis roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@cloud-redis.iam.gserviceaccount.com" --role="roles/redis.serviceAgent" --quiet

# ============================================
# CLOUD RUN
# ============================================
echo "Restoring Cloud Run roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/run.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:product-research-460317@appspot.gserviceaccount.com" --role="roles/run.invoker" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@serverless-robot-prod.iam.gserviceaccount.com" --role="roles/run.serviceAgent" --quiet

# ============================================
# SECRET MANAGER
# ============================================
echo "Restoring Secret Manager roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/secretmanager.viewer" --quiet

# ============================================
# SERVICE NETWORKING
# ============================================
echo "Restoring Service Networking roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@service-networking.iam.gserviceaccount.com" --role="roles/servicenetworking.serviceAgent" --quiet

# ============================================
# SERVICE USAGE
# ============================================
echo "Restoring Service Usage roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/serviceusage.serviceUsageConsumer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/serviceusage.serviceUsageConsumer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-app@product-research-460317.iam.gserviceaccount.com" --role="roles/serviceusage.serviceUsageConsumer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-aiplatform-re.iam.gserviceaccount.com" --role="roles/serviceusage.serviceUsageConsumer" --quiet

# ============================================
# STORAGE
# ============================================
echo "Restoring Storage roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:adk-deployment-service@product-research-460317.iam.gserviceaccount.com" --role="roles/storage.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-app@product-research-460317.iam.gserviceaccount.com" --role="roles/storage.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:agents-cb@product-research-460317.iam.gserviceaccount.com" --role="roles/storage.admin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/storage.bucketViewer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/storage.objectAdmin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:cloud-storage-migration-sa-gcs@product-research-460317.iam.gserviceaccount.com" --role="roles/storage.objectAdmin" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:setu-discovery-sa@product-research-460317.iam.gserviceaccount.com" --role="roles/storage.objectUser" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/storage.objectViewer" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:cloud-storage-migration-sa-gcs@product-research-460317.iam.gserviceaccount.com" --role="roles/storage.objectViewer" --quiet

# ============================================
# VPC ACCESS
# ============================================
echo "Restoring VPC Access roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-vpcaccess.iam.gserviceaccount.com" --role="roles/vpcaccess.serviceAgent" --quiet
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:845835740344-compute@developer.gserviceaccount.com" --role="roles/vpcaccess.user" --quiet

# ============================================
# WORKSTATIONS
# ============================================
echo "Restoring Workstations roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-workstations.iam.gserviceaccount.com" --role="roles/workstations.serviceAgent" --quiet

# ============================================
# BINARY AUTHORIZATION
# ============================================
echo "Restoring Binary Authorization roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-binaryauthorization.iam.gserviceaccount.com" --role="roles/binaryauthorization.serviceAgent" --quiet

# ============================================
# CLOUD AI COMPANION
# ============================================
echo "Restoring Cloud AI Companion roles..."
gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:service-845835740344@gcp-sa-cloudaicompanion.iam.gserviceaccount.com" --role="roles/cloudaicompanion.serviceAgent" --quiet

echo ""
echo "========================================"
echo "IAM POLICY RESTORATION COMPLETE"
echo "========================================"
echo ""
echo "Please verify by running:"
echo "gcloud projects get-iam-policy $PROJECT --format=json | jq '.bindings | length'"
