import os
from google.cloud import storage

def download_jsonl_files_to_gcs_data(bucket_name: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    jsonl_files = [blob for blob in blobs if blob.name.lower().endswith(".jsonl")]
    destination_folder = os.path.join(os.getcwd(), "gcs_data")
    os.makedirs(destination_folder, exist_ok=True)

    for blob in jsonl_files:
        destination_path = os.path.join(destination_folder, os.path.basename(blob.name))
        blob.download_to_filename(destination_path)
        print(f"✅ Downloaded: {blob.name} -> {destination_path}")


if __name__ == "__main__":
    BUCKET = "evaluation-research"
    download_jsonl_files_to_gcs_data(BUCKET)
