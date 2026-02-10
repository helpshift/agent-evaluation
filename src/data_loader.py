import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.cloud import storage
from .utils import safe_json_loads

class GCSClient:
    def __init__(self, project: Optional[str] = None):
        self.client = storage.Client(project=project)

    def blob_exists(self, bucket: str, blob_name: str) -> bool:
        b = self.client.bucket(bucket)
        blob = b.blob(blob_name)
        return blob.exists(self.client)

    def download_blob_to_path(self, bucket: str, blob_name: str, local_path: Path) -> None:
        b = self.client.bucket(bucket)
        blob = b.blob(blob_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

def choose_input_source(local_input_name: str) -> str:
    local_path = Path(__file__).resolve().parent.parent / local_input_name
    if local_path.exists() and local_path.is_file():
        logging.info("Using local input file: %s", local_path)
        return f"LOCAL::{str(local_path)}"
    return "GCS_FALLBACK"

def load_trace_rows(project: str, bucket: str, source: str, fallback_blob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if source.startswith("LOCAL::"):
        local_file = Path(source.split("LOCAL::", 1)[1])
        with local_file.open("r", encoding="utf-8") as f:
            for line in f:
                r = safe_json_loads(line)
                if r is not None:
                    rows.append(r)
        return rows

    gcs = GCSClient(project=project)
    if not gcs.blob_exists(bucket, fallback_blob):
        raise RuntimeError(f"GCS blob not found: gs://{bucket}/{fallback_blob}")

    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / Path(fallback_blob).name
        gcs.download_blob_to_path(bucket, fallback_blob, local)
        with local.open("r", encoding="utf-8") as f:
            for line in f:
                r = safe_json_loads(line)
                if r is not None:
                    rows.append(r)
    return rows
