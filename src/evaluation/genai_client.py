"""
GenAI Client Setup for Vertex AI Evaluation.

This module initializes the official Vertex AI GenAI Client
for production-grade agent evaluation using the latest SDK.
"""

import os
from typing import Optional

import vertexai
from vertexai import Client
from google.genai import types as genai_types


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
GCS_DEST_BUCKET = os.environ.get("GCS_EVAL_BUCKET", "gs://your-bucket/eval-results/")


# ============================================================
# CLIENT INITIALIZATION
# ============================================================

def get_genai_client(
    project_id: Optional[str] = None,
    location: Optional[str] = None
) -> Client:
    """
    Initialize and return the Vertex AI GenAI Client.
    
    Args:
        project_id: GCP Project ID. Defaults to env var GOOGLE_CLOUD_PROJECT.
        location: GCP Region. Defaults to env var GOOGLE_CLOUD_LOCATION.
    
    Returns:
        Initialized Vertex AI Client for evaluation operations.
    """
    project = project_id or PROJECT_ID
    loc = location or LOCATION
    
    # Initialize Vertex AI SDK
    vertexai.init(project=project, location=loc)
    
    # Create GenAI Client with v1beta1 API version for latest features
    client = Client(
        project=project,
        location=loc,
        http_options=genai_types.HttpOptions(api_version="v1beta1"),
    )
    
    return client


# ============================================================
# SINGLETON CLIENT INSTANCE
# ============================================================

_client_instance: Optional[Client] = None


def get_client() -> Client:
    """Get or create the singleton GenAI client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = get_genai_client()
    return _client_instance


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_model_inference(
    model: str,
    dataset,
    dest: Optional[str] = None
):
    """
    Generate model responses for a dataset using run_inference().
    
    Args:
        model: Model name (e.g., "gemini-2.5-flash", "gpt-4o")
        dataset: Pandas DataFrame or GCS path to evaluation prompts
        dest: Optional GCS destination for saving results
    
    Returns:
        EvaluationDataset with generated responses
    """
    client = get_client()
    
    config = {"dest": dest} if dest else None
    
    return client.evals.run_inference(
        model=model,
        src=dataset,
        config=config
    )


def run_agent_inference(
    agent_resource_name: str,
    dataset,
    dest: Optional[str] = None
):
    """
    Generate agent responses for a dataset.
    
    Args:
        agent_resource_name: Deployed agent resource name from Agent Engine
        dataset: Pandas DataFrame with prompts and session_inputs
        dest: Optional GCS destination
    
    Returns:
        EvaluationDataset with agent responses and intermediate_events
    """
    client = get_client()
    
    config = {"dest": dest} if dest else None
    
    return client.evals.run_inference(
        agent=agent_resource_name,
        src=dataset,
        config=config
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "get_genai_client",
    "get_client",
    "run_model_inference",
    "run_agent_inference",
    "PROJECT_ID",
    "LOCATION",
    "GCS_DEST_BUCKET",
]
