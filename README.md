# Agent Evaluation Pipeline

Modularized evaluation pipeline using Vertex AI GenAI Evaluation (EvalTask).

## Structure
- `src/`: Core logic and modular utilities.
- `instruction/`: YAML prompt templates for each metric.
- `output/`: Evaluation results (JSONL).
- `requirements.txt`: Project dependencies.

## Setup
1. Create a `.env` file from `.env.example`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the evaluation pipeline:
```bash
python -m src.main --project <GCP_PROJECT_ID>
```

### Arguments
- `--project`: GCP Project ID.
- `--location`: Vertex AI location (default: us-central1).
- `--judge-model`: LLM model to use as judge (default: gemini-1.5-pro).
- `--out`: Output filename (default: evaluation_results.jsonl).
- `--dry-run`: Parse data without calling Vertex API.
