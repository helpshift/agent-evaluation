import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

import vertexai
from vertexai.preview.evaluation import EvalTask

from .utils import gen_eval_id, extract_confidence_and_rationale
from .data_loader import choose_input_source, load_trace_rows
from .session_builder import group_traces_to_sessions, build_eval_dataframe
from .metrics_factory import build_pointwise_metrics, load_agent_instruction_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

METRIC_NAMES = [
    "safety_score",
    "pii_score",
    "hallucination_score",
    "action_completion_score",
    "groundedness_score",
    "multi_turn_chat_quality_score",
]

def run_eval_task(
    project: str,
    location: str,
    judge_model: str,
    experiment: str,
    df: pd.DataFrame,
    metric_list: list,
    qps: float,
    retry_timeout: int,
) -> pd.DataFrame:
    vertexai.init(project=project, location=location)
    # BYOR: No model parameter in evaluate() when dataset has 'response' column.
    # The judge is handled by the backend evaluation service.
    task = EvalTask(
        dataset=df, 
        metrics=metric_list, 
        experiment=experiment
    )
    result = task.evaluate(
        evaluation_service_qps=qps,
        retry_timeout=retry_timeout,
    )
    return result.metrics_table

def build_output_rows(
    base_df: pd.DataFrame,
    table_main: pd.DataFrame,
    table_ground: Optional[pd.DataFrame],
    judge_model: str,
    metrics_config: dict,
) -> List[Dict[str, Any]]:
    main_idx = {str(r["issue_id"]): r for _, r in table_main.iterrows()} if table_main is not None else {}
    ground_idx = {str(r["issue_id"]): r for _, r in table_ground.iterrows()} if table_ground is not None else {}

    out_rows = []
    for _, base in base_df.iterrows():
        issue_id = str(base["issue_id"])
        
        # Base metadata as requested
        record = {
            "timestamp": base.get("timestamp_start"),
            "evaluation_id": gen_eval_id("session"),
            "domain": base.get("domain"),
            "issue_id": issue_id,
            "agent_version": base.get("agent_version"),
            "latency for evaluation": json.loads(base.get("latency_json", "{}")),
            "time_to_first_response": base.get("ttfr", 0.0),
            "judge_total_tokens_cost": 0.0, # Placeholder (not directly provided by SDK per session)
            "turn_count": int(base.get("turn_count", 0)),
            "count_user_msg": int(base.get("count_user_msg", 0)),
            "count_assistant_msg": int(base.get("count_assistant_msg", 0)),
            "count_tool_call": int(base.get("count_tool_call", 0)),
            "count_tool_error": int(base.get("count_tool_error", 0)),
            "count_model_call": int(base.get("count_model_call", 0)),
            "count_model_error": int(base.get("count_model_error", 0)),
        }

        row_main = main_idx.get(issue_id, {})
        row_ground = ground_idx.get(issue_id, {})

        # Mapping for output keys (consistent with user request)
        metric_keys = {
            "safety_score": "safety_score",
            "pii_score": "pii_score",
            "hallucination_score": "hallucination_score",
            "action_completion_score": "action_completion_score",
            "groundedness_score": "groundedness_score",
            "multi_turn_chat_quality_score": "multi-turn_chat_quality_score"
        }

        for m_id, out_key in metric_keys.items():
            row = row_ground if m_id == "groundedness_score" else row_main
            score_col = f"{m_id}/score"
            expl_col = f"{m_id}/explanation"

            if score_col in row:
                conf, rationale = extract_confidence_and_rationale(row.get(expl_col))
                
                # Fetch tokens if available in SDK output table
                # Note: Exact token reporting varies by SDK version
                in_tokens = row.get(f"{m_id}/input_token_count") or row.get(f"{m_id}/input_tokens")
                out_tokens = row.get(f"{m_id}/output_token_count") or row.get(f"{m_id}/output_tokens")

                record[out_key] = {
                    "judge_model": judge_model,
                    "judge_prompt_version": metrics_config.get(m_id).prompt_version if metrics_config.get(m_id) else "1.0",
                    "judge_input_tokens": int(in_tokens) if in_tokens is not None else None,
                    "judge_output_tokens": int(out_tokens) if out_tokens is not None else None,
                    "score": float(row[score_col]) if row[score_col] is not None else None,
                    "confidence": float(conf) if conf is not None else None,
                    "rationale": rationale,
                }
            else:
                record[out_key] = None

        out_rows.append(record)
    return out_rows

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="GCP Project ID")
    ap.add_argument("--location", help="Vertex AI Location")
    ap.add_argument("--judge-model", help="Judge model name")
    ap.add_argument("--out", help="Output filename")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    project = args.project or os.getenv("PROJECT_ID")
    location = args.location or os.getenv("VERTEX_LOCATION", "us-central1")
    judge_model = args.judge_model or os.getenv("JUDGE_MODEL", "gemini-1.5-pro")
    out_file = args.out or os.getenv("OUTPUT_FILE", "evaluation_results.jsonl")
    bucket = os.getenv("EVAL_BUCKET", "evaluation-research")
    fallback_blob = os.getenv("FALLBACK_BLOB", "agent_traces_bq.jsonl")

    if not project:
        raise RuntimeError("PROJECT_ID is required.")

    instruction_dir = Path(__file__).resolve().parent.parent / "instruction"
    agent_instr = load_agent_instruction_text(instruction_dir)
    
    # 1. Load and build dataset (one row per session)
    source = choose_input_source("input.jsonl")
    trace_rows = load_trace_rows(project, bucket, source, fallback_blob)
    sessions_map = group_traces_to_sessions(trace_rows)
    df = build_eval_dataframe(sessions_map, agent_instr)

    if args.dry_run:
        logging.info("Dry run: skipping Vertex calls.")
        print(df.head())
        return

    # 2. Initialize metrics
    metrics_map = build_pointwise_metrics(METRIC_NAMES, instruction_dir)
    metric_objs = [m[0] for m in metrics_map.values()]

    # 3. Run evaluation in one task
    logging.info("Starting session-level evaluation for %d sessions...", len(df))
    table_main = None
    try:
        table_main = run_eval_task(
            project, location, judge_model, "agentic-eval-session-run", 
            df, metric_objs, 0.5, 900
        )
    except Exception as e:
        logging.error("Evaluation task failed: %s", e)
        # Fallback to empty results so we can still see metadata
        table_main = pd.DataFrame(columns=["issue_id"])

    # 4. Construct final output records
    cfgs = {k: v[1] for k, v in metrics_map.items()}
    out_rows = build_output_rows(df, table_main, None, judge_model, cfgs)

    # 5. Write results
    out_path = Path(__file__).resolve().parent.parent / "output" / out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    logging.info("✅ Done. Wrote results to %s", out_path)

if __name__ == "__main__":
    main()
