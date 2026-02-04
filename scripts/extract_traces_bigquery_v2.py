import os
import json
import argparse
import subprocess
from collections import defaultdict, OrderedDict
from google.oauth2 import credentials as oauth2_credentials
from google.cloud import bigquery

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "product-research-460317")
BQ_LOGS_TABLE = "product-research-460317.all_logs._AllLogs"


def normalize_trace(raw_trace):
    """
    Transform raw BigQuery log entries into a rich, story-like agent_traces format.
    Each trace tells the complete story of user->agents->tools->response.
    """
    log_entries = raw_trace.get("log_entries", [])
    if not log_entries:
        return None

    # === PHASE 1: Extract all data from logs ===

    # Agent data keyed by span_id
    agents = defaultdict(lambda: {
        "agent_name": "",
        "system_prompt": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": [],  # List of {name, args, output, status}
        "text_output": None
    })

    # Pending tool calls waiting for response (global list for cross-span matching)
    pending_tools_by_name = defaultdict(list)  # tool_name -> [{span_id, args, output, status}, ...]

    # Knowledge base content
    kb_usecases = None
    kb_faqs = None
    kb_instructions = []

    # Extracted metadata
    metadata = {
        "domain": "",
        "profile_id": "",
        "issue_id": "",
        "user_message": "",
        "final_response": ""
    }

    # Tool failures
    tool_failures = []

    for entry in log_entries:
        log_type = entry.get("log_name", "").split("/")[-1]
        span_id = entry.get("span_id", "")
        payload = entry.get("json_payload")

        if not payload:
            continue
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except:
                continue

        # --- SYSTEM MESSAGE: Extract agent name ---
        if log_type == "gen_ai.system.message":
            content = payload.get("content", "")
            if 'internal name is "' in content:
                agent_name = content.split('internal name is "')[1].split('"')[0]
                agents[span_id]["agent_name"] = agent_name
                agents[span_id]["system_prompt"] = content[:500]  # First 500 chars

        # --- USER MESSAGE: Extract user input and function responses ---
        elif log_type == "gen_ai.user.message":
            content = payload.get("content", {})

            if isinstance(content, dict):
                parts = content.get("parts", [])
                for part in parts:
                    if not isinstance(part, dict):
                        continue

                    # User text input
                    text = part.get("text")
                    if text and not metadata["user_message"]:
                        metadata["user_message"] = text

                    # Function response (tool output)
                    func_resp = part.get("function_response")
                    if func_resp:
                        func_name = func_resp.get("name", "")
                        response = func_resp.get("response", {})

                        # Match with pending tool call (by name, across all spans)
                        for tool in pending_tools_by_name.get(func_name, []):
                            if tool.get("output") is None:
                                # Build rich tool output
                                is_error = response.get("isError", False)
                                tool["output"] = {
                                    "content": response.get("content", [{"type": "text", "text": response}]),
                                    "structuredContent": response.get("structuredContent", response),
                                    "isError": is_error
                                }
                                tool["status"] = "failure" if is_error else "success"

                                # Extract knowledge base from successful tool responses
                                structured = response.get("structuredContent", response)
                                if isinstance(structured, dict):
                                    if func_name == "list_available_usecases" and structured.get("usecases_list"):
                                        kb_usecases = structured
                                    elif func_name == "get_faqs" and structured.get("faqs_list"):
                                        kb_faqs = structured
                                    elif func_name == "get_usecase_instruction" and structured.get("instruction"):
                                        kb_instructions.append(structured)

                                # Track failures
                                if is_error:
                                    tool_failures.append({
                                        "agent": agents[span_id]["agent_name"],
                                        "tool": func_name,
                                        "error": response
                                    })
                                break

                        # Also add to agent's tool list if not already there
                        found = False
                        for t in agents[span_id]["tool_calls"]:
                            if t["name"] == func_name and t.get("output") is None:
                                t["output"] = {
                                    "content": response.get("content", [{"type": "text", "text": response}]),
                                    "structuredContent": response.get("structuredContent", response),
                                    "isError": response.get("isError", False)
                                }
                                t["status"] = "failure" if response.get("isError") else "success"
                                found = True
                                break

            elif isinstance(content, str) and not metadata["user_message"]:
                metadata["user_message"] = content

        # --- CHOICE: Extract function calls and text responses ---
        elif log_type == "gen_ai.choice":
            content = payload.get("content", {})

            if isinstance(content, dict):
                parts = content.get("parts", [])
                for part in parts:
                    if not isinstance(part, dict):
                        continue

                    # Function call
                    func_call = part.get("function_call")
                    if func_call:
                        tool_name = func_call.get("name", "")
                        tool_args = func_call.get("args", {})

                        tool_entry = {
                            "name": tool_name,
                            "args": tool_args,
                            "output": None,
                            "status": "pending"
                        }

                        # Track for matching with response (by name for cross-span matching)
                        pending_tools_by_name[tool_name].append(tool_entry)
                        agents[span_id]["tool_calls"].append(tool_entry)

                        # Extract metadata from tool args
                        if tool_args.get("domain") and not metadata["domain"]:
                            metadata["domain"] = tool_args["domain"]
                        if tool_args.get("profile_id") and not metadata["profile_id"]:
                            metadata["profile_id"] = tool_args["profile_id"]
                        if tool_args.get("issue_pid") and not metadata["issue_id"]:
                            metadata["issue_id"] = str(tool_args["issue_pid"])
                        if tool_args.get("issue_id") and not metadata["issue_id"]:
                            metadata["issue_id"] = str(tool_args["issue_id"])

                    # Text response (agent output)
                    text = part.get("text")
                    if text and not part.get("function_call"):
                        agents[span_id]["text_output"] = text

                        # Set as final response if it's user-facing (not JSON)
                        is_json = text.strip().startswith("{") and text.strip().endswith("}")
                        current_is_json = metadata["final_response"].strip().startswith("{") if metadata["final_response"] else True

                        if not is_json and (current_is_json or len(text) > len(metadata["final_response"])):
                            metadata["final_response"] = text

    # === PHASE 2: Build the output structure ===

    # Build knowledge_base array with rich content
    knowledge_base = []
    if kb_usecases:
        knowledge_base.append([{"type": "text", "text": kb_usecases}])
    if kb_faqs:
        knowledge_base.append([{"type": "text", "text": kb_faqs}])
    for instr in kb_instructions:
        knowledge_base.append([{"type": "text", "text": instr}])

    # Build agent_info with full tool details
    agent_info = []
    all_tool_calls = []
    total_input_tokens = 0
    total_output_tokens = 0

    # Process agents in order they appeared
    seen_agents = set()
    for span_id, agent_data in agents.items():
        agent_name = agent_data["agent_name"]
        if not agent_name or agent_name in seen_agents:
            continue
        seen_agents.add(agent_name)

        # Build tool_info with rich output
        tool_info = []
        for tool in agent_data["tool_calls"]:
            tool_entry = {
                "tool_name": tool["name"],
                "tool_args": tool["args"],
                "tool_output": tool["output"] if tool["output"] else {"result": "Analysis processed successfully."},
                "tool_status": tool["status"]
            }
            tool_info.append(tool_entry)
            all_tool_calls.append(tool_entry)

        agent_entry = {
            "agent_name": agent_name,
            "agent_input": metadata["user_message"],
            "agent_output": agent_data["text_output"],
            "prompt_version": None,
            "total_input_tokens": agent_data["input_tokens"],
            "total_output_tokens": agent_data["output_tokens"],
            "tool_info": tool_info
        }
        agent_info.append(agent_entry)
        total_input_tokens += agent_data["input_tokens"]
        total_output_tokens += agent_data["output_tokens"]

    # Add orchestrator agent as first entry (aggregates everything)
    if agent_info:
        orchestrator = {
            "agent_name": "sequential_orchestrator_agent",
            "agent_input": metadata["user_message"],
            "agent_output": metadata["final_response"],
            "prompt_version": None,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tool_info": all_tool_calls.copy()
        }
        agent_info.insert(0, orchestrator)

    # If no agents found, create default
    if not agent_info:
        agent_info.append({
            "agent_name": "unknown_agent",
            "agent_input": metadata["user_message"],
            "agent_output": metadata["final_response"],
            "prompt_version": None,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "tool_info": []
        })

    # Build final result
    result = {
        "timestamp": str(raw_trace.get("start_time", "")),
        "trace_id": raw_trace.get("trace_id", ""),
        "session_id": "",  # Not available in logs
        "issue_id": metadata["issue_id"],
        "domain": metadata["domain"],
        "message_id": "",  # Not available in logs
        "profile_id": metadata["profile_id"],
        "user_message": metadata["user_message"],
        "agent_response": metadata["final_response"],
        "agent_type": "multi_agent_system",
        "knowledge_base": knowledge_base,
        "agent_info": agent_info,
        "tool_failures": tool_failures,
        "model_failures": []
    }

    return result


def fetch_full_distributed_trace(project_id, days, limit, agent_engine_id):
    """Fetch trace data from BigQuery."""
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        text=True
    ).strip()
    credentials = oauth2_credentials.Credentials(token=token)
    client = bigquery.Client(project=project_id, credentials=credentials, location="US")

    query = f"""
    WITH
    TargetTraces AS (
        SELECT DISTINCT
            trace,
            REGEXP_EXTRACT(trace, r'/traces/(.+)$') as trace_id
        FROM `{BQ_LOGS_TABLE}`
        WHERE
            timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
            AND JSON_VALUE(resource.labels, '$.reasoning_engine_id') = @agent_engine_id
            AND trace IS NOT NULL
        LIMIT @limit
    ),
    TraceLogs AS (
        SELECT
            t.trace_id,
            l.trace,
            l.span_id,
            l.timestamp,
            l.log_name,
            l.severity,
            l.json_payload
        FROM `{BQ_LOGS_TABLE}` l
        INNER JOIN TargetTraces t ON l.trace = t.trace
        WHERE l.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    )
    SELECT
        trace_id,
        trace,
        MIN(timestamp) as start_time,
        MAX(timestamp) as end_time,
        COUNT(*) as log_count,
        ARRAY_AGG(STRUCT(
            timestamp, log_name, severity, span_id, json_payload
        ) ORDER BY timestamp ASC) as log_entries
    FROM TraceLogs
    GROUP BY trace_id, trace
    ORDER BY start_time DESC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("days", "INT64", days),
            bigquery.ScalarQueryParameter("agent_engine_id", "STRING", agent_engine_id),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )

    print(f"Executing Query against {BQ_LOGS_TABLE}...")
    try:
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())
        print(f"Success! Found {len(results)} trace chains.")
        return results
    except Exception as e:
        print(f"\n[!] BigQuery Error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract agent traces from BigQuery")
    parser.add_argument("--limit", type=int, default=100, help="Max traces to fetch")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    parser.add_argument("--output", type=str, default="full_distributed_traces.jsonl")
    parser.add_argument("--raw", action="store_true", help="Output raw traces")
    args = parser.parse_args()

    AGENT_ID = "3122210601628073984"
    data = fetch_full_distributed_trace(PROJECT_ID, args.days, args.limit, AGENT_ID)

    if data:
        count = 0
        with open(args.output, "w") as f:
            for row in data:
                raw_trace = dict(row)
                raw_trace["log_entries"] = [dict(e) for e in raw_trace.get("log_entries", [])]

                if args.raw:
                    f.write(json.dumps(raw_trace, default=str) + "\n")
                    count += 1
                else:
                    normalized = normalize_trace(raw_trace)
                    if normalized and normalized.get("user_message") and normalized.get("agent_response"):
                        f.write(json.dumps(normalized, default=str) + "\n")
                        count += 1

        print(f"Saved {count} traces to {args.output}")


if __name__ == "__main__":
    main()
