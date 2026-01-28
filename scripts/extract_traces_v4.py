"""
extract_traces_final.py - comprehensive_single_line_extraction

Output: One JSON object per Trace ID containing the full story:
- Global Context (User Message, Final Response, IDs)
- Agent Chain (Chronological list of agents, their inputs, outputs, and tools)
- Failures & Knowledge Base
"""

import os
import re
import json
import uuid
import argparse
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from collections import defaultdict

from google.cloud import trace_v1
from google.cloud import storage
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Constants
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "product-research-460317")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "evaluation-research")
AGENT_ENGINE_ID = os.environ.get("AGENT_ENGINE_ID", "3122210601628073984")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def try_parse_json(data: Any) -> Any:
    """Recursively parses JSON strings into Python objects."""
    if isinstance(data, str):
        stripped = data.strip()
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            try: return try_parse_json(json.loads(stripped))
            except: return data
    elif isinstance(data, dict):
        return {k: try_parse_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [try_parse_json(i) for i in data]
    return data

def safe_get_text(parts: List[Dict]) -> str:
    """Safely extracts text, handling parsed JSON dicts gracefully."""
    text_list = []
    for p in parts:
        if "text" in p:
            val = p["text"]
            if isinstance(val, (dict, list)):
                text_list.append(json.dumps(val))
            elif val is not None:
                text_list.append(str(val))
    return " ".join(text_list)

def extract_ids_from_instruction(instruction: str) -> Dict[str, str]:
    ids = {}
    patterns = {
        "domain_id": r"domain:\s*([^\n]+)",
        "issue_id": r"(?:issue_pid|issue_id):\s*(\d+)",
        "profile_id": r"profile_id:\s*([^\n]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, str(instruction), re.IGNORECASE)
        if match: ids[key] = match.group(1).strip()
    return ids

# ============================================================================
# TRACE PROCESSING
# ============================================================================

def list_traces_safe(project_id: str, limit: int = 50, days: int = 0, agent_engine_id: Optional[str] = None):
    client = trace_v1.TraceServiceClient()
    start_time = None
    if days > 0:
        past_date = datetime.now(timezone.utc) - timedelta(days=days)
        start_time = Timestamp()
        start_time.FromDatetime(past_date)

    filter_str = f"+service.name:{agent_engine_id}" if agent_engine_id else None

    request = trace_v1.ListTracesRequest(
        project_id=project_id,
        view=trace_v1.ListTracesRequest.ViewType.COMPLETE,
        page_size=100,
        start_time=start_time,
        filter=filter_str
    )
    
    traces = []
    print(f"Fetching up to {limit} traces...")
    pager = client.list_traces(request=request)
    
    try:
        count = 0
        for trace in pager:
            traces.append(MessageToDict(trace._pb))
            count += 1
            if count >= limit: break
            if count % 100 == 0: time.sleep(1) 
    except (ResourceExhausted, ServiceUnavailable) as e:
        print(f"WARNING: API Quota hit. Continuing with {len(traces)} traces.")
        
    return traces

def process_consolidated_trace(trace):
    trace_id = trace["traceId"]
    spans = trace.get("spans", [])
    
    # Build Span Tree
    nodes = {span["spanId"]: span for span in spans}
    children = defaultdict(list)
    for span in spans:
        parent = span.get("parentSpanId")
        if parent and parent in nodes: children[parent].append(span["spanId"])

    # Identify Top-Level Agents (Orchestrator, Guard, Core)
    agent_spans = []
    for span in spans:
        labels = span.get("labels", {})
        # Only spans that are agent invocations
        if "invoke_agent" in span.get("name", "") or labels.get("gen_ai.agent.name"):
             # Heuristic: exclude nested tool agents if they don't have a distinct agent name
             agent_spans.append(span)

    # Sort agents chronologically
    agent_spans.sort(key=lambda x: x.get("startTime", ""))

    # --- Initialize The Single Line Record ---
    record = {
        "timestamp": None,
        "trace_id": trace_id,
        "session_id": None,
        "issue_id": None,
        "domain": None,
        "message_id": None,
        "user_message": None,      # Global: The first user input
        "agent_response": None,    # Global: The final agent output
        "agent_type": "multi_agent_system",
        "knowledge_base": [],
        "agent_info": [],          # Detailed timeline
        "tool_failures": [],
        "model_failures": []
    }

    if agent_spans:
        record["timestamp"] = agent_spans[0].get("startTime")

    # --- Process Each Agent in the Chain ---
    for span in agent_spans:
        labels = span.get("labels", {})
        agent_name = labels.get("gen_ai.agent.name")
        if not agent_name: continue

        # Capture Session ID (from any agent)
        if not record["session_id"]: 
            record["session_id"] = labels.get("gcp.vertex.agent.session_id") or labels.get("gen_ai.conversation.id")

        agent_obj = {
            "agent_name": agent_name,
            "agent_input": None,
            "agent_output": None,
            "prompt_version": None,
            "tool_info": []
        }

        # Get all sub-operations (Tools/LLM calls) for this agent
        stack = [span["spanId"]]
        descendants = []
        while stack:
            curr = stack.pop()
            kids = children.get(curr, [])
            descendants.extend(kids)
            stack.extend(kids)
        
        all_sub_spans = [nodes[sid] for sid in descendants]
        all_sub_spans.sort(key=lambda x: x.get("startTime", ""))

        for sub in all_sub_spans:
            sub_labels = sub.get("labels", {})
            name = sub.get("name", "")
            
            # --- LLM Interactions ---
            if "call_llm" in name:
                req = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_request"))
                res = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_response"))
                
                # 1. Extract IDs from Config
                if isinstance(req, dict) and "config" in req:
                     instr = req.get("config", {}).get("system_instruction", "")
                     ids = extract_ids_from_instruction(instr)
                     if ids.get("issue_id"): record["issue_id"] = ids["issue_id"]
                     if ids.get("domain_id"): record["domain"] = ids["domain_id"]

                # 2. Extract User Message
                if isinstance(req, dict):
                    for c in reversed(req.get("contents", [])):
                        if c.get("role") == "user":
                            txt = safe_get_text(c.get("parts", []))
                            if txt and not txt.startswith("For context:"):
                                # If global is empty, take this one
                                if not record["user_message"]: record["user_message"] = txt
                                # Always record for this specific agent
                                agent_obj["agent_input"] = txt
                                break
                
                # 3. Extract Agent Response
                if isinstance(res, dict) and "content" in res:
                    resp_txt = safe_get_text(res["content"].get("parts", []))
                    if resp_txt:
                        record["agent_response"] = resp_txt # Overwrite with latest response
                        agent_obj["agent_output"] = resp_txt

            # --- Tool Executions ---
            if "execute_tool" in name:
                tool_name = sub_labels.get("gen_ai.tool.name") or name.replace("execute_tool ", "")
                args = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_call_args"))
                output = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_response"))
                
                # Status Check
                status = "success"
                if isinstance(output, dict):
                    if output.get("isError") or output.get("status") == "Failure":
                        status = "failure"
                        record["tool_failures"].append({
                            "agent": agent_name,
                            "tool": tool_name,
                            "error": output
                        })

                # IDs from Tool Args
                if isinstance(args, dict):
                    if "issue_pid" in args: record["issue_id"] = str(args["issue_pid"])
                    if "domain" in args: record["domain"] = args["domain"]

                # Knowledge Base Harvesting
                if tool_name in ["get_faqs", "list_available_usecases", "get_usecase_instruction"] and isinstance(output, dict):
                    content = output.get("content") or output.get("structuredContent")
                    if content:
                         # Simple dedupe check
                         content_str = json.dumps(content)
                         curr_kb = [json.dumps(k) for k in record["knowledge_base"]]
                         if content_str not in curr_kb:
                             record["knowledge_base"].append(content)

                agent_obj["tool_info"].append({
                    "tool_name": tool_name,
                    "tool_args": args,
                    "tool_output": output,
                    "tool_status": status
                })

        record["agent_info"].append(agent_obj)

    return record

# ============================================================================
# PROPAGATION & OUTPUT
# ============================================================================

def propagate_and_output(merged_traces: List[Dict]):
    """Groups by Session to share IDs, then returns the sorted list."""
    sessions = defaultdict(list)
    for tr in merged_traces:
        sess_id = tr["session_id"] or f"unknown_{uuid.uuid4().hex[:8]}"
        sessions[sess_id].append(tr)
    
    final_output = []
    
    for sess_id, traces in sessions.items():
        # 1. Find best IDs in the session
        best_ids = {"issue_id": None, "domain": None}
        for tr in traces:
            if tr["issue_id"]: best_ids["issue_id"] = tr["issue_id"]
            if tr["domain"]: best_ids["domain"] = tr["domain"]
        
        # 2. Sort traces chronologically
        traces.sort(key=lambda x: x.get("timestamp") or "")
        
        # 3. Apply IDs and flatten
        for tr in traces:
            tr["issue_id"] = best_ids["issue_id"]
            tr["domain"] = best_ids["domain"]
            tr["session_id"] = sess_id
            final_output.append(tr)
            
    return final_output

def write_to_gcs(data: List[Dict], bucket_name: str, file_name: str):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        # Final safety sort
        data.sort(key=lambda x: x.get("timestamp") or "")
        jsonl_string = "\n".join([json.dumps(record) for record in data])
        blob.upload_from_string(jsonl_string, content_type="application/json")
        print(f"Uploaded {len(data)} consolidated records to gs://{bucket_name}/{file_name}")
    except Exception as e:
        print(f"GCS Upload Failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    # 1. List
    traces = list_traces_safe(PROJECT_ID, limit=args.limit, days=args.days, agent_engine_id=AGENT_ENGINE_ID)
    
    # 2. Process (Vertical Merge)
    merged_traces = []
    print(f"Processing {len(traces)} traces...")
    for trace in traces:
        try:
            merged_traces.append(process_consolidated_trace(trace))
        except Exception as e:
            print(f"Error extracting trace: {e}")
            
    # 3. Propagate (Horizontal ID Sharing)
    print("Propagating Session IDs...")
    final_rows = propagate_and_output(merged_traces)
    
    # 4. Save
    output_file = "agent_traces.jsonl"
    with open(output_file, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")
            
    write_to_gcs(final_rows, GCS_BUCKET, output_file)
    print(f"Done. Saved {len(final_rows)} consolidated records to {output_file}")

if __name__ == "__main__":
    main()