"""
extract_traces_v7.py - Consolidated Trace Extraction (1 Row per Trace ID)

Changes:
1. Vertically merges Orchestrator, Guard, and Core agents into a single record.
2. Intelligent selection of 'User Message' and 'Final Response'.
3. Aggregates all tool calls from all agents in the trace.
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

# Google Cloud imports
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
# 1. HELPER FUNCTIONS
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
# 2. TRACE FETCHING & MERGING
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
    """
    Parses a GCP trace and merges all agent activities into ONE record.
    """
    trace_id = trace["traceId"]
    spans = trace.get("spans", [])
    
    # 1. Build Span Tree
    nodes = {span["spanId"]: span for span in spans}
    children = defaultdict(list)
    roots = []
    for span in spans:
        parent = span.get("parentSpanId")
        if parent and parent in nodes: children[parent].append(span["spanId"])
        else: roots.append(span["spanId"])

    # 2. Identify all Agents in this trace
    agent_spans = []
    for span in spans:
        labels = span.get("labels", {})
        if "invoke_agent" in span.get("name", "") or labels.get("gen_ai.agent.name"):
             # Filter to ensure we capture the main agents (Orchestrator, Guard, Core)
             agent_spans.append(span)

    # 3. Initialize Unified Record
    consolidated = {
        "trace_id": trace_id,
        "session_id": None,
        "timestamp": None,  # Will be earliest timestamp
        "user_message": None,
        "agent_response": None, # Will be final response
        "ids": {},
        "tool_calls": [],
        "agent_chain": [] # Metadata about who ran
    }

    # 4. Extract data from each agent and merge
    timestamps = []
    
    for span in agent_spans:
        labels = span.get("labels", {})
        agent_name = labels.get("gen_ai.agent.name")
        consolidated["agent_chain"].append(agent_name)
        
        if span.get("startTime"): timestamps.append(span["startTime"])
        
        # Session ID (Any agent can provide it)
        if not consolidated["session_id"]:
            consolidated["session_id"] = labels.get("gcp.vertex.agent.session_id") or labels.get("gen_ai.conversation.id")

        # Get descendants for this agent
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
            
            # --- LLM Calls (User Msg / Response) ---
            if "call_llm" in name:
                req = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_request"))
                res = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_response"))
                
                # Extract IDs
                if isinstance(req, dict) and "config" in req:
                     instr = req.get("config", {}).get("system_instruction", "")
                     consolidated["ids"].update(extract_ids_from_instruction(instr))

                # User Message: Take the first non-empty user message we find
                if not consolidated["user_message"] and isinstance(req, dict):
                    for c in reversed(req.get("contents", [])):
                        if c.get("role") == "user":
                            txt = safe_get_text(c.get("parts", []))
                            if txt and not txt.startswith("For context:"):
                                consolidated["user_message"] = txt
                                break
                
                # Agent Response: We keep updating this, so the LAST agent (Core/Orchestrator) wins
                if isinstance(res, dict) and "content" in res:
                    resp_txt = safe_get_text(res["content"].get("parts", []))
                    if resp_txt:
                        consolidated["agent_response"] = resp_txt

            # --- Tool Calls ---
            if "execute_tool" in name:
                tool_name = sub_labels.get("gen_ai.tool.name") or name.replace("execute_tool ", "")
                args = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_call_args"))
                output = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_response"))
                
                # IDs from Tools
                if isinstance(args, dict):
                    if "issue_pid" in args: consolidated["ids"]["issue_id"] = str(args["issue_pid"])
                    if "domain" in args: consolidated["ids"]["domain_id"] = args["domain"]
                
                consolidated["tool_calls"].append({
                    "agent": agent_name,
                    "tool": tool_name,
                    "input": args,
                    "output": output
                })

    # Set final timestamp to earliest start time
    if timestamps:
        consolidated["timestamp"] = min(timestamps)
        
    return consolidated

# ============================================================================
# 3. PROPAGATION & OUTPUT
# ============================================================================

def propagate_and_output(merged_traces: List[Dict]):
    """
    Propagates IDs across the session (between different Trace IDs).
    """
    
    # Group by Session
    sessions = defaultdict(list)
    for tr in merged_traces:
        sess_id = tr["session_id"] or f"unknown_{uuid.uuid4().hex[:8]}"
        sessions[sess_id].append(tr)
    
    final_output = []
    
    for sess_id, traces in sessions.items():
        # Find best IDs in this session
        best_ids = {"issue_id": None, "domain_id": None, "profile_id": None}
        for tr in traces:
            for k, v in tr["ids"].items():
                if v and not best_ids.get(k): best_ids[k] = v
        
        # Sort traces by time
        traces.sort(key=lambda x: x.get("timestamp") or "")
        
        # Apply IDs and flatten
        for tr in traces:
            tr["issue_id"] = best_ids["issue_id"]
            tr["domain_id"] = best_ids["domain_id"]
            tr["session_id"] = sess_id
            
            # Clean up internal fields
            del tr["ids"]
            
            final_output.append(tr)
            
    return final_output

def write_to_gcs(data: List[Dict], bucket_name: str, file_name: str):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        # Sort by timestamp for clean log reading
        data.sort(key=lambda x: x.get("timestamp") or "")
        
        jsonl_string = "\n".join([json.dumps(record) for record in data])
        blob.upload_from_string(jsonl_string, content_type="application/json")
        print(f"Uploaded {len(data)} consolidated traces to gs://{bucket_name}/{file_name}")
    except Exception as e:
        print(f"GCS Upload Failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    # 1. List
    traces = list_traces_safe(PROJECT_ID, limit=args.limit, days=args.days, agent_engine_id=AGENT_ENGINE_ID)
    
    # 2. Process (Consolidate Agents -> 1 Object per Trace)
    merged_traces = []
    print(f"Processing {len(traces)} traces...")
    for trace in traces:
        try:
            merged_traces.append(process_consolidated_trace(trace))
        except Exception as e:
            print(f"Error extracting trace: {e}")
            
    # 3. Propagate (Share IDs between traces in the same session)
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