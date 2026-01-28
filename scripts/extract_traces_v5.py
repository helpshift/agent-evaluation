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
from google.cloud import logging_v2
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
CLOUD_RUN_SERVICE = os.environ.get("CLOUD_RUN_SERVICE", None)  # Optional service filter
LOG_QUERY_LIMIT = int(os.environ.get("LOG_QUERY_LIMIT", "500"))

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

def extract_ids_from_log_entry(entry) -> Dict[str, str]:
    """Extracts issue_id, domain, and message_id from a Cloud Logging entry."""
    ids = {}
    
    # Try jsonPayload first (structured logs)
    payload = getattr(entry, 'payload', None)
    if isinstance(payload, dict):
        # Check common field names for issue_id
        for issue_key in ['issue_id', 'issue_pid', 'issueId', 'issuePid']:
            if issue_key in payload and payload[issue_key]:
                ids['issue_id'] = str(payload[issue_key])
                break
        # Check common field names for domain
        for domain_key in ['domain', 'domain_id', 'domainId']:
            if domain_key in payload and payload[domain_key]:
                ids['domain'] = str(payload[domain_key])
                break
        # Check common field names for message_id
        for msg_key in ['message_id', 'messageId', 'msg_id', 'msgId']:
            if msg_key in payload and payload[msg_key]:
                ids['message_id'] = str(payload[msg_key])
                break
        # Check nested structures
        if 'request' in payload and isinstance(payload['request'], dict):
            req = payload['request']
            if not ids.get('issue_id') and req.get('issue_id'):
                ids['issue_id'] = str(req['issue_id'])
            if not ids.get('domain') and req.get('domain'):
                ids['domain'] = str(req['domain'])
            if not ids.get('message_id') and req.get('message_id'):
                ids['message_id'] = str(req['message_id'])
    
    # Fallback to text payload with regex
    text = getattr(entry, 'text_payload', '') or ''
    if not ids.get('issue_id'):
        match = re.search(r'issue[_-]?(?:id|pid)[":\s]+([\d]+)', text, re.IGNORECASE)
        if match:
            ids['issue_id'] = match.group(1)
    if not ids.get('domain'):
        match = re.search(r'domain[":\s]+([\w.-]+)', text, re.IGNORECASE)
        if match:
            ids['domain'] = match.group(1)
    if not ids.get('message_id'):
        match = re.search(r'message[_-]?id[":\s]+([\w-]+)', text, re.IGNORECASE)
        if match:
            ids['message_id'] = match.group(1)
    
    return ids

def fetch_logs_by_trace_ids(project_id: str, trace_ids: List[str], 
                            session_ids: List[str], days: int = 7,
                            profile_ids: List[str] = None,
                            issue_ids: List[str] = None) -> Dict[str, Dict]:
    """
    Fetches Cloud Run logs using multiple correlation strategies:
    1. profile_id exact match (most reliable)
    2. issue_id embedded in log app_id/session_id
    3. trace_id correlation
    
    Returns mapping: {trace_id: {issue_id, domain, message_id}}
    """
    log_data = {}
    
    if not trace_ids:
        return log_data
    
    try:
        client = logging_v2.Client(project=project_id)
        
        # Build time filter
        past_date = datetime.now(timezone.utc) - timedelta(days=days)
        time_filter = f'timestamp >= "{past_date.isoformat()}"'
        
        # Build mappings for correlation
        profile_to_traces = {}
        issue_to_traces = {}
        
        for i, tid in enumerate(trace_ids):
            if profile_ids and i < len(profile_ids) and profile_ids[i]:
                profile_to_traces.setdefault(profile_ids[i], []).append(tid)
            if issue_ids and i < len(issue_ids) and issue_ids[i]:
                issue_to_traces.setdefault(str(issue_ids[i]), []).append(tid)
        
        # === STRATEGY 1: Query by profile_id (EXACT MATCH - most reliable) ===
        if profile_to_traces:
            print(f"  Querying logs by profile_id for {len(profile_to_traces)} unique profiles...")
            for profile_id, related_traces in profile_to_traces.items():
                filters = [
                    f'jsonPayload.profile_id="{profile_id}"',
                    time_filter
                ]
                filter_str = ' AND '.join(filters)
                
                try:
                    entries = list(client.list_entries(
                        filter_=filter_str,
                        max_results=20
                    ))
                    
                    for entry in entries:
                        ids = extract_ids_from_log_entry(entry)
                        if ids.get('message_id') or ids.get('issue_id') or ids.get('domain'):
                            for tid in related_traces:
                                if tid not in log_data:
                                    log_data[tid] = ids.copy()
                                else:
                                    # Merge missing fields
                                    for key in ['message_id', 'issue_id', 'domain']:
                                        if not log_data[tid].get(key) and ids.get(key):
                                            log_data[tid][key] = ids[key]
                except Exception as e:
                    continue
        
        # === STRATEGY 2: Query by issue_id embedded in app_id/session_id ===
        if issue_to_traces:
            print(f"  Querying logs by issue_id for {len(issue_to_traces)} unique issues...")
            for issue_id, related_traces in issue_to_traces.items():
                # Skip if all related traces already have message_id
                if all(log_data.get(tid, {}).get('message_id') for tid in related_traces):
                    continue
                    
                # Search for logs where session_id contains the issue_id
                # Log session_id format: {domain}_app_{long_issue_id}-{hash}_{short_issue_id}
                filters = [
                    f'jsonPayload.session_id:"{issue_id}"',  # Contains issue_id in session_id
                    'jsonPayload.message_id:*',  # Must have message_id
                    time_filter
                ]
                filter_str = ' AND '.join(filters)
                
                try:
                    entries = list(client.list_entries(
                        filter_=filter_str,
                        max_results=20
                    ))
                    
                    for entry in entries:
                        ids = extract_ids_from_log_entry(entry)
                        if ids.get('message_id'):
                            for tid in related_traces:
                                if tid not in log_data:
                                    log_data[tid] = ids.copy()
                                elif not log_data[tid].get('message_id'):
                                    log_data[tid]['message_id'] = ids['message_id']
                            break  # Found message_id for this issue
                except Exception as e:
                    continue
        
        # === STRATEGY 3: Broad search for message_id and correlate ===
        traces_without_message_id = [
            tid for tid in trace_ids 
            if tid not in log_data or not log_data.get(tid, {}).get('message_id')
        ]
        
        if traces_without_message_id and (profile_to_traces or issue_to_traces):
            print(f"  Broad search for message_id to fill {len(traces_without_message_id)} remaining traces...")
            
            filters = [
                time_filter,
                'jsonPayload.message_id:*'
            ]
            if CLOUD_RUN_SERVICE:
                filters.append(f'resource.labels.service_name="{CLOUD_RUN_SERVICE}"')
            
            filter_str = ' AND '.join(filters)
            
            try:
                entries = list(client.list_entries(
                    filter_=filter_str,
                    max_results=LOG_QUERY_LIMIT
                ))
                
                print(f"  Found {len(entries)} log entries with message_id")
                
                for entry in entries:
                    payload = getattr(entry, 'payload', {})
                    if not isinstance(payload, dict):
                        continue
                        
                    ids = extract_ids_from_log_entry(entry)
                    if not ids.get('message_id'):
                        continue
                    
                    # Try to correlate by profile_id
                    log_profile = payload.get('profile_id')
                    if log_profile and log_profile in profile_to_traces:
                        for tid in profile_to_traces[log_profile]:
                            if tid not in log_data:
                                log_data[tid] = ids.copy()
                            elif not log_data[tid].get('message_id'):
                                log_data[tid]['message_id'] = ids['message_id']
                    
                    # Try to correlate by issue_id in app_id or session_id
                    # Log session_id format: {domain}_app_{issue_id}-{hash}_{seq}
                    log_app_id = payload.get('app_id', '') or ''
                    log_session_id = payload.get('session_id', '') or ''
                    combined_ids = str(log_app_id) + str(log_session_id)
                    
                    for issue_id, related_traces in issue_to_traces.items():
                        if issue_id in combined_ids:
                            for tid in related_traces:
                                if tid not in log_data:
                                    log_data[tid] = ids.copy()
                                elif not log_data[tid].get('message_id'):
                                    log_data[tid]['message_id'] = ids['message_id']
                            break
                            
            except Exception as e:
                print(f"  Warning: Broad message_id search failed: {e}")
        
        print(f"Fetched log data for {len(log_data)} traces")
        
    except Exception as e:
        print(f"WARNING: Failed to fetch logs: {e}")
    
    return log_data

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
    agent_spans= []
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
        "profile_id": None,      # For log correlation
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
            "total_input_tokens": 0,
            "total_output_tokens": 0,
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
                    resp_txt = safe_get_text(res["content"].get("parts", []))
                    if resp_txt:
                        record["agent_response"] = resp_txt # Overwrite with latest response
                        agent_obj["agent_output"] = resp_txt

                # 4. Extract Token Usage
                try:
                    in_tok = int(sub_labels.get("gen_ai.usage.input_tokens", 0))
                    out_tok = int(sub_labels.get("gen_ai.usage.output_tokens", 0))
                    agent_obj["total_input_tokens"] += in_tok
                    agent_obj["total_output_tokens"] += out_tok
                except (ValueError, TypeError):
                    pass

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
                    if "profile_id" in args: record["profile_id"] = args["profile_id"]

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

def propagate_and_output(merged_traces: List[Dict], log_data: Dict[str, Dict] = None):
    """Groups by Session to share IDs, then returns the sorted list."""
    sessions = defaultdict(list)
    for tr in merged_traces:
        sess_id = tr["session_id"] or f"unknown_{uuid.uuid4().hex[:8]}"
        sessions[sess_id].append(tr)
    
    final_output = []
    
    for sess_id, traces in sessions.items():
        # 1. Find best IDs in the session (from traces first)
        best_ids = {"issue_id": None, "domain": None, "message_id": None}
        for tr in traces:
            if tr["issue_id"]: best_ids["issue_id"] = tr["issue_id"]
            if tr["domain"]: best_ids["domain"] = tr["domain"]
            if tr.get("message_id"): best_ids["message_id"] = tr["message_id"]
        
        # 1.5 Check log_data for missing IDs
        if log_data:
            for tr in traces:
                if tr["trace_id"] in log_data:
                    log_ids = log_data[tr["trace_id"]]
                    if not best_ids["issue_id"] and log_ids.get("issue_id"):
                        best_ids["issue_id"] = log_ids["issue_id"]
                    if not best_ids["domain"] and log_ids.get("domain"):
                        best_ids["domain"] = log_ids["domain"]
                    if not best_ids["message_id"] and log_ids.get("message_id"):
                        best_ids["message_id"] = log_ids["message_id"]
        
        # 2. Sort traces chronologically
        traces.sort(key=lambda x: x.get("timestamp") or "")
        
        # 3. Apply IDs and flatten
        for tr in traces:
            tr["issue_id"] = best_ids["issue_id"]
            tr["domain"] = best_ids["domain"]
            tr["message_id"] = best_ids["message_id"]
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
    
    # 2.5 Fetch additional data from Cloud Run logs
    print("Fetching additional data from Cloud Logging...")
    trace_ids = [t["trace_id"] for t in merged_traces]
    session_ids = [t["session_id"] for t in merged_traces]
    profile_ids = [t.get("profile_id") for t in merged_traces]
    issue_ids = [t.get("issue_id") for t in merged_traces]
    log_data = fetch_logs_by_trace_ids(
        PROJECT_ID, trace_ids, session_ids, args.days,
        profile_ids=profile_ids, issue_ids=issue_ids
    )
            
    # 3. Propagate (Horizontal ID Sharing)
    print("Propagating Session IDs...")
    final_rows = propagate_and_output(merged_traces, log_data)
    
    # 4. Save
    output_file = "agent_traces.jsonl"
    with open(output_file, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")
            
    write_to_gcs(final_rows, GCS_BUCKET, output_file)
    print(f"Done. Saved {len(final_rows)} consolidated records to {output_file}")

if __name__ == "__main__":
    main()