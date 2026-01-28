"""
extract_traces_all_agents.py - Cloud Trace API extraction for ALL Agent Engines

Derived from extract_traces_v6.py.
Changes:
- Removes specific Agent Engine ID filter.
- Defaults to looking back 60 days.
- Segregates traces by Agent Engine / Agent Name.
- Outputs traces for all agents found.

Output: 
- Consolidated JSONL file
- Optionally split files per agent
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
# Default Agent Engine ID removed, we want all.
# No default Bifrost URL filter
BIFROST_URL = os.environ.get("BIFROST_URL", None)
CLOUD_RUN_SERVICE = os.environ.get("CLOUD_RUN_SERVICE", None)
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
    
    payload = getattr(entry, 'payload', None)
    text = getattr(entry, 'text_payload', '') or ''
    
    # Fallback: Parse text_payload if payload is missing but text looks like JSON
    if not payload and text: # and ('{' in text):
        parsed = try_parse_json(text)
        if isinstance(parsed, dict):
            payload = parsed

    if isinstance(payload, dict):
        for issue_key in ['issue_id', 'issue_pid', 'issueId', 'issuePid']:
            if issue_key in payload and payload[issue_key]:
                ids['issue_id'] = str(payload[issue_key])
                break
        for domain_key in ['domain', 'domain_id', 'domainId']:
            if domain_key in payload and payload[domain_key]:
                ids['domain'] = str(payload[domain_key])
                break
        for msg_key in ['message_id', 'messageId', 'msg_id', 'msgId']:
            if msg_key in payload and payload[msg_key]:
                ids['message_id'] = str(payload[msg_key])
                break
        if 'request' in payload and isinstance(payload['request'], dict):
            req = payload['request']
            if not ids.get('issue_id') and req.get('issue_id'):
                ids['issue_id'] = str(req['issue_id'])
            if not ids.get('domain') and req.get('domain'):
                ids['domain'] = str(req['domain'])
            if not ids.get('message_id') and req.get('message_id'):
                ids['message_id'] = str(req['message_id'])
    
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

    # 5. Check nested 'data' object structure (User payload)
    if isinstance(payload, dict):
        data = payload.get('data')
        if isinstance(data, dict):
            if not ids.get('issue_id') and data.get('id'):
                ids['issue_id'] = str(data['id'])
            
            # Check messages for message_id
            messages = data.get('messages')
            if isinstance(messages, list) and messages:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    if not ids.get('message_id') and first_msg.get('id'):
                        ids['message_id'] = str(first_msg['id'])
    
    return ids

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_logs_by_trace_ids(project_id: str, trace_ids: List[str], 
                            session_ids: List[str], days: int = 7,
                            profile_ids: List[str] = None,
                            issue_ids: List[str] = None,
                            merged_traces: List[Dict] = None) -> Dict[str, Dict]:
    """Fetches Cloud Run logs using batched queries to avoid API quotas and slowness."""
    log_data = {}
    if not trace_ids:
        return log_data
    
    # Initialize client once
    try:
        client = logging_v2.Client(project=project_id)
    except Exception as e:
        print(f"  ERROR: Could not initialize Logging Client: {e}")
        return log_data

    past_date = datetime.now(timezone.utc) - timedelta(days=days)
    # Use a stricter timestamp format to prevent parsing errors
    time_filter = f'timestamp >= "{past_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}"'
    
    # --- Helper to process entries from a batch ---
    def process_entries(entries, related_map, key_field_in_payload):
        for entry in entries:
            try:
                ids = extract_ids_from_log_entry(entry)
                payload = getattr(entry, 'payload', {}) or {}
                
                # Identify which key this log belongs to
                found_key = None
                
                # Check extracted IDs first
                if key_field_in_payload == 'profile_id' and ids.get('profile_id'):
                    found_key = ids['profile_id']
                elif key_field_in_payload == 'session_id' and ids.get('issue_id'):
                    found_key = ids['issue_id']
                
                # Fallback to checking payload directly
                if not found_key and isinstance(payload, dict):
                    found_key = str(payload.get(key_field_in_payload) or "")
                
                # ADK specific fallback
                if key_field_in_payload == 'adk_session_id' and isinstance(payload, dict):
                    found_key = str(payload.get('adk_session_id') or "")

                # Map back to traces
                if found_key and found_key in related_map:
                    for tid in related_map[found_key]:
                        if tid not in log_data:
                            log_data[tid] = ids.copy()
                        else:
                            # Merge existing data
                            for k, v in ids.items():
                                if not log_data[tid].get(k) and v:
                                    log_data[tid][k] = v
                                    
                        # Also merge from payload if needed
                        if isinstance(payload, dict):
                             if not log_data[tid].get('issue_id') and payload.get('issue_id'):
                                 log_data[tid]['issue_id'] = str(payload['issue_id'])
                             if not log_data[tid].get('domain') and payload.get('domain'):
                                 log_data[tid]['domain'] = str(payload['domain'])
                             if not log_data[tid].get('profile_id') and payload.get('user_id'):
                                 log_data[tid]['profile_id'] = str(payload['user_id'])
            except Exception:
                continue 

    # 1. Profile IDs
    profile_to_traces = {}
    if profile_ids:
        for i, tid in enumerate(trace_ids):
            if i < len(profile_ids) and profile_ids[i]:
                profile_to_traces.setdefault(profile_ids[i], []).append(tid)
    
    if profile_to_traces:
        unique_profiles = list(profile_to_traces.keys())
        print(f"  Querying logs for {len(unique_profiles)} profiles (batched)...")
        for i, chunk in enumerate(chunk_list(unique_profiles, 20)):
            # Progress dot
            if i % 5 == 0: print(f"    Processing profile chunk {i+1}...", end='\r')
            
            or_clauses = [f'jsonPayload.profile_id="{pid}"' for pid in chunk]
            filter_str = f'({" OR ".join(or_clauses)}) AND {time_filter}'
            try:
                entries = list(client.list_entries(filter_=filter_str, max_results=200))
                process_entries(entries, profile_to_traces, 'profile_id')
            except Exception as e:
                print(f"    [!] Error fetching profile logs: {e}")

    # 2. Issue IDs (Mapped to jsonPayload.session_id)
    issue_to_traces = {}
    if issue_ids:
        for i, tid in enumerate(trace_ids):
            if i < len(issue_ids) and issue_ids[i]:
                issue_to_traces.setdefault(str(issue_ids[i]), []).append(tid)
    
    if issue_to_traces:
        unique_issues = list(issue_to_traces.keys())
        print(f"\n  Querying logs for {len(unique_issues)} issues (batched)...")
        for i, chunk in enumerate(chunk_list(unique_issues, 20)):
            if i % 5 == 0: print(f"    Processing issue chunk {i+1}...", end='\r')

            # REMOVED: jsonPayload.message_id:* (Too expensive/slow)
            or_clauses = [f'jsonPayload.session_id:"{iid}"' for iid in chunk]
            filter_str = f'({" OR ".join(or_clauses)}) AND {time_filter}'
            
            try:
                # Reduced max_results to 100 to prevent timeouts on large payloads
                entries = list(client.list_entries(filter_=filter_str, max_results=100))
                process_entries(entries, issue_to_traces, 'session_id')
            except Exception as e:
                print(f"    [!] Error fetching issue logs: {e}")

    # 3. Session IDs (ADK)
    adk_session_to_traces = {}
    if session_ids:
        for i, sid in enumerate(session_ids):
            if sid:
                adk_session_to_traces.setdefault(sid, []).append(trace_ids[i])
    
    if adk_session_to_traces:
        unique_sids = list(adk_session_to_traces.keys())
        print(f"\n  Querying logs for {len(unique_sids)} ADK sessions (batched)...")
        for i, chunk in enumerate(chunk_list(unique_sids, 20)):
            if i % 5 == 0: print(f"    Processing ADK chunk {i+1}...", end='\r')
            
            or_clauses = [f'jsonPayload.adk_session_id="{sid}"' for sid in chunk]
            filter_str = f'({" OR ".join(or_clauses)}) AND {time_filter}'
            try:
                entries = list(client.list_entries(filter_=filter_str, max_results=100))
                process_entries(entries, adk_session_to_traces, 'adk_session_id')
            except Exception as e:
                print(f"    [!] Error fetching ADK logs: {e}")
    
    print("\n  Log enrichment complete.")
    return log_data

def trace_matches_bifrost_url(trace: Dict, bifrost_url: str) -> bool:
    """
    Checks if any span in the trace contains the Bifrost webhook URL
    in its labels, matching how Cloud Trace filters work.
    """
    if not bifrost_url:
        return True
        
    spans = trace.get("spans", [])
    for span in spans:
        labels = span.get("labels", {})
        # Check all labels for the URL
        for key, value in labels.items():
            if bifrost_url in str(value):
                return True
    return False

def list_traces_with_bifrost_filter(
    project_id: str, 
    limit: int = 50, 
    days: int = 7, 
    agent_engine_id: Optional[str] = None,
    bifrost_url: Optional[str] = None
) -> List[Dict]:
    """
    Fetches traces using Cloud Trace API and filters by Bifrost URL.
    
    1. First fetches traces. If agent_engine_id provided, filters by it.
    2. Then post-filters for traces containing the Bifrost URL
    """
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
    
    all_traces = []
    filtered_traces = []
    
    print(f"Fetching traces... Filter: {filter_str if filter_str else 'NONE (All Agents)'}")
    if bifrost_url:
        print(f"Will post-filter for Bifrost URL: {bifrost_url}")
    
    pager = client.list_traces(request=request)
    
    try:
        count = 0
        # Fetch more traces than needed since we'll filter them
        max_fetch = limit * 5 if bifrost_url else limit
        
        for trace in pager:
            trace_dict = MessageToDict(trace._pb)
            all_traces.append(trace_dict)
            
            # Apply Bifrost URL filter if specified
            if bifrost_url:
                if trace_matches_bifrost_url(trace_dict, bifrost_url):
                    filtered_traces.append(trace_dict)
                    if len(filtered_traces) >= limit:
                        break
            else:
                filtered_traces.append(trace_dict)
                if len(filtered_traces) >= limit:
                    break
            
            count += 1
            if count >= max_fetch:
                break
            if count % 100 == 0:
                time.sleep(1)
                
    except (ResourceExhausted, ServiceUnavailable) as e:
        print(f"WARNING: API Quota hit. Continuing with {len(filtered_traces)} traces.")
    
    print(f"Fetched {len(all_traces)} total traces, {len(filtered_traces)} match Bifrost filter")
    return filtered_traces

# ============================================================================
# TRACE PROCESSING
# ============================================================================

def process_consolidated_trace(trace):
    trace_id = trace["traceId"]
    spans = trace.get("spans", [])
    
    nodes = {span["spanId"]: span for span in spans}
    children = defaultdict(list)
    for span in spans:
        parent = span.get("parentSpanId")
        if parent and parent in nodes: 
            children[parent].append(span["spanId"])

    agent_spans = []
    engine_id = None
    services_used = set()
    
    for span in spans:
        labels = span.get("labels", {})
        if "invoke_agent" in span.get("name", "") or labels.get("gen_ai.agent.name"):
            agent_spans.append(span)

        # Collect Service Names
        if "g.co/gae/app/module" in labels:
            services_used.add(labels["g.co/gae/app/module"])
        elif "service.name" in labels:
            services_used.add(labels["service.name"])
        
        # Collect External Hosts
        host = labels.get("/http/host") or labels.get("http.host")
        if host:
            services_used.add(f"external:{host}")
        
        # Try to find Agent Engine ID / Service Name
        if not engine_id:
            # Common label for service name in various GCP contexts
            if "g.co/gae/app/module" in labels:
                engine_id = labels["g.co/gae/app/module"]
            elif "service.name" in labels:
                 engine_id = labels["service.name"]
            elif labels.get("agent_id"): # Hypothetical
                 engine_id = labels["agent_id"]

    agent_spans.sort(key=lambda x: x.get("startTime", ""))

    record = {
        "timestamp": None,
        "trace_id": trace_id,
        "session_id": None,
        "issue_id": None,
        "domain": None,
        "message_id": None,
        "profile_id": None,
        "user_message": None,
        "agent_response": None,
        "agent_type": "multi_agent_system",
        "agent_engine_id": engine_id,
        "services_used": sorted(list(services_used)),
        "knowledge_base": [],
        "agent_info": [],
        "tool_failures": [],
        "model_failures": []
    }

    if agent_spans:
        record["timestamp"] = agent_spans[0].get("startTime")

    for span in agent_spans:
        labels = span.get("labels", {})
        agent_name = labels.get("gen_ai.agent.name")
        if not agent_name: 
            continue

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

        stack = [span["spanId"]]
        descendants = []
        while stack:
            curr = stack.pop()
            kids = children.get(curr, [])
            descendants.extend(kids)
            stack.extend(kids)
        
        all_sub_spans = [nodes[sid] for sid in descendants if sid in nodes]
        all_sub_spans.sort(key=lambda x: x.get("startTime", ""))

        for sub in all_sub_spans:
            sub_labels = sub.get("labels", {})
            name = sub.get("name", "")
            
            if "call_llm" in name:
                req = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_request"))
                res = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_response"))
                
                if isinstance(req, dict) and "config" in req:
                    instr = req.get("config", {}).get("system_instruction", "")
                    ids = extract_ids_from_instruction(instr)
                    if ids.get("issue_id"): record["issue_id"] = ids["issue_id"]
                    if ids.get("domain_id"): record["domain"] = ids["domain_id"]

                if isinstance(req, dict):
                    for c in reversed(req.get("contents", [])):
                        if c.get("role") == "user":
                            txt = safe_get_text(c.get("parts", []))
                            if txt and not txt.startswith("For context:"):
                                if not record["user_message"]: 
                                    record["user_message"] = txt
                                agent_obj["agent_input"] = txt
                                break
                
                if isinstance(res, dict) and "content" in res:
                    resp_txt = safe_get_text(res["content"].get("parts", []))
                    if resp_txt:
                        record["agent_response"] = resp_txt
                        agent_obj["agent_output"] = resp_txt

                try:
                    in_tok = int(sub_labels.get("gen_ai.usage.input_tokens", 0))
                    out_tok = int(sub_labels.get("gen_ai.usage.output_tokens", 0))
                    agent_obj["total_input_tokens"] += in_tok
                    agent_obj["total_output_tokens"] += out_tok
                except (ValueError, TypeError):
                    pass

            if "execute_tool" in name:
                tool_name = sub_labels.get("gen_ai.tool.name") or name.replace("execute_tool ", "")
                args = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_call_args"))
                output = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_response"))
                
                status = "success"
                if isinstance(output, dict):
                    if output.get("isError") or output.get("status") == "Failure":
                        status = "failure"
                        record["tool_failures"].append({
                            "agent": agent_name,
                            "tool": tool_name,
                            "error": output
                        })

                if isinstance(args, dict):
                    if "issue_pid" in args: record["issue_id"] = str(args["issue_pid"])
                    if "domain" in args: record["domain"] = args["domain"]
                    if "profile_id" in args: record["profile_id"] = args["profile_id"]

                if tool_name in ["get_faqs", "list_available_usecases", "get_usecase_instruction"] and isinstance(output, dict):
                    content = output.get("content") or output.get("structuredContent")
                    if content:
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
        best_ids = {
            "issue_id": None, 
            "domain": None, 
            "message_id": None,
            "profile_id": None
        }
        for tr in traces:
            if tr["issue_id"]: best_ids["issue_id"] = tr["issue_id"]
            if tr["domain"]: best_ids["domain"] = tr["domain"]
            if tr.get("message_id"): best_ids["message_id"] = tr["message_id"]
            if tr.get("profile_id"): best_ids["profile_id"] = tr["profile_id"]
        
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
                    if not best_ids["profile_id"] and log_ids.get("profile_id"):
                        best_ids["profile_id"] = log_ids["profile_id"]
        
        traces.sort(key=lambda x: x.get("timestamp") or "")
        
        for tr in traces:
            tr["issue_id"] = best_ids["issue_id"]
            tr["domain"] = best_ids["domain"]
            tr["message_id"] = best_ids["message_id"]
            tr["profile_id"] = best_ids["profile_id"]
            tr["session_id"] = sess_id
            final_output.append(tr)
            
    return final_output

def write_to_gcs(data: List[Dict], bucket_name: str, file_name: str):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        data.sort(key=lambda x: x.get("timestamp") or "")
        jsonl_string = "\n".join([json.dumps(record) for record in data])
        blob.upload_from_string(jsonl_string, content_type="application/json")
        print(f"Uploaded {len(data)} consolidated records to gs://{bucket_name}/{file_name}")
    except Exception as e:
        print(f"GCS Upload Failed: {e}")

def discover_active_services(project_id: str, days: int) -> List[str]:
    services = set()
    print("Scanning recent traces to discover active services...")
    # Scan last 2 days (or requested days if less), up to 2000 traces
    scan_days = min(days, 20)
    traces = list_traces_with_bifrost_filter(
        project_id, limit=2000, days=scan_days, agent_engine_id=None
    )
    for trace in traces:
        spans = trace.get("spans", [])
        for span in spans:
            labels = span.get("labels", {})
            # Prefer specific app module or service name
            if "g.co/gae/app/module" in labels:
                services.add(labels["g.co/gae/app/module"])
            elif "service.name" in labels:
                 services.add(labels["service.name"])
    
    results = sorted(list(services))
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract traces for ALL Agent Engines (Batched by Agent)"
    )
    parser.add_argument("--limit", type=int, default=500, help="Max number of traces PER AGENT")
    parser.add_argument("--days", type=int, default=60, help="Days to look back")
    parser.add_argument("--bifrost-url", type=str, default=BIFROST_URL,
                        help="Bifrost webhook URL to filter by")
    parser.add_argument("--no-bifrost-filter", action="store_true",
                        help="Disable Bifrost URL filtering")
    parser.add_argument("--output", type=str, default="all_agents_traces.jsonl",
                        help="Output file name")
    parser.add_argument("--skip-gcs", action="store_true", help="Skip GCS upload")
    args = parser.parse_args()
    
    bifrost_filter = None if args.no_bifrost_filter else args.bifrost_url
    
    # 1. Discover Agents
    agents = discover_active_services(PROJECT_ID, args.days)
    if not agents:
        print("No active agents found in recent traces.")
        agents = [None] 
    else:
        print(f"Found {len(agents)} active agents: {agents}")

    all_consolidated_rows = []
    
    # 2. Batch Fetch per Agent
    for i, agent_id in enumerate(agents):
        print(f"\n[{i+1}/{len(agents)}] Fetching traces for Agent: {agent_id if agent_id else 'ALL'}")
        
        try:
            # Fetch
            traces = list_traces_with_bifrost_filter(
                PROJECT_ID, 
                limit=args.limit, 
                days=args.days, 
                agent_engine_id=agent_id,
                bifrost_url=bifrost_filter
            )
            
            if not traces:
                print(f"No traces found for {agent_id}")
                continue
                
            # Process
            print(f"Processing {len(traces)} traces for {agent_id}...")
            merged = []
            for t in traces:
                try:
                    merged.append(process_consolidated_trace(t))
                except Exception:
                    pass
            
            # Fetch Logs
            print(f"Fetching logs for {len(merged)} traces...")
            t_ids = [t["trace_id"] for t in merged]
            s_ids = [t["session_id"] for t in merged]
            p_ids = [t.get("profile_id") for t in merged]
            i_ids = [t.get("issue_id") for t in merged]
            
            log_data = fetch_logs_by_trace_ids(
                PROJECT_ID, t_ids, s_ids, args.days,
                profile_ids=p_ids, issue_ids=i_ids, merged_traces=merged
            )
            
            # Propagate
            final = propagate_and_output(merged, log_data)
            all_consolidated_rows.extend(final)
            
            # Save Individual File (Incremental Save)
            if agent_id:
                safe_key = "".join([c if c.isalnum() else "_" for c in str(agent_id)])
                filename = f"agent_traces_{safe_key}.jsonl"
                with open(filename, "w") as f:
                    for row in final:
                        f.write(json.dumps(row) + "\n")
                if not args.skip_gcs:
                    write_to_gcs(final, GCS_BUCKET, filename)
        
        except Exception as e:
            print(f"CRITICAL ERROR processing agent {agent_id}: {e}")
            continue

    # 3. Save Final Consolidated
    print(f"\n=== Saving Consolidated Output ({len(all_consolidated_rows)} traces) ===")
    with open(args.output, "w") as f:
        for row in all_consolidated_rows:
            f.write(json.dumps(row) + "\n")
    
    if not args.skip_gcs and all_consolidated_rows:
        write_to_gcs(all_consolidated_rows, GCS_BUCKET, args.output)

    # Summary
    print("\n=== Summary ===")
    print(f"Total traces: {len(all_consolidated_rows)}")
    with_issue_id = sum(1 for r in all_consolidated_rows if r.get("issue_id"))
    with_message_id = sum(1 for r in all_consolidated_rows if r.get("message_id"))
    with_domain = sum(1 for r in all_consolidated_rows if r.get("domain"))
    print(f"With issue_id: {with_issue_id}")
    print(f"With message_id: {with_message_id}")
    print(f"With domain: {with_domain}")

if __name__ == "__main__":
    main()
