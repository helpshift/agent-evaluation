"""
extract_traces_v6.py - Cloud Trace API with Bifrost URL filter

Uses Cloud Trace API (like v5) but adds post-processing filter for Bifrost webhook URL.
This ensures we only get traces that hit the Bifrost service.

Output: One JSON object per Trace ID containing:
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
import concurrent.futures
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
BIFROST_URL = os.environ.get(
    "BIFROST_URL",
    "https://bifrost-service-qa-845835740344.us-central1.run.app/webhook/bifrost_service"
)
CLOUD_RUN_SERVICE = os.environ.get("CLOUD_RUN_SERVICE", None)
LOG_QUERY_LIMIT = int(os.environ.get("LOG_QUERY_LIMIT", "500"))
CONTENT_MAPPING_FILE = os.environ.get("CONTENT_MAPPING_FILE", "gcs_data/content_mapping.json")
STATE_FILE_GCS = os.environ.get("STATE_FILE_GCS", "state/last_processed.json")  # GCS path for state

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_state_from_gcs(bucket_name: str, state_file: str) -> Dict[str, Any]:
    """Loads the last processed state from GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(state_file)

        if blob.exists():
            state_json = blob.download_as_text()
            state = json.loads(state_json)
            print(f"Loaded state from GCS: last_processed_time={state.get('last_processed_time')}")
            return state
        else:
            print("No previous state found, starting fresh")
            return {}
    except Exception as e:
        print(f"Warning: Could not load state from GCS: {e}")
        return {}

def save_state_to_gcs(bucket_name: str, state_file: str, state: Dict[str, Any]):
    """Saves the last processed state to GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(state_file)

        state_json = json.dumps(state, indent=2)
        blob.upload_from_string(state_json, content_type="application/json")
        print(f"Saved state to GCS: {state}")
    except Exception as e:
        print(f"Warning: Could not save state to GCS: {e}")

def load_content_mapping():
    """Loads the content mapping file if it exists."""
    if os.path.exists(CONTENT_MAPPING_FILE):
        try:
            with open(CONTENT_MAPPING_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load content mapping: {e}")
    return {"instructions": {}, "faqs": {}}

def save_content_mapping(mapping):
    """Saves the content mapping to file."""
    try:
        os.makedirs(os.path.dirname(CONTENT_MAPPING_FILE), exist_ok=True)
        with open(CONTENT_MAPPING_FILE, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved content mapping to {CONTENT_MAPPING_FILE}")
    except Exception as e:
        print(f"Warning: Could not save content mapping: {e}")

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

def fetch_logs_by_trace_ids(project_id: str, trace_ids: List[str], 
                            session_ids: List[str], days: int = 7,
                            profile_ids: List[str] = None,
                            issue_ids: List[str] = None,
                            merged_traces: List[Dict] = None) -> Dict[str, Dict]:
    """Fetches Cloud Run logs using multiple correlation strategies in parallel."""
    log_data = {}
    
    if not trace_ids:
        return log_data
    
    try:
        client = logging_v2.Client(project=project_id)
        past_date = datetime.now(timezone.utc) - timedelta(days=days)
        time_filter = f'timestamp >= "{past_date.isoformat()}"'
        
        # Prepare groupings
        profile_to_traces = {}
        issue_to_traces = {}
        adk_session_to_traces = {}
        
        for i, tid in enumerate(trace_ids):
            if profile_ids and i < len(profile_ids) and profile_ids[i]:
                profile_to_traces.setdefault(profile_ids[i], []).append(tid)
            if issue_ids and i < len(issue_ids) and issue_ids[i]:
                issue_to_traces.setdefault(str(issue_ids[i]), []).append(tid)
            if session_ids and i < len(session_ids) and session_ids[i]:
                adk_session_to_traces.setdefault(session_ids[i], []).append(tid)

        def execute_log_query(filter_str, related_traces, strategy):
            try:
                entries = list(client.list_entries(filter_=filter_str, max_results=20))
                return entries, related_traces, strategy
            except Exception as e:
                # print(f"Error in {strategy} query: {e}")
                return [], related_traces, strategy

        def process_entries(entries, related_traces, strategy):
            found_data = {}
            for entry in entries:
                ids = extract_ids_from_log_entry(entry)
                
                # Extract extra fields for session strategy
                if strategy == 'session' or strategy == 'text':
                    payload = getattr(entry, 'payload', {})
                    if isinstance(payload, dict):
                        if not ids.get('issue_id') and payload.get('issue_id'):
                            ids['issue_id'] = str(payload['issue_id'])
                        if not ids.get('domain') and payload.get('domain'):
                            ids['domain'] = str(payload['domain'])
                        if not ids.get('profile_id') and payload.get('user_id'):
                            ids['profile_id'] = str(payload['user_id'])
                        if not ids.get('message_id') and payload.get('message_id'):
                            ids['message_id'] = str(payload['message_id'])

                if ids:
                    for tid in related_traces:
                        if tid not in log_data:
                            log_data[tid] = ids.copy()
                        else:
                            for k, v in ids.items():
                                if not log_data[tid].get(k) and v:
                                    log_data[tid][k] = v
                        found_data[tid] = True
            return found_data

        # --- PHASE 1: ID-based Queries (Parallel) ---
        print("  Launching Phase 1 queries (Profile, Issue, Session)...")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # 1. Profile ID
            if profile_to_traces:
                for profile_id, related in profile_to_traces.items():
                    f = f'jsonPayload.profile_id="{profile_id}" AND {time_filter}'
                    futures.append(executor.submit(execute_log_query, f, related, 'profile'))
            
            # 2. Issue ID
            if issue_to_traces:
                for issue_id, related in issue_to_traces.items():
                    f = f'(jsonPayload.session_id:"{issue_id}" OR jsonPayload.issue_id="{issue_id}" OR jsonPayload.issue_pid="{issue_id}" OR jsonPayload.issueId="{issue_id}") AND {time_filter}'
                    futures.append(executor.submit(execute_log_query, f, related, 'issue'))

            # 3. Session ID
            if adk_session_to_traces:
                for adk_sid, related in adk_session_to_traces.items():
                    f = f'jsonPayload.adk_session_id="{adk_sid}" AND {time_filter}'
                    futures.append(executor.submit(execute_log_query, f, related, 'session'))
            
            for future in concurrent.futures.as_completed(futures):
                entries, related, strategy = future.result()
                process_entries(entries, related, strategy)

        # --- PHASE 2: Text Correlation (Parallel) ---
        traces_needing_logs = [t["trace_id"] for t in merged_traces 
                              if t["trace_id"] not in log_data or not log_data[t["trace_id"]].get("issue_id")]
        
        if traces_needing_logs:
            print(f"  Launching Phase 2 queries (Text Correlation) for {len(traces_needing_logs)} traces...")
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                for tr in merged_traces:
                    if tr["trace_id"] not in traces_needing_logs:
                        continue
                    
                    user_msg = tr.get("user_message")
                    if not user_msg or len(user_msg) < 10:
                        continue
                        
                    search_text = user_msg[:50].replace('"', '\\"')
                    
                    if tr.get("timestamp"):
                        try:
                            ts = datetime.fromisoformat(tr["timestamp"].replace('Z', '+00:00'))
                            start = (ts - timedelta(seconds=10)).isoformat()
                            end = (ts + timedelta(seconds=10)).isoformat()
                            time_filter_tight = f'timestamp >= "{start}" AND timestamp <= "{end}"'
                            
                            f = f'"{search_text}" AND {time_filter_tight}'
                            futures.append(executor.submit(execute_log_query, f, [tr["trace_id"]], 'text'))
                        except Exception:
                            continue
            
            for future in concurrent.futures.as_completed(futures):
                entries, related, strategy = future.result()
                process_entries(entries, related, strategy)
        
        print(f"Fetched log data for {len(log_data)} traces")
        
    except Exception as e:
        print(f"WARNING: Failed to fetch logs: {e}")
    
    return log_data

# ============================================================================
# TRACE FETCHING WITH BIFROST URL FILTER
# ============================================================================

def trace_matches_bifrost_url(trace: Dict, bifrost_url: str) -> bool:
    """Check if any span in the trace has the Bifrost URL in its attributes."""
    spans = trace.get("spans", [])
    for span in spans:
        labels = span.get("labels", {})
        
        # Check various URL attribute locations
        http_url = labels.get("/http/url") or labels.get("http.url") or ""
        if bifrost_url in http_url:
            return True
        
        # Also check in attributes if present
        attributes = span.get("attributes", {})
        for key, val in attributes.items():
            if "url" in key.lower() and bifrost_url in str(val):
                return True
    
    return False

def list_traces_with_bifrost_filter(
    project_id: str,
    limit: int = 50,
    days: int = 7,
    agent_engine_id: Optional[str] = None,
    bifrost_url: Optional[str] = None,
    start_from_time: Optional[datetime] = None
) -> List[Dict]:
    """
    Fetches traces using Cloud Trace API and filters by Bifrost URL.

    1. First fetches traces filtered by Agent Engine ID (via API filter)
    2. Then post-filters for traces containing the Bifrost URL
    3. Supports incremental fetching via start_from_time parameter
    """
    client = trace_v1.TraceServiceClient()
    start_time = None

    if start_from_time:
        # Use the provided start time for incremental processing
        start_time = Timestamp()
        start_time.FromDatetime(start_from_time)
        print(f"Incremental mode: fetching traces from {start_from_time.isoformat()}")
    elif days > 0:
        # Fallback to days parameter
        past_date = datetime.now(timezone.utc) - timedelta(days=days)
        start_time = Timestamp()
        start_time.FromDatetime(past_date)

    filter_str = f"+service.name:{agent_engine_id}" if agent_engine_id else None

    request = trace_v1.ListTracesRequest(
        project_id=project_id,
        view=trace_v1.ListTracesRequest.ViewType.COMPLETE,
        page_size=20,
        start_time=start_time,
        filter=filter_str
    )
    
    all_traces = []
    filtered_traces = []
    
    print(f"Fetching traces with Agent Engine filter: {agent_engine_id}")
    if bifrost_url:
        print(f"Will post-filter for Bifrost URL: {bifrost_url}")
    
    # Increase timeout to handle large time ranges
    pager = client.list_traces(request=request, timeout=120.0)
    
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

def process_consolidated_trace(trace, content_mapping):
    trace_id = trace["traceId"]
    spans = trace.get("spans", [])
    
    nodes = {span["spanId"]: span for span in spans}
    children = defaultdict(list)
    for span in spans:
        parent = span.get("parentSpanId")
        if parent and parent in nodes: 
            children[parent].append(span["spanId"])

    agent_spans = []
    for span in spans:
        labels = span.get("labels", {})
        if "invoke_agent" in span.get("name", "") or labels.get("gen_ai.agent.name"):
            agent_spans.append(span)

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
            "instruction_id": None,
            "instruction_version": None,
            "usecase_id": None,
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

                    # Remove isError if it is False (cleanup)
                    if "isError" in output and output["isError"] is False:
                        del output["isError"]

                if isinstance(args, dict):
                    if "issue_pid" in args: record["issue_id"] = str(args["issue_pid"])
                    if "domain" in args: record["domain"] = args["domain"]
                    if "profile_id" in args: record["profile_id"] = args["profile_id"]

                    # Capture instruction metadata for this agent
                    if tool_name == "get_usecase_instruction":
                        if args.get("instruction_id"):
                            agent_obj["instruction_id"] = args["instruction_id"]
                        if args.get("instruction_version"):
                            agent_obj["instruction_version"] = args["instruction_version"]
                        if args.get("usecase_id"):
                            agent_obj["usecase_id"] = args["usecase_id"]

                # --- Content Mapping Logic ---
                if tool_name == "get_usecase_instruction" and isinstance(args, dict):
                    instruction_id = args.get("instruction_id")
                    if instruction_id and isinstance(output, dict):
                        # Store only metadata in content_mapping (no instruction text)
                        content_list = output.get("content", [])
                        structured = output.get("structuredContent", {})

                        instruction_metadata = {
                            "instruction_id": instruction_id,
                            "domain": args.get("domain") or record.get("domain"),
                            "instruction_version": args.get("instruction_version"),
                            "usecase_id": args.get("usecase_id")
                        }

                        # Check structuredContent for metadata
                        if isinstance(structured, dict):
                            if structured.get("version"):
                                instruction_metadata["instruction_version"] = structured["version"]
                            if structured.get("domain"):
                                instruction_metadata["domain"] = structured["domain"]
                            if structured.get("usecase_id"):
                                instruction_metadata["usecase_id"] = structured["usecase_id"]

                        # Check content list for metadata
                        for item in content_list:
                            if item.get("type") == "text":
                                txt_data = item.get("text", {})
                                if isinstance(txt_data, dict):
                                    if txt_data.get("version"):
                                        instruction_metadata["instruction_version"] = txt_data["version"]
                                    if txt_data.get("domain"):
                                        instruction_metadata["domain"] = txt_data["domain"]
                                    if txt_data.get("usecase_id"):
                                        instruction_metadata["usecase_id"] = txt_data["usecase_id"]

                        # Remove None values to keep it clean
                        instruction_metadata = {k: v for k, v in instruction_metadata.items() if v is not None}

                        # Store only the 4 identifier fields (instruction text stays in JSONL)
                        content_mapping["instructions"][instruction_id] = instruction_metadata

                # --- FAQ Identifier Mapping (lightweight, no body content) ---
                if tool_name == "get_faqs" and isinstance(output, dict):
                    content_list = output.get("content", [])
                    for item in content_list:
                        if item.get("type") == "text":
                            txt_data = item.get("text", {})
                            if isinstance(txt_data, dict):
                                faqs_list = txt_data.get("faqs_list", [])
                                for faq in faqs_list:
                                    faq_id = faq.get("id")
                                    if faq_id:
                                        # Store only identifiers (no body content)
                                        faq_metadata = {
                                            "faq_id": faq_id,
                                            "faq_slug": faq.get("faq_slug"),
                                            "title": faq.get("title"),
                                            "domain": args.get("domain") or record.get("domain")
                                        }

                                        # Remove None values
                                        faq_metadata = {k: v for k, v in faq_metadata.items() if v is not None}

                                        # Store only identifiers (faq_body stays in JSONL)
                                        content_mapping["faqs"][faq_id] = faq_metadata

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

def append_to_gcs(data: List[Dict], bucket_name: str, file_name: str):
    """Appends new data to an existing GCS file (for incremental updates)."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        existing_data = []

        # Load existing data if file exists
        if blob.exists():
            print(f"Loading existing data from gs://{bucket_name}/{file_name}...")
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if line:
                    try:
                        existing_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(existing_data)} existing records")

        # Create a set of existing trace IDs to avoid duplicates
        existing_trace_ids = {record.get("trace_id") for record in existing_data if record.get("trace_id")}

        # Filter out duplicates from new data
        new_records = [record for record in data if record.get("trace_id") not in existing_trace_ids]

        if not new_records:
            print("No new records to append (all were duplicates)")
            return

        # Combine and sort all data
        all_data = existing_data + new_records
        all_data.sort(key=lambda x: x.get("timestamp") or "")

        # Upload combined data
        jsonl_string = "\n".join([json.dumps(record) for record in all_data])
        blob.upload_from_string(jsonl_string, content_type="application/json")
        print(f"Appended {len(new_records)} new records to gs://{bucket_name}/{file_name} (total: {len(all_data)})")
    except Exception as e:
        print(f"GCS Append Failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract traces with Agent Engine ID and Bifrost URL filters"
    )
    parser.add_argument("--limit", type=int, default=100, help="Max number of traces")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    parser.add_argument("--bifrost-url", type=str, default=BIFROST_URL,
                        help="Bifrost webhook URL to filter by")
    parser.add_argument("--no-bifrost-filter", action="store_true",
                        help="Disable Bifrost URL filtering")
    parser.add_argument("--output", type=str, default="agent_traces.jsonl",
                        help="Output file name")
    parser.add_argument("--skip-gcs", action="store_true", help="Skip GCS upload")
    parser.add_argument("--incremental", action="store_true",
                        help="Run in incremental mode (process only new traces since last run)")
    parser.add_argument("--state-file", type=str, default=STATE_FILE_GCS,
                        help="GCS path for state file (for incremental mode)")
    args = parser.parse_args()

    bifrost_filter = None if args.no_bifrost_filter else args.bifrost_url

    # 0. Load state for incremental processing
    state = {}
    start_from_time = None
    if args.incremental:
        state = load_state_from_gcs(GCS_BUCKET, args.state_file)
        if state.get("last_processed_time"):
            # Add 1 second to avoid reprocessing the same timestamp
            last_time = datetime.fromisoformat(state["last_processed_time"].replace('Z', '+00:00'))
            start_from_time = last_time + timedelta(seconds=1)

    # 0b. Load Content Mapping
    content_mapping = load_content_mapping()

    # 1. Fetch traces with both filters
    traces = list_traces_with_bifrost_filter(
        PROJECT_ID,
        limit=args.limit,
        days=args.days,
        agent_engine_id=AGENT_ENGINE_ID,
        bifrost_url=bifrost_filter,
        start_from_time=start_from_time
    )
    
    if not traces:
        print("No traces found matching the filters.")
        return
    
    # 2. Process traces
    merged_traces = []
    print(f"Processing {len(traces)} traces...")
    for trace in traces:
        try:
            merged_traces.append(process_consolidated_trace(trace, content_mapping))
        except Exception as e:
            print(f"Error extracting trace: {e}")
    
    # 2a. Save Content Mapping
    save_content_mapping(content_mapping)

    # 3. Fetch additional data from Cloud Logging
    print("Fetching additional data from Cloud Logging...")
    trace_ids = [t["trace_id"] for t in merged_traces]
    session_ids = [t["session_id"] for t in merged_traces]
    profile_ids = [t.get("profile_id") for t in merged_traces]
    issue_ids = [t.get("issue_id") for t in merged_traces]
    log_data = fetch_logs_by_trace_ids(
        PROJECT_ID, trace_ids, session_ids, args.days,
        profile_ids=profile_ids, issue_ids=issue_ids,
        merged_traces=merged_traces
    )
            
    # 4. Propagate IDs
    print("Propagating Session IDs...")
    final_rows = propagate_and_output(merged_traces, log_data)
    
    # 5. Save locally
    with open(args.output, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(final_rows)} consolidated records to {args.output}")

    # 6. Upload to GCS
    if not args.skip_gcs:
        if args.incremental:
            # In incremental mode, append to existing file
            append_to_gcs(final_rows, GCS_BUCKET, args.output)
        else:
            # In full mode, overwrite the file
            write_to_gcs(final_rows, GCS_BUCKET, args.output)

    # 7. Update state for incremental processing
    if args.incremental and final_rows:
        # Find the latest timestamp from processed traces
        latest_time = max(
            (row.get("timestamp") for row in final_rows if row.get("timestamp")),
            default=None
        )
        if latest_time:
            state["last_processed_time"] = latest_time
            state["last_run_time"] = datetime.now(timezone.utc).isoformat()
            state["traces_processed"] = len(final_rows)
            save_state_to_gcs(GCS_BUCKET, args.state_file, state)

    # 8. Summary
    print("\n=== Summary ===")
    print(f"Total traces: {len(final_rows)}")
    with_issue_id = sum(1 for r in final_rows if r.get("issue_id"))
    with_message_id = sum(1 for r in final_rows if r.get("message_id"))
    with_domain = sum(1 for r in final_rows if r.get("domain"))
    print(f"With issue_id: {with_issue_id}")
    print(f"With message_id: {with_message_id}")
    print(f"With domain: {with_domain}")
    if args.incremental:
        print(f"Incremental mode: {len(final_rows)} new traces since {start_from_time if start_from_time else 'beginning'}")

if __name__ == "__main__":
    main()
