"""
extract_traces_v2.py - Robust Agent Trace Extraction Script
"""

import os
import re
import json
import time
import argparse
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
# HELPER FUNCTIONS
# ============================================================================

def is_system_message(msg: Any) -> bool:
    if not msg: return False
    if isinstance(msg, dict):
        system_indicators = ['action_type', 'request', 'config', 'tool_call', 'escalate_issue', 'resolve_issue']
        return any(k in msg for k in system_indicators)
    
    if not isinstance(msg, str): return False
    msg_stripped = msg.strip()
    
    if msg_stripped.startswith(('{', '[')):
        try:
            return is_system_message(json.loads(msg_stripped))
        except: pass
            
    internal_keywords = r'(action_type|issue_id|issue_pid|instruction_id|domain_id|profile_id)'
    if re.search(internal_keywords + r'\s*[:=]\s*', msg_stripped, re.IGNORECASE):
        return True
    return False

def try_parse_json(data: Any) -> Any:
    """Safely attempts to parse JSON strings into objects."""
    if isinstance(data, str):
        stripped = data.strip()
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            try:
                parsed = json.loads(stripped)
                return try_parse_json(parsed)
            except:
                return data
    elif isinstance(data, dict):
        return {k: try_parse_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [try_parse_json(i) for i in data]
    return data

def extract_ids_from_instruction(instruction: str) -> Dict[str, str]:
    ids = {}
    patterns = {
        "domain_id": r"domain:\s*([^\n]+)",
        "issue_id": r"(?:issue_pid|issue_id):\s*(\d+)",
        "profile_id": r"profile_id:\s*([^\n]+)",
        "app_id": r"app_id:\s*([^\n]+)",
        "sandbox_id": r"sandbox:\s*([^\n]+)",
        "instruction_id": r"instruction_id:\s*([^\n]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, str(instruction), re.IGNORECASE)
        if match:
            ids[key] = match.group(1).strip()
    return ids

def find_kb_items_recursive(data: Any) -> List[Dict]:
    items = []
    if isinstance(data, dict):
        kb_keys = {"goal", "instruction_version", "usecase_name", "faq_body", "id", "title"}
        if any(k in data for k in kb_keys):
            items.append(data)
        for v in data.values():
            items.extend(find_kb_items_recursive(v))
    elif isinstance(data, list):
        for item in data:
            items.extend(find_kb_items_recursive(item))
    return items

def clean_and_deduplicate_kb(kb_list: list) -> list:
    if not kb_list: return kb_list
    unique_items = []
    seen = set()
    for item in kb_list:
        if not isinstance(item, dict): continue
        if str(item.get('usecase_name', '')).startswith('DO_NOT_USE'): continue
        fingerprint = json.dumps(item, sort_keys=True)
        if fingerprint not in seen:
            unique_items.append(item)
            seen.add(fingerprint)
    return unique_items

# ============================================================================
# TRACE PROCESSING
# ============================================================================

def list_traces_safe(project_id: str, limit: int = 50, days: int = 0, agent_engine_id: Optional[str] = None):
    """Fetches traces with rate-limit handling."""
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
        page_size=100, # Small page size to avoid timeouts
        start_time=start_time,
        filter=filter_str
    )
    
    traces = []
    print(f"Fetching up to {limit} traces (days={days}, filter={agent_engine_id})...")
    
    # Manually iterate pages to handle rate limits
    pager = client.list_traces(request=request)
    
    try:
        count = 0
        for trace in pager:
            traces.append(MessageToDict(trace._pb))
            count += 1
            if count >= limit:
                break
            
            # Simple rate limiting: Sleep slightly every 100 traces
            if count % 100 == 0:
                print(f"  ...fetched {count} traces, sleeping to respect quota...")
                time.sleep(2)
                
    except (ResourceExhausted, ServiceUnavailable) as e:
        print(f"\nWARNING: API Quota hit or Service Unavailable after fetching {len(traces)} traces.")
        print(f"Saving what we have so far. Error: {e}")
        
    return traces

def build_span_tree(spans):
    nodes = {span["spanId"]: span for span in spans}
    children = defaultdict(list)
    roots = []
    for span in spans:
        sid = span["spanId"]
        parent_id = span.get("parentSpanId")
        if parent_id and parent_id in nodes:
            children[parent_id].append(sid)
        else:
            roots.append(sid)
    return nodes, children, roots

def process_trace(trace, target_agent=None):
    trace_id = trace["traceId"]
    spans = trace.get("spans", [])
    nodes, children_map, roots = build_span_tree(spans)
    
    agent_roots = set()
    for span in spans:
        labels = span.get("labels", {})
        span_name = span.get("name", "")
        agent_name = labels.get("gen_ai.agent.name")
        if not agent_name and span_name.startswith("invoke_agent "):
            agent_name = span_name.split(" ", 1)[1]
        if agent_name:
            agent_roots.add(span["spanId"])

    def get_exclusive_descendants(start_span_id, children_map, all_agent_roots):
        descendants = []
        stack = [start_span_id]
        while stack:
            curr = stack.pop()
            kids = children_map.get(curr, [])
            for k in kids:
                if k in all_agent_roots:
                    descendants.append(k)
                else:
                    descendants.append(k)
                    stack.append(k)
        return descendants

    agent_executions = []

    for span in spans:
        if span["spanId"] not in agent_roots: continue
             
        labels = span.get("labels", {})
        span_name = span.get("name", "")
        agent_name = labels.get("gen_ai.agent.name")
        if not agent_name and span_name.startswith("invoke_agent "):
            agent_name = span_name.split(" ", 1)[1]
            
        if target_agent and agent_name != target_agent: continue

        descendant_ids = get_exclusive_descendants(span["spanId"], children_map, agent_roots)
        
        # Calculate Latency
        start_time_str = span.get("startTime")
        end_time_str = span.get("endTime")
        response_latency_ms = 0
        if start_time_str and end_time_str:
            try:
                start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                response_latency_ms = int((end_dt - start_dt).total_seconds() * 1000)
            except: pass

        extracted_data = {
            "trace_id": trace_id,
            "agent_name": agent_name,
            "session_id": labels.get("gcp.vertex.agent.session_id") or labels.get("gen_ai.conversation.id"), 
            "message_id": labels.get("gcp.vertex.agent.event_id"),
            "invocation_id": labels.get("gcp.vertex.agent.invocation_id"),
            "timestamp": start_time_str,
            "user_message": None,
            "agent_response": None,
            "tool_info": [],
            "knowledge_base": [],
            "llm_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "model_failures": [],
            "tool_failures": [],
            "ids": {
                "domain_id": None, "issue_id": None, "profile_id": None, 
                "app_id": None, "sandbox_id": None
            },
            "category": None,
            "prompt_version": None,
            "tool_count": 0,
            "llm_call_count": 0,
            "response_latency_ms": response_latency_ms,
            "guard_analysis": None,
            "action_type": None,
            "escalation_info": None,
            "conversation_context": None,
            "final_status": None,
            "inferred_category": None,
            "rag_context": [],
            "is_system_message": False,
            "extracted_spans": []
        }

        all_relevant_spans = [nodes[sid] for sid in descendant_ids]
        all_relevant_spans.sort(key=lambda s: s.get("startTime", ""))

        for sub_span in all_relevant_spans:
            sub_labels = sub_span.get("labels", {})
            sub_name = sub_span.get("name", "")

            # 1. Capture Visualization Spans
            if sub_span.get("startTime") and sub_span.get("endTime"):
                extracted_data["extracted_spans"].append({
                    "name": sub_name,
                    "start": sub_span["startTime"],
                    "end": sub_span["endTime"],
                    "type": "tool" if "execute_tool" in sub_name else "llm" if "call_llm" in sub_name else "agent"
                })

            # 2. Capture Session IDs
            if not extracted_data["session_id"]:
                extracted_data["session_id"] = sub_labels.get("gcp.vertex.agent.session_id") or sub_labels.get("gen_ai.conversation.id")

            # 3. Process Tools
            if "execute_tool" in sub_name:
                extracted_data["tool_count"] += 1
                tool_name = sub_labels.get("gen_ai.tool.name") or sub_name.replace("execute_tool ", "")
                
                req_args = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_call_args"))
                tool_out = try_parse_json(sub_labels.get("gcp.vertex.agent.tool_response"))
                
                tool_status = "unknown"
                if isinstance(tool_out, dict):
                    tool_status = tool_out.get("status", "success")
                
                if str(tool_status).lower() in ["failure", "error"]:
                    extracted_data["tool_failures"].append({"tool": tool_name, "error": tool_out})

                extracted_data["tool_info"].append({
                    "tool": tool_name,
                    "requested_args": req_args,
                    "tool_output": tool_out,
                    "status": tool_status,
                    "timestamp": sub_span.get("endTime")
                })

                if isinstance(req_args, dict):
                    if "domain" in req_args: extracted_data["ids"]["domain_id"] = req_args["domain"]
                    if "issue_pid" in req_args: extracted_data["ids"]["issue_id"] = str(req_args["issue_pid"])
                    elif "issue_id" in req_args: extracted_data["ids"]["issue_id"] = str(req_args["issue_id"])
                
                if isinstance(req_args, dict) and "action_type" in req_args: 
                    extracted_data["action_type"] = req_args["action_type"]
                
                kb_items = find_kb_items_recursive(tool_out)
                if kb_items: extracted_data["knowledge_base"].extend(kb_items)

            # 4. Process LLM Calls
            if "call_llm" in sub_name or sub_labels.get("gen_ai.operation.name") == "generate_content":
                extracted_data["llm_call_count"] += 1
                i_tok = int(sub_labels.get("gen_ai.usage.input_tokens", 0))
                o_tok = int(sub_labels.get("gen_ai.usage.output_tokens", 0))
                extracted_data["llm_usage"]["total_tokens"] += (i_tok + o_tok)

                llm_req = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_request"))
                llm_res = try_parse_json(sub_labels.get("gcp.vertex.agent.llm_response"))

                if isinstance(llm_req, dict) and "config" in llm_req:
                    if "system_instruction" in llm_req["config"]:
                        ids = extract_ids_from_instruction(str(llm_req["config"]["system_instruction"]))
                        for k, v in ids.items():
                            if v and not extracted_data["ids"][k]: extracted_data["ids"][k] = v

                # FIX: User Message Extraction with Type Safety
                if not extracted_data["user_message"] and isinstance(llm_req, dict) and "contents" in llm_req:
                    for content in reversed(llm_req["contents"]):
                        if content.get("role") == "user":
                            parts = content.get("parts", [])
                            text_parts = []
                            for p in parts:
                                txt = p.get("text")
                                if isinstance(txt, str):
                                    text_parts.append(txt)
                                elif isinstance(txt, (dict, list)):
                                    # If it was parsed to a dict, dump it back to string
                                    text_parts.append(json.dumps(txt))
                            
                            if text_parts:
                                msg = " ".join(text_parts)
                                if not msg.startswith("For context:"):
                                    extracted_data["user_message"] = try_parse_json(msg)
                                    extracted_data["is_system_message"] = is_system_message(msg)
                                    break
                
                # FIX: Agent Response Extraction with Type Safety
                if isinstance(llm_res, dict) and "content" in llm_res:
                    parts = llm_res["content"].get("parts", [])
                    text_parts = []
                    for p in parts:
                        txt = p.get("text")
                        if isinstance(txt, str):
                            text_parts.append(txt)
                        elif isinstance(txt, (dict, list)):
                            text_parts.append(json.dumps(txt))
                    
                    if text_parts:
                        extracted_data["agent_response"] = try_parse_json(" ".join(text_parts))

        extracted_data["knowledge_base"] = clean_and_deduplicate_kb(extracted_data["knowledge_base"])
        if extracted_data["agent_response"]: extracted_data["final_status"] = "responded"
        elif extracted_data["action_type"] == "ESCALATE_ISSUE": extracted_data["final_status"] = "escalated"
        else: extracted_data["final_status"] = "processing"

        agent_executions.append(extracted_data)

    return agent_executions

def write_to_gcs(data: List[Dict], bucket_name: str, file_name: str):
    """Writes JSONL to GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        jsonl_string = "\n".join([json.dumps(record) for record in data])
        blob.upload_from_string(jsonl_string, content_type="application/json")
        print(f"Uploaded {len(data)} records to gs://{bucket_name}/{file_name}")
    except Exception as e:
        print(f"GCS Upload Failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    
    # Use the safe version of list_traces
    traces = list_traces_safe(PROJECT_ID, limit=args.limit, days=args.days, agent_engine_id=AGENT_ENGINE_ID)
    
    if not traces:
        print("No traces found or API failed.")
        return

    all_data = []
    print(f"Processing {len(traces)} traces...")
    for trace in traces:
        try:
            all_data.extend(process_trace(trace))
        except Exception as e:
            print(f"Skipping trace {trace.get('traceId')} due to error: {e}")
    
    with open("agent_traces.jsonl", "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
    
    write_to_gcs(all_data, GCS_BUCKET, "agent_traces.jsonl")

if __name__ == "__main__":
    main()