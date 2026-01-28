
import os
import re
import json
import uuid
import argparse
from typing import List, Dict, Any, Optional
from collections import defaultdict
from google.cloud import trace_v1
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.protobuf.json_format import MessageToDict
from dotenv import load_dotenv

# Constants
PROJECT_ID = "product-research-460317"
OUTPUT_FILE = "agent_traces.jsonl"
GCS_BUCKET = "evaluation-research"

# ============================================================================
# GLOBAL HELPER FUNCTIONS (Universal - work for ANY agent engine)
# ============================================================================

def is_system_message(msg: Any) -> bool:
    """
    Detect if a user_message is actually a system/internal payload (not human input).
    Enhanced for Phase 2.3 to handle already-parsed dictionaries/lists.
    """
    if not msg:
        return False
        
    # If it's already a dict, check for system indicators
    if isinstance(msg, dict):
        system_indicators = [
            'action_type', 'request', 'config', 'tool_call', 
            'message_to_user', 'pvt_note_content', 'private_note',
            'escalate_issue', 'resolve_issue', 'reject_issue'
        ]
        return any(k in msg for k in system_indicators)
        
    if not isinstance(msg, str):
        return False
        
    msg_stripped = msg.strip()
    
    # 1. Check if it looks like JSON
    if msg_stripped.startswith('{') or msg_stripped.startswith('['):
        try:
            obj = json.loads(msg_stripped)
            return is_system_message(obj)
        except:
            pass
            
    # 2. Check for key=value or key:value patterns
    internal_keywords = r'(action_type|issue_id|issue_pid|instruction_id|domain_id|profile_id|type|status)'
    if re.search(internal_keywords + r'\s*[:=]\s*', msg_stripped, re.IGNORECASE):
        return True

    # 3. Check for comma-separated IDs/Actions patterns
    if ',' in msg_stripped:
        parts = msg_stripped.split(',')
        if len(parts) >= 2 and (parts[0].upper() in ["ESCALATE_ISSUE", "RESOLVE_ISSUE", "REJECT_ISSUE", "TOOL_CALL"]):
            return True
        if all(re.match(r'^[a-zA-Z0-9_\-]{8,}$', p.strip()) for p in parts if p.strip()):
            return True

    # 4. Check for technical density (UUIDs, hashes)
    if re.search(r'[a-f0-9]{8}-([a-f0-9]{4}-){3}[a-f0-9]{12}', msg_stripped): # UUID
        return True
    if len(msg_stripped) > 24 and re.match(r'^[a-f0-9]+$', msg_stripped.lower()): # Long hex hash
        return True

    return False

def try_parse_json(data: Any) -> Any:
    """
    Recursively attempt to parse strings as JSON if they look like JSON.
    This ensures structured data is captured as objects, not strings.
    """
    if isinstance(data, str):
        stripped = data.strip()
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            try:
                parsed = json.loads(stripped)
                # Recursively parse the result in case of nested stringified JSON
                return try_parse_json(parsed)
            except:
                return data
        return data
    elif isinstance(data, dict):
        return {k: try_parse_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [try_parse_json(i) for i in data]
    return data

def get_category_priority(category: str) -> int:
    """
    Assign priority score to categories (higher = better/more specific).
    This ensures trace-level propagation picks the most meaningful category.
    """
    if not category:
        return 0
    # Lowest priority - generic fallbacks
    if category in ["general_inquiry", "internal_agent_call", "greetings_or_chitchat"]:
        return 1
    # Medium priority - action-derived
    if category in ["escalation", "resolved", "rejected"]:
        return 2
    # High priority - semantic fallback categories
    if category in ["gameplay_questions", "technical_issues", "policy_questions", "feedback_or_suggestion", "missing_rewards", "profile_or_account_issues", "purchase_or_payment_issues"]:
        return 3
    # Highest priority - explicitly matched usecases
    return 4

def propagate_metadata_within_trace(executions: List[Dict]) -> List[Dict]:
    """
    Phase 2: Propagate identity metadata (issue_id, domain_id, etc.) across the trace.
    Ensures that if one agent extracts metadata, all agents in the same trace share it.
    """
    if not executions:
        return executions
        
    by_trace = defaultdict(list)
    for ex in executions:
        by_trace[ex.get('trace_id', 'unknown')].append(ex)
        
    for trace_id, group in by_trace.items():
        # Find best IDs in this trace
        best_ids = {
            "domain_id": None,
            "issue_id": None,
            "profile_id": None,
            "app_id": None,
            "sandbox_id": None
        }
        
        for ex in group:
            ids = ex.get("ids", {})
            for k, v in ids.items():
                if v and not best_ids.get(k):
                    best_ids[k] = v
                    
        # Apply best IDs to all in trace
        for ex in group:
            for k, v in best_ids.items():
                if v and not ex["ids"].get(k):
                    ex["ids"][k] = v
                    
    return executions


def propagate_categories_within_trace(executions: List[Dict]) -> List[Dict]:
    """
    Post-process to propagate the best category within each trace_id.
    This ensures consistency - all agents in same trace share the best category.
    Universal solution that works for ANY agent engine.
    """
    if not executions:
        return executions
    
    # Group by trace_id
    by_trace = defaultdict(list)
    for ex in executions:
        by_trace[ex.get('trace_id', 'unknown')].append(ex)
    
    # For each trace group, find and propagate the best category
    for trace_id, group in by_trace.items():
        # Find the best category in this trace
        best_category = None
        best_priority = 0
        original_user_message = None
        
        for ex in group:
            # Track original user message (from non-system messages)
            if not ex.get('is_system_message') and ex.get('user_message'):
                if not original_user_message:
                    original_user_message = ex['user_message']
            
            # Check explicit category first
            cat = ex.get('category') or ex.get('inferred_category')
            priority = get_category_priority(cat)
            if priority > best_priority:
                best_priority = priority
                best_category = cat
        
        # Propagate best category and original_user_message to all in trace
        for ex in group:
            current_priority = get_category_priority(ex.get('inferred_category'))
            # Only upgrade if current is weaker
            if best_category and current_priority < best_priority:
                ex['inferred_category'] = best_category
            # Add original_user_message for system message entries
            if ex.get('is_system_message') and original_user_message:
                ex['original_user_message'] = original_user_message
    
    return executions

def find_kb_items_recursive(data: Any) -> List[Dict]:
    """
    Recursively find all dictionaries that look like Knowledge Base items
    (contain goal, instruction_version, usecase_name, or faq_body).
    Phase 2.2: Scavenges 100+ items from any nesting level.
    """
    items = []
    if isinstance(data, dict):
        # If this dict looks like a KB item, capture it
        kb_keys = {"goal", "instruction_version", "usecase_name", "faq_body", "id", "title"}
        if any(k in data for k in kb_keys):
            items.append(data)
        
        # Recurse into all values
        for v in data.values():
            items.extend(find_kb_items_recursive(v))
    elif isinstance(data, list):
        for item in data:
            items.extend(find_kb_items_recursive(item))
    return items


def clean_and_deduplicate_kb(kb_list: list) -> list:
    """
    Filter out DO_NOT_USE entries and perform deep deduplication.
    Uses JSON string as a fingerprint for stable uniqueness.
    """
    if not kb_list:
        return kb_list
        
    unique_items = []
    seen_fingerprints = set()
    
    for item in kb_list:
        if not isinstance(item, dict):
            # If it's a raw string (e.g. instruction text), deduplicate it too
            if isinstance(item, str):
                item_str = item.strip()
                if item_str and item_str not in seen_fingerprints:
                    unique_items.append(item_str)
                    seen_fingerprints.add(item_str)
            continue
            
        # Skip internal/test categories
        u_name = str(item.get('usecase_name', ''))
        if u_name.startswith('DO_NOT_USE'):
            continue
            
        # Create a stable fingerprint for the dictionary
        # We sort keys to ensure identical dicts have identical fingerprints
        fingerprint = json.dumps(item, sort_keys=True)
        if fingerprint not in seen_fingerprints:
            unique_items.append(item)
            seen_fingerprints.add(fingerprint)
            
    return unique_items


# Action type to category mapping (derived from trace action_type field)
ACTION_TYPE_CATEGORY_MAP = {
    "ESCALATE_ISSUE": "escalation",
    "RESOLVE_ISSUE": "resolved",
    "REJECT_ISSUE": "rejected",
}

# Semantic fallback patterns (used when no usecases match)
# Order matters: more specific categories should come first
SEMANTIC_FALLBACK_PATTERNS = {
    "policy_questions": ["gdpr", "privacy", "rights", "data", "terms", "policy", "delete", "erase", "personal information", "dsa", "legal"],
    "purchase_or_payment_issues": ["purchase", "buy", "pay", "charge", "receipt", "billing", "money", "transaction", "payment", "refund", "store", "bought", "cost", "price"],
    "missing_rewards": ["reward", "missing", "chips", "award", "prize", "gift", "bonus", "claim", "earning", "win"],
    "profile_or_account_issues": ["account", "login", "password", "sign in", "profile", "avatar", "email", "reset", "access", "username", "identit", "id", "link"],
    "technical_issues": ["wifi", "network", "connect", "internet", "loading", "crash", "freeze", "lag", "update", "install", "download", "server", "error", "not working", "bug", "technical", "stuck"],
    "feedback_or_suggestion": ["suggest", "feedback", "improve", "wish", "would be nice", "should add", "feature request", "opinion", "idea"],
    "greetings_or_chitchat": ["hello", "hi", "hey", "how are you", "how u doing", "thanks", "thank you", "bye", "greeting", "morning", "evening"],
    "gameplay_questions": ["level", "how to", "what is", "how do", "where can", "when", "feature", "unlock", "vault", "club", "league", "event", "spin", "play", "gameplay", "game", "quest", "task"],
}

def infer_category_dynamic(user_message: Any, usecases_list: list, action_type: str = None) -> str:
    """
    Dynamically infer category from trace data without hardcoded keywords.
    Enhanced for Phase 2.3 to handle both strings and parsed objects.
    """
    if not user_message:
        msg_lower = ""
    elif isinstance(user_message, str):
        msg_lower = user_message.lower()
    else:
        # If it's a dict/list, convert to string for keyword matching
        msg_lower = json.dumps(user_message).lower()
    
    # Priority 1: Match user message against dynamic usecases_list vocabulary
    if usecases_list and msg_lower:
        best_match = None
        best_score = 0
        
        for usecase in usecases_list:
            if not isinstance(usecase, dict):
                continue
            
            usecase_name = usecase.get("usecase_name", "")
            goal = usecase.get("goal", "")
            
            # Skip placeholder/system usecases
            if usecase_name.startswith("DO_NOT_USE"):
                continue
            
            # Calculate match score based on word overlap
            score = 0
            
            # Check usecase name words (higher weight)
            name_words = usecase_name.lower().replace("_", " ").split()
            for word in name_words:
                if len(word) > 2 and word in msg_lower:
                    score += 3
            
            # Check goal words (lower weight)
            goal_words = goal.lower().replace("_", " ").split()
            for word in goal_words:
                if len(word) > 3 and word in msg_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = usecase_name
        
        if best_match and best_score >= 1:
            return best_match
    
    # Priority 2: Derive from action_type (only if user message didn't match any usecase)
    if action_type and action_type in ACTION_TYPE_CATEGORY_MAP:
        return ACTION_TYPE_CATEGORY_MAP[action_type]
    
    # Priority 3: Semantic pattern matching for common unhandled question types
    if msg_lower:
        # Standardize apostrophes for better matching
        msg_standardized = msg_lower.replace("’", "'").replace("n't", " not").replace("'s", " is")
        for category, patterns in SEMANTIC_FALLBACK_PATTERNS.items():
            for pattern in patterns:
                # Use regex for word boundaries to avoid partial matches (e.g., "hi" in "this")
                if re.search(r'\b' + re.escape(pattern) + r'\b', msg_standardized):
                    return category
    
    # Priority 4: Fallback to general_inquiry
    if msg_lower.strip():
        return "general_inquiry"
    
    return None



def get_trace_client():
    return trace_v1.TraceServiceClient()

from datetime import datetime, timedelta
from google.protobuf.timestamp_pb2 import Timestamp

def list_traces(project_id: str, limit: int = 50, days: int = 0) -> List[Dict[str, Any]]:
    client = get_trace_client()
    
    # Use view=2 for COMPLETE/ROOTSPAN data 
    try:
        view_type = trace_v1.ListTracesRequest.ViewType.COMPLETE
    except AttributeError:
        view_type = 2

    start_time = None
    if days > 0:
        # Calculate start time
        past_date = datetime.utcnow() - timedelta(days=days)
        start_time = Timestamp()
        start_time.FromDatetime(past_date)

    request = trace_v1.ListTracesRequest(
        project_id=project_id,
        view=view_type,
        page_size=limit if limit < 1000 else 1000,
        start_time=start_time
    )
    
    # Iterate and fetch
    traces = []
    print(f"Fetching {limit} traces (filter days={days})...")
    for trace in client.list_traces(request=request):
        traces.append(MessageToDict(trace._pb))
        if len(traces) >= limit:
            break
            
    return traces

def extract_ids_from_instruction(instruction: str) -> Dict[str, str]:
    """Extracts domain, issue_pid, profile_id from system instruction text."""
    ids = {}
    
    # Regex patterns based on the inspected trace
    patterns = {
        "domain_id": r"domain:\s*([^\n]+)",
        "issue_id": r"issue_pid:\s*(\d+)",
        "profile_id": r"profile_id:\s*([^\n]+)",
        "app_id": r"app_id:\s*([^\n]+)",
        "sandbox_id": r"sandbox:\s*([^\n]+)",
        "instruction_id": r"instruction_id:\s*([^\n]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, instruction)
        if match:
            ids[key] = match.group(1).strip()
            
    return ids

def parse_llm_request(request_json_str: str) -> Dict[str, Any]:
    try:
        return json.loads(request_json_str)
    except:
        return {}

def parse_llm_response(response_json_str: str) -> Dict[str, Any]:
    try:
        return json.loads(response_json_str)
    except:
        return {}

def build_span_tree(spans):
    """
    Builds a tree representation of spans.
    Returns:
        nodes: dict {span_id: span}
        children: dict {parent_span_id: [child_span_ids]}
        roots: list of span_ids that have no parent in the provided list
    """
    nodes = {}
    children = {}
    span_ids = set()
    
    for span in spans:
        sid = span["spanId"]
        nodes[sid] = span
        span_ids.add(sid)
        children[sid] = []

    roots = []
    for span in spans:
        sid = span["spanId"]
        parent_id = span.get("parentSpanId")
        if parent_id and parent_id in nodes:
            children[parent_id].append(sid)
        else:
            roots.append(sid)
            
    return nodes, children, roots

def get_descendants(span_id, children_map):
    """
    Recursively get all descendant span IDs.
    """
    descendants = []
    stack = [span_id]
    while stack:
        curr = stack.pop()
        kids = children_map.get(curr, [])
        descendants.extend(kids)
        stack.extend(kids)
    return descendants

def process_trace(trace, target_agent=None):
    """
    Process a trace and return a list of agent execution objects.
    If target_agent is specified, only return objects for that agent.
    """
    trace_id = trace["traceId"]
    spans = trace.get("spans", [])
    agent_roots = set()
    for span in spans:
        labels = span.get("labels", {})
        span_name = span.get("name", "")
        agent_name = labels.get("gen_ai.agent.name")
        if not agent_name and span_name.startswith("invoke_agent "):
            agent_name = span_name.split(" ", 1)[1]
        
        if agent_name:
            agent_roots.add(span["spanId"])

    nodes, children_map, roots = build_span_tree(spans)
    print(f"DEBUG: Found {len(agent_roots)} agent roots: {list(agent_roots)[:3]}")


    # Helper for exclusive descendants
    def get_exclusive_descendants(start_span_id, children_map, all_agent_roots):
        descendants = []
        stack = [start_span_id]
        
        while stack:
            curr = stack.pop()
            # If curr is an agent root (and not the one we started with), 
            # we include it (to show the call) but DO NOT recurse.
            # However, the `start_span_id` IS an agent root, so we must allow it to expand.
            
            kids = children_map.get(curr, [])
            
            for k in kids:
                if k in all_agent_roots:
                    descendants.append(k)
                    if start_span_id in ['8155261932500545236', '11326816054061857699']:
                         print(f"DEBUG: Stopped recursion at sub-agent {k}")
                else:
                    descendants.append(k)
                    stack.append(k)
                    if start_span_id in ['8155261932500545236', '11326816054061857699']:
                         print(f"DEBUG: Recursing into {k}")
        
        return descendants

    agent_executions = []
    
    # Pre-collect usecases from ALL spans in this trace (shared across agents)
    trace_level_usecases = []
    for span in spans:
        labels = span.get("labels", {})
        tool_output_str = labels.get("gcp.vertex.agent.tool_response")
        if tool_output_str:
            try:
                tool_output = json.loads(tool_output_str)
                if isinstance(tool_output, dict):
                    structured = tool_output.get("structuredContent", tool_output)
                    if isinstance(structured, dict) and "usecases_list" in structured:
                        for uc in structured["usecases_list"]:
                            if isinstance(uc, dict) and uc not in trace_level_usecases:
                                # Filter DO_NOT_USE at trace level collection
                                if not str(uc.get("usecase_name", "")).startswith("DO_NOT_USE"):
                                    trace_level_usecases.append(uc)
            except:
                pass


    # Find all 'invoke_agent' spans
    for span in spans:
        if span["spanId"] not in agent_roots:
             continue
             
        labels = span.get("labels", {})
        span_name = span.get("name", "")
        agent_name = labels.get("gen_ai.agent.name")
        if not agent_name and span_name.startswith("invoke_agent "):
            agent_name = span_name.split(" ", 1)[1]
            
        # If filtering is requested
        if target_agent and agent_name != target_agent:
            continue

        # Found an agent execution! Extract separate data for it.
        descendant_ids = get_exclusive_descendants(span["spanId"], children_map, agent_roots)
        
        # Calculate response latency from span timestamps
        start_time_str = span.get("startTime")
        end_time_str = span.get("endTime")
        response_latency_ms = None
        if start_time_str and end_time_str:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                response_latency_ms = int((end_dt - start_dt).total_seconds() * 1000)
            except:
                pass

        extracted_data = {
            "trace_id": trace_id,
            "agent_name": agent_name,
            "session_id": labels.get("gcp.vertex.agent.session_id"), 
            "message_id": labels.get("gcp.vertex.agent.event_id"),
            "invocation_id": labels.get("gcp.vertex.agent.invocation_id"),
            "timestamp": span.get("startTime"),
            "user_message": None,
            "agent_response": None,
            "tool_info": [],
            "knowledge_base": None,
            "llm_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "model_failures": [],
            "tool_failures": [],
            "ids": {
                "domain_id": None,
                "issue_id": None,
                "profile_id": None,
                "app_id": None,
                "sandbox_id": None
            },
            "category": None,
            "prompt_version": None,
            # New generic fields
            "tool_count": 0,
            "llm_call_count": 0,
            "response_latency_ms": response_latency_ms,
            "guard_analysis": None,
            "action_type": None,
            "escalation_info": None,
            "conversation_context": None,
            "final_status": None,
            # Category and RAG fields
            "inferred_category": None,
            "rag_context": [],
            "has_rag_context": False,
            # Global flags for universal trace handling
            "is_system_message": False,
            "original_user_message": None
        }
        
        instruction_id_map = {}
        collected_usecases_list = []  # Collect available usecases for dynamic category inference
        # Helper to update IDs safely
        def update_ids(new_ids):
            for k, v in new_ids.items():
                if v and not extracted_data["ids"].get(k):
                    if k == "instruction_id":
                         check_category_by_id(v)
                    else:
                         extracted_data["ids"][k] = v

        # Helper to update category if instruction_id found
        def check_category_by_id(instr_id):
             if instr_id and instr_id in instruction_id_map and not extracted_data["category"]:
                 extracted_data["category"] = instruction_id_map[instr_id]

        all_relevant_spans = [nodes[sid] for sid in descendant_ids]
        # Sort by start time to get chronological order (input -> tool -> output)
        all_relevant_spans.sort(key=lambda s: s.get("startTime", ""))

        # We also want to look at the AGENT span itself for potential session ID
        if not extracted_data["session_id"] and "gen_ai.conversation.id" in labels:
             extracted_data["session_id"] = labels["gen_ai.conversation.id"]
             
        # Iterate through descendants to find LLM/Tool calls
        first_llm_request_found = False
        
        for sub_span in all_relevant_spans:
            sub_labels = sub_span.get("labels", {})
            sub_name = sub_span.get("name", "")

            # Message ID / Session ID fallback
            if not extracted_data["message_id"] and "gcp.vertex.agent.event_id" in sub_labels:
                extracted_data["message_id"] = sub_labels["gcp.vertex.agent.event_id"]
            if not extracted_data["session_id"]:
                if "gcp.vertex.agent.session_id" in sub_labels:
                    extracted_data["session_id"] = sub_labels["gcp.vertex.agent.session_id"]
                elif "gen_ai.conversation.id" in sub_labels:
                    extracted_data["session_id"] = sub_labels["gen_ai.conversation.id"]
            if not extracted_data["invocation_id"] and "gcp.vertex.agent.invocation_id" in sub_labels:
                extracted_data["invocation_id"] = sub_labels["gcp.vertex.agent.invocation_id"]

            # Tools - Process tool executions
            if "execute_tool" in sub_name or sub_labels.get("gen_ai.operation.name") == "execute_tool":
                tool_name = sub_labels.get("gen_ai.tool.name") or sub_name.replace("execute_tool ", "")
                requested_args_str = sub_labels.get("gcp.vertex.agent.tool_call_args")
                tool_output_str = sub_labels.get("gcp.vertex.agent.tool_response")
                
                # Parse args to JSON if possible
                requested_args = requested_args_str
                try:
                    if requested_args_str:
                        requested_args = json.loads(requested_args_str)
                except:
                    pass

                # Parse tool output to JSON if possible
                tool_output = tool_output_str
                try:
                    if tool_output_str:
                        tool_output = json.loads(tool_output_str)
                except:
                    pass

                tool_status = "unknown"
                structured_data = {}
                resp_json = {}
                
                # Parse tool status and extract structured data
                if tool_output_str:
                    try:
                        resp_json = tool_output if isinstance(tool_output, dict) else json.loads(tool_output_str)

                        if isinstance(resp_json, dict):
                            if "status" in resp_json:
                                tool_status = resp_json["status"]
                            elif "isError" in resp_json:
                                tool_status = "failure" if resp_json["isError"] else "success"
                            
                            # Extract structured content
                            structured_data = resp_json
                            if "structuredContent" in resp_json and isinstance(resp_json["structuredContent"], dict):
                                structured_data = resp_json["structuredContent"]
                                if "status" in structured_data: 
                                    tool_status = structured_data["status"]
                    except:
                        pass

                # Add tool info with parsed JSON (not strings)
                extracted_data["tool_info"].append({
                    "tool": tool_name,
                    "requested_args": try_parse_json(requested_args),
                    "tool_output": try_parse_json(tool_output),
                    "status": tool_status,
                    "timestamp": sub_span.get("endTime") or sub_span.get("startTime")
                })

                # UNIVERSAL KB EXTRACTION (Phase 2.2): Scavenge everything recursively
                kb_items = find_kb_items_recursive(structured_data)
                if kb_items:
                    if extracted_data["knowledge_base"] is None:
                        extracted_data["knowledge_base"] = []
                    extracted_data["knowledge_base"].extend(kb_items)

                    # Also update mapping and inference list for usecases found anywhere
                    for item in kb_items:
                        if isinstance(item, dict):
                            i_id = item.get("instruction_id")
                            u_name = item.get("usecase_name")
                            if i_id and u_name:
                                instruction_id_map[i_id] = u_name
                            if u_name and item not in collected_usecases_list:
                                collected_usecases_list.append(item)

                # Determine Category from get_usecase_instruction
                if tool_name == "get_usecase_instruction":
                    try:
                        args_dict = requested_args if isinstance(requested_args, dict) else json.loads(requested_args_str)
                        i_id = args_dict.get("instruction_id")
                        if i_id and i_id in instruction_id_map:
                            extracted_data["category"] = instruction_id_map[i_id]
                    except:
                        pass

                # Extract prompt version from tool outputs
                def find_version(obj):
                    if isinstance(obj, dict):
                        if "instruction_version" in obj:
                            return obj["instruction_version"]
                        for k, v in obj.items():
                            res = find_version(v)
                            if res: return res
                    elif isinstance(obj, list):
                        for item in obj:
                            res = find_version(item)
                            if res: return res
                    return None

                ver = find_version(resp_json)
                if ver and extracted_data.get("prompt_version") is None:
                    extracted_data["prompt_version"] = str(ver)

                # Check for tool errors
                if tool_status.lower() in ["failure", "error"]:
                    extracted_data["tool_failures"].append({
                        "tool": tool_name,
                        "error": tool_output
                    })

                # ID Fallback from tools
                if isinstance(requested_args, dict):
                    if "domain" in requested_args: update_ids({"domain_id": requested_args["domain"]})
                    if "issue_pid" in requested_args: update_ids({"issue_id": str(requested_args["issue_pid"])})
                    elif "issue_id" in requested_args: update_ids({"issue_id": str(requested_args["issue_id"])})
                    if "profile_id" in requested_args: update_ids({"profile_id": requested_args["profile_id"]})
                    if "app_id" in requested_args: update_ids({"app_id": requested_args["app_id"]})
                    if "sandbox_id" in requested_args: update_ids({"sandbox_id": requested_args["sandbox_id"]})

                # Generic extraction: guard_analysis from tool args
                if isinstance(requested_args, dict) and "guard_results" in requested_args:
                    extracted_data["guard_analysis"] = requested_args["guard_results"]
                
                # Generic extraction: action_type from tool request
                if isinstance(requested_args, dict):
                    if "action_type" in requested_args:
                        extracted_data["action_type"] = requested_args["action_type"]
                    elif "request" in requested_args:
                        # Try to parse request string for action_type
                        req_str = requested_args["request"]
                        if isinstance(req_str, str):
                            try:
                                req_obj = json.loads(req_str)
                                if isinstance(req_obj, dict) and "action_type" in req_obj:
                                    extracted_data["action_type"] = req_obj["action_type"]
                            except:
                                # Check if it's a simple action string like "ESCALATE_ISSUE"
                                if req_str in ["ESCALATE_ISSUE", "RESOLVE_ISSUE", "REJECT_ISSUE"]:
                                    extracted_data["action_type"] = req_str
                
                # Generic extraction: conversation_context from tool outputs
                if isinstance(tool_output, dict):
                    ctx_fields = ["previous_messages_count", "guardrail_breach_count", "has_chat_history", 
                                  "is_followup_to_question", "leniency_level", "regex_risk_level"]
                    ctx = {k: tool_output.get(k) for k in ctx_fields if k in tool_output}
                    if ctx and not extracted_data["conversation_context"]:
                        extracted_data["conversation_context"] = ctx
                
                # Generic extraction: escalation_info from tool outputs
                if isinstance(tool_output, dict):
                    structured = tool_output.get("structuredContent", tool_output)
                    if isinstance(structured, dict):
                        if "escalate" in str(structured).lower() or structured.get("status") == "Failure":
                            if not extracted_data["escalation_info"]:
                                extracted_data["escalation_info"] = {
                                    "tool": tool_name,
                                    "status": structured.get("status"),
                                    "message": structured.get("message")
                                }

                # Increment tool count
                extracted_data["tool_count"] += 1

            # LLM Calls
            if "call_llm" in sub_name or sub_labels.get("gen_ai.operation.name") == "generate_content":
                extracted_data["llm_call_count"] += 1
                i_tokens = int(sub_labels.get("gen_ai.usage.input_tokens", 0))
                o_tokens = int(sub_labels.get("gen_ai.usage.output_tokens", 0))
                extracted_data["llm_usage"]["input_tokens"] += i_tokens
                extracted_data["llm_usage"]["output_tokens"] += o_tokens
                extracted_data["llm_usage"]["total_tokens"] += (i_tokens + o_tokens)

                llm_req_str = sub_labels.get("gcp.vertex.agent.llm_request")
                llm_res_str = sub_labels.get("gcp.vertex.agent.llm_response")
                
                if llm_req_str:
                    llm_req = parse_llm_request(llm_req_str)
                    
                    # 1. System Instruction -> IDs
                    if "config" in llm_req and "system_instruction" in llm_req["config"]:
                        instr = llm_req["config"]["system_instruction"]
                        ids = extract_ids_from_instruction(instr)
                        update_ids(ids)
                        
                    # 2. User Message (Input)
                    # We capture the user message from the FIRST LLM call of this agent
                    if not first_llm_request_found and "contents" in llm_req:
                        # Find last user content in this request
                        for content in reversed(llm_req["contents"]):
                            if content.get("role") == "user":
                                parts = content.get("parts", [])
                                text_parts = [p.get("text") for p in parts if "text" in p]
                                if text_parts:
                                    msg = " ".join(text_parts)
                                    if not msg.startswith("For context:"):
                                        extracted_data["user_message"] = try_parse_json(msg)
                                        # Detect if this is a system/internal call
                                        if is_system_message(msg):
                                            extracted_data["is_system_message"] = True
                                        
                                        first_llm_request_found = True
                                        break
                
                if llm_res_str:
                    llm_res = parse_llm_response(llm_res_str)
                    if "content" in llm_res:
                        parts = llm_res["content"].get("parts", [])
                        text_parts = [p.get("text") for p in parts if "text" in p]
                        if text_parts:
                            # We keep overwriting this, so we get the LAST response
                            agent_resp_msg = " ".join(text_parts)
                            extracted_data["agent_response"] = try_parse_json(agent_resp_msg)

        # Derive final_status from available data
        if extracted_data["agent_response"]:
            extracted_data["final_status"] = "responded"
        elif extracted_data["escalation_info"] or extracted_data["action_type"] == "ESCALATE_ISSUE":
            extracted_data["final_status"] = "escalated"
        elif extracted_data["tool_count"] > 0:
            extracted_data["final_status"] = "processing"
        else:
            extracted_data["final_status"] = "unknown"

        # Infer category dynamically if not explicitly set
        if not extracted_data["category"]:
            # Even for system messages, we try to infer from action_type first
            inferred = infer_category_dynamic(
                extracted_data["user_message"],
                trace_level_usecases + collected_usecases_list,
                extracted_data["action_type"]
            )
            
            if inferred:
                extracted_data["inferred_category"] = inferred
            elif extracted_data.get("is_system_message"):
                # If it's a system message and nothing matched, label as internal
                extracted_data["inferred_category"] = "internal_agent_call"
        else:
            extracted_data["inferred_category"] = extracted_data["category"]
        
        # UNIVERSAL KB DEDUPLICATION (Phase 2.2)
        extracted_data["knowledge_base"] = clean_and_deduplicate_kb(extracted_data["knowledge_base"])
        if not extracted_data["knowledge_base"]:
            extracted_data["knowledge_base"] = None

        # Extract rag_context from knowledge_base (just faq_body strings)
        if extracted_data["knowledge_base"]:
            for item in extracted_data["knowledge_base"]:
                if isinstance(item, dict) and "faq_body" in item:
                    extracted_data["rag_context"].append(item["faq_body"])
        extracted_data["has_rag_context"] = len(extracted_data["rag_context"]) > 0

        agent_executions.append(extracted_data)

    return agent_executions

def map_to_evaluation_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map internal extraction format to the strict EvaluationInput schema.
    This ensures exact match with the production evaluator's expectations.
    """
    # Allowed enums for strict validation match
    VALID_BENCHMARK_CATEGORIES = ["math", "reasoning", "tool_use", "multi_turn", "safety", "general", "code", "retrieval"]
    VALID_TOOL_STATUSES = ["success", "failure", "timeout", "skipped", "pending"]

    # 1. Flatten IDs
    ids = data.get("ids", {})
    
    # 2. Extract response stats
    response = data.get("agent_response") or ""
    # Ensure response is a string for length checks
    if not isinstance(response, str):
        response_str = json.dumps(response)
    else:
        response_str = response
        
    response_length = len(response_str)
    response_word_count = len(response_str.split())
    
    # 3. Format Knowledge Base (Object, not List)
    kb_list = data.get("knowledge_base")
    kb_obj = None
    if kb_list and isinstance(kb_list, list) and len(kb_list) > 0:
        # Aggregate content from all items
        contents = []
        sources = set()
        for item in kb_list:
            if not isinstance(item, dict):
                contents.append(str(item))
                continue
            
            content = item.get("goal") or item.get("faq_body") or item.get("content") or item.get("instruction")
            if content:
                contents.append(str(content))
            
            source = item.get("usecase_name") or item.get("title") or item.get("faq_slug") or item.get("source")
            if source:
                sources.add(str(source))
        
        if contents:
            kb_obj = {
                "content": "\n---\n".join(contents),
                "source": ", ".join(sorted(list(sources))) if sources else "extracted_context",
                "relevance_score": 1.0,
                "chunk_ids": [],
                "retrieval_method": "vector_similarity" if data.get("has_rag_context") else "instruction"
            }
    
    # 4. Construct tool_info with latency/retry
    tool_info = []
    for tool in data.get("tool_info", []):
        raw_status = str(tool.get("status") or "success").lower()
        status = raw_status if raw_status in VALID_TOOL_STATUSES else "success"

        tool_info.append({
            "tool": str(tool.get("tool")),
            "tool_id": str(tool.get("tool_id") or f"tool_{uuid.uuid4().hex[:8]}"),
            "requested_args": tool.get("requested_args", {}),
            "tool_output": tool.get("tool_output"),
            "status": status,
            "latency_ms": float(tool.get("latency_ms") or 1.0),
            "retry_count": int(tool.get("retry_count") or 0)
        })

    # 5. Build conversation history
    history = []
    if data.get("user_message"):
        history.append({
            "role": "user",
            "content": str(data["user_message"]),
            "timestamp": data.get("timestamp"),
            "tool_calls": []
        })
    if data.get("agent_response"):
        history.append({
            "role": "assistant",
            "content": str(data["agent_response"]),
            "timestamp": None,
            "tool_calls": tool_info
        })

    # 6. Normalize Category
    category = str(data.get("inferred_category") or "general").lower()
    # Map common sub-categories to standard evaluator categories
    cat_map = {
        "gameplay_questions": "retrieval",
        "technical_issues": "general",
        "purchase_or_payment_issues": "general",
        "profile_or_account_issues": "general",
        "missing_rewards": "general",
        "greetings_or_chitchat": "general",
        "rewards_and_awards": "general",
        "account_and_profile": "general",
        "billing_and_purchase": "general"
    }
    category = cat_map.get(category, category)
    
    # Auto-switch to retrieval if knowledge base is present
    if kb_obj and category == "general":
        category = "retrieval"
    if category not in VALID_BENCHMARK_CATEGORIES:
        category = "general"

    # 7. Normalize tool_failures and model_failures
    normalized_tool_failures = []
    for tf in data.get("tool_failures", []):
        if isinstance(tf, dict):
            # Smart error extraction from nested objects often found in agent traces
            error_obj = tf.get("error") or tf or {}
            msg = "Tool execution failed"
            if isinstance(error_obj, dict):
                struct = error_obj.get("structuredContent")
                if isinstance(struct, dict):
                    msg = struct.get("message") or struct.get("status") or msg
                else:
                    msg = error_obj.get("message") or error_obj.get("error_message") or error_obj.get("status") or msg

            normalized_tool_failures.append({
                "tool_name": str(tf.get("tool_name") or tf.get("tool") or "unknown_tool"),
                "error_type": str(tf.get("error_type") or "execution_error"),
                "error_message": str(msg),
                "timestamp": tf.get("timestamp") or data.get("timestamp"),
                "retry_count": int(tf.get("retry_count") or 0),
                "args": tf.get("args") or tf.get("requested_args") or {},
                "stack_trace": tf.get("stack_trace")
            })

    normalized_model_failures = []
    for mf in data.get("model_failures", []):
        if isinstance(mf, dict):
            # Handle both internal extraction format and potential raw gcp labels
            msg = mf.get("error_message") or mf.get("message") or mf.get("status") or "Model call failed"
            normalized_model_failures.append({
                "model_name": str(mf.get("model_name") or mf.get("model") or "unknown_model"),
                "error_type": str(mf.get("error_type") or "model_error"),
                "error_message": str(msg),
                "timestamp": mf.get("timestamp") or data.get("timestamp"),
                "retry_count": int(mf.get("retry_count") or 0),
                "prompt_preview": mf.get("prompt_preview"),
                "http_status_code": mf.get("http_status_code")
            })

    # 8. Final Mapping
    mapped = {
        "domain": str(ids.get("domain_id") or "general"),
        "issue_id": str(ids.get("issue_id") or f"ISSUE-{uuid.uuid4().hex[:8].upper()}"),
        "message_id": str(data.get("message_id") or f"msg_{uuid.uuid4().hex[:8]}"),
        "session_id": str(data.get("session_id") or f"session_{uuid.uuid4().hex[:8]}"),
        "invocation_id": str(data.get("invocation_id") or f"inv_{uuid.uuid4().hex[:8]}"),
        "timestamp": data.get("timestamp"),
        "user_message": str(data.get("user_message") or ""),
        "agent_name": str(data.get("agent_name") or "unknown_agent"),
        "agent_model": "gemini-2.5-flash", 
        "agent_type": "llm_agent",
        "agent_version": "1.0.0",
        "expected_tools": [],
        "expected_response": None,
        "benchmark_category": category,
        "tool_info": tool_info,
        "knowledge_base": kb_obj,
        "prompt_version": str(data.get("prompt_version") or "v1.0"),
        "llm_token_usage": data.get("llm_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        "response": response_str,
        "usage_metadata": data.get("llm_usage", {}),
        "latency_ms": float(data.get("response_latency_ms") or 0.0),
        "time_to_first_token_ms": None,
        "conversation_history": history,
        "session_state_before": {},
        "session_state_after": {},
        "guardrail_checks": [],
        "tool_failures": normalized_tool_failures,
        "model_failures": normalized_model_failures,
        "response_length": response_length,
        "response_word_count": response_word_count,
        "thinking_steps": []
    }
    
    return mapped
# Check if the bucket exists or not, if not create it
def check_bucket_exists(bucket_name: str) -> bool:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket.exists()

def create_bucket(bucket_name: str):
    client = storage.Client()
    bucket = client.create_bucket(bucket_name)
    return bucket

# Code to write all data into GCS bucket
def write_to_gcs(data: List[Dict], bucket_name: str, file_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1200, help="Number of traces to fetch")
    parser.add_argument("--days", type=int, default=30, help="Filter traces from the last N days")
    parser.add_argument("--agent-name", type=str, help="Specific agent name to filter for (e.g. guard_agent)")
    args = parser.parse_args()
    
    print(f"Fetching {args.limit} traces...")
    traces = list_traces(PROJECT_ID, limit=args.limit, days=args.days)
    print(f"Processing {len(traces)} traces...")
    
    all_agent_executions = []
    
    for trace in traces:
        try:
            executions = process_trace(trace, target_agent=args.agent_name)
            for ex in executions:
                if ex["user_message"] or ex["tool_info"] or ex["agent_response"]:
                     all_agent_executions.append(ex)
        except Exception as e:
            print(f"Error processing trace {trace.get('traceId')}: {e}")
            
    # GLOBAL FIX: Propagate categories and metadata across ALL extracted executions
    print("Applying global trace-level metadata and category propagation...")
    all_agent_executions = propagate_metadata_within_trace(all_agent_executions)
    all_agent_executions = propagate_categories_within_trace(all_agent_executions)
    
    # FINAL STEP: Map to EvaluationInput exact match schema
    print("Applying final EvaluationInput schema mapping...")
    final_output = [map_to_evaluation_input(ex) for ex in all_agent_executions]
    
    output_filename = "agent_traces.jsonl"
    with open(output_filename, "w") as f:
        for ex in final_output:
            f.write(json.dumps(ex) + "\n")
    #code to write everything added in function in main as code is executed
    write_to_gcs(final_output, GCS_BUCKET, output_filename)
                     
    print(f"Extraction complete. Saved {len(all_agent_executions)} agent execution records to {output_filename}")

if __name__ == "__main__":
    main()
