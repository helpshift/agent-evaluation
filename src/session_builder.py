import json
import statistics
import pandas as pd
import datetime as dt
from typing import Any, Dict, List, Optional
from .utils import parse_timestamp, as_float_seconds, percentile

def group_traces_to_sessions(trace_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in trace_rows:
        issue_id = r.get("issue_id")
        session_id = r.get("session_id")
        trace_id = r.get("trace_id")
        key = (
            str(issue_id)
            if issue_id not in [None, "", "null"]
            else (str(session_id) if session_id else str(trace_id))
        )
        grouped.setdefault(key, []).append(r)

    for k, rows in grouped.items():
        rows.sort(key=lambda x: (parse_timestamp(x.get("timestamp")) or dt.datetime.min))
    print("Grouped Traces as sessions:-----------------\n", grouped, "\n\n")
    return grouped

def build_multiturn_conversation(session_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    convo: List[Dict[str, str]] = []
    for r in session_rows:
        um = r.get("user_message") or r.get("prompt") or r.get("user_query")
        ar = r.get("agent_response") or r.get("response")
        if isinstance(um, str) and um.strip():
            convo.append({"role": "user", "content": um.strip()})
        if isinstance(ar, str) and ar.strip():
            convo.append({"role": "assistant", "content": ar.strip()})
    return convo

def conversation_to_transcript(convo: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    turn = 1
    i = 0
    while i < len(convo):
        if convo[i].get("role") == "user":
            lines.append(f"Turn {turn} - USER:\n{convo[i].get('content','')}\n")
            if i + 1 < len(convo) and convo[i + 1].get("role") == "assistant":
                lines.append(f"Turn {turn} - ASSISTANT:\n{convo[i+1].get('content','')}\n")
                i += 2
            else:
                i += 1
            turn += 1
        else:
            lines.append(f"ASSISTANT:\n{convo[i].get('content','')}\n")
            i += 1
    return "\n".join(lines).strip()

def extract_tool_events(session_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for r in session_rows:
        agent_info = r.get("agent_info") or []
        if isinstance(agent_info, list):
            for a in agent_info:
                if isinstance(a, dict):
                    events.extend(a.get("tool_info") or [])
        events.extend(r.get("tool_failures") or [])
        events.extend(r.get("model_failures") or [])
    return [e for e in events if isinstance(e, dict)]

def compute_session_metrics(session_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    convo = build_multiturn_conversation(session_rows)
    count_user = sum(1 for m in convo if m["role"] == "user")
    count_asst = sum(1 for m in convo if m["role"] == "assistant")
    latencies: List[float] = []
    ttfr: Optional[float] = None
    first_user_ts: Optional[dt.datetime] = None
    first_asst_ts: Optional[dt.datetime] = None

    tool_calls = 0
    tool_errors = 0
    model_calls = 0
    model_errors = 0

    for r in session_rows:
        ts = parse_timestamp(r.get("timestamp"))
        
        # Tool/Model info
        agent_info = r.get("agent_info") or []
        if isinstance(agent_info, list):
            model_calls += len(agent_info)
            for a in agent_info:
                if isinstance(a, dict):
                    t_info = a.get("tool_info") or []
                    tool_calls += len(t_info)
                    for t in t_info:
                        st = str(t.get("tool_status", "")).lower()
                        if st and st not in ["success", "ok", "succeeded"]:
                            tool_errors += 1
        
        tool_errors += len(r.get("tool_failures") or [])
        model_errors += len(r.get("model_failures") or [])

        # Latency
        for key in ["latency_sec", "latency_s", "latency_seconds", "latency_ms", "total_latency_ms"]:
            val = as_float_seconds(r.get(key))
            if val is not None:
                latencies.append(val)
        
        # TTFR calculation
        um = r.get("user_message") or r.get("prompt") or r.get("user_query")
        ar = r.get("agent_response") or r.get("response")
        if um and first_user_ts is None:
            first_user_ts = ts
        if ar and first_asst_ts is None and first_user_ts is not None:
            first_asst_ts = ts
            if first_user_ts and first_asst_ts:
                ttfr = (first_asst_ts - first_user_ts).total_seconds()

    import statistics
    latency_stats = {
        "avg": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "max": round(max(latencies), 3) if latencies else 0.0,
        "min": round(min(latencies), 3) if latencies else 0.0,
        "median": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p90": round(percentile(latencies, 90.0), 3) if latencies else 0.0,
    }

    return {
        "turn_count": len(convo) // 2,
        "count_user_msg": count_user,
        "count_assistant_msg": count_asst,
        "count_tool_call": tool_calls,
        "count_tool_error": tool_errors,
        "count_model_call": model_calls,
        "count_model_error": model_errors,
        "latency_stats": latency_stats,
        "ttfr": ttfr or 0.0
    }

def extract_tool_events(session_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for r in session_rows:
        agent_info = r.get("agent_info") or []
        if isinstance(agent_info, list):
            for a in agent_info:
                if isinstance(a, dict):
                    events.extend(a.get("tool_info") or [])
        events.extend(r.get("tool_failures") or [])
        events.extend(r.get("model_failures") or [])
    return [e for e in events if isinstance(e, dict)]

def extract_kb(session_rows: List[Dict[str, Any]]) -> List[Any]:
    kb_merged: List[Any] = []
    for r in session_rows:
        kb = r.get("knowledge_base") or r.get("context") or []
        if isinstance(kb, list):
            kb_merged.extend(kb)
        elif isinstance(kb, str) and kb.strip():
            kb_merged.append(kb.strip())
    return kb_merged

def build_eval_dataframe(
    sessions_map: Dict[str, List[Dict[str, Any]]],
    agent_instruction_text: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for session_key, session_rows in sessions_map.items():
        convo = build_multiturn_conversation(session_rows)
        transcript = conversation_to_transcript(convo)
        
        s_m = compute_session_metrics(session_rows)

        latest_prompt = ""
        for m in reversed(convo):
            if m.get("role") == "user" and (m.get("content") or "").strip():
                latest_prompt = m["content"].strip()
                break
        
        final_resp = ""
        for m in reversed(convo):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                final_resp = m["content"].strip()
                break

        kb = extract_kb(session_rows)
        events = extract_tool_events(session_rows)

        issue_id = next((r.get("issue_id") for r in session_rows if r.get("issue_id") not in [None, ""]), None)
        domain = next((r.get("domain") for r in session_rows if r.get("domain")), None)
        agent_v = next((r.get("agent_version") or r.get("version") for r in session_rows if r.get("agent_version") or r.get("version")), "unknown")
        first_ts = parse_timestamp(session_rows[0].get("timestamp")) if session_rows else None

        rows.append(
            {
                "issue_id": str(issue_id or session_key),
                "domain": domain or "unknown",
                "timestamp_start": first_ts.isoformat() if first_ts else None,
                "agent_version": agent_v,
                "history": transcript,
                "prompt": latest_prompt,
                "response": final_resp,
                "context": json.dumps(kb, ensure_ascii=False) if kb else "",
                "instructions": agent_instruction_text or "",
                "intermediate_events": json.dumps(events, ensure_ascii=False) if events else "",
                "reference": "",
                "turn_count": s_m["turn_count"],
                "count_user_msg": s_m["count_user_msg"],
                "count_assistant_msg": s_m["count_assistant_msg"],
                "count_tool_call": s_m["count_tool_call"],
                "count_tool_error": s_m["count_tool_error"],
                "count_model_call": s_m["count_model_call"],
                "count_model_error": s_m["count_model_error"],
                "latency_json": json.dumps(s_m["latency_stats"]),
                "ttfr": s_m["ttfr"]
            }
        )

    return pd.DataFrame(rows)
