#!/usr/bin/env python3
"""
Run complete evaluation on agent_traces.jsonl with SESSION-LEVEL GROUPING.

Key improvement: Groups traces by session_id to properly evaluate multi-turn conversations.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation import (
    ComprehensiveEvaluator,
    EvaluationInput,
    EvaluationOutput,
    ToolCallInfo,
    ToolCallStatus,
    TokenUsage,
    KnowledgeBase,
    ToolFailure,
    generate_summary_report
)
from src.evaluation.schemas import ConversationTurn


def group_traces_by_session(traces: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group traces by session_id and sort by timestamp.
    
    Args:
        traces: List of trace dictionaries
        
    Returns:
        Dictionary mapping session_id to list of traces (sorted by timestamp)
    """
    sessions = defaultdict(list)
    
    for trace in traces:
        # Use session_id if available, otherwise use trace_id as session
        session_id = trace.get("session_id") or trace.get("trace_id") or "unknown"
        sessions[session_id].append(trace)
    
    # Sort each session's traces by timestamp
    for session_id in sessions:
        sessions[session_id].sort(key=lambda t: t.get("timestamp", ""))
    
    return dict(sessions)


def extract_tool_info_from_agent_info(agent_info: List[Dict]) -> List[ToolCallInfo]:
    """
    Extract all tool calls from agent_info structure.
    
    The agent_info contains multiple agents, each with their own tool_info.
    """
    all_tools = []
    
    for agent in agent_info:
        for tc in agent.get("tool_info", []):
            status_str = tc.get("tool_status", "success")
            try:
                status = ToolCallStatus(status_str)
            except ValueError:
                status = ToolCallStatus.SUCCESS
            
            all_tools.append(ToolCallInfo(
                tool=tc.get("tool_name", "unknown"),
                tool_id=None,
                requested_args=tc.get("tool_args", {}),
                tool_output=tc.get("tool_output"),
                status=status,
                latency_ms=None,
                retry_count=0
            ))
    
    return all_tools


def parse_session_to_evaluation_input(session_id: str, traces: List[Dict]) -> EvaluationInput:
    """
    Parse multiple traces from a session into a single EvaluationInput.
    
    This builds the conversation history from all turns and aggregates:
    - Tool calls from all turns
    - Knowledge base from all turns
    - Tool failures from all turns
    """
    if not traces:
        raise ValueError("No traces provided for session")
    
    # Build conversation history from all turns
    conversation_history = []
    all_tool_info = []
    all_tool_failures = []
    all_knowledge_base_content = []
    
    # Get metadata from first trace with values
    domain = None
    issue_id = None
    agent_name = None
    agent_model = None
    
    for trace in traces:
        if not domain and trace.get("domain"):
            domain = trace["domain"]
        if not issue_id and trace.get("issue_id"):
            issue_id = trace["issue_id"]
        if not agent_name:
            agent_name = trace.get("agent_type", "multi_agent_system")
        if not agent_model:
            agent_model = trace.get("agent_model", "gemini-2.5-flash")
    
    # Process each turn
    for trace in traces:
        # Parse timestamp
        timestamp = None
        if trace.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(trace["timestamp"].replace("Z", "+00:00"))
            except:
                timestamp = datetime.now()
        
        user_msg = trace.get("user_message", "")
        agent_resp = trace.get("agent_response", "")
        
        # Add user turn
        if user_msg:
            conversation_history.append(ConversationTurn(
                role="user",
                content=user_msg,
                timestamp=timestamp,
                tool_calls=[]
            ))
        
        # Extract tool info from agent_info structure
        agent_info = trace.get("agent_info", [])
        turn_tools = extract_tool_info_from_agent_info(agent_info)
        all_tool_info.extend(turn_tools)
        
        # Add assistant turn with tool calls
        if agent_resp or turn_tools:
            conversation_history.append(ConversationTurn(
                role="assistant",
                content=agent_resp or "",
                timestamp=timestamp,
                tool_calls=turn_tools
            ))
        
        # Collect knowledge base content
        kb_content = trace.get("knowledge_base", [])
        if kb_content:
            all_knowledge_base_content.extend(kb_content)
        
        # Collect tool failures
        for tf in trace.get("tool_failures", []):
            all_tool_failures.append(ToolFailure(
                tool_name=tf.get("tool_name", "unknown"),
                error_type=tf.get("error_type", "unknown"),
                error_message=tf.get("error_message", ""),
                timestamp=datetime.now(),
                retry_count=tf.get("retry_count", 0),
                args=tf.get("args", {}),
                stack_trace=tf.get("stack_trace")
            ))
    
    # Get final response (last agent response)
    final_response = ""
    for trace in reversed(traces):
        if trace.get("agent_response"):
            final_response = trace["agent_response"]
            break
    
    # Get last user message (the one determining the final response)
    last_user_message = ""
    for trace in reversed(traces):
        if trace.get("user_message"):
            last_user_message = trace["user_message"]
            break
    
    # Build knowledge base from aggregated content
    kb = None
    if all_knowledge_base_content:
        # Flatten knowledge base content
        kb_text_parts = []
        for kb_item in all_knowledge_base_content:
            if isinstance(kb_item, list):
                for item in kb_item:
                    if isinstance(item, dict) and "text" in item:
                        text_data = item.get("text", {})
                        if isinstance(text_data, dict):
                            kb_text_parts.append(json.dumps(text_data, default=str)[:500])
            elif isinstance(kb_item, dict) and "text" in kb_item:
                text_data = kb_item.get("text", {})
                if isinstance(text_data, dict):
                    kb_text_parts.append(json.dumps(text_data, default=str)[:500])
        
        if kb_text_parts:
            kb = KnowledgeBase(
                content="\n".join(kb_text_parts[:5]),
                source="multi_agent_system",
                relevance_score=0.85
            )
    
    # Parse timestamp from first trace
    first_timestamp = datetime.now()
    if traces[0].get("timestamp"):
        try:
            first_timestamp = datetime.fromisoformat(traces[0]["timestamp"].replace("Z", "+00:00"))
        except:
            pass
    
    return EvaluationInput(
        domain=domain or "unknown",
        issue_id=issue_id or "unknown",
        message_id=traces[-1].get("message_id") if traces else None,
        session_id=session_id,
        invocation_id=traces[0].get("trace_id"),
        timestamp=first_timestamp,
        user_message=last_user_message,
        agent_name=agent_name,
        agent_model=agent_model,
        agent_version="1.0.0",
        expected_tools=[],
        expected_response=None,
        benchmark_category=None,
        tool_info=all_tool_info,
        knowledge_base=kb,
        prompt_version="v1.0",
        llm_token_usage=TokenUsage(),
        response=final_response,
        usage_metadata={},
        latency_ms=None,
        time_to_first_token_ms=None,
        conversation_history=conversation_history,
        session_state_before={},
        session_state_after={},
        guardrail_checks=[],
        tool_failures=all_tool_failures,
        model_failures=[],
        response_length=len(final_response) if final_response else 0,
        response_word_count=len(final_response.split()) if final_response else 0,
        thinking_steps=[]
    )


def output_to_jsonl(output: EvaluationOutput) -> str:
    """Convert EvaluationOutput to JSONL line."""
    return output.model_dump_json()


async def run_evaluation(input_file: str, output_file: str):
    """
    Run complete evaluation on all traces in the input file.
    
    IMPROVED: Groups traces by session_id for proper multi-turn evaluation.
    
    Args:
        input_file: Path to agent_traces.jsonl
        output_file: Path to output evaluation results JSONL
    """
    print(f"\n{'='*70}")
    print("SESSION-LEVEL AGENT TRACE EVALUATION")
    print(f"{'='*70}\n")
    
    # Get project ID from environment
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        project_id=project_id,
        location="us-central1"
    )
    
    # Read traces
    traces = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    
    print(f"📂 Loaded {len(traces)} individual traces from {input_file}")
    
    # Group traces by session
    sessions = group_traces_by_session(traces)
    print(f"📊 Grouped into {len(sessions)} sessions\n")
    
    # Show session breakdown
    print("Session Breakdown:")
    for session_id, session_traces in list(sessions.items())[:10]:
        issue_ids = set(t.get("issue_id") for t in session_traces if t.get("issue_id"))
        print(f"  • {session_id[:20]}...: {len(session_traces)} turns, issues: {issue_ids or 'N/A'}")
    if len(sessions) > 10:
        print(f"  ... and {len(sessions) - 10} more sessions\n")
    else:
        print()
    
    # Run evaluation per session
    results: List[EvaluationOutput] = []
    
    # Initialize output file (clear it first)
    with open(output_file, 'w') as f:
        pass
        
    for i, (session_id, session_traces) in enumerate(sessions.items(), 1):
        # Skip empty sessions
        if not session_traces:
            continue
        
        # Skip sessions with no user message
        has_user_msg = any(t.get("user_message") for t in session_traces)
        if not has_user_msg:
            print(f"⏭️  [{i}/{len(sessions)}] Skipping session {session_id[:20]}... (no user message)")
            continue
        
        issue_id = next((t.get("issue_id") for t in session_traces if t.get("issue_id")), "N/A")
        turn_count = len(session_traces)
        
        print(f"⏳ [{i}/{len(sessions)}] Evaluating session {session_id[:20]}... ({turn_count} turns, issue: {issue_id})")
        
        try:
            # Parse session to input
            eval_input = parse_session_to_evaluation_input(session_id, session_traces)
            
            # Run evaluation
            eval_output = await evaluator.evaluate(
                input_data=eval_input,
                expected_trajectory=None  # Could be extracted from expected_tools
            )
            
            # Add session metadata
            eval_output.session_id = session_id
            eval_output.agent_name = eval_input.agent_name
            eval_output.agent_model = eval_input.agent_model
            eval_output.turn_count = turn_count
            eval_output.count_user_msg = len([t for t in session_traces if t.get("user_message")])
            eval_output.count_assistant_msg = len([t for t in session_traces if t.get("agent_response")])
            
            results.append(eval_output)
            
            verdict_emoji = "✅" if eval_output.verdict.value == "PASS" else ("⚠️" if eval_output.verdict.value == "PARTIAL" else "❌")
            print(f"   {verdict_emoji} Verdict: {eval_output.verdict.value} | Safety: {eval_output.safety_score.score:.2f} | Actions: {eval_output.action_completion_score.score:.2f}")
            
            # Write result immediately
            with open(output_file, 'a') as f:
                f.write(output_to_jsonl(eval_output) + "\n")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Write summary at the end
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"📄 Results written to: {output_file}")
    print(f"📊 Sessions evaluated: {len(results)} (from {len(traces)} traces)")
    
    # Generate summary
    if results:
        summary = generate_summary_report(results)
        print(f"\n📈 Summary:")
        print(f"   Pass Rate: {summary['pass_rate']*100:.1f}%")
        print(f"   Passed: {summary['passed']}, Partial: {summary['partial']}, Failed: {summary['failed']}")
        print(f"   Avg Safety: {summary['average_metrics']['safety_score']:.2f}")
        print(f"   Avg Hallucination: {summary['average_metrics']['hallucination_score']:.2f}")
        print(f"   Avg Action Completion: {summary['average_metrics']['action_completion_score']:.2f}")
        print(f"   Total Judge Tokens: {summary['total_judge_tokens']:,}")
        print(f"   Total Judge Cost: ${summary['total_judge_cost']:.4f}")
        
        # Write summary
        summary_file = output_file.replace(".jsonl", "_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n📋 Summary written to: {summary_file}")
    
    return results


async def main():
    # Use the scripts directory trace file (user has it open)
    input_file = "/Users/rinit.lulla/Documents/GitHub/agent-evaluation/scripts/agent_traces.jsonl"
    output_file = "/Users/rinit.lulla/Documents/GitHub/agent-evaluation/session_evaluation_results.jsonl"
    
    # Fallback to root directory if scripts version doesn't exist
    if not os.path.exists(input_file):
        input_file = "/Users/rinit.lulla/Documents/GitHub/agent-evaluation/agent_traces.jsonl"
    
    await run_evaluation(input_file, output_file)


if __name__ == "__main__":
    asyncio.run(main())
