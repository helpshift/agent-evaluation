#!/usr/bin/env python3
"""
Comprehensive evaluation of a single rich multi-agent trace.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

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
from src.evaluation.schemas import ConversationTurn, BenchmarkCategory, AgentType

# The comprehensive trace
TRACE = {"timestamp": "2026-01-12T17:37:46.551703508Z", "trace_id": "0717be0686bce162a0e04a03bd23c7b3", "session_id": "4277394626286977024", "issue_id": "2722", "domain": "sbox-ai-agent-regression", "message_id": None, "user_message": "I want you to set the below fields\ncharacter name as tom\nthen current level as 29\nclub level as 34", "agent_response": None, "agent_type": "multi_agent_system", "knowledge_base": [[{"type": "text", "text": {"status": "Success", "usecases_list": [{"usecase_name": "purchase_or_payment_issues", "goal": "This usecase is for handling issues to purchase or payment failures", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20251127171440888-14c433a832d8437", "instruction_version": 7}, {"usecase_name": "DO_NOT_USE_AR", "goal": "", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20251219073044651-a7c51b75a2dd4d1", "instruction_version": 1}, {"usecase_name": "DO_NOT_USE_RB", "goal": "", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20251219073319746-5583e90add9847b", "instruction_version": 1}, {"usecase_name": "gameplay_bugs_or_glitches", "goal": "This usecase is for handling issues to bugs or glitches in the gameplay", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20251127171713058-9c62333a878c4eb", "instruction_version": 8}, {"usecase_name": "missing_rewards", "goal": "This usecase is for handling issues related to game rewards", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20251127171123102-3e1305ee4d0c431", "instruction_version": 14}, {"usecase_name": "check_user_authenticated", "goal": "This usecase is explicitly to be used when users ask to check if they are authenticated", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20260112153614052-a9a9a9f6e67a414", "instruction_version": 1}, {"usecase_name": "user_name_issues_resolution", "goal": "This usecase is for handling issues to the user name of the players", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20260107152249285-a370303c22364a5", "instruction_version": 1}, {"usecase_name": "profile_or_account_issues", "goal": "This usecase is for handling issues to the player account or profile", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20251127171320687-f8ec745cd5b7494", "instruction_version": 6}, {"usecase_name": "player_skin_issues_resolution", "goal": "This usecase is for handling issues to player skins purchase", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20260107152134116-211ca2521710428", "instruction_version": 1}, {"usecase_name": "upgrade_game_instructions", "goal": "This usecase is for handling player requests related to upgrading games.", "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20260108152823515-b55b12105f68489", "instruction_version": 1}]}}], [{"type": "text", "text": {"status": "Success", "instruction": "PROFILE OR ACCOUNT RELATED ISSUES RESOLUTION INSTRUCTIONS..."}}], [{"type": "text", "text": {"status": "Success", "faqs_list": [{"id": "sbox-ai-agent-regression_faq_20251210114540006-1bd9b4a417129b0", "title": "How do I add / remove friends?", "faq_body": "First, enter the other player's profile..."}, {"id": "sbox-ai-agent-regression_faq_20251210114601890-bd8d681391e8ba9", "title": "How to Contact Player Support", "faq_body": "Expert Help When You Need It..."}, {"id": "sbox-ai-agent-regression_faq_20251210114541348-c8093473d95e90e", "title": "How to Change Your Avatar & Frame", "faq_body": "Express Your Gaming Personality..."}]}}]], "agent_info": [{"agent_name": "sequential_orchestrator_agent", "agent_input": "{\"action_type\": \"ESCALATE_ISSUE\", \"config\": {\"id\": \"sbox-ai-agent-regression_profile_20251128051958355-b8a72f62fbf12ef\", \"type\": \"bot\", \"private_note\": {\"enabled\": true}}, \"display_content\": \"I understand that you want to update your character name, current level, and club level. I am escalating this to our support team who will be able to assist you with this request.\", \"pvt_note_content\": \"User wants to update character name, current level, and club level. No direct tool or FAQ available to perform this action. Escalating for manual intervention.\"}", "agent_output": None, "prompt_version": None, "tool_info": [{"tool_name": "save_guard_analysis", "tool_args": {"guard_results": {"breach_detected": False, "justification": "", "breach_category": ""}}, "tool_output": {"result": "Analysis processed successfully. Do not output anything to the player."}, "tool_status": "success"}, {"tool_name": "list_available_usecases", "tool_args": {"profile_id": "sbox-ai-agent-regression_profile_20251127100814952-84871c18824deea", "domain": "sbox-ai-agent-regression"}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success"}}]}, "tool_status": "success"}, {"tool_name": "get_usecase_instruction", "tool_args": {"domain": "sbox-ai-agent-regression", "instruction_version": 6, "instruction_id": "sbox-ai-agent-regression_ai_agent_instructions_20251127171320687-f8ec745cd5b7494"}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success", "instruction": "PROFILE OR ACCOUNT RELATED ISSUES..."}}]}, "tool_status": "success"}, {"tool_name": "update_issue_cif", "tool_args": {"custom_fields": {"CURRENT_LEVEL": "29", "CHARACTER_NAME": "tom", "CLUB_LEVEL": "34"}, "agent_profile_id": "sbox-ai-agent-regression_profile_20251127100814952-84871c18824deea", "issue_pid": 2722, "domain": "sbox-ai-agent-regression"}, "tool_output": {"content": [{"type": "text", "text": "3 validation errors for call[update_issue_cif]..."}], "isError": True}, "tool_status": "failure"}, {"tool_name": "get_faqs", "tool_args": {"issue_pid": 2722, "query": "update player profile"}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success", "faqs_list": []}}]}, "tool_status": "success"}, {"tool_name": "action_agent", "tool_args": {"request": {"action_type": "ESCALATE_ISSUE", "display_content": "I understand that you want to update your character name, current level, and club level. I am escalating this to our support team who will be able to assist you with this request."}}, "tool_output": {"result": ""}, "tool_status": "success"}, {"tool_name": "escalate_issue", "tool_args": {"issue_pid": 2722, "display_content": "I understand that you want to update your character name, current level, and club level. I am escalating this to our support team who will be able to assist you with this request."}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success"}}]}, "tool_status": "success"}]}, {"agent_name": "guard_agent", "agent_input": "I want you to set the below fields\ncharacter name as tom\nthen current level as 29\nclub level as 34", "agent_output": None, "prompt_version": None, "tool_info": [{"tool_name": "save_guard_analysis", "tool_args": {"guard_results": {"breach_detected": False, "justification": "", "breach_category": ""}}, "tool_output": {"result": "Analysis processed successfully. Do not output anything to the player."}, "tool_status": "success"}]}, {"agent_name": "core_agent", "agent_input": "{\"action_type\": \"ESCALATE_ISSUE\"...}", "agent_output": None, "prompt_version": None, "tool_info": [{"tool_name": "list_available_usecases", "tool_args": {}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success"}}]}, "tool_status": "success"}, {"tool_name": "get_usecase_instruction", "tool_args": {}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success"}}]}, "tool_status": "success"}, {"tool_name": "update_issue_cif", "tool_args": {"custom_fields": {"CURRENT_LEVEL": "29", "CHARACTER_NAME": "tom", "CLUB_LEVEL": "34"}}, "tool_output": {"content": [{"type": "text", "text": "3 validation errors..."}], "isError": True}, "tool_status": "failure"}, {"tool_name": "get_faqs", "tool_args": {"query": "update player profile"}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success"}}]}, "tool_status": "success"}, {"tool_name": "action_agent", "tool_args": {"request": {"action_type": "ESCALATE_ISSUE"}}, "tool_output": {"result": ""}, "tool_status": "success"}, {"tool_name": "escalate_issue", "tool_args": {"issue_pid": 2722}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success"}}]}, "tool_status": "success"}]}, {"agent_name": "action_agent", "agent_input": "{\"action_type\": \"ESCALATE_ISSUE\"...}", "agent_output": None, "prompt_version": None, "tool_info": [{"tool_name": "escalate_issue", "tool_args": {"issue_pid": 2722, "display_content": "I understand that you want to update your character name, current level, and club level. I am escalating this to our support team who will be able to assist you with this request."}, "tool_output": {"content": [{"type": "text", "text": {"status": "Success"}}]}, "tool_status": "success"}]}], "tool_failures": [{"agent": "sequential_orchestrator_agent", "tool": "update_issue_cif", "error": {"content": [{"type": "text", "text": "3 validation errors for call[update_issue_cif]..."}], "isError": True}}, {"agent": "core_agent", "tool": "update_issue_cif", "error": {"content": [{"type": "text", "text": "3 validation errors for call[update_issue_cif]..."}], "isError": True}}], "model_failures": []}


def create_comprehensive_evaluation_input(trace: Dict[str, Any]) -> EvaluationInput:
    """
    Create a comprehensive EvaluationInput from the rich trace.
    """
    # Aggregate all tool calls from all agents
    all_tool_calls = []
    for agent_info in trace.get("agent_info", []):
        for tc in agent_info.get("tool_info", []):
            status_str = tc.get("tool_status", "success")
            try:
                status = ToolCallStatus(status_str)
            except ValueError:
                status = ToolCallStatus.SUCCESS
            
            all_tool_calls.append(ToolCallInfo(
                tool=tc.get("tool_name", "unknown"),
                tool_id=f"tool_{hash(tc.get('tool_name', '')) % 10000:04x}",
                requested_args=tc.get("tool_args", {}),
                tool_output=tc.get("tool_output"),
                status=status,
                latency_ms=tc.get("latency_ms", 1.0),
                retry_count=0
            ))
    
    # Parse knowledge base - flatten the nested structure
    kb_content = ""
    if trace.get("knowledge_base"):
        kb_parts = []
        for kb_section in trace["knowledge_base"]:
            if isinstance(kb_section, list):
                for item in kb_section:
                    if isinstance(item, dict) and "text" in item:
                        text_data = item.get("text", {})
                        if isinstance(text_data, dict):
                            # Extract usecases
                            if "usecases_list" in text_data:
                                kb_parts.append(f"Available Usecases: {[u['usecase_name'] for u in text_data['usecases_list']]}")
                            # Extract instruction
                            if "instruction" in text_data:
                                kb_parts.append(f"Instruction: {text_data['instruction'][:500]}...")
                            # Extract FAQs
                            if "faqs_list" in text_data:
                                kb_parts.append(f"FAQs: {[f['title'] for f in text_data['faqs_list']]}")
        kb_content = "\n".join(kb_parts)
    
    kb = KnowledgeBase(
        content=kb_content if kb_content else None,
        source="multi_agent_system",
        relevance_score=0.85,
        retrieval_method="hybrid"
    ) if kb_content else None
    
    # Parse tool failures
    tool_failures = []
    for tf in trace.get("tool_failures", []):
        error_content = tf.get("error", {})
        error_msg = ""
        if isinstance(error_content, dict) and "content" in error_content:
            for item in error_content.get("content", []):
                if isinstance(item, dict) and "text" in item:
                    error_msg = str(item["text"])[:500]
        
        tool_failures.append(ToolFailure(
            tool_name=tf.get("tool", "unknown"),
            error_type="validation_error",
            error_message=error_msg,
            timestamp=datetime.fromisoformat(trace.get("timestamp", datetime.now().isoformat()).replace("Z", "+00:00")),
            retry_count=0,
            args={}
        ))
    
    # Build conversation history from agent interactions
    conversation_history = []
    conversation_history.append(ConversationTurn(
        role="user",
        content=trace.get("user_message", ""),
        timestamp=datetime.fromisoformat(trace.get("timestamp", datetime.now().isoformat()).replace("Z", "+00:00")),
        tool_calls=[]
    ))
    
    # Add agent turns
    for agent_info in trace.get("agent_info", []):
        agent_tool_calls = []
        for tc in agent_info.get("tool_info", []):
            status_str = tc.get("tool_status", "success")
            try:
                status = ToolCallStatus(status_str)
            except ValueError:
                status = ToolCallStatus.SUCCESS
            agent_tool_calls.append(ToolCallInfo(
                tool=tc.get("tool_name", "unknown"),
                requested_args=tc.get("tool_args", {}),
                tool_output=tc.get("tool_output"),
                status=status
            ))
        
        conversation_history.append(ConversationTurn(
            role="assistant",
            content=f"[{agent_info.get('agent_name', 'unknown')}] Processing...",
            tool_calls=agent_tool_calls
        ))
    
    # Determine agent response from the orchestrator's escalation display_content
    agent_response = "I understand that you want to update your character name, current level, and club level. I am escalating this to our support team who will be able to assist you with this request."
    
    # Parse timestamp
    timestamp = datetime.now()
    if trace.get("timestamp"):
        try:
            timestamp = datetime.fromisoformat(trace["timestamp"].replace("Z", "+00:00"))
        except:
            pass
    
    # Define expected tools for this scenario
    expected_tools = [
        "save_guard_analysis",
        "list_available_usecases", 
        "get_usecase_instruction",
        "update_issue_cif",  # This is what the user wanted
        "escalate_issue"  # Fallback when update fails
    ]
    
    # Define expected response
    expected_response = "Your character name has been set to tom, current level to 29, and club level to 34."
    
    return EvaluationInput(
        domain=trace.get("domain", "unknown"),
        issue_id=trace.get("issue_id", "unknown"),
        message_id=trace.get("message_id"),
        session_id=trace.get("session_id"),
        invocation_id=trace.get("trace_id"),
        timestamp=timestamp,
        user_message=trace.get("user_message", ""),
        agent_name="multi_agent_system",
        agent_model="gemini-2.5-flash",
        agent_type=AgentType.MULTI_AGENT,
        agent_version="1.0.0",
        expected_tools=expected_tools,
        expected_response=expected_response,
        benchmark_category=BenchmarkCategory.TOOL_USE,
        tool_info=all_tool_calls,
        knowledge_base=kb,
        prompt_version="v1.0",
        llm_token_usage=TokenUsage(input_tokens=15000, output_tokens=500, total_tokens=15500),
        response=agent_response,
        usage_metadata={"input_tokens": 15000, "output_tokens": 500, "total_tokens": 15500},
        latency_ms=5000.0,
        time_to_first_token_ms=800.0,
        conversation_history=conversation_history,
        session_state_before={},
        session_state_after={"escalated": True},
        guardrail_checks=[],
        tool_failures=tool_failures,
        model_failures=[],
        response_length=len(agent_response),
        response_word_count=len(agent_response.split()),
        thinking_steps=[]
    )


async def run_comprehensive_evaluation():
    """Run comprehensive evaluation on the single rich trace."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MULTI-AGENT TRACE EVALUATION")
    print("="*70 + "\n")
    
    # Get project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        project_id=project_id,
        location="us-central1"
    )
    
    # Create evaluation input
    eval_input = create_comprehensive_evaluation_input(TRACE)
    
    print(f"📋 Trace ID: {TRACE.get('trace_id')}")
    print(f"📋 Session: {TRACE.get('session_id')}")
    print(f"📋 Issue: {TRACE.get('issue_id')}")
    print(f"📋 Domain: {TRACE.get('domain')}")
    print(f"📋 Agent Type: {TRACE.get('agent_type')}")
    print(f"📋 User Message: {TRACE.get('user_message')[:80]}...")
    print(f"📋 Agents Involved: {[a['agent_name'] for a in TRACE.get('agent_info', [])]}")
    print(f"📋 Total Tool Calls: {len(eval_input.tool_info)}")
    print(f"📋 Tool Failures: {len(eval_input.tool_failures)}")
    print(f"📋 Knowledge Base: {'Yes' if eval_input.knowledge_base else 'No'}")
    print(f"📋 Expected Tools: {eval_input.expected_tools}")
    print()
    
    print("⏳ Running LLM-as-Judge evaluation with Gemini 2.5 Pro...")
    print()
    
    # Run evaluation
    result = await evaluator.evaluate(
        input_data=eval_input,
        expected_trajectory=eval_input.expected_tools
    )
    
    # Add metadata
    result.agent_name = eval_input.agent_name
    result.agent_model = eval_input.agent_model
    result.benchmark_category = eval_input.benchmark_category.value if eval_input.benchmark_category else None
    result.session_id = eval_input.session_id
    result.invocation_id = eval_input.invocation_id
    
    # Print results
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📊 VERDICT: {result.verdict.value}")
    print()
    
    print("📈 CORE QUALITY SCORES:")
    print(f"   • Safety Score:            {result.safety_score:.2f}")
    print(f"   • PII Score:               {result.pii_score:.2f}")
    print(f"   • Hallucination Score:     {result.hallucination_score:.2f}")
    print(f"   • Action Completion Score: {result.action_completion_score:.2f}")
    print(f"   • Toxicity Score:          {result.toxicity_score:.2f}" if result.toxicity_score else "   • Toxicity Score:          N/A")
    print()
    
    print("📝 TEXT QUALITY SCORES:")
    print(f"   • Fluency Score:           {result.text_quality_metrics.fluency_score:.2f}")
    print(f"   • Coherence Score:         {result.text_quality_metrics.coherence_score:.2f}")
    print(f"   • Relevance Score:         {result.text_quality_metrics.relevance_score:.2f}")
    print(f"   • Conciseness Score:       {result.text_quality_metrics.conciseness_score:.2f}")
    print()
    
    print("🔧 TOOL TRAJECTORY METRICS:")
    print(f"   • Exact Match:             {result.tool_selection_metrics.exact_match:.2f}")
    print(f"   • In-Order Match:          {result.tool_selection_metrics.in_order:.2f}")
    print(f"   • Any-Order Match:         {result.tool_selection_metrics.any_order:.2f}")
    print(f"   • Precision:               {result.tool_selection_metrics.precision:.2f}")
    print(f"   • Recall:                  {result.tool_selection_metrics.recall:.2f}")
    print()
    
    if result.similarity_metrics:
        print("📐 SIMILARITY METRICS:")
        print(f"   • BLEU Score:              {result.similarity_metrics.bleu_score:.4f}" if result.similarity_metrics.bleu_score else "   • BLEU Score:              N/A")
        print(f"   • ROUGE-1:                 {result.similarity_metrics.rouge_1:.4f}" if result.similarity_metrics.rouge_1 else "   • ROUGE-1:                 N/A")
        print(f"   • ROUGE-2:                 {result.similarity_metrics.rouge_2:.4f}" if result.similarity_metrics.rouge_2 else "   • ROUGE-2:                 N/A")
        print(f"   • ROUGE-L:                 {result.similarity_metrics.rouge_l:.4f}" if result.similarity_metrics.rouge_l else "   • ROUGE-L:                 N/A")
        print(f"   • Embedding Similarity:    {result.similarity_metrics.embedding_similarity:.4f}" if result.similarity_metrics.embedding_similarity else "   • Embedding Similarity:    N/A")
        print(f"   • Exact Match:             {result.similarity_metrics.exact_match}")
        print()
    
    if result.trajectory_analysis:
        print("🎯 TRAJECTORY ANALYSIS:")
        print(f"   • Expected:                {result.trajectory_analysis.expected_trajectory}")
        print(f"   • Actual:                  {result.trajectory_analysis.actual_trajectory}")
        print(f"   • Missing Tools:           {result.trajectory_analysis.missing_tools}")
        print(f"   • Extra Tools:             {result.trajectory_analysis.extra_tools}")
        print(f"   • Correct Order:           {result.trajectory_analysis.correct_order}")
        print()
    
    print("💰 COST BREAKDOWN:")
    print(f"   • Agent Input Cost:        ${result.cost_breakdown.agent_input_tokens_cost:.6f}")
    print(f"   • Agent Output Cost:       ${result.cost_breakdown.agent_output_tokens_cost:.6f}")
    print(f"   • Judge Input Cost:        ${result.cost_breakdown.judge_input_tokens_cost:.6f}")
    print(f"   • Judge Output Cost:       ${result.cost_breakdown.judge_output_tokens_cost:.6f}")
    print(f"   • Total Agent Cost:        ${result.cost_breakdown.total_agent_cost:.6f}")
    print(f"   • Total Judge Cost:        ${result.cost_breakdown.total_judge_cost:.6f}")
    print(f"   • TOTAL COST:              ${result.cost_breakdown.total_cost:.6f}")
    print()
    
    print("📊 STATISTICS:")
    print(f"   • Turn Count:              {result.turn_count}")
    print(f"   • Tool Calls:              {result.count_tool_call}")
    print(f"   • Tool Errors:             {result.count_tool_error}")
    print(f"   • Response Length:         {result.response_length} chars")
    print(f"   • Response Words:          {result.response_word_count}")
    print()
    
    print(f"⭐ OVERALL SCORE: {result.overall_score:.2f}")
    print()
    
    print("💬 EVALUATION EXPLANATION:")
    print(f"   {result.evaluation_explanations}")
    print()
    
    # Write to JSONL
    output_file = "/Users/rinit.lulla/Documents/GitHub/agent-evaluation/single_trace_evaluation_result.jsonl"
    with open(output_file, 'w') as f:
        f.write(result.model_dump_json() + "\n")
    
    print(f"📄 Results written to: {output_file}")
    print()
    
    return result


if __name__ == "__main__":
    asyncio.run(run_comprehensive_evaluation())
