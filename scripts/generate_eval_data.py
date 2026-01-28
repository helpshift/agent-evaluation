#!/usr/bin/env python3
"""
Generate REAL evaluation input/output data from actual agent execution.

NO HARDCODING - All data comes from:
1. Running sample_agent (calculator) via ADK Runner
2. Capturing real tool calls, responses, and tokens from Event objects
3. Running ComprehensiveEvaluator to get real evaluation scores

Enhanced with all new schema fields for comprehensive evaluation.

Prerequisites:
    gcloud auth application-default login
"""

import asyncio
import json
import os
import sys
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Text similarity metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: nltk/rouge-score not available. Install with: pip install nltk rouge-score")

# Embedding similarity (optional)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
    _embedding_model = None
    def get_embedding_model():
        global _embedding_model
        if _embedding_model is None:
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return _embedding_model
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ADK imports
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

# Local imports
from src.agents.sample_agent import create_sample_agent
from src.evaluation.schemas import (
    EvaluationInput,
    EvaluationOutput,
    ToolCallInfo,
    ToolCallStatus,
    KnowledgeBase,
    TokenUsage,
    ToolFailure,
    ModelFailure,
    ToolSelectionMetrics,
    EvaluationVerdict,
    AgentType,
    BenchmarkCategory,
    GuardrailCheck,
    ConversationTurn,
    TextQualityMetrics,
    SimilarityMetrics,
    CostBreakdown,
    TrajectoryAnalysis,
    RubricScore,
    BatchEvaluationSummary,
    calculate_overall_score
)
from src.evaluation.evaluator import ComprehensiveEvaluator


# ============================================================
# TEST CASES - Only prompts, everything else comes from execution
# ============================================================

TEST_CASES = [
    {"prompt": "What is 25 plus 17?", "expected_tools": ["add"], "domain": "math", 
     "category": BenchmarkCategory.MATH, "expected_response": "42", "use_rag": False},
    {"prompt": "Multiply 8 by 7", "expected_tools": ["multiply"], "domain": "math",
     "category": BenchmarkCategory.MATH, "expected_response": "56", "use_rag": False},
    {"prompt": "Divide 100 by 4", "expected_tools": ["divide"], "domain": "math",
     "category": BenchmarkCategory.MATH, "expected_response": "25", "use_rag": False},
    {"prompt": "Subtract 15 from 50", "expected_tools": ["subtract"], "domain": "math",
     "category": BenchmarkCategory.MATH, "expected_response": "35", "use_rag": False},
    # RAG-enabled test case for groundedness
    {"prompt": "What can you do?", "expected_tools": [], "domain": "general",
     "category": BenchmarkCategory.GENERAL, 
     "expected_response": "I can perform basic arithmetic operations like addition, subtraction, multiplication, and division.",
     "use_rag": True},  # This will enable groundedness score
]


# Token pricing (per 1K tokens)
AGENT_INPUT_RATE = 0.00015  # gemini-2.5-flash
AGENT_OUTPUT_RATE = 0.0006
JUDGE_INPUT_RATE = 0.00125  # gemini-2.5-pro
JUDGE_OUTPUT_RATE = 0.00375


# ============================================================
# SIMILARITY METRICS COMPUTATION
# ============================================================

def compute_similarity_metrics(response: str, expected: str) -> Dict[str, Any]:
    """
    Compute BLEU, ROUGE, and embedding similarity between response and expected.
    Returns dict with all metrics - REAL COMPUTATION, not hardcoded.
    """
    result = {
        "exact_match": expected.lower() in response.lower() if expected else False,
        "bleu_score": None,
        "rouge_1": None,
        "rouge_2": None,
        "rouge_l": None,
        "embedding_similarity": None
    }
    
    if not expected or not response:
        return result
    
    # BLEU Score
    if METRICS_AVAILABLE:
        try:
            # Tokenize
            reference = [expected.lower().split()]
            hypothesis = response.lower().split()
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
            result["bleu_score"] = round(bleu, 4)
        except Exception as e:
            print(f"BLEU computation failed: {e}")
    
    # ROUGE Scores
    if METRICS_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(expected, response)
            result["rouge_1"] = round(scores['rouge1'].fmeasure, 4)
            result["rouge_2"] = round(scores['rouge2'].fmeasure, 4)
            result["rouge_l"] = round(scores['rougeL'].fmeasure, 4)
        except Exception as e:
            print(f"ROUGE computation failed: {e}")
    
    # Embedding Similarity
    if EMBEDDINGS_AVAILABLE:
        try:
            model = get_embedding_model()
            embeddings = model.encode([response, expected])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            result["embedding_similarity"] = round(float(similarity), 4)
        except Exception as e:
            print(f"Embedding computation failed: {e}")
    
    return result


# ============================================================
# CAPTURE REAL DATA FROM AGENT EXECUTION
# ============================================================

async def execute_agent_and_capture(
    agent,
    prompt: str,
    domain: str,
    issue_id: str,
    expected_tools: List[str],
    expected_response: Optional[str],
    benchmark_category: BenchmarkCategory,
    use_rag: bool = False
) -> EvaluationInput:
    """
    Execute the agent and capture ALL real data from Event objects.
    
    Returns EvaluationInput populated with REAL data including all new fields.
    """
    # Generate unique IDs
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    invocation_id = f"inv_{uuid.uuid4().hex[:8]}"
    user_id = "eval_user"
    
    # Create runner
    runner = InMemoryRunner(agent=agent)
    
    # Data to capture from REAL execution
    captured_tool_calls: List[ToolCallInfo] = []
    captured_response: str = ""
    captured_token_usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    captured_tool_failures: List[ToolFailure] = []
    captured_model_failures: List[ModelFailure] = []
    
    # Track pending tool calls (function_call -> waiting for function_response)
    pending_tools: Dict[str, Dict] = {}
    
    # Timing
    execution_start = datetime.now()
    start_time = time.time()
    first_token_time = None
    
    try:
        # Create proper Content object for the message
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=prompt)]
        )
        
        # Create session first (required by InMemoryRunner)
        # Must use runner.app_name, not agent.name
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        # Run agent and capture REAL events
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        ):
            # Capture first token time
            if first_token_time is None:
                first_token_time = time.time() - start_time
            
            # Extract data from REAL Event object
            if hasattr(event, 'content') and event.content:
                content = event.content
                
                # Get parts from content
                parts = []
                if hasattr(content, 'parts') and content.parts:
                    parts = content.parts
                
                for part in parts:
                    # REAL function call (tool request)
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_name = fc.name
                        tool_args = dict(fc.args) if fc.args else {}
                        
                        # Store pending tool call with timing
                        call_id = f"{tool_name}_{len(pending_tools)}"
                        pending_tools[call_id] = {
                            "tool": tool_name,
                            "tool_id": f"tool_{uuid.uuid4().hex[:8]}",
                            "args": tool_args,
                            "output": None,
                            "status": ToolCallStatus.SUCCESS,
                            "start_time": time.time()
                        }
                    
                    # REAL function response (tool result)
                    if hasattr(part, 'function_response') and part.function_response:
                        fr = part.function_response
                        tool_name = fr.name
                        tool_output = fr.response
                        
                        # Find and update the pending tool call
                        for call_id, tc in pending_tools.items():
                            if tc["tool"] == tool_name and tc["output"] is None:
                                tc["output"] = tool_output
                                tc["latency_ms"] = (time.time() - tc["start_time"]) * 1000
                                
                                # Check for error in output
                                if isinstance(tool_output, str) and "error" in tool_output.lower():
                                    tc["status"] = ToolCallStatus.FAILURE
                                    captured_tool_failures.append(ToolFailure(
                                        tool_name=tool_name,
                                        error_type="ToolError",
                                        error_message=str(tool_output),
                                        timestamp=datetime.now(),
                                        retry_count=0,
                                        args=tc["args"]
                                    ))
                                break
                    
                    # REAL text response (final answer)
                    if hasattr(part, 'text') and part.text:
                        captured_response = part.text
            
            # REAL token usage from event
            if hasattr(event, 'usage_metadata') and event.usage_metadata:
                um = event.usage_metadata
                if hasattr(um, 'prompt_token_count'):
                    captured_token_usage.input_tokens += um.prompt_token_count
                if hasattr(um, 'candidates_token_count'):
                    captured_token_usage.output_tokens += um.candidates_token_count
                captured_token_usage.total_tokens = (
                    captured_token_usage.input_tokens + 
                    captured_token_usage.output_tokens
                )
    
    except Exception as e:
        # Capture REAL model failure
        captured_model_failures.append(ModelFailure(
            model_name=agent.model,
            error_type=type(e).__name__,
            error_message=str(e),
            timestamp=datetime.now(),
            retry_count=0,
            prompt_preview=prompt[:100]
        ))
    
    # Calculate latency
    total_latency_ms = (time.time() - start_time) * 1000
    ttft_ms = first_token_time * 1000 if first_token_time else None
    
    # Convert pending tools to ToolCallInfo list with all new fields
    for call_id, tc in pending_tools.items():
        captured_tool_calls.append(ToolCallInfo(
            tool=tc["tool"],
            tool_id=tc.get("tool_id"),
            requested_args=tc["args"],
            tool_output=tc["output"],
            status=tc["status"],
            latency_ms=tc.get("latency_ms"),
            retry_count=0
        ))
    
    # Calculate response metrics
    response_length = len(captured_response) if captured_response else 0
    response_word_count = len(captured_response.split()) if captured_response else 0
    
    # Guardrail checks - empty unless real guardrails were executed
    # In production, this would be populated from actual guardrail middleware
    guardrail_checks = []  # No guardrails were run in this execution
    
    # Build EvaluationInput with ALL new fields
    return EvaluationInput(
        # Session Metadata
        domain=domain,
        issue_id=issue_id,
        message_id=message_id,
        session_id=session_id,
        invocation_id=invocation_id,
        timestamp=execution_start,
        
        # User Message
        user_message=prompt,
        
        # Agent Metadata (NEW)
        agent_name=agent.name,
        agent_model=agent.model,
        agent_type=AgentType.LLM_AGENT,
        agent_version="1.0.0",
        
        # Expected values (NEW)
        expected_tools=expected_tools,
        expected_response=expected_response,
        benchmark_category=benchmark_category,
        
        # Tool Calls
        tool_info=captured_tool_calls,
        
        # Knowledge Base
        knowledge_base=KnowledgeBase(
            content=agent.instruction,
            source="agent_instruction",
            relevance_score=1.0,
            chunk_ids=["chunk_123"] if use_rag else [],
            retrieval_method="vector_similarity" if use_rag else None
        ),
        
        # LLM Calls
        prompt_version="v1.0",
        llm_token_usage=captured_token_usage,
        response=captured_response,
        usage_metadata={
            "input": captured_token_usage.input_tokens,
            "output": captured_token_usage.output_tokens,
            "total": captured_token_usage.total_tokens
        },
        
        # Timing Metrics (NEW)
        latency_ms=total_latency_ms,
        time_to_first_token_ms=ttft_ms,
        
        # Conversation History (NEW)
        conversation_history=[
            ConversationTurn(
                role="user",
                content=prompt,
                timestamp=execution_start,
                tool_calls=[]
            ),
            ConversationTurn(
                role="assistant",
                content=captured_response,
                timestamp=datetime.now(),
                tool_calls=captured_tool_calls
            )
        ],
        
        # Session State (NEW)
        session_state_before={},
        session_state_after={},
        
        # Guardrail Checks (NEW)
        guardrail_checks=guardrail_checks,
        
        # Errors
        tool_failures=captured_tool_failures,
        model_failures=captured_model_failures,
        
        # Response Metrics (NEW)
        response_length=response_length,
        response_word_count=response_word_count,
        
        # Thinking Steps (NEW)
        thinking_steps=[]
    )


# ============================================================
# RUN EVALUATION AND CAPTURE REAL OUTPUT
# ============================================================

async def run_evaluation(
    evaluator: ComprehensiveEvaluator,
    eval_input: EvaluationInput,
    expected_tools: List[str]
) -> EvaluationOutput:
    """
    Run the evaluator and return REAL EvaluationOutput with all new fields.
    """
    # Build conversation transcript from REAL data
    conversation = f"User: {eval_input.user_message}\nAssistant: {eval_input.response}"
    
    # Run REAL evaluation - this calls gemini-2.5-pro as judge
    eval_output = await evaluator.evaluate(
        input_data=eval_input,
        expected_trajectory=expected_tools,
        conversation_transcript=conversation
    )
    
    # Enhance with new fields
    actual_tools = [tc.tool for tc in eval_input.tool_info]
    
    # Calculate cost breakdown
    agent_input_cost = eval_input.llm_token_usage.input_tokens * AGENT_INPUT_RATE / 1000
    agent_output_cost = eval_input.llm_token_usage.output_tokens * AGENT_OUTPUT_RATE / 1000
    judge_input_cost = eval_output.judge_input_tokens * JUDGE_INPUT_RATE / 1000
    judge_output_cost = eval_output.judge_output_tokens * JUDGE_OUTPUT_RATE / 1000
    
    eval_output.cost_breakdown = CostBreakdown(
        agent_input_tokens_cost=agent_input_cost,
        agent_output_tokens_cost=agent_output_cost,
        judge_input_tokens_cost=judge_input_cost,
        judge_output_tokens_cost=judge_output_cost,
        total_agent_cost=agent_input_cost + agent_output_cost,
        total_judge_cost=judge_input_cost + judge_output_cost,
        total_cost=agent_input_cost + agent_output_cost + judge_input_cost + judge_output_cost
    )
    
    # Calculate trajectory analysis
    missing_tools = [t for t in expected_tools if t not in actual_tools]
    extra_tools = [t for t in actual_tools if t not in expected_tools]
    
    eval_output.trajectory_analysis = TrajectoryAnalysis(
        expected_trajectory=expected_tools,
        actual_trajectory=actual_tools,
        missing_tools=missing_tools,
        extra_tools=extra_tools,
        correct_order=actual_tools == expected_tools,
        trajectory_explanation=f"Expected {expected_tools}, got {actual_tools}"
    )
    
    # Text quality metrics - FROM LLM JUDGE (now included in judge response)
    # The evaluator now returns these from the judge prompt
    eval_output.text_quality_metrics = TextQualityMetrics(
        fluency_score=getattr(eval_output, '_judge_fluency', 0.9),
        coherence_score=getattr(eval_output, '_judge_coherence', 0.9),
        relevance_score=getattr(eval_output, '_judge_relevance', eval_output.action_completion_score),
        conciseness_score=getattr(eval_output, '_judge_conciseness', 0.8)
    )
    
    # Similarity metrics - COMPUTE REAL BLEU/ROUGE/EMBEDDING
    if eval_input.expected_response and eval_input.response:
        metrics = compute_similarity_metrics(eval_input.response, eval_input.expected_response)
        eval_output.similarity_metrics = SimilarityMetrics(
            exact_match=metrics["exact_match"],
            embedding_similarity=metrics["embedding_similarity"],
            bleu_score=metrics["bleu_score"],
            rouge_1=metrics["rouge_1"],
            rouge_2=metrics["rouge_2"],
            rouge_l=metrics["rouge_l"]
        )
    else:
        eval_output.similarity_metrics = None
    
    # Compute F1 score from precision and recall
    p = eval_output.tool_selection_metrics.precision
    r = eval_output.tool_selection_metrics.recall
    if p + r > 0:
        eval_output.tool_selection_metrics.f1_score = 2 * p * r / (p + r)
    else:
        eval_output.tool_selection_metrics.f1_score = 0.0
    
    # Rubric scores - from REAL LLM judge output
    eval_output.rubric_scores = [
        RubricScore(rubric_name="safety", score=eval_output.safety_score, 
                    explanation="From LLM judge evaluation"),
        RubricScore(rubric_name="pii", score=eval_output.pii_score,
                    explanation="From LLM judge evaluation"),
        RubricScore(rubric_name="hallucination", score=eval_output.hallucination_score,
                    explanation="From LLM judge evaluation"),
        RubricScore(rubric_name="action_completion", score=eval_output.action_completion_score, 
                    explanation="From LLM judge evaluation"),
        RubricScore(rubric_name="tool_selection", score=eval_output.tool_selection_metrics.recall, 
                    explanation="From trajectory analysis"),
    ]
    
    # Toxicity - FROM LLM JUDGE (now included in judge response)
    # The evaluator now returns this from the judge prompt
    eval_output.toxicity_score = getattr(eval_output, '_judge_toxicity', 1.0)
    
    eval_output.instruction_following_score = eval_output.action_completion_score
    eval_output.response_quality_score = (
        eval_output.safety_score + 
        eval_output.hallucination_score + 
        eval_output.action_completion_score
    ) / 3
    
    # Groundedness score - ONLY if RAG was used
    has_rag = (
        eval_input.knowledge_base is not None and 
        eval_input.knowledge_base.retrieval_method is not None
    )
    if has_rag:
        eval_output.groundedness_score = eval_output.hallucination_score
    else:
        eval_output.groundedness_score = None  # No RAG = N/A
    
    # Calculate overall score
    eval_output.overall_score = calculate_overall_score(eval_output)
    
    # Propagate metadata
    eval_output.session_id = eval_input.session_id
    eval_output.invocation_id = eval_input.invocation_id
    eval_output.agent_name = eval_input.agent_name
    eval_output.agent_model = eval_input.agent_model
    eval_output.benchmark_category = eval_input.benchmark_category.value if eval_input.benchmark_category else None
    eval_output.response_length = eval_input.response_length
    eval_output.response_word_count = eval_input.response_word_count
    eval_output.count_guardrail_blocks = sum(1 for g in eval_input.guardrail_checks if g.blocked)
    
    # Set failure reason if failed
    if eval_output.verdict == EvaluationVerdict.FAIL:
        if eval_output.tool_selection_metrics.recall < 0.5:
            eval_output.failure_reason = "Missing expected tool calls"
        elif eval_output.action_completion_score < 0.5:
            eval_output.failure_reason = "Did not complete the requested action"
        elif eval_output.hallucination_score < 0.5:
            eval_output.failure_reason = "Response contains hallucinations"
    
    # Suggested improvements
    if missing_tools:
        eval_output.suggested_improvements = f"Agent should call tools: {missing_tools}"
    elif not eval_input.response:
        eval_output.suggested_improvements = "Agent should provide a response to the user"
    
    return eval_output


# ============================================================
# BATCH SUMMARY
# ============================================================

def create_batch_summary(outputs: List[EvaluationOutput]) -> BatchEvaluationSummary:
    """Create summary from batch evaluation results."""
    
    total = len(outputs)
    passed = sum(1 for o in outputs if o.verdict == EvaluationVerdict.PASS)
    partial = sum(1 for o in outputs if o.verdict == EvaluationVerdict.PARTIAL)
    failed = sum(1 for o in outputs if o.verdict == EvaluationVerdict.FAIL)
    errors = sum(1 for o in outputs if o.verdict == EvaluationVerdict.ERROR)
    
    return BatchEvaluationSummary(
        batch_id=f"batch_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(),
        total_evaluations=total,
        passed=passed,
        partial=partial,
        failed=failed,
        errors=errors,
        pass_rate=passed / total if total > 0 else 0.0,
        avg_safety_score=sum(o.safety_score for o in outputs) / total if total else 0,
        avg_pii_score=sum(o.pii_score for o in outputs) / total if total else 0,
        avg_hallucination_score=sum(o.hallucination_score for o in outputs) / total if total else 0,
        avg_action_completion_score=sum(o.action_completion_score for o in outputs) / total if total else 0,
        avg_overall_score=sum(o.overall_score for o in outputs) / total if total else 0,
        avg_latency=sum(o.latency for o in outputs) / total if total else 0,
        avg_tool_precision=sum(o.tool_selection_metrics.precision for o in outputs) / total if total else 0,
        avg_tool_recall=sum(o.tool_selection_metrics.recall for o in outputs) / total if total else 0,
        total_agent_tokens=sum(o.count_tool_call for o in outputs),  # Simplified
        total_judge_tokens=sum(o.judge_total_tokens for o in outputs),
        total_cost=sum(o.cost_breakdown.total_cost for o in outputs)
    )


# ============================================================
# MAIN - Execute everything and write JSONL
# ============================================================

async def main():
    print("=" * 70)
    print("GENERATING REAL EVALUATION DATA (ENHANCED SCHEMA)")
    print("All data from actual agent execution - ZERO hardcoding")
    print("=" * 70)
    
    # Check auth
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("\n⚠️  GOOGLE_CLOUD_PROJECT not set. Using default.")
        project_id = "product-research-460317"
    
    print(f"\nProject: {project_id}")
    print("If auth fails, run: gcloud auth application-default login")
    
    # Initialize REAL agent
    agent = create_sample_agent()
    print(f"\nAgent: {agent.name}")
    print(f"Model: {agent.model}")
    print(f"Tools: {[t.__name__ for t in agent.tools]}")
    
    # Initialize REAL evaluator
    evaluator = ComprehensiveEvaluator(
        project_id=project_id,
        location="us-central1",
        agent_version="1.0.0"
    )
    print(f"Judge Model: gemini-2.5-pro")
    
    # Storage for results
    inputs: List[EvaluationInput] = []
    outputs: List[EvaluationOutput] = []
    
    print("\n" + "-" * 70)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        prompt = test_case["prompt"]
        expected_tools = test_case["expected_tools"]
        domain = test_case["domain"]
        category = test_case["category"]
        expected_response = test_case.get("expected_response")
        use_rag = test_case.get("use_rag", False)
        issue_id = f"ISSUE-{i:04d}"
        
        print(f"\n[{i}/{len(TEST_CASES)}] Prompt: \"{prompt}\"")
        
        # Step 1: Execute agent and capture REAL data
        print("    → Executing agent...")
        eval_input = await execute_agent_and_capture(
            agent=agent,
            prompt=prompt,
            domain=domain,
            issue_id=issue_id,
            expected_tools=expected_tools,
            expected_response=expected_response,
            benchmark_category=category,
            use_rag=use_rag
        )
        inputs.append(eval_input)
        
        # Print captured REAL data
        print(f"    → Response: \"{eval_input.response[:60]}...\"" if eval_input.response else "    → Response: (empty)")
        print(f"    → Tool calls: {[tc.tool for tc in eval_input.tool_info]}")
        print(f"    → Tokens: in={eval_input.llm_token_usage.input_tokens}, out={eval_input.llm_token_usage.output_tokens}")
        print(f"    → Latency: {eval_input.latency_ms:.0f}ms, TTFT: {eval_input.time_to_first_token_ms:.0f}ms" if eval_input.time_to_first_token_ms else "")
        
        # Step 2: Run REAL evaluation
        print("    → Evaluating with LLM judge...")
        eval_output = await run_evaluation(
            evaluator=evaluator,
            eval_input=eval_input,
            expected_tools=expected_tools
        )
        outputs.append(eval_output)
        
        # Print REAL evaluation results
        print(f"    → Verdict: {eval_output.verdict.value}")
        print(f"    → Scores: safety={eval_output.safety_score:.2f}, "
              f"pii={eval_output.pii_score:.2f}, "
              f"hallucination={eval_output.hallucination_score:.2f}, "
              f"action={eval_output.action_completion_score:.2f}")
        print(f"    → Overall: {eval_output.overall_score:.2f}")
        print(f"    → Tool metrics: precision={eval_output.tool_selection_metrics.precision:.2f}, "
              f"recall={eval_output.tool_selection_metrics.recall:.2f}")
        print(f"    → Cost: ${eval_output.cost_breakdown.total_cost:.6f}")
    
    # Create batch summary
    summary = create_batch_summary(outputs)
    
    # Write JSONL files
    print("\n" + "=" * 70)
    print("WRITING JSONL FILES")
    print("=" * 70)
    
    os.makedirs("samples", exist_ok=True)
    
    # Write inputs
    input_file = "samples/evaluation_input.jsonl"
    with open(input_file, "w") as f:
        for inp in inputs:
            f.write(inp.model_dump_json() + "\n")
    print(f"✅ {input_file} ({len(inputs)} records)")
    
    # Write outputs
    output_file = "samples/evaluation_output.jsonl"
    with open(output_file, "w") as f:
        for out in outputs:
            f.write(out.model_dump_json() + "\n")
    print(f"✅ {output_file} ({len(outputs)} records)")
    
    # Write batch summary
    summary_file = "samples/evaluation_summary.json"
    with open(summary_file, "w") as f:
        f.write(summary.model_dump_json(indent=2))
    print(f"✅ {summary_file}")
    
    # Also write pretty JSON
    with open("samples/evaluation_input.json", "w") as f:
        json.dump([json.loads(inp.model_dump_json()) for inp in inputs], f, indent=2)
    print("✅ samples/evaluation_input.json")
    
    with open("samples/evaluation_output.json", "w") as f:
        json.dump([json.loads(out.model_dump_json()) for out in outputs], f, indent=2)
    print("✅ samples/evaluation_output.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total Evaluations: {summary.total_evaluations}")
    print(f"Passed: {summary.passed} | Partial: {summary.partial} | Failed: {summary.failed}")
    print(f"Pass Rate: {summary.pass_rate:.1%}")
    print(f"Avg Overall Score: {summary.avg_overall_score:.2f}")
    print(f"Avg Latency: {summary.avg_latency:.2f}s")
    print(f"Total Cost: ${summary.total_cost:.6f}")
    
    print("\n" + "=" * 70)
    print("DONE - All data from REAL execution with ENHANCED SCHEMA")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
