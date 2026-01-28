"""
Evaluation Schemas for Agent Evaluation Framework.

Comprehensive Pydantic models for:
- Input Schema: Session metadata, user messages, tool calls, LLM calls, errors
- Output Schema: Evaluation results with all metrics

Enhanced with additional metrics for comprehensive agent evaluation.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, computed_field
from datetime import datetime
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class ToolCallStatus(str, Enum):
    """Status of a tool call execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    PENDING = "pending"


class EvaluationVerdict(str, Enum):
    """Overall evaluation verdict."""
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    ERROR = "ERROR"


class AgentType(str, Enum):
    """Type of agent being evaluated."""
    LLM_AGENT = "llm_agent"
    SEQUENTIAL_AGENT = "sequential_agent"
    PARALLEL_AGENT = "parallel_agent"
    LOOP_AGENT = "loop_agent"
    CUSTOM_AGENT = "custom_agent"
    MULTI_AGENT = "multi_agent"


class BenchmarkCategory(str, Enum):
    """Benchmark category for evaluation."""
    MATH = "math"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    MULTI_TURN = "multi_turn"
    SAFETY = "safety"
    GENERAL = "general"
    CODE = "code"
    RETRIEVAL = "retrieval"


# ============================================================
# INPUT SCHEMA COMPONENTS
# ============================================================

class SessionMetadata(BaseModel):
    """Session-level metadata for evaluation input."""
    
    domain: str = Field(..., description="Domain of the session")
    issue_id: str = Field(..., description="Unique identifier for the issue (Primary Key)")
    message_id: Optional[str] = Field(
        None, 
        description="Unique identifier for the message (null for issue-level)"
    )
    session_id: Optional[str] = Field(None, description="Session identifier")
    invocation_id: Optional[str] = Field(None, description="Unique invocation identifier")
    timestamp: datetime = Field(..., description="Message timestamp")


class UserMessage(BaseModel):
    """User message content."""
    
    user_message: str = Field(..., description="Content of the user message")


class ToolCallInfo(BaseModel):
    """Information about a single tool call."""
    
    tool: str = Field(..., description="Tool name")
    tool_id: Optional[str] = Field(None, description="Unique tool call identifier")
    requested_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments requested for the tool"
    )
    tool_output: Optional[Any] = Field(
        None, 
        description="Output from the tool execution"
    )
    status: ToolCallStatus = Field(
        ToolCallStatus.SUCCESS, 
        description="Status of the tool call (success, failure, timeout, skipped)"
    )
    latency_ms: Optional[float] = Field(None, description="Tool execution latency in ms")
    retry_count: int = Field(0, description="Number of retries for this tool call")


class KnowledgeBase(BaseModel):
    """Knowledge base content retrieved for the LLM."""
    
    content: Optional[str] = Field(None, description="Retrieved knowledge given to LLM")
    source: Optional[str] = Field(None, description="Source of knowledge")
    relevance_score: Optional[float] = Field(None, description="Relevance score 0-1")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of retrieved chunks")
    retrieval_method: Optional[str] = Field(None, description="Method used for retrieval (vector, keyword, hybrid)")


class TokenUsage(BaseModel):
    """Token usage metrics for LLM calls."""
    
    input_tokens: int = Field(0, description="Input tokens")
    output_tokens: int = Field(0, description="Output tokens")
    total_tokens: int = Field(0, description="Total tokens")
    cached_tokens: int = Field(0, description="Cached/context tokens")
    reasoning_tokens: int = Field(0, description="Reasoning/thinking tokens")


class GuardrailCheck(BaseModel):
    """Result of a guardrail/safety check."""
    
    guardrail_name: str = Field(..., description="Name of the guardrail")
    passed: bool = Field(..., description="Whether the check passed")
    score: Optional[float] = Field(None, description="Guardrail score if applicable")
    blocked: bool = Field(False, description="Whether request was blocked")
    details: Optional[str] = Field(None, description="Additional details")


class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    
    role: str = Field(..., description="Role: user, assistant, system, tool")
    content: str = Field(..., description="Content of the turn")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of turn")
    tool_calls: List[ToolCallInfo] = Field(default_factory=list, description="Tool calls in this turn")


class LLMCallInfo(BaseModel):
    """Information about an LLM call."""
    
    prompt_version: str = Field(..., description="Input prompt version to the model")
    llm_token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Total i/p-o/p token usage"
    )
    response: str = Field(..., description="Model response")
    usage_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage metadata (input, output, total)"
    )


class ToolFailure(BaseModel):
    """Details of a tool failure."""
    
    tool_name: str = Field(..., description="Name of the failed tool")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = Field(0, description="Number of retries attempted")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments that caused failure")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")


class ModelFailure(BaseModel):
    """Details of a model failure."""
    
    model_name: str = Field(..., description="Name of the failed model")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = Field(0, description="Number of retries attempted")
    prompt_preview: Optional[str] = Field(None, description="Preview of prompt that failed")
    http_status_code: Optional[int] = Field(None, description="HTTP status code if API error")


# ============================================================
# MAIN INPUT SCHEMA
# ============================================================

class EvaluationInput(BaseModel):
    """
    Complete input schema for agent evaluation.
    
    Contains all information needed to evaluate an agent's performance:
    - Session metadata (domain, issue_id, message_id, timestamp)
    - User message
    - Tool calls with arguments, outputs, and statuses
    - Knowledge base content
    - LLM call information
    - Error/retry information
    - Agent metadata
    - Conversation history
    - Guardrail checks
    """
    
    # Session Metadata
    domain: str = Field(..., description="Domain of the session")
    issue_id: str = Field(..., description="Unique identifier for the issue (Primary Key)")
    message_id: Optional[str] = Field(
        None, 
        description="Unique identifier for the message (null for issue-level)"
    )
    session_id: Optional[str] = Field(None, description="Session identifier")
    invocation_id: Optional[str] = Field(None, description="Unique invocation identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="Message timestamp"
    )
    
    # User Message
    user_message: str = Field(..., description="Content of the user message")
    
    # Agent Metadata (NEW)
    agent_name: Optional[str] = Field(None, description="Name of the agent")
    agent_model: Optional[str] = Field(None, description="Model used by the agent (e.g., gemini-2.5-flash)")
    agent_type: AgentType = Field(AgentType.LLM_AGENT, description="Type of agent")
    agent_version: str = Field("1.0.0", description="Version of the agent")
    
    # Expected values for evaluation (NEW)
    expected_tools: List[str] = Field(default_factory=list, description="Expected tool trajectory")
    expected_response: Optional[str] = Field(None, description="Expected response for comparison")
    benchmark_category: Optional[BenchmarkCategory] = Field(None, description="Benchmark category")
    
    # Tool Calls
    tool_info: List[ToolCallInfo] = Field(
        default_factory=list,
        description="Array of tool calls with name, args, output, status"
    )
    
    # Knowledge Base
    knowledge_base: Optional[KnowledgeBase] = Field(
        None,
        description="Retrieved knowledge given to LLM"
    )
    
    # LLM Calls
    prompt_version: str = Field("v1.0", description="Input prompt version to the model")
    llm_token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Total i/p-o/p token usage"
    )
    response: str = Field("", description="Model response")
    usage_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage (input, output, total)"
    )
    
    # Timing Metrics (NEW)
    latency_ms: Optional[float] = Field(None, description="Total response latency in milliseconds")
    time_to_first_token_ms: Optional[float] = Field(None, description="Time to first token in milliseconds")
    
    # Conversation History (NEW)
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Previous turns in multi-turn conversations"
    )
    
    # Session State (NEW)
    session_state_before: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session state before execution"
    )
    session_state_after: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session state after execution"
    )
    
    # Guardrail Checks (NEW)
    guardrail_checks: List[GuardrailCheck] = Field(
        default_factory=list,
        description="Results of guardrail/safety checks"
    )
    
    # Errors/Retries
    tool_failures: List[ToolFailure] = Field(
        default_factory=list,
        description="Details of tool failures"
    )
    model_failures: List[ModelFailure] = Field(
        default_factory=list,
        description="Details of model failures"
    )
    
    # Response Metrics (NEW)
    response_length: Optional[int] = Field(None, description="Character count of response")
    response_word_count: Optional[int] = Field(None, description="Word count of response")
    
    # Thinking/Reasoning (NEW)
    thinking_steps: List[str] = Field(
        default_factory=list,
        description="Chain-of-thought reasoning steps"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================
# OUTPUT SCHEMA COMPONENTS
# ============================================================

class MetricResult(BaseModel):
    """Score with judge metadata."""
    judge_model: str = Field(..., description="Model used for judging")
    judge_prompt_version: str = Field(..., description="Version of judge prompt")
    judge_input_tokens: int = Field(0, description="Judge input tokens for this metric")
    judge_output_tokens: int = Field(0, description="Judge output tokens for this metric")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Score value")

class VerdictResult(BaseModel):
    """Boolean verdict with judge metadata."""
    judge_model: str = Field(..., description="Model used for judging")
    judge_prompt_version: str = Field(..., description="Version of judge prompt")
    judge_input_tokens: int = Field(0, description="Judge input tokens for this metric")
    judge_output_tokens: int = Field(0, description="Judge output tokens for this metric")
    verdict: bool = Field(..., description="Verdict value")

class LatencyStats(BaseModel):
    """Latency statistics."""
    avg: float = Field(0.0, description="Average latency")
    max: float = Field(0.0, description="Maximum latency")
    min: float = Field(0.0, description="Minimum latency")
    median: float = Field(0.0, description="Median latency")
    p90: float = Field(0.0, description="P90 latency")

class ToolSelectionMetrics(BaseModel):
    """Tool selection evaluation metrics."""
    # Included directly in user request as "tool_selection_score" with judge metadata? 
    # The user request has "tool_selection_score" as json. 
    # But later in the request description: "{judge_model... message_id, in_order, any_order...}"
    # So we'll adapt this or wrap it.
    
    judge_model: str = Field("gemini-2.5-pro", description="Judge model")
    judge_prompt_version: str = Field("v1.0", description="Judge prompt version") 
    judge_input_tokens: int = Field(0, description="Judge input tokens")
    judge_output_tokens: int = Field(0, description="Judge output tokens")
    message_id: Optional[str] = Field(None, description="Message ID if scoped")
    
    in_order: float = Field(0.0, ge=0.0, le=1.0, description="In-order match score")
    any_order: float = Field(0.0, ge=0.0, le=1.0, description="Any-order match score")
    exact_match: float = Field(0.0, ge=0.0, le=1.0, description="Exact match score")
    precision: float = Field(0.0, ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(0.0, ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(0.0, ge=0.0, le=1.0, description="F1 score")


class TextQualityMetrics(BaseModel):
    """Text quality evaluation metrics."""
    fluency_score: float = Field(0.0, ge=0.0, le=1.0, description="Language fluency score")
    coherence_score: float = Field(0.0, ge=0.0, le=1.0, description="Response coherence score")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance to query score")
    conciseness_score: float = Field(0.0, ge=0.0, le=1.0, description="Response conciseness score")


class RubricScore(BaseModel):
    """Score from a specific rubric evaluation."""
    rubric_name: str = Field(..., description="Name of the rubric")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Score for this rubric")
    max_score: float = Field(1.0, description="Maximum possible score")
    explanation: Optional[str] = Field(None, description="Explanation for this score")


class SimilarityMetrics(BaseModel):
    """Similarity metrics comparing response to expected."""
    embedding_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Semantic similarity score")
    bleu_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="BLEU score")
    rouge_1: Optional[float] = Field(None, ge=0.0, le=1.0, description="ROUGE-1 score")
    rouge_2: Optional[float] = Field(None, ge=0.0, le=1.0, description="ROUGE-2 score")
    rouge_l: Optional[float] = Field(None, ge=0.0, le=1.0, description="ROUGE-L score")
    exact_match: bool = Field(False, description="Whether response exactly matches expected")


class CostBreakdown(BaseModel):
    """Detailed cost breakdown for the evaluation."""
    agent_input_tokens_cost: float = Field(0.0, description="Cost for agent input tokens")
    agent_output_tokens_cost: float = Field(0.0, description="Cost for agent output tokens")
    judge_input_tokens_cost: float = Field(0.0, description="Cost for judge input tokens")
    judge_output_tokens_cost: float = Field(0.0, description="Cost for judge output tokens")
    total_agent_cost: float = Field(0.0, description="Total agent cost")
    total_judge_cost: float = Field(0.0, description="Total judge cost")
    total_cost: float = Field(0.0, description="Total cost")


class TrajectoryAnalysis(BaseModel):
    """Detailed analysis of tool call trajectory."""
    expected_trajectory: List[str] = Field(default_factory=list, description="Expected tool sequence")
    actual_trajectory: List[str] = Field(default_factory=list, description="Actual tool sequence")
    missing_tools: List[str] = Field(default_factory=list, description="Tools that should have been called")
    extra_tools: List[str] = Field(default_factory=list, description="Unnecessary tools called")
    correct_order: bool = Field(False, description="Whether tools were called in correct order")
    trajectory_explanation: Optional[str] = Field(None, description="Explanation of trajectory analysis")


# ============================================================
# MAIN OUTPUT SCHEMA
# ============================================================

class EvaluationOutput(BaseModel):
    """
    Complete output schema matching user requirements.
    """
    
    # Evaluation Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Conversation start timestamp (issue-level)")
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    domain: str = Field(..., description="Domain of the session (issue-level)")
    issue_id: str = Field(..., description="Unique issue identifier (primary key)")
    agent_version: str = Field(..., description="Version of the agent being evaluated")
    
    # Additional Metadata
    session_id: Optional[str] = Field(None, description="Session identifier")
    invocation_id: Optional[str] = Field(None, description="Invocation identifier")
    message_id: Optional[str] = Field(None, description="Message identifier")
    agent_name: Optional[str] = Field(None, description="Name of the agent")
    agent_model: Optional[str] = Field(None, description="Model used by the agent")
    benchmark_category: Optional[str] = Field(None, description="Benchmark category")
    
    # Performance Stats
    latency: LatencyStats = Field(..., description="Aggregate latency stats")
    time_to_first_token: Optional[float] = Field(None, description="TTFT in seconds")
    judge_total_tokens_cost: float = Field(0.0, description="Total judge token cost")
    
    # Counts
    turn_count: int = Field(0, description="Total conversation turns")
    count_user_msg: int = Field(0, description="Count of user messages")
    count_assistant_msg: int = Field(0, description="Count of assistant messages")
    count_tool_call: int = Field(0, description="Count of tool calls")
    count_tool_error: int = Field(0, description="Count of tool errors")
    count_model_call: int = Field(0, description="Count of model calls")
    count_model_error: int = Field(0, description="Count of model errors")
    
    # Metric Results
    safety_score: MetricResult = Field(..., description="Safety evaluation result")
    pii_score: MetricResult = Field(..., description="PII evaluation result")
    hallucination_score: MetricResult = Field(..., description="Hallucination evaluation result")
    action_completion_score: MetricResult = Field(..., description="Action completion evaluation")
    groundedness_score: Optional[MetricResult] = Field(None, description="Groundedness evaluation")
    
    # Boolean Checks
    loop_presence: Optional[VerdictResult] = Field(None, description="Loop detection verdict")
    
    # Additional Scores
    multi_turn_chat_quality_score: Optional[MetricResult] = Field(None, alias="multi-turn_chat_quality_score", description="Multi-turn quality evaluation")
    tool_selection_score: Optional[ToolSelectionMetrics] = Field(None, description="Tool selection evaluation")
    
    # Other (kept for backward compatibility or internal use if needed, but made optional)
    evaluation_explanations: Optional[str] = Field(None, description="Short explanations")
    verdict: EvaluationVerdict = Field(EvaluationVerdict.FAIL, description="Overall verdict")
    conversation: Optional[str] = Field(None, description="Full Transcript")
    
    # Hidden/Internal fields required for `run_trace_evaluation.py` logic or summary report?
    # The summary report uses `safety_score` as float. We changed it to `MetricResult`.
    # This implies we MUST update `generate_summary_report` as well in `evaluator.py`.

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        populate_by_name = True


# ============================================================
# BATCH EVALUATION SCHEMAS
# ============================================================

class BatchEvaluationSummary(BaseModel):
    """Summary of a batch evaluation run."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch timestamp")
    total_evaluations: int = Field(0, description="Total number of evaluations")
    passed: int = Field(0, description="Number passed")
    partial: int = Field(0, description="Number partial")
    failed: int = Field(0, description="Number failed")
    errors: int = Field(0, description="Number with errors")
    pass_rate: float = Field(0.0, description="Pass rate (0-1)")
    
    # Aggregate Metrics
    avg_safety_score: float = Field(0.0, description="Average safety score")
    avg_pii_score: float = Field(0.0, description="Average PII score")
    avg_hallucination_score: float = Field(0.0, description="Average hallucination score")
    avg_action_completion_score: float = Field(0.0, description="Average action completion score")
    avg_overall_score: float = Field(0.0, description="Average overall score")
    avg_latency: float = Field(0.0, description="Average latency in seconds")
    
    # Tool Metrics
    avg_tool_precision: float = Field(0.0, description="Average tool precision")
    avg_tool_recall: float = Field(0.0, description="Average tool recall")
    
    # Cost
    total_agent_tokens: int = Field(0, description="Total agent tokens used")
    total_judge_tokens: int = Field(0, description="Total judge tokens used")
    total_cost: float = Field(0.0, description="Total cost for batch")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_evaluation_input(
    domain: str,
    issue_id: str,
    user_message: str,
    tool_calls: List[Dict[str, Any]] = None,
    response: str = "",
    **kwargs
) -> EvaluationInput:
    """Helper to create EvaluationInput from raw data."""
    
    tool_info = []
    if tool_calls:
        for tc in tool_calls:
            tool_info.append(ToolCallInfo(
                tool=tc.get("tool", "unknown"),
                requested_args=tc.get("requested_args", tc.get("args", {})),
                tool_output=tc.get("tool_output", tc.get("output")),
                status=ToolCallStatus(tc.get("status", "success"))
            ))
    
    # Calculate response metrics
    response_length = len(response) if response else None
    response_word_count = len(response.split()) if response else None
    
    return EvaluationInput(
        domain=domain,
        issue_id=issue_id,
        user_message=user_message,
        tool_info=tool_info,
        response=response,
        response_length=response_length,
        response_word_count=response_word_count,
        **kwargs
    )


def create_evaluation_output(
    evaluation_id: str,
    domain: str,
    issue_id: str,
    judge_model: str = "gemini-2.5-pro",
    **kwargs
) -> EvaluationOutput:
    """Helper to create EvaluationOutput with defaults."""
    
    return EvaluationOutput(
        evaluation_id=evaluation_id,
        domain=domain,
        issue_id=issue_id,
        judge_model=judge_model,
        **kwargs
    )


def calculate_overall_score(output: EvaluationOutput, weights: Dict[str, float] = None) -> float:
    """Calculate weighted overall score from individual scores."""
    
    if weights is None:
        weights = {
            "safety": 0.20,
            "pii": 0.15,
            "hallucination": 0.20,
            "action_completion": 0.25,
            "tool_precision": 0.10,
            "tool_recall": 0.10
        }
    
    score = (
        output.safety_score * weights.get("safety", 0.2) +
        output.pii_score * weights.get("pii", 0.15) +
        output.hallucination_score * weights.get("hallucination", 0.2) +
        output.action_completion_score * weights.get("action_completion", 0.25) +
        output.tool_selection_metrics.precision * weights.get("tool_precision", 0.1) +
        output.tool_selection_metrics.recall * weights.get("tool_recall", 0.1)
    )
    
    return min(1.0, max(0.0, score))


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Enums
    "ToolCallStatus",
    "EvaluationVerdict",
    "AgentType",
    "BenchmarkCategory",
    # Input Components
    "SessionMetadata",
    "UserMessage",
    "ToolCallInfo",
    "KnowledgeBase",
    "TokenUsage",
    "LLMCallInfo",
    "ToolFailure",
    "ModelFailure",
    "GuardrailCheck",
    "ConversationTurn",
    # Output Components
    "ToolSelectionMetrics",
    "TextQualityMetrics",
    "RubricScore",
    "SimilarityMetrics",
    "CostBreakdown",
    "TrajectoryAnalysis",
    # Main Schemas
    "EvaluationInput",
    "EvaluationOutput",
    "BatchEvaluationSummary",
    # Helpers
    "create_evaluation_input",
    "create_evaluation_output",
    "calculate_overall_score"
]
