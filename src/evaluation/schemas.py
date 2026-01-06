"""
Evaluation Schemas for Agent Evaluation Framework.

Comprehensive Pydantic models for:
- Input Schema: Session metadata, user messages, tool calls, LLM calls, errors
- Output Schema: Evaluation results with all metrics
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
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


class EvaluationVerdict(str, Enum):
    """Overall evaluation verdict."""
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


# ============================================================
# INPUT SCHEMA COMPONENTS
# ============================================================

class SessionMetadata(BaseModel):
    """Session-level metadata for evaluation input."""
    
    domain: str = Field(..., description="Domain of the session")
    issue_id: str = Field(..., description="Unique identifier for the issue")
    message_id: Optional[str] = Field(
        None, 
        description="Unique identifier for the message (null for issue-level)"
    )
    timestamp: datetime = Field(..., description="Message timestamp")


class UserMessage(BaseModel):
    """User message content."""
    
    user_message: str = Field(..., description="Content of the user message")


class ToolCallInfo(BaseModel):
    """Information about a single tool call."""
    
    tool: str = Field(..., description="Tool name")
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


class KnowledgeBase(BaseModel):
    """Knowledge base content retrieved for the LLM."""
    
    content: Optional[str] = Field(None, description="Retrieved knowledge given to LLM")
    source: Optional[str] = Field(None, description="Source of knowledge")
    relevance_score: Optional[float] = Field(None, description="Relevance score 0-1")


class TokenUsage(BaseModel):
    """Token usage metrics for LLM calls."""
    
    input_tokens: int = Field(0, description="Input tokens")
    output_tokens: int = Field(0, description="Output tokens")
    total_tokens: int = Field(0, description="Total tokens")


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


class ModelFailure(BaseModel):
    """Details of a model failure."""
    
    model_name: str = Field(..., description="Name of the failed model")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = Field(0, description="Number of retries attempted")
    prompt_preview: Optional[str] = Field(None, description="Preview of prompt that failed")


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
    """
    
    # Session Metadata
    domain: str = Field(..., description="Domain of the session")
    issue_id: str = Field(..., description="Unique identifier for the issue")
    message_id: Optional[str] = Field(
        None, 
        description="Unique identifier for the message (null for issue-level)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="Message timestamp"
    )
    
    # User Message
    user_message: str = Field(..., description="Content of the user message")
    
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
    
    # Errors/Retries
    tool_failures: List[ToolFailure] = Field(
        default_factory=list,
        description="Details of tool failures"
    )
    model_failures: List[ModelFailure] = Field(
        default_factory=list,
        description="Details of model failures"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================
# OUTPUT SCHEMA COMPONENTS
# ============================================================

class ToolSelectionMetrics(BaseModel):
    """Tool selection evaluation metrics."""
    
    in_order: float = Field(0.0, ge=0.0, le=1.0, description="In-order match score")
    any_order: float = Field(0.0, ge=0.0, le=1.0, description="Any-order match score")
    exact_match: float = Field(0.0, ge=0.0, le=1.0, description="Exact match score")
    precision: float = Field(0.0, ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(0.0, ge=0.0, le=1.0, description="Recall score")


# ============================================================
# MAIN OUTPUT SCHEMA
# ============================================================

class EvaluationOutput(BaseModel):
    """
    Complete output schema for agent evaluation.
    
    Contains all evaluation results:
    - Evaluation metadata (id, timestamp, judge info)
    - Session info (domain, issue_id, message_id)
    - Conversation statistics (turn counts, message counts)
    - Quality scores (safety, PII, hallucination, action completion)
    - Tool selection metrics
    - Full explanations
    """
    
    # Evaluation Metadata
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of evaluation"
    )
    latency: float = Field(0.0, description="Latency in seconds")
    time_to_first_token: Optional[float] = Field(
        None,
        alias="ttft",
        description="Time to first token (TTFT)"
    )
    
    # Judge Information
    judge_model: str = Field(..., description="Model used for judging")
    judge_prompt_version: str = Field("v1.0", description="Version of judge prompt")
    judge_input_tokens: int = Field(0, description="Judge input tokens")
    judge_output_tokens: int = Field(0, description="Judge output tokens")
    judge_total_tokens: int = Field(0, description="Judge total tokens")
    judge_total_tokens_cost: float = Field(0.0, description="Judge total tokens cost")
    
    # Session Information
    domain: str = Field(..., description="Domain")
    issue_id: str = Field(..., description="Issue identifier")
    message_id: Optional[str] = Field(None, description="Unique identifier for messages")
    agent_version: str = Field("1.0.0", description="Version of the agent")
    
    # Conversation Statistics
    turn_count: int = Field(0, description="Total conversation turns")
    count_user_msg: int = Field(0, description="Count of user messages")
    count_assistant_msg: int = Field(0, description="Count of assistant messages")
    count_tool_call: int = Field(0, description="Count of tool calls")
    count_tool_error: int = Field(0, description="Count of tool errors")
    count_model_call: int = Field(0, description="Count of model calls")
    count_model_error: int = Field(0, description="Count of model errors")
    
    # Conversation Content
    conversation: str = Field("", description="Full conversation transcript")
    kb: Optional[Dict[str, Any]] = Field(None, description="Knowledge base content used")
    
    # Quality Scores (0.0 - 1.0)
    safety_score: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0, 
        description="Safety evaluation score"
    )
    pii_score: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0, 
        description="PII detection score (higher = less PII leaked)"
    )
    hallucination_score: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0, 
        description="Hallucination score (higher = less hallucination)"
    )
    action_completion_score: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0, 
        description="Action completion score"
    )
    
    # Tool Selection Metrics
    tool_selection_metrics: ToolSelectionMetrics = Field(
        default_factory=ToolSelectionMetrics,
        description="Tool selection metrics (in_order, any_order, exact_match, precision, recall)"
    )
    
    # Explanations
    evaluation_explanations: str = Field(
        "",
        description="Short explanations for metric scores"
    )
    
    # Verdict
    verdict: EvaluationVerdict = Field(
        EvaluationVerdict.FAIL,
        description="Overall evaluation verdict"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        populate_by_name = True


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
    
    return EvaluationInput(
        domain=domain,
        issue_id=issue_id,
        user_message=user_message,
        tool_info=tool_info,
        response=response,
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


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Enums
    "ToolCallStatus",
    "EvaluationVerdict",
    # Input Components
    "SessionMetadata",
    "UserMessage",
    "ToolCallInfo",
    "KnowledgeBase",
    "TokenUsage",
    "LLMCallInfo",
    "ToolFailure",
    "ModelFailure",
    # Main Schemas
    "EvaluationInput",
    "EvaluationOutput",
    "ToolSelectionMetrics",
    # Helpers
    "create_evaluation_input",
    "create_evaluation_output"
]
