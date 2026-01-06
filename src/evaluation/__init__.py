"""
Evaluation module for Agent Evaluation Framework.

Provides:
- Schemas: Input/Output Pydantic models
- Evaluators: Trajectory, LLM Judge, Comprehensive
- GenAI SDK: Official Vertex AI GenAI Evaluation Client
- Batch evaluation utilities
"""

from .schemas import (
    # Enums
    ToolCallStatus,
    EvaluationVerdict,
    # Input Components
    SessionMetadata,
    UserMessage,
    ToolCallInfo,
    KnowledgeBase,
    TokenUsage,
    LLMCallInfo,
    ToolFailure,
    ModelFailure,
    # Main Schemas
    EvaluationInput,
    EvaluationOutput,
    ToolSelectionMetrics,
    # Helpers
    create_evaluation_input,
    create_evaluation_output
)

from .evaluator import (
    TrajectoryEvaluator,
    LLMJudgeEvaluator,
    ComprehensiveEvaluator,
    evaluate_batch,
    generate_summary_report,
    JUDGE_MODEL,
    JUDGE_PROMPT_VERSION
)

# New GenAI SDK Modules
from .genai_client import (
    get_genai_client,
    get_client,
    run_model_inference,
    run_agent_inference,
)

from .rubric_evaluator import (
    AdaptiveRubricMetrics,
    StaticRubricMetrics,
    AgentMetrics,
    MultiTurnMetrics,
    ComputationMetrics,
    GenAIEvaluator,
    MetricPresets,
    create_custom_metric,
    create_llm_metric,
)

from .agent_evaluator import (
    AgentEvaluator,
    TrajectoryMetrics,
    create_agent_dataset,
    create_agent_info_from_adk_agent,
    create_trajectory_dataset,
    quick_agent_eval,
    quick_trajectory_eval,
)

from .multi_turn_evaluator import (
    MultiTurnEvaluator,
    create_multi_turn_dataset,
    format_gemini_conversation,
    format_openai_conversation,
    quick_multi_turn_eval,
)

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
    "create_evaluation_output",
    # Evaluators
    "TrajectoryEvaluator",
    "LLMJudgeEvaluator",
    "ComprehensiveEvaluator",
    "evaluate_batch",
    "generate_summary_report",
    "JUDGE_MODEL",
    "JUDGE_PROMPT_VERSION",
    # GenAI Client
    "get_genai_client",
    "get_client",
    "run_model_inference",
    "run_agent_inference",
    # Rubric Metrics
    "AdaptiveRubricMetrics",
    "StaticRubricMetrics",
    "AgentMetrics",
    "MultiTurnMetrics",
    "ComputationMetrics",
    "GenAIEvaluator",
    "MetricPresets",
    "create_custom_metric",
    "create_llm_metric",
    # Agent Evaluation
    "AgentEvaluator",
    "TrajectoryMetrics",
    "create_agent_dataset",
    "create_agent_info_from_adk_agent",
    "create_trajectory_dataset",
    "quick_agent_eval",
    "quick_trajectory_eval",
    # Multi-Turn Evaluation
    "MultiTurnEvaluator",
    "create_multi_turn_dataset",
    "format_gemini_conversation",
    "format_openai_conversation",
    "quick_multi_turn_eval",
]

