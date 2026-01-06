"""
Rubric-Based Evaluation Module using Official Vertex AI GenAI SDK.

This module provides comprehensive rubric-based evaluation including:
- Adaptive rubrics (auto-generated per prompt)
- Static rubrics (fixed scoring criteria)
- Agent-specific metrics (FINAL_RESPONSE_QUALITY, TOOL_USE_QUALITY, etc.)
- Computation-based metrics (BLEU, ROUGE, exact_match)
- Custom function metrics
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from vertexai import types

from src.evaluation.genai_client import get_client, GCS_DEST_BUCKET


# ============================================================
# METRIC TYPE DEFINITIONS
# ============================================================

@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""
    metrics: List[Any]
    dest: Optional[str] = None
    agent_info: Optional[Any] = None
    include_rubric_generation: bool = True


# ============================================================
# ADAPTIVE RUBRIC METRICS
# ============================================================

class AdaptiveRubricMetrics:
    """
    Adaptive rubrics dynamically generate pass/fail tests for each prompt.
    These are unit-test-like evaluations tailored to each specific input.
    """
    
    @staticmethod
    def general_quality(guidelines: Optional[str] = None) -> Any:
        """
        GENERAL_QUALITY - Comprehensive adaptive rubric covering:
        - Instruction following
        - Formatting
        - Tone/Style
        
        Args:
            guidelines: Optional natural language guidelines to focus rubrics.
        
        Returns:
            RubricMetric configuration for evaluation.
        """
        if guidelines:
            return types.RubricMetric.GENERAL_QUALITY(
                metric_spec_parameters={"guidelines": guidelines}
            )
        return types.RubricMetric.GENERAL_QUALITY
    
    @staticmethod
    def text_quality() -> Any:
        """
        TEXT_QUALITY - Linguistic quality evaluation:
        - Fluency
        - Coherence
        - Grammar
        """
        return types.RubricMetric.TEXT_QUALITY
    
    @staticmethod
    def instruction_following() -> Any:
        """
        INSTRUCTION_FOLLOWING - Adherence to constraints:
        - Follows specific requirements
        - Meets format specifications
        """
        return types.RubricMetric.INSTRUCTION_FOLLOWING


# ============================================================
# STATIC RUBRIC METRICS
# ============================================================

class StaticRubricMetrics:
    """
    Static rubrics apply fixed scoring guidelines to all examples.
    Useful for consistent benchmarking across prompts.
    """
    
    @staticmethod
    def grounding() -> Any:
        """
        GROUNDING - Factuality check against provided context.
        Score: 0-1 (rate of claims that are supported or no_rad)
        """
        return types.RubricMetric.GROUNDING
    
    @staticmethod
    def safety() -> Any:
        """
        SAFETY - Check for policy violations:
        - PII & Demographic Data
        - Hate Speech
        - Dangerous Content
        - Harassment
        - Sexually Explicit
        
        Score: 0 (unsafe) or 1 (safe)
        """
        return types.RubricMetric.SAFETY
    
    @staticmethod
    def fluency() -> Any:
        """FLUENCY - Language mastery evaluation."""
        return types.RubricMetric.FLUENCY
    
    @staticmethod
    def coherence() -> Any:
        """COHERENCE - Logical flow and organization."""
        return types.RubricMetric.COHERENCE


# ============================================================
# AGENT-SPECIFIC METRICS
# ============================================================

class AgentMetrics:
    """
    Metrics designed specifically for evaluating AI agents.
    Require agent_info and intermediate_events in the dataset.
    """
    
    @staticmethod
    def final_response_match() -> Any:
        """
        FINAL_RESPONSE_MATCH - Compare agent answer to reference.
        Score: 1 (match) or 0 (no match)
        """
        return types.RubricMetric.FINAL_RESPONSE_MATCH
    
    @staticmethod
    def final_response_quality() -> Any:
        """
        FINAL_RESPONSE_QUALITY - Adaptive rubric based on:
        - Agent configuration (developer instruction)
        - Tool declarations
        - Intermediate events (tool usage)
        """
        return types.RubricMetric.FINAL_RESPONSE_QUALITY
    
    @staticmethod
    def final_response_reference_free() -> Any:
        """
        FINAL_RESPONSE_REFERENCE_FREE - Quality without reference answer.
        Note: Requires pre-generated rubrics.
        """
        return types.RubricMetric.FINAL_RESPONSE_REFERENCE_FREE
    
    @staticmethod
    def tool_use_quality() -> Any:
        """
        TOOL_USE_QUALITY - Evaluate tool usage:
        - Correct tool selection
        - Correct parameters
        - Proper sequence
        """
        return types.RubricMetric.TOOL_USE_QUALITY
    
    @staticmethod
    def hallucination() -> Any:
        """
        HALLUCINATION - Grounding check based on tool responses.
        Segments response into atomic claims and verifies each.
        Score: 0-1 (rate of supported claims)
        """
        return types.RubricMetric.HALLUCINATION


# ============================================================
# MULTI-TURN METRICS
# ============================================================

class MultiTurnMetrics:
    """
    Metrics for evaluating multi-turn conversations.
    Automatically extracts conversation_history from supported formats.
    """
    
    @staticmethod
    def general_quality() -> Any:
        """MULTI_TURN_GENERAL_QUALITY - Overall conversational quality."""
        return types.RubricMetric.MULTI_TURN_GENERAL_QUALITY
    
    @staticmethod
    def text_quality() -> Any:
        """MULTI_TURN_TEXT_QUALITY - Text quality within dialogue."""
        return types.RubricMetric.MULTI_TURN_TEXT_QUALITY


# ============================================================
# COMPUTATION-BASED METRICS
# ============================================================

class ComputationMetrics:
    """
    Deterministic metrics using ground truth comparison.
    Require 'reference' column in dataset.
    """
    
    @staticmethod
    def bleu() -> Any:
        """BLEU - N-gram matching for translation quality."""
        return types.Metric(name='bleu')
    
    @staticmethod
    def rouge_l() -> Any:
        """ROUGE-L - Longest common subsequence overlap."""
        return types.Metric(name='rouge_l')
    
    @staticmethod
    def rouge_1() -> Any:
        """ROUGE-1 - Unigram overlap."""
        return types.Metric(name='rouge_1')
    
    @staticmethod
    def exact_match() -> Any:
        """EXACT_MATCH - Identical response to reference."""
        return types.Metric(name='exact_match')


# ============================================================
# CUSTOM FUNCTION METRICS
# ============================================================

def create_custom_metric(name: str, func: Callable[[dict], dict]) -> Any:
    """
    Create a custom metric using a Python function.
    
    Args:
        name: Metric name for results
        func: Function that takes instance dict and returns {"score": float}
    
    Returns:
        Custom metric configuration
    
    Example:
        def keyword_check(instance):
            return {"score": 1.0 if "magic" in instance.get("response", "") else 0.0}
        
        metric = create_custom_metric("keyword_check", keyword_check)
    """
    return types.Metric(name=name, custom_function=func)


def create_llm_metric(
    name: str,
    instruction: str,
    criteria: Dict[str, str],
    rating_scores: Dict[str, str]
) -> Any:
    """
    Create a custom LLM-based metric with static rubric.
    
    Args:
        name: Metric name
        instruction: Task instruction for the judge model
        criteria: Dict of criterion name -> description
        rating_scores: Dict of score -> meaning
    
    Returns:
        LLMMetric configuration
    
    Example:
        metric = create_llm_metric(
            name="simplicity",
            instruction="Evaluate story simplicity for a 5-year-old",
            criteria={"Vocabulary": "Uses simple words"},
            rating_scores={"5": "Excellent", "1": "Very Poor"}
        )
    """
    return types.LLMMetric(
        name=name,
        prompt_template=types.MetricPromptBuilder(
            instruction=instruction,
            criteria=criteria,
            rating_scores=rating_scores
        )
    )


# ============================================================
# MAIN EVALUATION RUNNER
# ============================================================

class GenAIEvaluator:
    """
    Main evaluator class using official Vertex AI GenAI SDK.
    
    Features:
    - Model and agent evaluation
    - Rubric generation and validation
    - Batch evaluation for large datasets
    - Interactive results visualization
    """
    
    def __init__(self, dest: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            dest: GCS destination for saving results
        """
        self.client = get_client()
        self.dest = dest or GCS_DEST_BUCKET
    
    def evaluate(
        self,
        dataset: pd.DataFrame,
        metrics: List[Any],
        agent_info: Optional[Any] = None
    ):
        """
        Run evaluation on a dataset.
        
        Args:
            dataset: DataFrame with prompt, response, (optional) reference columns
            metrics: List of metric configurations (RubricMetric, Metric, etc.)
            agent_info: Optional AgentInfo for agent-specific metrics
        
        Returns:
            EvaluationResult with .show() method for visualization
        """
        if agent_info:
            return self.client.evals.create_evaluation_run(
                dataset=dataset,
                agent_info=agent_info,
                metrics=metrics,
                dest=self.dest
            )
        else:
            return self.client.evals.evaluate(
                dataset=dataset,
                metrics=metrics
            )
    
    def generate_rubrics(
        self,
        dataset: pd.DataFrame,
        rubric_group_name: str,
        metric: Any = None
    ):
        """
        Pre-generate rubrics for review before evaluation.
        
        Args:
            dataset: DataFrame with prompts
            rubric_group_name: Name for the rubric group
            metric: Predefined metric spec (defaults to GENERAL_QUALITY)
        
        Returns:
            Dataset with rubric_groups column
        """
        predefined_spec = metric or types.RubricMetric.GENERAL_QUALITY
        
        return self.client.evals.generate_rubrics(
            src=dataset,
            rubric_group_name=rubric_group_name,
            predefined_spec_name=predefined_spec
        )
    
    def batch_evaluate(
        self,
        dataset: pd.DataFrame,
        metrics: List[Any]
    ):
        """
        Asynchronous batch evaluation for large datasets.
        
        Args:
            dataset: Large DataFrame
            metrics: Metrics to evaluate
        
        Returns:
            Batch evaluation job that can be polled for status
        """
        return self.client.evals.batch_evaluate(
            dataset=dataset,
            metrics=metrics,
            dest=self.dest
        )
    
    def compare_models(
        self,
        prompts: pd.DataFrame,
        models: List[str],
        metrics: List[Any]
    ):
        """
        Compare multiple models on the same prompts.
        
        Args:
            prompts: DataFrame with prompts
            models: List of model names to compare
            metrics: Evaluation metrics
        
        Returns:
            Comparison result with win/tie rates
        """
        inference_results = []
        for model in models:
            result = self.client.evals.run_inference(
                model=model,
                src=prompts
            )
            inference_results.append(result)
        
        return self.client.evals.evaluate(
            dataset=inference_results,
            metrics=metrics
        )


# ============================================================
# PRESET METRIC BUNDLES
# ============================================================

class MetricPresets:
    """Pre-configured metric bundles for common use cases."""
    
    @staticmethod
    def text_generation() -> List[Any]:
        """Metrics for general text generation tasks."""
        return [
            AdaptiveRubricMetrics.general_quality(),
            AdaptiveRubricMetrics.text_quality(),
            StaticRubricMetrics.safety(),
        ]
    
    @staticmethod
    def summarization() -> List[Any]:
        """Metrics for summarization tasks."""
        return [
            AdaptiveRubricMetrics.instruction_following(),
            StaticRubricMetrics.grounding(),
            ComputationMetrics.rouge_l(),
        ]
    
    @staticmethod
    def agent_evaluation() -> List[Any]:
        """Metrics for AI agent evaluation."""
        return [
            AgentMetrics.final_response_quality(),
            AgentMetrics.tool_use_quality(),
            AgentMetrics.hallucination(),
            StaticRubricMetrics.safety(),
        ]
    
    @staticmethod
    def qa_with_reference() -> List[Any]:
        """Metrics for QA with ground truth."""
        return [
            AdaptiveRubricMetrics.instruction_following(),
            ComputationMetrics.exact_match(),
            ComputationMetrics.bleu(),
        ]
    
    @staticmethod
    def multi_turn_chat() -> List[Any]:
        """Metrics for multi-turn conversations."""
        return [
            MultiTurnMetrics.general_quality(),
            MultiTurnMetrics.text_quality(),
            StaticRubricMetrics.safety(),
        ]


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Metric Classes
    "AdaptiveRubricMetrics",
    "StaticRubricMetrics",
    "AgentMetrics",
    "MultiTurnMetrics",
    "ComputationMetrics",
    
    # Custom Metric Builders
    "create_custom_metric",
    "create_llm_metric",
    
    # Main Evaluator
    "GenAIEvaluator",
    
    # Presets
    "MetricPresets",
    
    # Config
    "EvaluationConfig",
]
