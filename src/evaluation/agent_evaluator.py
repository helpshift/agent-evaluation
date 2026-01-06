"""
Agent Evaluation Module using Official Vertex AI GenAI SDK.

This module provides comprehensive agent evaluation including:
- Agent inference via Agent Engine
- Trajectory-based evaluation
- Tool use quality assessment
- Hallucination detection
- Final response quality
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from vertexai import types

from src.evaluation.genai_client import get_client, GCS_DEST_BUCKET
from src.evaluation.rubric_evaluator import (
    AgentMetrics,
    StaticRubricMetrics,
    GenAIEvaluator,
)


# ============================================================
# AGENT DATASET PREPARATION
# ============================================================

def create_agent_dataset(
    prompts: List[str],
    user_id: str = "eval_user",
    initial_state: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Create a properly formatted dataset for agent evaluation.
    
    Args:
        prompts: List of user prompts
        user_id: User identifier for session tracking
        initial_state: Initial state for each session
    
    Returns:
        DataFrame ready for agent evaluation
    """
    session_inputs = types.evals.SessionInput(
        user_id=user_id,
        state=initial_state or {},
    )
    
    return pd.DataFrame({
        "prompt": prompts,
        "session_inputs": [session_inputs] * len(prompts),
    })


def create_agent_info_from_adk_agent(adk_agent, resource_name: Optional[str] = None):
    """
    Create AgentInfo from an ADK agent for evaluation.
    
    Args:
        adk_agent: ADK Agent instance
        resource_name: Optional deployed agent resource name
    
    Returns:
        AgentInfo for evaluation
    """
    return types.evals.AgentInfo.load_from_agent(adk_agent, resource_name)


# ============================================================
# TRAJECTORY METRICS (COMPUTATION-BASED)
# ============================================================

class TrajectoryMetrics:
    """
    Computation-based trajectory evaluation metrics.
    These compare predicted_trajectory against reference_trajectory.
    """
    
    @staticmethod
    def exact_match() -> Any:
        """
        TRAJECTORY_EXACT_MATCH - Identical trajectory match.
        Score: 1 if predicted == reference, else 0
        """
        return types.Metric(name='trajectory_exact_match')
    
    @staticmethod
    def in_order_match() -> Any:
        """
        TRAJECTORY_IN_ORDER_MATCH - All actions in correct order.
        Allows extra actions but requires correct sequence.
        """
        return types.Metric(name='trajectory_in_order_match')
    
    @staticmethod
    def any_order_match() -> Any:
        """
        TRAJECTORY_ANY_ORDER_MATCH - All actions present, any order.
        More lenient than in_order_match.
        """
        return types.Metric(name='trajectory_any_order_match')
    
    @staticmethod
    def precision() -> Any:
        """
        TRAJECTORY_PRECISION - Correct actions / total predicted.
        Measures accuracy of agent's tool choices.
        """
        return types.Metric(name='trajectory_precision')
    
    @staticmethod
    def recall() -> Any:
        """
        TRAJECTORY_RECALL - Found actions / expected actions.
        Measures completeness of agent's tool usage.
        """
        return types.Metric(name='trajectory_recall')
    
    @staticmethod
    def single_tool_use(tool_name: str) -> Any:
        """
        TRAJECTORY_SINGLE_TOOL_USE - Check if specific tool was used.
        
        Args:
            tool_name: Name of the tool to check for
        
        Returns:
            Metric that scores 1 if tool was used, 0 otherwise
        """
        return types.Metric(
            name='trajectory_single_tool_use',
            config={'tool_name': tool_name}
        )


# ============================================================
# TRAJECTORY DATASET PREPARATION
# ============================================================

def create_trajectory_dataset(
    predicted_trajectories: List[List[Dict[str, Any]]],
    reference_trajectories: List[List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Create dataset for trajectory evaluation.
    
    Args:
        predicted_trajectories: List of actual tool call sequences
        reference_trajectories: List of expected tool call sequences
    
    Each trajectory is a list of dicts with:
        - tool_name: str
        - tool_input: dict
    
    Returns:
        DataFrame for trajectory evaluation
    
    Example:
        predicted = [[
            {"tool_name": "search", "tool_input": {"query": "weather"}}
        ]]
        reference = [[
            {"tool_name": "search", "tool_input": {"query": "weather"}}
        ]]
        df = create_trajectory_dataset(predicted, reference)
    """
    return pd.DataFrame({
        "predicted_trajectory": predicted_trajectories,
        "reference_trajectory": reference_trajectories,
    })


# ============================================================
# FULL AGENT EVALUATOR
# ============================================================

class AgentEvaluator:
    """
    Complete agent evaluator using official Vertex AI GenAI SDK.
    
    Supports:
    - Deployed agents on Agent Engine
    - Local ADK agents
    - Trajectory evaluation
    - Response quality evaluation
    """
    
    def __init__(self, dest: Optional[str] = None):
        """
        Initialize the agent evaluator.
        
        Args:
            dest: GCS destination for saving results
        """
        self.client = get_client()
        self.dest = dest or GCS_DEST_BUCKET
        self.base_evaluator = GenAIEvaluator(dest)
    
    def run_agent_inference(
        self,
        agent_resource_name: str,
        prompts: List[str],
        user_id: str = "eval_user",
        initial_state: Optional[Dict[str, Any]] = None
    ):
        """
        Run inference on a deployed agent.
        
        Args:
            agent_resource_name: Agent Engine resource name
            prompts: List of prompts to test
            user_id: User ID for sessions
            initial_state: Initial session state
        
        Returns:
            EvaluationDataset with responses and intermediate_events
        """
        dataset = create_agent_dataset(prompts, user_id, initial_state)
        
        return self.client.evals.run_inference(
            agent=agent_resource_name,
            src=dataset,
        )
    
    def evaluate_agent(
        self,
        dataset_with_responses,
        agent_info,
        include_safety: bool = True,
        include_hallucination: bool = True
    ):
        """
        Evaluate agent responses with comprehensive metrics.
        
        Args:
            dataset_with_responses: Dataset from run_agent_inference
            agent_info: AgentInfo from create_agent_info_from_adk_agent
            include_safety: Include safety evaluation
            include_hallucination: Include hallucination detection
        
        Returns:
            Evaluation run with .show() for visualization
        """
        metrics = [
            AgentMetrics.final_response_quality(),
            AgentMetrics.tool_use_quality(),
        ]
        
        if include_hallucination:
            metrics.append(AgentMetrics.hallucination())
        
        if include_safety:
            metrics.append(StaticRubricMetrics.safety())
        
        return self.client.evals.create_evaluation_run(
            dataset=dataset_with_responses,
            agent_info=agent_info,
            metrics=metrics,
            dest=self.dest,
        )
    
    def evaluate_trajectories(
        self,
        predicted: List[List[Dict[str, Any]]],
        reference: List[List[Dict[str, Any]]],
        metrics: Optional[List[Any]] = None
    ):
        """
        Evaluate agent trajectories against reference.
        
        Args:
            predicted: Actual tool call sequences
            reference: Expected tool call sequences
            metrics: Optional custom metrics (defaults to all trajectory metrics)
        
        Returns:
            Trajectory evaluation result
        """
        dataset = create_trajectory_dataset(predicted, reference)
        
        if metrics is None:
            metrics = [
                TrajectoryMetrics.exact_match(),
                TrajectoryMetrics.in_order_match(),
                TrajectoryMetrics.any_order_match(),
                TrajectoryMetrics.precision(),
                TrajectoryMetrics.recall(),
            ]
        
        return self.client.evals.evaluate(
            dataset=dataset,
            metrics=metrics,
        )
    
    def full_evaluation(
        self,
        agent_resource_name: str,
        adk_agent,
        prompts: List[str],
        reference_trajectories: Optional[List[List[Dict[str, Any]]]] = None
    ):
        """
        Run complete agent evaluation pipeline.
        
        Args:
            agent_resource_name: Deployed agent resource name
            adk_agent: Original ADK agent for AgentInfo
            prompts: Test prompts
            reference_trajectories: Optional expected trajectories
        
        Returns:
            Complete evaluation results
        """
        # Step 1: Run inference
        inference_result = self.run_agent_inference(
            agent_resource_name=agent_resource_name,
            prompts=prompts,
        )
        
        # Step 2: Create agent info
        agent_info = create_agent_info_from_adk_agent(adk_agent, agent_resource_name)
        
        # Step 3: Run evaluation
        eval_result = self.evaluate_agent(
            dataset_with_responses=inference_result,
            agent_info=agent_info,
        )
        
        # Step 4: Optional trajectory evaluation
        trajectory_result = None
        if reference_trajectories:
            # Extract predicted trajectories from inference result
            predicted = self._extract_trajectories(inference_result)
            trajectory_result = self.evaluate_trajectories(predicted, reference_trajectories)
        
        return {
            "inference": inference_result,
            "evaluation": eval_result,
            "trajectory": trajectory_result,
        }
    
    def _extract_trajectories(self, inference_result) -> List[List[Dict[str, Any]]]:
        """Extract trajectories from inference result intermediate_events."""
        trajectories = []
        
        # Access the DataFrame
        df = inference_result.to_dataframe() if hasattr(inference_result, 'to_dataframe') else inference_result
        
        for _, row in df.iterrows():
            trajectory = []
            events = row.get('intermediate_events', [])
            
            for event in events:
                if 'function_call' in event:
                    trajectory.append({
                        'tool_name': event['function_call'].get('name'),
                        'tool_input': event['function_call'].get('args', {}),
                    })
            
            trajectories.append(trajectory)
        
        return trajectories


# ============================================================
# QUICK EVALUATION FUNCTIONS
# ============================================================

def quick_agent_eval(
    agent_resource_name: str,
    prompts: List[str],
    adk_agent=None
):
    """
    Quick one-liner agent evaluation.
    
    Example:
        result = quick_agent_eval(
            "projects/xxx/locations/xxx/agents/xxx",
            ["What's the weather?", "Book a flight"]
        )
        result.show()
    """
    evaluator = AgentEvaluator()
    
    # Run inference
    inference = evaluator.run_agent_inference(agent_resource_name, prompts)
    
    # Get agent info if ADK agent provided
    agent_info = None
    if adk_agent:
        agent_info = create_agent_info_from_adk_agent(adk_agent, agent_resource_name)
    
    # Evaluate
    if agent_info:
        return evaluator.evaluate_agent(inference, agent_info)
    else:
        return evaluator.base_evaluator.evaluate(
            dataset=inference,
            metrics=[
                StaticRubricMetrics.safety(),
                StaticRubricMetrics.grounding(),
            ]
        )


def quick_trajectory_eval(
    predicted: List[List[Dict[str, Any]]],
    reference: List[List[Dict[str, Any]]]
):
    """
    Quick trajectory comparison.
    
    Example:
        result = quick_trajectory_eval(
            [[{"tool_name": "search", "tool_input": {}}]],
            [[{"tool_name": "search", "tool_input": {}}]]
        )
        result.show()
    """
    evaluator = AgentEvaluator()
    return evaluator.evaluate_trajectories(predicted, reference)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Dataset Preparation
    "create_agent_dataset",
    "create_agent_info_from_adk_agent",
    "create_trajectory_dataset",
    
    # Metrics
    "TrajectoryMetrics",
    
    # Evaluator
    "AgentEvaluator",
    
    # Quick Functions
    "quick_agent_eval",
    "quick_trajectory_eval",
]
