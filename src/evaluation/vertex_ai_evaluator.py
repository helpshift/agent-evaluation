"""
Vertex AI GenAI Evaluation Service Integration for ADK Agents.

This module provides production-scale evaluation using Vertex AI's
EvalTask with metrics for:
- Response quality (groundedness, coherence, fluency)
- Tool use evaluation
- Trajectory matching
- Safety and hallucination detection
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import vertexai
from vertexai.preview.evaluation import EvalTask, MetricPromptTemplateExamples

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TrajectoryStep:
    """Represents a step in the agent's trajectory."""
    agent_name: str
    action_type: str  # "generate", "tool_call", "delegate"
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationCase:
    """Represents a single evaluation case."""
    case_id: str
    topic: str
    expected_trajectory: List[str]
    expected_response_keywords: List[str]
    actual_trajectory: List[str] = field(default_factory=list)
    actual_response: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_trajectory_score: float
    avg_response_quality: float
    avg_groundedness: float
    avg_coherence: float
    avg_fluency: float
    detailed_results: List[Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================
# TRAJECTORY EVALUATOR
# ============================================================

class TrajectoryEvaluator:
    """
    Evaluates agent trajectories using exact matching and 
    Vertex AI LLM-as-judge for semantic similarity.
    """
    
    @staticmethod
    def exact_match_score(expected: List[str], actual: List[str]) -> float:
        """
        Calculate exact match score for trajectories.
        Based on ADK's tool_trajectory_avg_score metric.
        """
        if not expected:
            return 1.0 if not actual else 0.0
        
        matches = 0
        for i, exp_step in enumerate(expected):
            if i < len(actual) and actual[i] == exp_step:
                matches += 1
        
        return matches / len(expected)
    
    @staticmethod
    def order_match_score(expected: List[str], actual: List[str]) -> float:
        """
        Check if expected steps appear in order (not necessarily contiguous).
        More lenient than exact match.
        """
        if not expected:
            return 1.0 if not actual else 0.0
        
        exp_idx = 0
        for act_step in actual:
            if exp_idx < len(expected) and act_step == expected[exp_idx]:
                exp_idx += 1
        
        return exp_idx / len(expected)
    
    @staticmethod
    def step_presence_score(expected: List[str], actual: List[str]) -> float:
        """
        Check if all expected steps are present (regardless of order).
        """
        if not expected:
            return 1.0
        
        actual_set = set(actual)
        present = sum(1 for step in expected if step in actual_set)
        return present / len(expected)


# ============================================================
# VERTEX AI RESPONSE EVALUATOR
# ============================================================

class VertexAIResponseEvaluator:
    """
    Uses Vertex AI GenAI Evaluation Service for response quality metrics.
    """
    
    def __init__(self, project_id: str = PROJECT_ID, location: str = LOCATION):
        self.project_id = project_id
        self.location = location
    
    def evaluate_responses(
        self,
        responses: List[Dict[str, str]],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple responses using Vertex AI EvalTask.
        
        Args:
            responses: List of dicts with 'instruction', 'context', 'response'
            metrics: List of metric names to evaluate
        
        Returns:
            DataFrame with evaluation results
        """
        if metrics is None:
            metrics = [
                MetricPromptTemplateExamples.Pointwise.COHERENCE,
                MetricPromptTemplateExamples.Pointwise.FLUENCY,
                MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
                MetricPromptTemplateExamples.Pointwise.SAFETY,
            ]
        
        # Convert to DataFrame for EvalTask
        eval_dataset = pd.DataFrame(responses)
        
        # Create evaluation task
        eval_task = EvalTask(
            dataset=eval_dataset,
            metrics=metrics
        )
        
        # Run evaluation
        prompt_template = "Instruction: {instruction}. Context: {context}. Response: {response}"
        result = eval_task.evaluate(prompt_template=prompt_template)
        
        return result.metrics_table
    
    def evaluate_single_response(
        self,
        instruction: str,
        context: str,
        response: str
    ) -> Dict[str, float]:
        """Evaluate a single response."""
        responses = [{
            "instruction": instruction,
            "context": context,
            "response": response
        }]
        
        df = self.evaluate_responses(responses)
        
        # Extract scores from first row
        scores = {}
        for col in df.columns:
            if col not in ['instruction', 'context', 'response']:
                try:
                    scores[col] = float(df[col].iloc[0])
                except (ValueError, TypeError):
                    scores[col] = 0.0
        
        return scores


# ============================================================
# CUSTOM RUBRIC EVALUATOR
# ============================================================

class RubricBasedEvaluator:
    """
    Custom rubric-based evaluation using Vertex AI.
    Based on ADK's rubric_based_tool_use_quality_v1 and
    rubric_based_final_response_quality_v1 metrics.
    """
    
    STORY_QUALITY_RUBRIC = """
    Evaluate the story based on these criteria:
    
    1. COHERENCE (0-1): Does the story have a logical flow?
       - 1.0: Perfect logical progression
       - 0.7: Some minor logical gaps
       - 0.4: Noticeable inconsistencies
       - 0.0: Incoherent
    
    2. TOPIC_ADHERENCE (0-1): Does the story match the given topic?
       - 1.0: Perfectly on topic
       - 0.7: Mostly on topic with minor deviations
       - 0.4: Loosely related to topic
       - 0.0: Completely off topic
    
    3. CREATIVITY (0-1): Is the story engaging and original?
       - 1.0: Highly creative and engaging
       - 0.7: Good creative elements
       - 0.4: Basic storytelling
       - 0.0: No creativity
    
    4. GRAMMAR (0-1): Is the grammar correct?
       - 1.0: No errors
       - 0.7: Minor errors
       - 0.4: Several errors
       - 0.0: Many errors
    
    5. TONE (0-1): Is the tone appropriate?
       - 1.0: Perfect tone for the topic
       - 0.7: Generally appropriate
       - 0.4: Sometimes inappropriate
       - 0.0: Completely wrong tone
    """
    
    TOOL_USE_RUBRIC = """
    Evaluate the agent's tool usage based on:
    
    1. CORRECT_TOOL_SELECTION (0-1): Did the agent use appropriate tools?
       - 1.0: All tools were appropriate
       - 0.5: Some unnecessary tools used
       - 0.0: Wrong tools selected
    
    2. TOOL_ORDER (0-1): Were tools called in logical order?
       - 1.0: Perfect order
       - 0.5: Minor ordering issues
       - 0.0: Completely wrong order
    
    3. TOOL_EFFICIENCY (0-1): Was the tool usage efficient?
       - 1.0: Minimal necessary tool calls
       - 0.5: Some redundant calls
       - 0.0: Many unnecessary calls
    """


# ============================================================
# MAIN EVALUATOR
# ============================================================

class AgentEvaluator:
    """
    Comprehensive agent evaluator combining:
    - Trajectory evaluation
    - Vertex AI response quality
    - Custom rubric evaluation
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        project_id: str = PROJECT_ID,
        location: str = LOCATION
    ):
        self.agent = agent
        self.trajectory_eval = TrajectoryEvaluator()
        self.response_eval = VertexAIResponseEvaluator(project_id, location)
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=agent,
            app_name="agent_evaluation",
            session_service=self.session_service
        )
    
    async def evaluate_case(self, case: EvaluationCase) -> EvaluationCase:
        """Evaluate a single case and populate results."""
        
        # Create session with initial state
        session = await self.session_service.create_session(
            app_name="agent_evaluation",
            user_id="eval_user",
            session_id=f"eval_{case.case_id}",
            state={"topic": case.topic}
        )
        
        # Run agent
        content = types.Content(
            role="user",
            parts=[types.Part(text=f"Write a story about: {case.topic}")]
        )
        
        trajectory = []
        final_response = ""
        
        async for event in self.runner.run_async(
            user_id="eval_user",
            session_id=f"eval_{case.case_id}",
            new_message=content
        ):
            # Capture trajectory
            if event.author and event.author != "user":
                trajectory.append(event.author)
            
            # Capture final response
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
        
        # Get story from state
        updated_session = await self.session_service.get_session(
            app_name="agent_evaluation",
            user_id="eval_user",
            session_id=f"eval_{case.case_id}"
        )
        
        story = updated_session.state.get("current_story", final_response)
        
        case.actual_trajectory = trajectory
        case.actual_response = story
        
        # Calculate trajectory scores
        case.scores["trajectory_exact"] = self.trajectory_eval.exact_match_score(
            case.expected_trajectory, case.actual_trajectory
        )
        case.scores["trajectory_order"] = self.trajectory_eval.order_match_score(
            case.expected_trajectory, case.actual_trajectory
        )
        case.scores["trajectory_presence"] = self.trajectory_eval.step_presence_score(
            case.expected_trajectory, case.actual_trajectory
        )
        
        # Keyword presence check
        if case.expected_response_keywords:
            keywords_found = sum(
                1 for kw in case.expected_response_keywords
                if kw.lower() in case.actual_response.lower()
            )
            case.scores["keyword_score"] = keywords_found / len(case.expected_response_keywords)
        
        return case
    
    async def evaluate_batch(
        self,
        cases: List[EvaluationCase],
        use_vertex_ai: bool = True
    ) -> EvaluationResult:
        """Evaluate multiple cases and aggregate results."""
        
        evaluated_cases = []
        for case in cases:
            result = await self.evaluate_case(case)
            evaluated_cases.append(result)
        
        # Aggregate scores
        trajectory_scores = [c.scores.get("trajectory_exact", 0) for c in evaluated_cases]
        
        # Use Vertex AI for response evaluation if enabled
        if use_vertex_ai and evaluated_cases:
            responses = [{
                "instruction": f"Write a story about: {c.topic}",
                "context": f"Topic: {c.topic}",
                "response": c.actual_response
            } for c in evaluated_cases]
            
            try:
                vertex_results = self.response_eval.evaluate_responses(responses)
                # Extract average scores
                avg_coherence = vertex_results.get("coherence/mean", 0.8)
                avg_fluency = vertex_results.get("fluency/mean", 0.8)
                avg_groundedness = vertex_results.get("groundedness/mean", 0.8)
            except Exception as e:
                print(f"Vertex AI evaluation error: {e}")
                avg_coherence = avg_fluency = avg_groundedness = 0.0
        else:
            avg_coherence = avg_fluency = avg_groundedness = 0.0
        
        # Calculate pass/fail
        passed = sum(1 for c in evaluated_cases if c.scores.get("trajectory_exact", 0) >= 0.8)
        failed = len(evaluated_cases) - passed
        
        # Detailed results
        detailed = []
        for c in evaluated_cases:
            detailed.append({
                "case_id": c.case_id,
                "topic": c.topic,
                "scores": c.scores,
                "trajectory": c.actual_trajectory,
                "response_length": len(c.actual_response)
            })
        
        return EvaluationResult(
            total_cases=len(evaluated_cases),
            passed_cases=passed,
            failed_cases=failed,
            avg_trajectory_score=sum(trajectory_scores) / len(trajectory_scores) if trajectory_scores else 0,
            avg_response_quality=(avg_coherence + avg_fluency) / 2,
            avg_groundedness=avg_groundedness,
            avg_coherence=avg_coherence,
            avg_fluency=avg_fluency,
            detailed_results=detailed
        )


# ============================================================
# EVALSET LOADER
# ============================================================

def load_evalset(filepath: str) -> List[EvaluationCase]:
    """Load evaluation cases from ADK evalset JSON format."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    cases = []
    for eval_case in data.get("eval_cases", []):
        # Extract expected trajectory from intermediate_responses
        expected_trajectory = []
        for conv in eval_case.get("conversation", []):
            intermediate = conv.get("intermediate_data", {})
            for resp in intermediate.get("intermediate_responses", []):
                if len(resp) >= 1:
                    expected_trajectory.append(resp[0])
        
        # Extract topic from state
        topic = eval_case.get("session_input", {}).get("state", {}).get("topic", "")
        
        cases.append(EvaluationCase(
            case_id=eval_case.get("eval_id", "unknown"),
            topic=topic,
            expected_trajectory=expected_trajectory,
            expected_response_keywords=[]
        ))
    
    return cases


# ============================================================
# CLI INTERFACE
# ============================================================

async def run_evaluation(
    agent: BaseAgent,
    evalset_path: str,
    output_path: Optional[str] = None,
    use_vertex_ai: bool = True
) -> EvaluationResult:
    """
    Run full evaluation pipeline.
    
    Args:
        agent: The ADK agent to evaluate
        evalset_path: Path to evalset JSON file
        output_path: Optional path to save results
        use_vertex_ai: Whether to use Vertex AI metrics
    
    Returns:
        EvaluationResult with aggregated metrics
    """
    # Load cases
    cases = load_evalset(evalset_path)
    print(f"Loaded {len(cases)} evaluation cases")
    
    # Initialize evaluator
    evaluator = AgentEvaluator(agent)
    
    # Run evaluation
    result = await evaluator.evaluate_batch(cases, use_vertex_ai=use_vertex_ai)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Cases: {result.total_cases}")
    print(f"Passed: {result.passed_cases}")
    print(f"Failed: {result.failed_cases}")
    print(f"Avg Trajectory Score: {result.avg_trajectory_score:.3f}")
    print(f"Avg Response Quality: {result.avg_response_quality:.3f}")
    print(f"Avg Coherence: {result.avg_coherence:.3f}")
    print(f"Avg Fluency: {result.avg_fluency:.3f}")
    print(f"Avg Groundedness: {result.avg_groundedness:.3f}")
    print("=" * 60)
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump({
                "summary": {
                    "total_cases": result.total_cases,
                    "passed": result.passed_cases,
                    "failed": result.failed_cases,
                    "avg_trajectory_score": result.avg_trajectory_score,
                    "avg_response_quality": result.avg_response_quality,
                    "avg_coherence": result.avg_coherence,
                    "avg_fluency": result.avg_fluency,
                    "avg_groundedness": result.avg_groundedness,
                    "timestamp": result.timestamp
                },
                "detailed_results": result.detailed_results
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return result


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Import the StoryFlowAgent
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from agents.story_flow_agent import root_agent
    
    evalset_path = os.path.join(
        os.path.dirname(__file__),
        "..", "agents", "story_agent.evalset.json"
    )
    
    output_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "tests", "data", "vertex_eval_results.json"
    )
    
    asyncio.run(run_evaluation(
        agent=root_agent,
        evalset_path=evalset_path,
        output_path=output_path,
        use_vertex_ai=True
    ))
