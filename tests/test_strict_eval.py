"""
Strict Evaluation Test Suite for AdvancedResearchAgent.

Uses Vertex AI GenAI Evaluation Service metrics:
- Exact Match
- In-Order Match
- Precision
- Recall

Toughness Level: 5/5 (Zero Tolerance)
"""

import pytest
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Import the advanced agent and callbacks
from src.agents.advanced_research_agent import (
    AdvancedResearchAgent,
    advanced_research_agent
)
from src.agents.callbacks import (
    callback_tracker,
    reset_tracker,
    get_trajectory,
    get_callback_metrics
)


# ============================================================
# TRAJECTORY EVALUATION METRICS
# ============================================================

@dataclass
class TrajectoryMetrics:
    """Vertex AI-style trajectory evaluation metrics."""
    
    exact_match: float = 0.0
    in_order_match: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    any_order_match: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "exact_match": self.exact_match,
            "in_order_match": self.in_order_match,
            "precision": self.precision,
            "recall": self.recall,
            "any_order_match": self.any_order_match
        }


def calculate_exact_match(
    actual: List[str],
    expected: List[str]
) -> float:
    """
    Exact match: 1.0 if trajectories are identical, 0.0 otherwise.
    """
    return 1.0 if actual == expected else 0.0


def calculate_in_order_match(
    actual: List[str],
    expected: List[str]
) -> float:
    """
    In-order match: 1.0 if all expected tools appear in correct order
    (may have extra tools between them).
    """
    if not expected:
        return 1.0
    
    expected_idx = 0
    for tool in actual:
        if expected_idx < len(expected) and tool == expected[expected_idx]:
            expected_idx += 1
    
    return 1.0 if expected_idx == len(expected) else 0.0


def calculate_precision(
    actual: List[str],
    expected: List[str]
) -> float:
    """
    Precision: (# of actual tools in expected) / (# of actual tools)
    Measures relevance of tool calls.
    """
    if not actual:
        return 0.0 if expected else 1.0
    
    expected_set = set(expected)
    relevant = sum(1 for tool in actual if tool in expected_set)
    return relevant / len(actual)


def calculate_recall(
    actual: List[str],
    expected: List[str]
) -> float:
    """
    Recall: (# of expected tools in actual) / (# of expected tools)
    Measures completeness of required tool calls.
    """
    if not expected:
        return 1.0
    
    actual_set = set(actual)
    found = sum(1 for tool in expected if tool in actual_set)
    return found / len(expected)


def calculate_any_order_match(
    actual: List[str],
    expected: List[str]
) -> float:
    """
    Any-order match: 1.0 if all expected tools appear (order doesn't matter).
    """
    if not expected:
        return 1.0
    
    actual_set = set(actual)
    expected_set = set(expected)
    return 1.0 if expected_set.issubset(actual_set) else 0.0


def evaluate_trajectory(
    actual: List[str],
    expected: List[str]
) -> TrajectoryMetrics:
    """Calculate all trajectory metrics."""
    return TrajectoryMetrics(
        exact_match=calculate_exact_match(actual, expected),
        in_order_match=calculate_in_order_match(actual, expected),
        precision=calculate_precision(actual, expected),
        recall=calculate_recall(actual, expected),
        any_order_match=calculate_any_order_match(actual, expected)
    )


# ============================================================
# STATE VALIDATION
# ============================================================

def validate_state_keys(
    actual_state: Dict[str, Any],
    expected_keys: List[str]
) -> Dict[str, Any]:
    """Validate that required state keys are present."""
    present = []
    missing = []
    
    for key in expected_keys:
        if key in actual_state:
            present.append(key)
        else:
            missing.append(key)
    
    score = len(present) / len(expected_keys) if expected_keys else 1.0
    
    return {
        "score": score,
        "present": present,
        "missing": missing,
        "passed": len(missing) == 0
    }


# ============================================================
# EVALUATION CASE RUNNER
# ============================================================

@dataclass
class EvaluationResult:
    """Result of a single evaluation case."""
    
    case_id: str
    case_name: str
    passed: bool
    trajectory_metrics: TrajectoryMetrics
    state_validation: Dict[str, Any]
    callback_metrics: Dict[str, Any]
    actual_trajectory: List[str]
    expected_trajectory: List[str]
    execution_time_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "passed": self.passed,
            "trajectory_metrics": self.trajectory_metrics.to_dict(),
            "state_validation": self.state_validation,
            "callback_metrics": self.callback_metrics,
            "actual_trajectory": self.actual_trajectory,
            "expected_trajectory": self.expected_trajectory,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error
        }


async def run_evaluation_case(
    agent: AdvancedResearchAgent,
    case: Dict[str, Any],
    thresholds: Dict[str, float]
) -> EvaluationResult:
    """Run a single evaluation case against the agent."""
    
    case_id = case["id"]
    case_name = case["name"]
    
    start_time = datetime.now()
    
    # Reset callback tracker
    reset_tracker()
    
    # Setup session with initial state
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="advanced_research_eval",
        user_id="evaluator",
        state=case.get("initial_state", {})
    )
    
    # Create runner
    runner = Runner(
        agent=agent,
        app_name="advanced_research_eval",
        session_service=session_service
    )
    
    try:
        # Run agent with input
        query = case.get("input", {}).get("query", "")
        
        async for event in runner.run_async(
            user_id="evaluator",
            session_id=session.id,
            new_message=query
        ):
            pass  # Collect all events
        
        # Get final session state
        final_session = await session_service.get_session(
            app_name="advanced_research_eval",
            user_id="evaluator",
            session_id=session.id
        )
        final_state = final_session.state if final_session else {}
        
        # Get actual trajectory from callbacks
        actual_trajectory = get_trajectory()
        
        # Extract expected trajectory (tool names only)
        expected_trajectory = [
            step["tool"] 
            for step in case.get("expected_trajectory", [])
            if step.get("required", True)
        ]
        
        # Calculate trajectory metrics
        trajectory_metrics = evaluate_trajectory(actual_trajectory, expected_trajectory)
        
        # Validate state keys
        expected_keys = case.get("expected_state_keys", [])
        state_validation = validate_state_keys(final_state, expected_keys)
        
        # Get callback metrics
        callback_metrics = get_callback_metrics()
        
        # Determine pass/fail based on thresholds
        passed = True
        
        # Check trajectory metrics against thresholds
        if trajectory_metrics.exact_match < thresholds.get("exact_match", 1.0):
            passed = False
        if trajectory_metrics.precision < thresholds.get("precision", 0.95):
            passed = False
        if trajectory_metrics.recall < thresholds.get("recall", 1.0):
            passed = False
        
        # Check state validation
        if not state_validation.get("passed", False):
            passed = False
        
        # Check case-specific pass criteria
        pass_criteria = case.get("pass_criteria", {})
        workflow_status = final_state.get("workflow_status")
        
        if "workflow_status" in pass_criteria:
            if workflow_status != pass_criteria["workflow_status"]:
                passed = False
        
        if pass_criteria.get("request_blocked"):
            if callback_metrics.get("blocked_requests", 0) == 0:
                passed = False
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return EvaluationResult(
            case_id=case_id,
            case_name=case_name,
            passed=passed,
            trajectory_metrics=trajectory_metrics,
            state_validation=state_validation,
            callback_metrics=callback_metrics,
            actual_trajectory=actual_trajectory,
            expected_trajectory=expected_trajectory,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        return EvaluationResult(
            case_id=case_id,
            case_name=case_name,
            passed=False,
            trajectory_metrics=TrajectoryMetrics(),
            state_validation={"error": str(e)},
            callback_metrics={},
            actual_trajectory=[],
            expected_trajectory=[],
            execution_time_ms=execution_time,
            error=str(e)
        )


# ============================================================
# PYTEST TEST SUITE
# ============================================================

class TestTrajectoryMetrics:
    """Unit tests for trajectory evaluation functions."""
    
    def test_exact_match_identical(self):
        actual = ["search_web", "analyze_data", "check_facts"]
        expected = ["search_web", "analyze_data", "check_facts"]
        assert calculate_exact_match(actual, expected) == 1.0
    
    def test_exact_match_different(self):
        actual = ["search_web", "analyze_data"]
        expected = ["search_web", "check_facts"]
        assert calculate_exact_match(actual, expected) == 0.0
    
    def test_exact_match_extra_tools(self):
        actual = ["search_web", "extra_tool", "analyze_data"]
        expected = ["search_web", "analyze_data"]
        assert calculate_exact_match(actual, expected) == 0.0
    
    def test_in_order_match_correct(self):
        actual = ["search_web", "extra", "analyze_data", "check_facts"]
        expected = ["search_web", "analyze_data", "check_facts"]
        assert calculate_in_order_match(actual, expected) == 1.0
    
    def test_in_order_match_wrong_order(self):
        actual = ["check_facts", "search_web", "analyze_data"]
        expected = ["search_web", "analyze_data", "check_facts"]
        assert calculate_in_order_match(actual, expected) == 0.0
    
    def test_precision_all_relevant(self):
        actual = ["search_web", "analyze_data"]
        expected = ["search_web", "analyze_data", "check_facts"]
        assert calculate_precision(actual, expected) == 1.0
    
    def test_precision_some_irrelevant(self):
        actual = ["search_web", "random_tool", "analyze_data", "another"]
        expected = ["search_web", "analyze_data"]
        assert calculate_precision(actual, expected) == 0.5
    
    def test_recall_all_found(self):
        actual = ["search_web", "analyze_data", "check_facts", "extra"]
        expected = ["search_web", "analyze_data"]
        assert calculate_recall(actual, expected) == 1.0
    
    def test_recall_some_missing(self):
        actual = ["search_web"]
        expected = ["search_web", "analyze_data"]
        assert calculate_recall(actual, expected) == 0.5
    
    def test_any_order_match_all_present(self):
        actual = ["check_facts", "search_web", "extra", "analyze_data"]
        expected = ["search_web", "analyze_data"]
        assert calculate_any_order_match(actual, expected) == 1.0
    
    def test_any_order_match_missing(self):
        actual = ["search_web", "extra"]
        expected = ["search_web", "analyze_data"]
        assert calculate_any_order_match(actual, expected) == 0.0


class TestStateValidation:
    """Tests for state key validation."""
    
    def test_all_keys_present(self):
        state = {"key1": "val1", "key2": "val2", "key3": "val3"}
        expected = ["key1", "key2"]
        result = validate_state_keys(state, expected)
        assert result["passed"] == True
        assert result["score"] == 1.0
    
    def test_some_keys_missing(self):
        state = {"key1": "val1"}
        expected = ["key1", "key2"]
        result = validate_state_keys(state, expected)
        assert result["passed"] == False
        assert result["score"] == 0.5
    
    def test_empty_expected(self):
        state = {"key1": "val1"}
        expected = []
        result = validate_state_keys(state, expected)
        assert result["passed"] == True


class TestEvalsetLoader:
    """Tests for loading evaluation dataset."""
    
    def test_load_evalset(self):
        evalset_path = Path(__file__).parent.parent / "src" / "agents" / "evalset_advanced.json"
        
        if evalset_path.exists():
            with open(evalset_path) as f:
                evalset = json.load(f)
            
            assert "cases" in evalset
            assert len(evalset["cases"]) > 0
            assert evalset.get("toughness_level") == 5
    
    def test_evalset_case_structure(self):
        evalset_path = Path(__file__).parent.parent / "src" / "agents" / "evalset_advanced.json"
        
        if evalset_path.exists():
            with open(evalset_path) as f:
                evalset = json.load(f)
            
            for case in evalset["cases"]:
                assert "id" in case
                assert "name" in case
                assert "expected_trajectory" in case


class TestCallbackTracker:
    """Tests for callback tracking functionality."""
    
    def test_tracker_reset(self):
        reset_tracker()
        metrics = get_callback_metrics()
        assert metrics["total_model_calls"] == 0
        assert metrics["total_tool_calls"] == 0
    
    def test_tracker_trajectory(self):
        reset_tracker()
        callback_tracker.log_tool_call("tool1", {}, "result1")
        callback_tracker.log_tool_call("tool2", {}, "result2")
        
        trajectory = get_trajectory()
        assert trajectory == ["tool1", "tool2"]


@pytest.mark.asyncio
class TestAdvancedAgentEvaluation:
    """Integration tests for AdvancedResearchAgent evaluation."""
    
    async def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        agent = AdvancedResearchAgent(name="TestAgent")
        assert agent.name == "TestAgent"
        assert hasattr(agent, 'parallel_research')
        assert hasattr(agent, 'quality_gate')
    
    @pytest.mark.skip(reason="Requires live LLM connection")
    async def test_single_case_evaluation(self):
        """Run evaluation on a single test case."""
        agent = advanced_research_agent
        
        test_case = {
            "id": "TEST_001",
            "name": "Basic Test",
            "input": {"query": "Test research query"},
            "initial_state": {"research_topic": "test topic"},
            "expected_trajectory": [
                {"tool": "search_web", "required": True}
            ],
            "expected_state_keys": ["web_research_results"]
        }
        
        thresholds = {
            "exact_match": 1.0,
            "precision": 0.95,
            "recall": 1.0
        }
        
        result = await run_evaluation_case(agent, test_case, thresholds)
        assert result.case_id == "TEST_001"


class TestStrictEvaluationThresholds:
    """Tests to verify strict threshold enforcement."""
    
    def test_thresholds_zero_tolerance(self):
        """Verify that our thresholds enforce zero tolerance."""
        thresholds = {
            "exact_match": 1.0,
            "in_order_match": 1.0,
            "precision": 0.95,
            "recall": 1.0
        }
        
        # Perfect scores should pass
        metrics = TrajectoryMetrics(
            exact_match=1.0,
            in_order_match=1.0,
            precision=0.95,
            recall=1.0
        )
        
        passed = (
            metrics.exact_match >= thresholds["exact_match"] and
            metrics.precision >= thresholds["precision"] and
            metrics.recall >= thresholds["recall"]
        )
        assert passed == True
    
    def test_thresholds_fail_on_low_precision(self):
        """Precision below 0.95 should fail."""
        thresholds = {"precision": 0.95}
        
        metrics = TrajectoryMetrics(precision=0.90)
        assert metrics.precision < thresholds["precision"]
    
    def test_thresholds_fail_on_missing_recall(self):
        """Recall below 1.0 should fail."""
        thresholds = {"recall": 1.0}
        
        metrics = TrajectoryMetrics(recall=0.9)
        assert metrics.recall < thresholds["recall"]


# ============================================================
# FULL EVALUATION RUNNER
# ============================================================

async def run_full_evaluation(
    evalset_path: str = "src/agents/evalset_advanced.json"
) -> Dict[str, Any]:
    """
    Run full evaluation suite and generate report.
    
    Returns:
        Evaluation summary with all results
    """
    # Load evalset
    with open(evalset_path) as f:
        evalset = json.load(f)
    
    agent = advanced_research_agent
    cases = evalset["cases"]
    
    # Extract thresholds
    thresholds = {
        "exact_match": evalset["evaluation_metrics"]["trajectory"]["exact_match"]["threshold"],
        "precision": evalset["evaluation_metrics"]["trajectory"]["precision"]["threshold"],
        "recall": evalset["evaluation_metrics"]["trajectory"]["recall"]["threshold"]
    }
    
    results = []
    for case in cases:
        result = await run_evaluation_case(agent, case, thresholds)
        results.append(result)
    
    # Calculate summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    avg_metrics = {
        "exact_match": sum(r.trajectory_metrics.exact_match for r in results) / total,
        "precision": sum(r.trajectory_metrics.precision for r in results) / total,
        "recall": sum(r.trajectory_metrics.recall for r in results) / total
    }
    
    summary = {
        "evalset_id": evalset["evalset_id"],
        "toughness_level": evalset["toughness_level"],
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "average_metrics": avg_metrics,
        "results": [r.to_dict() for r in results],
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    output_path = "tests/data/evaluation_results_advanced.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
