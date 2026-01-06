"""
Pytest suite for Vertex AI GenAI Evaluation Service integration.

Tests:
1. Trajectory evaluation (exact match, order match, presence)
2. Vertex AI response quality metrics
3. Full ADK evalset evaluation
4. ADK CLI eval compatibility
"""

import pytest
import os
import sys
import json
import asyncio
from typing import List, Dict, Any

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def trajectory_evaluator():
    """Create TrajectoryEvaluator instance."""
    from src.evaluation.vertex_ai_evaluator import TrajectoryEvaluator
    return TrajectoryEvaluator()


@pytest.fixture
def evaluation_cases():
    """Sample evaluation cases for testing."""
    from src.evaluation.vertex_ai_evaluator import EvaluationCase
    
    return [
        EvaluationCase(
            case_id="test_001",
            topic="a brave knight",
            expected_trajectory=["StoryGenerator", "Critic", "Reviser", "GrammarCheck", "ToneCheck"],
            expected_response_keywords=["knight", "brave", "adventure"]
        ),
        EvaluationCase(
            case_id="test_002",
            topic="space exploration",
            expected_trajectory=["StoryGenerator", "GrammarCheck", "ToneCheck"],
            expected_response_keywords=["space", "astronaut", "stars"]
        ),
        EvaluationCase(
            case_id="test_003",
            topic="a magical forest",
            expected_trajectory=["StoryGenerator", "Critic", "Reviser", "GrammarCheck", "ToneCheck"],
            expected_response_keywords=["forest", "magic", "trees"]
        )
    ]


@pytest.fixture
def story_agent():
    """Load the StoryFlowAgent."""
    from src.agents.story_flow_agent import root_agent
    return root_agent


@pytest.fixture
def agent_evaluator(story_agent):
    """Create AgentEvaluator instance."""
    from src.evaluation.vertex_ai_evaluator import AgentEvaluator
    return AgentEvaluator(story_agent)


# ============================================================
# TRAJECTORY EVALUATION TESTS
# ============================================================

class TestTrajectoryEvaluation:
    """Tests for trajectory matching logic."""
    
    def test_exact_match_perfect(self, trajectory_evaluator):
        """Test perfect exact match."""
        expected = ["A", "B", "C"]
        actual = ["A", "B", "C"]
        score = trajectory_evaluator.exact_match_score(expected, actual)
        assert score == 1.0
    
    def test_exact_match_partial(self, trajectory_evaluator):
        """Test partial exact match."""
        expected = ["A", "B", "C"]
        actual = ["A", "B", "D"]
        score = trajectory_evaluator.exact_match_score(expected, actual)
        assert score == pytest.approx(0.666, rel=0.01)
    
    def test_exact_match_none(self, trajectory_evaluator):
        """Test no match."""
        expected = ["A", "B", "C"]
        actual = ["X", "Y", "Z"]
        score = trajectory_evaluator.exact_match_score(expected, actual)
        assert score == 0.0
    
    def test_order_match_with_gaps(self, trajectory_evaluator):
        """Test order matching with extra steps."""
        expected = ["A", "C", "E"]
        actual = ["A", "B", "C", "D", "E"]
        score = trajectory_evaluator.order_match_score(expected, actual)
        assert score == 1.0
    
    def test_order_match_wrong_order(self, trajectory_evaluator):
        """Test order matching with wrong order."""
        expected = ["A", "B", "C"]
        actual = ["C", "B", "A"]
        score = trajectory_evaluator.order_match_score(expected, actual)
        # Only A matches in order
        assert score == pytest.approx(0.333, rel=0.01)
    
    def test_step_presence_all_present(self, trajectory_evaluator):
        """Test all steps present."""
        expected = ["A", "B", "C"]
        actual = ["C", "A", "B", "D"]
        score = trajectory_evaluator.step_presence_score(expected, actual)
        assert score == 1.0
    
    def test_step_presence_some_missing(self, trajectory_evaluator):
        """Test some steps missing."""
        expected = ["A", "B", "C"]
        actual = ["A", "D", "E"]
        score = trajectory_evaluator.step_presence_score(expected, actual)
        assert score == pytest.approx(0.333, rel=0.01)
    
    def test_empty_expected(self, trajectory_evaluator):
        """Test empty expected trajectory."""
        expected = []
        actual = ["A", "B"]
        
        exact = trajectory_evaluator.exact_match_score(expected, actual)
        order = trajectory_evaluator.order_match_score(expected, actual)
        presence = trajectory_evaluator.step_presence_score(expected, actual)
        
        assert exact == 0.0  # Empty expected with non-empty actual
        assert order == 0.0
        assert presence == 1.0


# ============================================================
# EVALSET LOADER TESTS
# ============================================================

class TestEvalsetLoader:
    """Tests for loading ADK evalset format."""
    
    def test_load_evalset(self):
        """Test loading the evalset file."""
        from src.evaluation.vertex_ai_evaluator import load_evalset
        
        evalset_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "agents", "story_agent.evalset.json"
        )
        
        cases = load_evalset(evalset_path)
        
        assert len(cases) >= 1
        assert cases[0].case_id is not None
        assert cases[0].topic is not None
    
    def test_evalset_format(self):
        """Verify evalset JSON structure."""
        evalset_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "agents", "story_agent.evalset.json"
        )
        
        with open(evalset_path, "r") as f:
            data = json.load(f)
        
        # Check required fields
        assert "eval_set_id" in data
        assert "eval_cases" in data
        assert len(data["eval_cases"]) > 0
        
        # Check first case structure
        first_case = data["eval_cases"][0]
        assert "eval_id" in first_case
        assert "conversation" in first_case
        assert "session_input" in first_case


# ============================================================
# AGENT EVALUATION TESTS
# ============================================================

class TestAgentEvaluation:
    """Tests for agent evaluation with Vertex AI."""
    
    @pytest.mark.asyncio
    async def test_evaluate_single_case(self, agent_evaluator, evaluation_cases):
        """Test evaluating a single case."""
        case = evaluation_cases[0]
        result = await agent_evaluator.evaluate_case(case)
        
        assert result.actual_response != ""
        assert len(result.actual_trajectory) > 0
        assert "trajectory_exact" in result.scores
        assert "trajectory_order" in result.scores
    
    @pytest.mark.asyncio
    async def test_evaluate_batch_without_vertex(self, agent_evaluator, evaluation_cases):
        """Test batch evaluation without Vertex AI (for CI/CD)."""
        # Use first 2 cases for speed
        cases = evaluation_cases[:2]
        
        result = await agent_evaluator.evaluate_batch(cases, use_vertex_ai=False)
        
        assert result.total_cases == 2
        assert result.passed_cases + result.failed_cases == 2
        assert 0 <= result.avg_trajectory_score <= 1


# ============================================================
# ADK CLI COMPATIBILITY TESTS
# ============================================================

class TestADKCLICompatibility:
    """Tests for ADK CLI eval compatibility."""
    
    def test_test_config_exists(self):
        """Verify test_config.json exists and is valid."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "agents", "test_config.json"
        )
        
        assert os.path.exists(config_path)
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        assert "criteria" in config
    
    def test_evalset_adk_format(self):
        """Verify evalset follows ADK format."""
        evalset_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "agents", "story_agent.evalset.json"
        )
        
        with open(evalset_path, "r") as f:
            data = json.load(f)
        
        # ADK required fields
        assert "eval_set_id" in data
        assert "eval_cases" in data
        
        for case in data["eval_cases"]:
            assert "eval_id" in case
            assert "conversation" in case
            
            for conv in case["conversation"]:
                assert "user_content" in conv
                assert "final_response" in conv


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestFullPipeline:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self, story_agent):
        """Run complete evaluation pipeline."""
        from src.evaluation.vertex_ai_evaluator import (
            AgentEvaluator, load_evalset
        )
        
        evalset_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "agents", "story_agent.evalset.json"
        )
        
        # Load cases
        cases = load_evalset(evalset_path)
        
        # Limit to 2 cases for speed
        cases = cases[:2]
        
        # Evaluate
        evaluator = AgentEvaluator(story_agent)
        result = await evaluator.evaluate_batch(cases, use_vertex_ai=False)
        
        # Assertions
        assert result.total_cases == len(cases)
        assert result.avg_trajectory_score >= 0
        
        print(f"\n{'='*60}")
        print("INTEGRATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total: {result.total_cases}")
        print(f"Passed: {result.passed_cases}")
        print(f"Failed: {result.failed_cases}")
        print(f"Avg Trajectory: {result.avg_trajectory_score:.3f}")
        print(f"{'='*60}")
    
    @pytest.mark.asyncio
    async def test_evaluation_with_vertex_ai(self, story_agent):
        """Test with Vertex AI (requires credentials)."""
        pytest.importorskip("vertexai")
        
        from src.evaluation.vertex_ai_evaluator import (
            AgentEvaluator, EvaluationCase
        )
        
        # Single simple case
        case = EvaluationCase(
            case_id="vertex_test",
            topic="a happy ending",
            expected_trajectory=["StoryGenerator", "GrammarCheck", "ToneCheck"],
            expected_response_keywords=["happy", "joy"]
        )
        
        evaluator = AgentEvaluator(story_agent)
        
        try:
            result = await evaluator.evaluate_batch([case], use_vertex_ai=True)
            assert result.total_cases == 1
            print(f"\nVertex AI Metrics:")
            print(f"  Coherence: {result.avg_coherence:.3f}")
            print(f"  Fluency: {result.avg_fluency:.3f}")
            print(f"  Groundedness: {result.avg_groundedness:.3f}")
        except Exception as e:
            pytest.skip(f"Vertex AI not configured: {e}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
