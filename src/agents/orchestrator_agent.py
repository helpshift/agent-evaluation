"""
Evaluation Orchestrator Agent - Runs the entire evaluation pipeline.

This agent orchestrates everything:
1. Generates stories using StoryFlowAgent
2. Evaluates them using the EvaluatorAgent
3. Produces a summary report
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path

from google.adk.agents import LlmAgent, SequentialAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from typing import AsyncGenerator
from typing_extensions import override

from src.agents.story_flow_agent import StoryFlowAgent, root_agent as story_agent
from src.agents.evaluator_agent import create_evaluator_agent, EvaluationInput


# --- Constants ---
APP_NAME = "evaluation_orchestrator"
GEMINI_MODEL = "gemini-2.5-flash"


def load_test_cases() -> list[dict]:
    """Load test cases from golden dataset."""
    # Use absolute path based on project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(project_root, "tests", "data", "story_eval_dataset.json")
    with open(dataset_path, "r") as f:
        return json.load(f)


# --- Tool Definitions ---
def run_story_generation(topic: str, tool_context: ToolContext) -> str:
    """
    Generate a story on the given topic using StoryFlowAgent.
    
    Args:
        topic: The story topic to write about
        
    Returns:
        The generated story text
    """
    print(f"[Tool] Generating story for topic: {topic}")
    
    # Store the topic in state for the workflow to use
    tool_context.state["current_topic"] = topic
    
    # For now, return a placeholder - actual execution happens via the workflow
    return f"Story generation initiated for: {topic}"


def evaluate_story(
    test_id: str,
    topic: str,
    story: str,
    expected_behavior: str,
    tool_context: ToolContext
) -> dict:
    """
    Evaluate a generated story using the LLM-as-judge evaluator.
    
    Args:
        test_id: Unique identifier for this test case
        topic: The original story topic
        story: The generated story text
        expected_behavior: What the story should achieve
        
    Returns:
        Evaluation result with scores and verdict
    """
    print(f"[Tool] Evaluating story for test: {test_id}")
    
    # Store evaluation request in state
    tool_context.state["eval_requests"] = tool_context.state.get("eval_requests", [])
    tool_context.state["eval_requests"].append({
        "test_id": test_id,
        "topic": topic,
        "story": story,
        "expected_behavior": expected_behavior
    })
    
    return {"status": "evaluation_queued", "test_id": test_id}


def save_results(results: list[dict], tool_context: ToolContext) -> str:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Path to the saved results file
    """
    results_path = Path(__file__).parent.parent / "tests" / "data" / "orchestrator_results.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(results),
        "pass_count": sum(1 for r in results if r.get("verdict") == "PASS"),
        "partial_count": sum(1 for r in results if r.get("verdict") == "PARTIAL"),
        "fail_count": sum(1 for r in results if r.get("verdict") == "FAIL"),
        "average_score": sum(r.get("overall_score", 0) for r in results) / len(results) if results else 0,
        "results": results
    }
    
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    tool_context.state["evaluation_summary"] = summary
    
    return f"Results saved to: {results_path}"


# --- Orchestrator Agent ---
ORCHESTRATOR_INSTRUCTIONS = """You are an Evaluation Orchestrator Agent.

Your job is to run the complete evaluation pipeline:

1. Load test cases from the golden dataset
2. For each test case:
   a. Generate a story using the story generation tool
   b. Evaluate the story using the evaluation tool
3. Compile all results
4. Save the final report

You have access to:
- run_story_generation: Generate stories on a topic
- evaluate_story: Evaluate a generated story
- save_results: Save evaluation results to file

Start by loading the test cases and then process them one by one.
Provide a summary at the end with pass/fail counts and average score.
"""


# Create the orchestrator agent
evaluation_orchestrator = LlmAgent(
    name="EvaluationOrchestrator",
    model=GEMINI_MODEL,
    instruction=ORCHESTRATOR_INSTRUCTIONS,
    description="Orchestrates the full story generation and evaluation pipeline",
    tools=[run_story_generation, evaluate_story, save_results]
)


# --- Full Pipeline Runner ---
class EvaluationPipelineAgent(BaseAgent):
    """
    Custom agent that runs the complete evaluation pipeline.
    
    This demonstrates the full ADK pattern:
    - Custom orchestration
    - Running sub-agents programmatically
    - Collecting and aggregating results
    """
    
    story_agent: StoryFlowAgent
    evaluator: LlmAgent
    max_tests: int = 5  # Limit for quick runs
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        name: str = "EvaluationPipelineAgent",
        max_tests: int = 5
    ):
        evaluator = create_evaluator_agent()
        
        super().__init__(
            name=name,
            story_agent=story_agent,
            evaluator=evaluator,
            max_tests=max_tests,
            sub_agents=[story_agent, evaluator]
        )
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Run the complete evaluation pipeline."""
        
        # Load test cases
        test_cases = load_test_cases()[:self.max_tests]
        results = []
        
        print(f"\n{'='*60}")
        print(f"EVALUATION PIPELINE - Processing {len(test_cases)} test cases")
        print(f"{'='*60}")
        
        for i, test_case in enumerate(test_cases):
            test_id = test_case["id"]
            topic = test_case["initial_state"]["topic"]
            
            print(f"\n[{i+1}/{len(test_cases)}] Test: {test_id}")
            print(f"Topic: {topic}")
            
            # Set topic in state for story agent
            ctx.session.state["topic"] = topic
            
            # Generate story
            print("  Generating story...")
            async for event in self.story_agent.run_async(ctx):
                yield event
            
            story = ctx.session.state.get("current_story", "")
            print(f"  Story generated ({len(story)} chars)")
            
            # Create evaluation input
            eval_input = EvaluationInput(
                test_id=test_id,
                topic=topic,
                generated_story=story if story else "No story generated",
                expected_behavior=test_case["expected_behavior"],
                evaluation_criteria=test_case["evaluation_criteria"]
            )
            
            # Run evaluator
            print("  Evaluating...")
            ctx.session.state["eval_input"] = eval_input.model_dump_json()
            
            eval_response = ""
            async for event in self.evaluator.run_async(ctx):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            eval_response = part.text
                yield event
            
            # Parse result
            try:
                if "```json" in eval_response:
                    json_str = eval_response.split("```json")[1].split("```")[0]
                elif "```" in eval_response:
                    json_str = eval_response.split("```")[1].split("```")[0]
                else:
                    json_str = eval_response
                
                result = json.loads(json_str.strip())
            except:
                result = {
                    "test_id": test_id,
                    "overall_score": 0.0,
                    "verdict": "ERROR",
                    "reasoning": "Failed to parse evaluation"
                }
            
            result["story_preview"] = story[:150] if story else "None"
            results.append(result)
            
            print(f"  Verdict: {result.get('verdict', 'N/A')} | Score: {result.get('overall_score', 'N/A')}")
        
        # Generate summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        pass_count = sum(1 for r in results if r.get("verdict") == "PASS")
        partial_count = sum(1 for r in results if r.get("verdict") == "PARTIAL")
        fail_count = sum(1 for r in results if r.get("verdict") == "FAIL")
        error_count = sum(1 for r in results if r.get("verdict") == "ERROR")
        avg_score = sum(r.get("overall_score", 0) for r in results) / len(results) if results else 0
        
        summary = f"""
## Evaluation Complete

- **Total Tests:** {len(results)}
- **PASS:** {pass_count}
- **PARTIAL:** {partial_count}
- **FAIL:** {fail_count}
- **ERROR:** {error_count}
- **Average Score:** {avg_score:.2f}

{'✅ EVALUATION PASSED' if avg_score >= 0.6 else '⚠️ NEEDS IMPROVEMENT'}
"""
        print(summary)
        
        # Save results
        results_path = Path(__file__).parent.parent / "tests" / "data" / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": len(results),
                    "pass": pass_count,
                    "partial": partial_count,
                    "fail": fail_count,
                    "error": error_count,
                    "average_score": avg_score
                },
                "results": results
            }, f, indent=2)
        
        # Store in state
        ctx.session.state["evaluation_results"] = results
        ctx.session.state["evaluation_summary"] = summary
        
        # Yield final summary as event
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=summary)]
            )
        )


# Module-level agent for adk web
root_agent = EvaluationPipelineAgent(name="EvaluationPipelineAgent", max_tests=5)
