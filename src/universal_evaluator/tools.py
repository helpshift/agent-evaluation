"""
Universal Evaluator Tools.

Tools for the Universal Evaluation Agent to:
1. Connect to an agent.
2. Generate golden datasets.
3. Run sessions.
4. Evaluate results.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import ADK components
from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import our previous work
from src.evaluation.evaluator import ComprehensiveEvaluator
from src.evaluation.schemas import EvaluationInput, EvaluationOutput, ToolCallInfo

# Import sample agents (simulated dynamic discovery)
from src.target_agents import samples

logger = logging.getLogger(__name__)

# Mock registry for "Remote" connection simulation
AGENT_REGISTRY = {
    "MathAgent": samples.math_agent,
    "StoryTeller": samples.story_teller,
    "ResearchAssistant": samples.research_assistant,
    "SupportRouter": samples.support_router,
    "ParallelBooker": samples.parallel_booker,
    "CodeFixer": samples.code_fixer
}

# Values specific to session management
APP_NAME = "universal_evaluator_app"
USER_ID = "evaluator_user"

# =========================================================================
# 1. Connection Tools
# =========================================================================

def list_available_agents() -> List[str]:
    """Lists all agents available for evaluation."""
    return list(AGENT_REGISTRY.keys())

def get_agent_details(agent_name: str) -> Dict[str, Any]:
    """Gets details about a specific agent (introspects it)."""
    agent = AGENT_REGISTRY.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found."}
    
    tools_info = []
    if hasattr(agent, "tools"):
         # This is a simplification; extracting tool metadata is complex
         tools_info = [str(t) for t in agent.tools]
    
    sub_agents = []
    if hasattr(agent, "sub_agents") and agent.sub_agents:
        sub_agents = [sa.name for sa in agent.sub_agents]

    return {
        "name": agent.name,
        "description": getattr(agent, "description", "No description provided."),
        "model": getattr(agent, "model", "Unknown"),
        "instruction_preview": getattr(agent, "instruction", "")[:200] + "...",
        "tools": tools_info,
        "sub_agents": sub_agents,
        "type": agent.__class__.__name__
    }

# =========================================================================
# 2. Golden Dataset Generation
# =========================================================================

# Note: In a real scenario, this would call an LLM. Here we simulate it
# because we are inside a tool implementation.
# However, the user wants the evaluator to do this.
# So we will return a structure that the Evaluator's LLM can simply FILL IN,
# or we can use a separate LLM call here if we had access to one.
# For this implementation, we will define the prompt for the *Evaluator Agent*
# to use to generate this.

GENERATION_PROMPT_TEMPLATE = """
You are generating a test dataset for an AI agent.
Agent Name: {name}
Description: {description}
Tools: {tools}

Generate 5 diverse test cases in JSON format.
Each case must have:
- "input": The user message.
- "expected_trajectory": A list of tool names that SHOULD be called (can be empty).
- "expected_intent": What the agent should achieve.

Output purely a JSON list of objects.
"""

# =========================================================================
# 3. Execution Tools
# =========================================================================

async def run_evaluation_session(
    agent_name: str, 
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Runs the agent against the provided test cases and evaluates performance.
    
    Args:
        agent_name: Name of the agent to test.
        test_cases: List of dicts with 'input' and 'expected_trajectory'.
        
    Returns:
        Summary report of the evaluation.
    """
    agent = AGENT_REGISTRY.get(agent_name)
    if not agent:
        return {"error": f"Agent '{agent_name}' not found."}

    # Initialize Evaluator Service
    # We use our previously built ComprehensiveEvaluator!
    comp_evaluator = ComprehensiveEvaluator()
    
    results = []
    
    for i, case in enumerate(test_cases):
        user_input = case.get("input")
        expected_traj = case.get("expected_trajectory", [])
        
        # Setup Session
        session_service = InMemorySessionService()
        session_id = f"eval_session_{agent_name}_{i}_{datetime.now().timestamp()}"
        await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        
        runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
        
        # Run Agent
        # Capture trajectory manually from events for now (or assume callback catches it)
        # Since we are "outside", we observe the events.
        
        actual_trajectory = []
        final_response = ""
        tool_info = []
        
        user_msg_content = types.Content(role="user", parts=[types.Part(text=user_input)])
        
        try:
            async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=user_msg_content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.function_call:
                            actual_trajectory.append(part.function_call.name)
                            tool_info.append(ToolCallInfo(
                                tool=part.function_call.name,
                                requested_args=part.function_call.args,
                                status="success" 
                            ))
                
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text

        except Exception as e:
            return {"error": f"Error running case {i}: {str(e)}"}

        # Construct Evaluation Input
        eval_input = EvaluationInput(
            domain="generic",
            issue_id=f"test_case_{i}",
            user_message=user_input,
            response=final_response,
            tool_info=tool_info
        )
        
        # Evaluate
        eval_output = await comp_evaluator.evaluate(
            input_data=eval_input,
            expected_trajectory=expected_traj
        )
        
        results.append(eval_output.dict())

    # Aggregate
    passed = sum(1 for r in results if r['verdict'] == 'PASS')
    
    return {
        "agent": agent_name,
        "total_cases": len(results),
        "passed_cases": passed,
        "pass_rate": passed / len(results) if results else 0,
        "detailed_results": results
    }
