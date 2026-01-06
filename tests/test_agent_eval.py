import pytest
import json
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

from google.adk.sessions import InMemorySessionService

from google.adk.runners import Runner
from google.genai import types
from src.agents.sample_agent import create_sample_agent
from src.agents.evaluator_agent import create_evaluator_agent, EvaluationOutput

GOLDEN_DATASET_PATH = "tests/data/golden_dataset.json"

@pytest.fixture
def sample_agent():
    return create_sample_agent()

@pytest.fixture
def evaluator_agent():
    return create_evaluator_agent()

@pytest.fixture
def session_service():
    return InMemorySessionService()

def load_dataset():
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)

@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_dataset())
async def test_evaluate_agent_interaction(case, sample_agent, evaluator_agent, session_service):
    user_query = case["query"]
    expected_response = case["expected_response"]
    expected_tools = case["expected_tools"]
    
    # --- Step 1: Run Sample Agent ---
    sample_runner = Runner(agent=sample_agent, app_name="eval_test", session_service=session_service)
    session_id = f"session_{hash(user_query)}" # Unique ID per test
    await session_service.create_session(app_name="eval_test", user_id="test_user", session_id=session_id)
    
    user_content = types.Content(role='user', parts=[types.Part(text=user_query)])
    
    actual_response_text = ""
    actual_tool_calls = []
    
    async for event in sample_runner.run_async(user_id="test_user", session_id=session_id, new_message=user_content):
        # Inspect events for tool calls and final response
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    actual_tool_calls.append(part.function_call.name)
                if part.text:
                    actual_response_text = part.text # Keep updating, last one is usually final answer
    
    # --- Step 2: Prepare Evaluation Input ---
    eval_input = {
        "user_query": user_query,
        "expected_response": expected_response,
        "actual_response": actual_response_text,
        "actual_tool_calls": actual_tool_calls,
        "expected_tool_calls": expected_tools
    }
    
    # --- Step 3: Run Evaluator Agent ---
    eval_runner = Runner(agent=evaluator_agent, app_name="eval_judge", session_service=session_service)
    eval_session_id = f"eval_{session_id}"
    await session_service.create_session(app_name="eval_judge", user_id="judge", session_id=eval_session_id)
    
    # The input schema expects a JSON string or dict? ADK normally takes text for LlmAgent unless handled specially.
    # But we defined input_schema, so we should send JSON string.
    eval_content = types.Content(role='user', parts=[types.Part(text=json.dumps(eval_input))])
    
    eval_result_json = ""
    async for event in eval_runner.run_async(user_id="judge", session_id=eval_session_id, new_message=eval_content):
        if event.content and event.content.parts:
             for part in event.content.parts:
                 if part.text:
                     eval_result_json = part.text

    # --- Step 4: Parse and Assert ---
    print(f"\nQUERY: {user_query}")
    print(f"ACTUAL TOOLS: {actual_tool_calls}")
    print(f"ACTUAL TEXT: {actual_response_text}")
    print(f"EVAL RESULT: {eval_result_json}")

    try:
        # Clean up markdown code blocks if present
        cleaned_json = eval_result_json.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(cleaned_json)
        
        # Verify Score Thresholds
        assert result_dict["tool_use_correctness"] >= 7, f"Tool use score too low: {result_dict['tool_use_correctness']}"
        assert result_dict["response_quality"] >= 7, f"Quality score too low: {result_dict['response_quality']}"
        assert result_dict["hallucination_percentage"] <= 20, f"Hallucination too high: {result_dict['hallucination_percentage']}"
        assert result_dict["json_compliance"] == True, "Judge said JSON was not compliant (ironic if this parsing worked)"

    except json.JSONDecodeError:
        pytest.fail(f"Evaluator failed to return valid JSON: {eval_result_json}")
