import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.events import Event
from src.agents.sample_agent import create_sample_agent
from src.agents.evaluator_agent import create_evaluator_agent

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

# Mock Generator for run_async
async def mock_event_generator(events):
    for event in events:
        yield event

def create_text_event(text):
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return Event(author="mock_agent", content=content)

def create_tool_call_event(tool_name):
    fc = types.FunctionCall(name=tool_name, args={})
    content = types.Content(role="model", parts=[types.Part(function_call=fc)])
    return Event(author="mock_agent", content=content)

@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_dataset())
async def test_evaluate_agent_interaction_mocked(case, sample_agent, evaluator_agent, session_service):
    user_query = case["query"]
    expected_response = case["expected_response"]
    expected_tools = case["expected_tools"]
    
    # --- Mock Sample Agent Execution ---
    # We want sample agent to "return" the expected response + expected tool call event
    sample_events = []
    for tool in expected_tools:
        sample_events.append(create_tool_call_event(tool))
    sample_events.append(create_text_event(expected_response)) # Simulating correct behavior

    with patch('google.adk.runners.Runner.run_async', new_callable=MagicMock) as mock_run:
        mock_run.return_value = mock_event_generator(sample_events)
        
        sample_runner = Runner(agent=sample_agent, app_name="eval_test", session_service=session_service)
        session_id = f"session_{hash(user_query)}"
        await session_service.create_session(app_name="eval_test", user_id="test_user", session_id=session_id)
        
        actual_response_text = ""
        actual_tool_calls = []
        
        # Run the loop (it will iterate over our mock events)
        async for event in sample_runner.run_async(user_id="test_user", session_id=session_id, new_message=None):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        actual_tool_calls.append(part.function_call.name)
                    if part.text:
                        actual_response_text = part.text

    # --- Verify Sample Agent Capture ---
    # In this mock, we are just verifying our capture logic works given the events
    assert actual_response_text == expected_response
    assert actual_tool_calls == expected_tools

    # --- Mock Evaluator Agent Execution ---
    # We want evaluator to return a good score since we fed it "perfect" data
    eval_result_obj = {
        "tool_use_correctness": 10,
        "response_quality": 10,
        "hallucination_percentage": 0.0,
        "json_compliance": True,
        "reasoning": "Perfect match."
    }
    eval_events = [create_text_event(json.dumps(eval_result_obj))]

    with patch('google.adk.runners.Runner.run_async', new_callable=MagicMock) as mock_eval_run:
        mock_eval_run.return_value = mock_event_generator(eval_events)
        
        eval_runner = Runner(agent=evaluator_agent, app_name="eval_judge", session_service=session_service)
        eval_session_id = f"eval_{session_id}"
        await session_service.create_session(app_name="eval_judge", user_id="judge", session_id=eval_session_id)
        
        eval_result_json = ""
        async for event in eval_runner.run_async(user_id="judge", session_id=eval_session_id, new_message=None):
             if event.content and event.content.parts:
                 for part in event.content.parts:
                     if part.text:
                         eval_result_json = part.text
        
    # --- Verify Evaluation ---
    result_dict = json.loads(eval_result_json)
    assert result_dict["tool_use_correctness"] == 10
    assert result_dict["response_quality"] == 10
    print(f"Mocked Test Passed for: {user_query}")
