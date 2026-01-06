"""
Script to demonstrate the Universal Evaluator Agent.
It initializes the evaluator and asks it to evaluate the 'StoryTeller' sample agent.
"""

import asyncio
import logging
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from src.universal_evaluator import universal_evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo_runner")

APP_NAME = "universal_eval_demo"
USER_ID = "admin"
SESSION_ID = "demo_session_001"

async def main():
    print("\n" + "="*60)
    print("UNIVERSAL EVALUATOR DEMO")
    print("="*60 + "\n")
    
    # 1. Setup Session
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    
    # 2. Setup Runner with the Evaluator Agent
    runner = Runner(
        agent=universal_evaluator,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # 3. Create instruction for the Evaluator
    # "Please evaluate the StoryTeller agent."
    user_msg = "Please evaluate the StoryTeller agent."
    content = types.Content(role="user", parts=[types.Part(text=user_msg)])
    
    print(f"User: {user_msg}\n")
    print("Evaluator is thinking...\n")
    
    # 4. Run loop
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    print(f"🛠️  Tool Call from {event.author}: {part.function_call.name}")
        
        if event.is_final_response():
             print(f"\n🤖 {event.author}: {event.content.parts[0].text}\n")
        elif event.content:
             pass

if __name__ == "__main__":
    asyncio.run(main())
