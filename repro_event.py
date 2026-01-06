from google.adk.events import Event
from google.genai import types
import json

try:
    content = types.Content(role="model", parts=[types.Part(text="test")])
    e = Event(source="model", author="mock_agent", content=content)
    print("Success!")
    print(e.model_dump_json())
except Exception as e:
    print(f"Validation Error: {e}")
