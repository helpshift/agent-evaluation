from google.adk.events import Event
from google.genai import types
import json

print("Event Schema:")
try:
    print(json.dumps(Event.model_json_schema(), indent=2))
except Exception as e:
    print(e)
