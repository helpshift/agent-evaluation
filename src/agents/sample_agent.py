from google.adk.agents import LlmAgent
from src.tools.sample_tools import add, subtract, multiply, divide

def create_sample_agent(model_name: str = "gemini-2.5-flash") -> LlmAgent:
    """Creates and returns the Sample Agent."""
    
    agent = LlmAgent(
        name="calculator_agent",
        model=model_name,
        instruction="""You are a helpful Calculator Agent. 
        You can perform basic arithmetic operations using your tools.
        Always answer the user's math questions by using the appropriate tool.
        If the question is not about math, politely decline or answer generally if simple.
        """,
        tools=[add, subtract, multiply, divide]
    )
    return agent
