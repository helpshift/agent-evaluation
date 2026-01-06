"""
Sample Agents for Universal Evaluator.

This module exports instances of 6 different agent types:
1. MathAgent (Simple)
2. StoryTeller (Loop)
3. ResearchAssistant (Hierarchical)
4. SupportRouter (Dispatcher)
5. ParallelBooker (Parallel)
6. CodeFixer (Custom)
"""

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent, ParallelAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool
from typing import AsyncGenerator
import json

# Common configuration
MODEL = "gemini-2.5-flash"

# =========================================================================
# 1. MathAgent (Simple Tool Use)
# =========================================================================

def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

math_agent = LlmAgent(
    name="MathAgent",
    model=MODEL,
    instruction="You are a helpful math assistant. Use the tools provided to answer math questions.",
    tools=[add, multiply],
    description="Can perform basic addition and multiplication."
)

# =========================================================================
# 2. StoryTeller (Loop: Draft -> Critique -> Refine)
# =========================================================================

drafter = LlmAgent(
    name="Drafter",
    model=MODEL,
    instruction="Write a very short (2 sentence) story draft about the topic in state['topic']. Save to state['draft'].",
    output_key="draft"
)

critique = LlmAgent(
    name="Critic",
    model=MODEL,
    instruction="Critique the story in state['draft']. If good enough, say 'GOOD'. Else, give feedback. Save to state['feedback'].",
    output_key="feedback"
)

# Check termination condition
class QualityCheck(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        feedback = ctx.session.state.get("feedback", "")
        if "GOOD" in feedback:
            # Signal to stop loop
            yield Event(author=self.name, actions=EventActions(escalate=True))

story_teller = LoopAgent(
    name="StoryTeller",
    sub_agents=[drafter, critique, QualityCheck(name="QualityCheck")],
    max_iterations=3,
    description="Iteratively writes and refines a story."
)


# =========================================================================
# 3. ResearchAssistant (Hierarchical: Parent -> SubAgents)
# =========================================================================

searcher = LlmAgent(
    name="WebSearcher",
    model=MODEL,
    instruction="Simulate searching the web. Just return mock facts about the query.",
    description="Searches the web for information."
)

summarizer = LlmAgent(
    name="Summarizer",
    model=MODEL,
    instruction="Summarize the text provided by the user.",
    description="Summarizes text."
)

research_assistant = LlmAgent(
    name="ResearchAssistant",
    model=MODEL,
    instruction="You are a research assistant. Delegate searching to WebSearcher and summarizing to Summarizer.",
    sub_agents=[searcher, summarizer],
    description="Hierarchical agent that can search and summarize topics."
)


# =========================================================================
# 4. SupportRouter (Dispatcher)
# =========================================================================

billing = LlmAgent(
    name="BillingSupport",
    model=MODEL,
    instruction="Handle billing questions. Ask for invoice ID if missing.",
    description="Handles questions about invoices, payments, and refunds."
)

tech = LlmAgent(
    name="TechSupport",
    model=MODEL,
    instruction="Handle technical issues. Ask for error logs.",
    description="Handles technical problems, bugs, and crashes."
)

support_router = LlmAgent(
    name="SupportRouter",
    model=MODEL,
    instruction="Route the user to the appropriate specialist: BillingSupport or TechSupport.",
    sub_agents=[billing, tech],
    description="Routes support tickets to the correct department."
)


# =========================================================================
# 5. ParallelBooker (Parallel Execution)
# =========================================================================

flight_agent = LlmAgent(
    name="FlightBooker",
    model=MODEL,
    instruction="Find flights for the destination. Return a mock flight.",
    output_key="flight_options"
)

hotel_agent = LlmAgent(
    name="HotelBooker",
    model=MODEL,
    instruction="Find hotels for the destination. Return a mock hotel.",
    output_key="hotel_options"
)

parallel_booker = ParallelAgent(
    name="ParallelBooker",
    sub_agents=[flight_agent, hotel_agent],
    description="Searches for flights and hotels simultaneously."
)


# =========================================================================
# 6. CodeFixer (Custom BaseAgent)
# =========================================================================

class CustomCodeAgent(BaseAgent):
    """Demonstrates a completely custom logic flow."""
    llm: LlmAgent
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str):
        llm = LlmAgent(name="InternalLLM", model=MODEL)
        super().__init__(name=name, llm=llm, sub_agents=[llm])
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Custom logic:
        # 1. Check if 'code' is in input
        # 2. If 'error' is present, call LLM to fix
        # 3. Else, return "No error provided"
        
        user_msg = ""
        # Extract text from last message
        # Simplified for demo
        
        yield Event(author=self.name, content="Analyzing code...")
        
        # Simulate logic
        if "error" in str(ctx.session.state):
             # Delegate to internal LLM
            async for event in self.llm.run_async(ctx):
                yield event
        else:
            yield Event(author=self.name, content="Code looks fine (Custom Logic decision).")

code_fixer = CustomCodeAgent(name="CodeFixer")


# Export all
__all__ = [
    "math_agent",
    "story_teller",
    "research_assistant",
    "support_router",
    "parallel_booker",
    "code_fixer"
]
