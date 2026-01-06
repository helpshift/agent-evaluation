"""
StoryFlowAgent - A sophisticated custom agent demonstrating ADK best practices.

This agent implements the StoryFlowAgent pattern from ADK documentation:
- Custom orchestration via BaseAgent._run_async_impl
- LoopAgent for iterative critique/revision
- SequentialAgent for post-processing
- Conditional regeneration based on tone analysis
"""

import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools.tool_context import ToolContext

# --- Constants ---
GEMINI_MODEL = "gemini-2.5-flash"
COMPLETION_PHRASE = "No major issues found."

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Tool Definition ---
def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the critique indicates no further changes are needed."""
    logger.info(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {}


# --- Sub-Agent Definitions ---
story_generator = LlmAgent(
    name="StoryGenerator",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction="""You are a Creative Writing Assistant.
Write a short story (3-5 sentences) based on the topic provided in session state key 'topic'.
If no topic is found, write about a brave kitten exploring a haunted house.
Output *only* the story text.
""",
    description="Writes the initial story draft based on the topic.",
    output_key="current_story"
)

critic = LlmAgent(
    name="Critic",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction=f"""You are a Story Critic AI.

**Story to Review:**
```
{{current_story}}
```

**Task:**
Review for clarity, engagement, and coherence.
IF you identify 1-2 clear improvements needed:
  Provide specific suggestions concisely.
ELSE IF the story is good:
  Respond exactly with: "{COMPLETION_PHRASE}"

Output only the critique OR the exact phrase.
""",
    description="Reviews the current draft and provides feedback.",
    output_key="criticism"
)

reviser = LlmAgent(
    name="Reviser",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction=f"""You are a Story Reviser.

**Current Story:**
```
{{current_story}}
```

**Critique:**
{{criticism}}

**Task:**
IF the critique is exactly "{COMPLETION_PHRASE}":
  Call the 'exit_loop' function. Do NOT output any text.
ELSE:
  Apply the suggestions to improve the story.
  Output only the revised story.
""",
    description="Refines the story or exits the loop.",
    tools=[exit_loop],
    output_key="current_story"
)

grammar_check = LlmAgent(
    name="GrammarCheck",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction="""You are a Grammar Checker.

**Story to Check:**
```
{current_story}
```

Check for grammar errors.
Output corrections as a list, or "Grammar is good!" if none.
""",
    description="Checks grammar of the story.",
    output_key="grammar_suggestions"
)

tone_check = LlmAgent(
    name="ToneCheck",
    model=GEMINI_MODEL,
    include_contents='none',
    instruction="""You are a Tone Analyzer.

**Story to Analyze:**
```
{current_story}
```

Analyze the tone. Output exactly one word:
- 'positive' if generally positive
- 'negative' if generally negative
- 'neutral' otherwise
""",
    description="Analyzes the tone of the story.",
    output_key="tone_check_result"
)


# --- Custom StoryFlowAgent ---
class StoryFlowAgent(BaseAgent):
    """
    Custom agent implementing sophisticated story generation workflow.
    
    Workflow:
    1. Generate initial story
    2. Loop: Critique -> Revise (until good or max iterations)
    3. Sequential: Grammar check -> Tone check
    4. Conditional: If tone negative, regenerate
    """
    
    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent
    loop_agent: LoopAgent
    sequential_agent: SequentialAgent
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        name: str = "StoryFlowAgent",
    ):
        # Create internal workflow agents
        loop_agent = LoopAgent(
            name="CriticReviserLoop",
            sub_agents=[critic, reviser],
            max_iterations=3
        )
        
        sequential_agent = SequentialAgent(
            name="PostProcessing",
            sub_agents=[grammar_check, tone_check]
        )
        
        # Define sub_agents for framework
        sub_agents_list = [
            story_generator,
            loop_agent,
            sequential_agent,
        ]
        
        super().__init__(
            name=name,
            story_generator=story_generator,
            critic=critic,
            reviser=reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,
            loop_agent=loop_agent,
            sequential_agent=sequential_agent,
            sub_agents=sub_agents_list,
        )
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Custom orchestration logic for story workflow."""
        
        logger.info(f"[{self.name}] Starting story generation workflow.")
        
        # 1. Initial Story Generation
        logger.info(f"[{self.name}] Step 1: Generating initial story...")
        async for event in self.story_generator.run_async(ctx):
            logger.debug(f"[{self.name}] Event from StoryGenerator")
            yield event
        
        if "current_story" not in ctx.session.state:
            logger.error(f"[{self.name}] Failed to generate initial story.")
            return
        
        logger.info(f"[{self.name}] Initial story generated.")
        
        # 2. Critic-Reviser Loop
        logger.info(f"[{self.name}] Step 2: Running critique/revision loop...")
        async for event in self.loop_agent.run_async(ctx):
            yield event
        
        logger.info(f"[{self.name}] Critique loop completed.")
        
        # 3. Post-Processing (Grammar + Tone)
        logger.info(f"[{self.name}] Step 3: Running post-processing...")
        async for event in self.sequential_agent.run_async(ctx):
            yield event
        
        # 4. Conditional: Regenerate if tone is negative
        tone_result = ctx.session.state.get("tone_check_result", "neutral")
        logger.info(f"[{self.name}] Tone check result: {tone_result}")
        
        if "negative" in tone_result.lower():
            logger.info(f"[{self.name}] Tone is negative. Regenerating story...")
            async for event in self.story_generator.run_async(ctx):
                yield event
        else:
            logger.info(f"[{self.name}] Tone is acceptable. Workflow complete.")
        
        logger.info(f"[{self.name}] Story workflow finished.")


# --- Create the root agent for ADK Web ---
root_agent = StoryFlowAgent(name="StoryFlowAgent")
