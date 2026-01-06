"""
Agent package for the evaluation project.

This package exposes agents for use with `adk web`.
Run: `adk web src/agents` to launch the web interface.
"""

# Import agents to make them discoverable by ADK
from src.agents.story_flow_agent import root_agent
from src.agents.evaluator_agent import evaluator_agent

# For backwards compatibility, also expose the calculator
from src.agents.sample_agent import create_sample_agent

__all__ = ["root_agent", "evaluator_agent", "create_sample_agent"]

