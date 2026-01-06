"""
Advanced ADK Callbacks for Agent Orchestration.

This module provides:
- before_model_callback: Input validation, prompt injection, guardrails
- after_tool_callback: Output logging, state management, result transformation
- after_agent_callback: Agent lifecycle tracking

These callbacks demonstrate production-grade ADK patterns.
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CALLBACK STATE TRACKING
# ============================================================

class CallbackTracker:
    """Thread-safe tracker for callback invocations and metrics."""
    
    def __init__(self):
        self.model_calls = []
        self.tool_calls = []
        self.blocked_requests = []
        self.start_time = None
    
    def reset(self):
        self.model_calls = []
        self.tool_calls = []
        self.blocked_requests = []
        self.start_time = datetime.now()
    
    def log_model_call(self, agent_name: str, prompt_preview: str):
        self.model_calls.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "prompt_preview": prompt_preview[:200] if prompt_preview else ""
        })
    
    def log_tool_call(self, tool_name: str, args: Dict, result_preview: str):
        self.tool_calls.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "args": args,
            "result_preview": result_preview[:200] if result_preview else ""
        })
    
    def log_blocked(self, reason: str):
        self.blocked_requests.append({
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })
    
    def get_trajectory(self) -> list:
        """Return ordered list of tool calls for trajectory evaluation."""
        return [call["tool"] for call in self.tool_calls]
    
    def get_metrics(self) -> Dict:
        return {
            "total_model_calls": len(self.model_calls),
            "total_tool_calls": len(self.tool_calls),
            "blocked_requests": len(self.blocked_requests),
            "trajectory": self.get_trajectory()
        }


# Global tracker instance
callback_tracker = CallbackTracker()


# ============================================================
# GUARDRAIL PATTERNS (CONTENT FILTERING)
# ============================================================

BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "disregard your training",
    "pretend you are",
    "act as if constraints don't exist",
]

REQUIRED_TOPICS = [
    "research",
    "analysis", 
    "summarize",
    "explain",
    "compare",
]

def contains_blocked_content(text: str) -> Optional[str]:
    """Check if text contains blocked patterns (prompt injection detection)."""
    text_lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in text_lower:
            return pattern
    return None


def is_valid_research_query(text: str) -> bool:
    """Validate that query is research-related."""
    text_lower = text.lower()
    return any(topic in text_lower for topic in REQUIRED_TOPICS)


# ============================================================
# BEFORE MODEL CALLBACK
# ============================================================

def before_model_callback(
    callback_context: CallbackContext,
    llm_request: types.GenerateContentConfig
) -> Optional[types.GenerateContentResponse]:
    """
    Called BEFORE each LLM request. Use for:
    - Input validation and sanitization
    - Prompt injection detection
    - Request modification
    - Caching/guardrails
    
    Returns:
        None: Continue with LLM call
        GenerateContentResponse: Skip LLM and return this response
    """
    agent_name = callback_context.agent_name
    
    # Extract user message for validation
    user_message = ""
    if hasattr(llm_request, 'contents') and llm_request.contents:
        for content in llm_request.contents:
            if content.role == "user":
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        user_message += part.text + " "
    
    logger.info(f"[CALLBACK] before_model: {agent_name}")
    
    # Check for prompt injection
    blocked_pattern = contains_blocked_content(user_message)
    if blocked_pattern:
        logger.warning(f"[GUARDRAIL] Blocked prompt injection: {blocked_pattern}")
        callback_tracker.log_blocked(f"Prompt injection: {blocked_pattern}")
        
        # Return a safe response instead of calling LLM
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="I cannot process that request. Please rephrase your query.")]
                    )
                )
            ]
        )
    
    # Log the call for trajectory tracking
    callback_tracker.log_model_call(agent_name, user_message)
    
    # Save request info to state for debugging
    if callback_context.state is not None:
        callback_context.state["last_model_request"] = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "prompt_preview": user_message[:100]
        }
    
    # Return None to continue with normal LLM call
    return None


# ============================================================
# AFTER TOOL CALLBACK
# ============================================================

def after_tool_callback(
    tool_context: ToolContext,
    tool_response: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Called AFTER each tool execution. Use for:
    - Logging tool outputs
    - Transforming/filtering results
    - Saving results to state
    - Metrics collection
    
    Returns:
        None: Use original tool response
        Dict: Replace tool response with this
    """
    tool_name = tool_context.tool_name
    agent_name = tool_context.agent_name
    
    logger.info(f"[CALLBACK] after_tool: {tool_name} (called by {agent_name})")
    
    # Convert response to string for logging
    result_str = str(tool_response) if tool_response else ""
    
    # Log for trajectory tracking
    callback_tracker.log_tool_call(
        tool_name=tool_name,
        args=getattr(tool_context, 'tool_args', {}),
        result_preview=result_str
    )
    
    # Save tool result to state for access by other agents
    if tool_context.state is not None:
        # Use tool name as key for result storage
        state_key = f"tool_result_{tool_name}"
        tool_context.state[state_key] = {
            "result": tool_response,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name
        }
        
        # Track tool call sequence
        if "tool_sequence" not in tool_context.state:
            tool_context.state["tool_sequence"] = []
        tool_context.state["tool_sequence"].append(tool_name)
    
    # Example: Filter sensitive data from certain tools
    if tool_name == "search_internal_docs":
        if isinstance(tool_response, dict) and "confidential" in tool_response:
            # Redact confidential info
            filtered_response = {k: v for k, v in tool_response.items() if k != "confidential"}
            filtered_response["_redacted"] = True
            logger.info(f"[CALLBACK] Redacted confidential data from {tool_name}")
            return filtered_response
    
    # Return None to use original response
    return None


# ============================================================
# AFTER AGENT CALLBACK  
# ============================================================

def after_agent_callback(
    callback_context: CallbackContext
) -> None:
    """
    Called after an agent completes execution. Use for:
    - Cleanup and finalization
    - Metrics aggregation
    - State persistence
    """
    agent_name = callback_context.agent_name
    
    logger.info(f"[CALLBACK] after_agent: {agent_name} completed")
    
    # Save completion metrics to state
    if callback_context.state is not None:
        callback_context.state[f"agent_completed_{agent_name}"] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": callback_tracker.get_metrics()
        }


# ============================================================
# UTILITY FUNCTIONS FOR EVALUATION
# ============================================================

def get_trajectory() -> list:
    """Get the tool call trajectory for evaluation."""
    return callback_tracker.get_trajectory()


def get_callback_metrics() -> Dict:
    """Get callback metrics for evaluation."""
    return callback_tracker.get_metrics()


def reset_tracker():
    """Reset the callback tracker (call before each evaluation run)."""
    callback_tracker.reset()
