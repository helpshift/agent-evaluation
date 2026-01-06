"""
Multi-Turn Conversation Evaluation Module.

Provides evaluation for multi-turn dialogues using:
- MULTI_TURN_GENERAL_QUALITY (adaptive rubrics)
- MULTI_TURN_TEXT_QUALITY (text quality in dialogue)
- Conversation history extraction from various formats
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from vertexai import types

from src.evaluation.genai_client import get_client
from src.evaluation.rubric_evaluator import MultiTurnMetrics, StaticRubricMetrics


# ============================================================
# CONVERSATION DATASET PREPARATION
# ============================================================

def create_multi_turn_dataset(
    conversations: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create dataset for multi-turn evaluation.
    
    Args:
        conversations: List of conversation dicts with:
            - prompt: Current user message
            - response: Model response to evaluate
            - history: List of previous turns (optional, extracted automatically)
    
    Returns:
        DataFrame formatted for multi-turn evaluation
    """
    return pd.DataFrame(conversations)


def format_gemini_conversation(
    turns: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Format a conversation in Gemini batch prediction style.
    
    Args:
        turns: List of {"role": "user"/"model", "text": "..."}
    
    Returns:
        Dict with prompt (current), response, and conversation_history
    
    Example:
        conv = format_gemini_conversation([
            {"role": "user", "text": "Hi!"},
            {"role": "model", "text": "Hello!"},
            {"role": "user", "text": "How are you?"},
            {"role": "model", "text": "I'm doing well!"}
        ])
    """
    if len(turns) < 2:
        raise ValueError("Need at least 2 turns (user + model)")
    
    # Last user message is the prompt
    # Last model message is the response
    # Everything before is history
    
    history = []
    for turn in turns[:-2]:
        history.append({
            "role": turn["role"],
            "parts": [{"text": turn["text"]}]
        })
    
    current_prompt = turns[-2]  # Second to last (user)
    current_response = turns[-1]  # Last (model)
    
    return {
        "prompt": {"role": "user", "parts": [{"text": current_prompt["text"]}]},
        "response": {"role": "model", "parts": [{"text": current_response["text"]}]},
        "conversation_history": history
    }


def format_openai_conversation(
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Format OpenAI Chat Completion format for evaluation.
    
    Args:
        messages: List of {"role": "system"/"user"/"assistant", "content": "..."}
    
    Returns:
        Dict with prompt, response, and history
    """
    history = []
    system_instruction = None
    
    for msg in messages[:-2]:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            history.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })
    
    current_prompt = messages[-2]  # User
    current_response = messages[-1]  # Assistant
    
    return {
        "prompt": current_prompt["content"],
        "response": current_response["content"],
        "conversation_history": history,
        "system_instruction": system_instruction
    }


# ============================================================
# MULTI-TURN EVALUATOR
# ============================================================

class MultiTurnEvaluator:
    """
    Evaluator for multi-turn conversations.
    
    Automatically handles:
    - Gemini format conversations
    - OpenAI Chat Completion format
    - Custom conversation formats
    """
    
    def __init__(self):
        self.client = get_client()
    
    def evaluate(
        self,
        conversations: List[Dict[str, Any]],
        include_safety: bool = True
    ):
        """
        Evaluate multi-turn conversations.
        
        Args:
            conversations: List of conversation dicts
            include_safety: Include safety evaluation
        
        Returns:
            Evaluation result with .show() method
        """
        dataset = create_multi_turn_dataset(conversations)
        
        metrics = [
            MultiTurnMetrics.general_quality(),
            MultiTurnMetrics.text_quality(),
        ]
        
        if include_safety:
            metrics.append(StaticRubricMetrics.safety())
        
        return self.client.evals.evaluate(
            dataset=dataset,
            metrics=metrics
        )
    
    def evaluate_from_gemini_format(
        self,
        conversations: List[List[Dict[str, str]]]
    ):
        """
        Evaluate conversations in Gemini format.
        
        Args:
            conversations: List of turn lists, each:
                [{"role": "user", "text": "..."}, ...]
        """
        formatted = [format_gemini_conversation(conv) for conv in conversations]
        return self.evaluate(formatted)
    
    def evaluate_from_openai_format(
        self,
        conversations: List[List[Dict[str, str]]]
    ):
        """
        Evaluate conversations in OpenAI format.
        
        Args:
            conversations: List of message lists:
                [{"role": "user"/"assistant", "content": "..."}, ...]
        """
        formatted = [format_openai_conversation(conv) for conv in conversations]
        return self.evaluate(formatted)


# ============================================================
# QUICK EVALUATION FUNCTIONS
# ============================================================

def quick_multi_turn_eval(
    current_prompt: str,
    response: str,
    history: List[Dict[str, str]]
):
    """
    Quick evaluation of a single multi-turn response.
    
    Args:
        current_prompt: Current user message
        response: Model response to evaluate
        history: Previous turns as list of {"role": "...", "text": "..."}
    
    Example:
        result = quick_multi_turn_eval(
            current_prompt="How's the weather?",
            response="The weather is sunny today!",
            history=[
                {"role": "user", "text": "Hello"},
                {"role": "model", "text": "Hi! How can I help?"}
            ]
        )
        result.show()
    """
    evaluator = MultiTurnEvaluator()
    
    formatted_history = [
        {"role": h["role"], "parts": [{"text": h["text"]}]}
        for h in history
    ]
    
    conversation = {
        "prompt": current_prompt,
        "response": response,
        "conversation_history": formatted_history
    }
    
    return evaluator.evaluate([conversation])


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Dataset Helpers
    "create_multi_turn_dataset",
    "format_gemini_conversation",
    "format_openai_conversation",
    
    # Evaluator
    "MultiTurnEvaluator",
    
    # Quick Functions
    "quick_multi_turn_eval",
]
