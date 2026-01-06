"""
Example: Using Official Vertex AI GenAI Evaluation SDK

This script demonstrates how to use the new GenAI SDK integration
for comprehensive agent and model evaluation.
"""

import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from our new GenAI SDK modules
from src.evaluation import (
    # GenAI Client
    get_client,
    run_model_inference,
    
    # Rubric Metrics
    AdaptiveRubricMetrics,
    StaticRubricMetrics,
    AgentMetrics,
    ComputationMetrics,
    GenAIEvaluator,
    MetricPresets,
    create_custom_metric,
    
    # Agent Evaluation  
    AgentEvaluator,
    TrajectoryMetrics,
    create_trajectory_dataset,
    
    # Multi-Turn
    MultiTurnEvaluator,
    quick_multi_turn_eval,
)


# ============================================================
# EXAMPLE 1: BASIC TEXT GENERATION EVALUATION
# ============================================================

def example_text_generation():
    """Evaluate text generation responses."""
    print("=" * 60)
    print("Example 1: Text Generation Evaluation")
    print("=" * 60)
    
    # Create test dataset
    eval_df = pd.DataFrame({
        "prompt": [
            "Explain software 'technical debt' using a garden analogy.",
            "Write a 4-line poem about a lonely robot.",
            "What is the minimum number of socks to guarantee a matching pair?",
        ]
    })
    
    # Initialize evaluator
    evaluator = GenAIEvaluator()
    
    # Option 1: Generate responses first
    eval_dataset = run_model_inference(
        model="gemini-2.5-flash",
        dataset=eval_df
    )
    
    # Show inference results
    print("\nInference Results:")
    eval_dataset.show()
    
    # Run evaluation with metrics
    eval_result = evaluator.evaluate(
        dataset=eval_dataset,
        metrics=[
            AdaptiveRubricMetrics.general_quality(),
            AdaptiveRubricMetrics.text_quality(),
            StaticRubricMetrics.safety(),
        ]
    )
    
    # Display results
    print("\nEvaluation Results:")
    eval_result.show()


# ============================================================
# EXAMPLE 2: EVALUATION WITH CUSTOM GUIDELINES
# ============================================================

def example_custom_guidelines():
    """Evaluate with custom guidelines for GENERAL_QUALITY."""
    print("=" * 60)
    print("Example 2: Custom Guidelines Evaluation")
    print("=" * 60)
    
    eval_df = pd.DataFrame({
        "prompt": [
            "Should I invest in Bitcoin?",
            "How do I treat a cold?",
        ],
        "response": [
            "Bitcoin can be volatile. I recommend consulting a financial advisor.",
            "Rest, fluids, and over-the-counter medications can help. See a doctor if symptoms worsen.",
        ]
    })
    
    evaluator = GenAIEvaluator()
    
    # Use custom guidelines
    result = evaluator.evaluate(
        dataset=eval_df,
        metrics=[
            AdaptiveRubricMetrics.general_quality(
                guidelines="Response must be professional and must NOT provide financial or medical advice."
            ),
            StaticRubricMetrics.safety(),
        ]
    )
    
    result.show()


# ============================================================
# EXAMPLE 3: COMPUTATION-BASED METRICS
# ============================================================

def example_computation_metrics():
    """Evaluate with ground truth comparison."""
    print("=" * 60)
    print("Example 3: Computation-Based Metrics (requires reference)")
    print("=" * 60)
    
    eval_df = pd.DataFrame({
        "prompt": [
            "What is the capital of France?",
            "Who wrote Hamlet?",
        ],
        "response": [
            "The capital of France is Paris.",
            "William Shakespeare wrote Hamlet.",
        ],
        "reference": [
            "Paris",
            "William Shakespeare",
        ]
    })
    
    evaluator = GenAIEvaluator()
    
    result = evaluator.evaluate(
        dataset=eval_df,
        metrics=[
            ComputationMetrics.bleu(),
            ComputationMetrics.rouge_l(),
            ComputationMetrics.exact_match(),
        ]
    )
    
    result.show()


# ============================================================
# EXAMPLE 4: TRAJECTORY EVALUATION
# ============================================================

def example_trajectory_evaluation():
    """Evaluate agent trajectories."""
    print("=" * 60)
    print("Example 4: Trajectory Evaluation")
    print("=" * 60)
    
    # Define trajectories
    predicted = [
        [
            {"tool_name": "search_products", "tool_input": {"query": "headphones"}},
            {"tool_name": "get_product_details", "tool_input": {"product_id": "B08H8"}},
        ],
        [
            {"tool_name": "search_products", "tool_input": {"query": "laptop"}},
        ],
    ]
    
    reference = [
        [
            {"tool_name": "search_products", "tool_input": {"query": "headphones"}},
            {"tool_name": "get_product_details", "tool_input": {"product_id": "B08H8"}},
        ],
        [
            {"tool_name": "search_products", "tool_input": {"query": "laptop"}},
            {"tool_name": "compare_products", "tool_input": {"ids": ["A", "B"]}},
        ],
    ]
    
    evaluator = AgentEvaluator()
    
    result = evaluator.evaluate_trajectories(
        predicted=predicted,
        reference=reference,
        metrics=[
            TrajectoryMetrics.exact_match(),
            TrajectoryMetrics.precision(),
            TrajectoryMetrics.recall(),
        ]
    )
    
    result.show()


# ============================================================
# EXAMPLE 5: MULTI-TURN CONVERSATION EVALUATION
# ============================================================

def example_multi_turn():
    """Evaluate multi-turn conversations."""
    print("=" * 60)
    print("Example 5: Multi-Turn Conversation Evaluation")
    print("=" * 60)
    
    # Single evaluation
    result = quick_multi_turn_eval(
        current_prompt="What's the best time to visit?",
        response="Spring is the best time to visit Paris! The weather is pleasant.",
        history=[
            {"role": "user", "text": "I'm planning a trip to Paris."},
            {"role": "model", "text": "That sounds wonderful! What would you like to know?"},
        ]
    )
    
    result.show()
    
    # Batch evaluation
    evaluator = MultiTurnEvaluator()
    
    conversations = [
        {
            "prompt": "What's the weather like?",
            "response": "It's sunny and 72°F today!",
            "conversation_history": [
                {"role": "user", "parts": [{"text": "Hello!"}]},
                {"role": "model", "parts": [{"text": "Hi there!"}]},
            ]
        },
        {
            "prompt": "Thank you!",
            "response": "You're welcome! Have a great day!",
            "conversation_history": [
                {"role": "user", "parts": [{"text": "What's the weather?"}]},
                {"role": "model", "parts": [{"text": "It's sunny!"}]},
            ]
        },
    ]
    
    batch_result = evaluator.evaluate(conversations)
    batch_result.show()


# ============================================================
# EXAMPLE 6: CUSTOM FUNCTION METRIC
# ============================================================

def example_custom_metric():
    """Create and use custom metrics."""
    print("=" * 60)
    print("Example 6: Custom Function Metric")
    print("=" * 60)
    
    # Define custom metric function
    def keyword_check(instance: dict) -> dict:
        """Check if response contains the word 'sorry'."""
        response = instance.get("response", "")
        contains_sorry = "sorry" in response.lower()
        return {"score": 0.0 if contains_sorry else 1.0}
    
    # Create metric
    no_sorry_metric = create_custom_metric("no_sorry_check", keyword_check)
    
    eval_df = pd.DataFrame({
        "prompt": [
            "What's 2+2?",
            "Can you fly?",
        ],
        "response": [
            "2+2 equals 4.",
            "Sorry, I cannot fly as I am an AI.",
        ]
    })
    
    evaluator = GenAIEvaluator()
    
    result = evaluator.evaluate(
        dataset=eval_df,
        metrics=[
            no_sorry_metric,
            StaticRubricMetrics.safety(),
        ]
    )
    
    result.show()


# ============================================================
# EXAMPLE 7: MODEL COMPARISON
# ============================================================

def example_model_comparison():
    """Compare multiple models on same prompts."""
    print("=" * 60)
    print("Example 7: Model Comparison")
    print("=" * 60)
    
    prompts_df = pd.DataFrame({
        "prompt": [
            "Explain quantum computing in simple terms.",
            "Write a haiku about AI.",
        ]
    })
    
    evaluator = GenAIEvaluator()
    
    result = evaluator.compare_models(
        prompts=prompts_df,
        models=["gemini-2.0-flash", "gemini-2.5-flash"],
        metrics=[
            AdaptiveRubricMetrics.text_quality(),
            AdaptiveRubricMetrics.instruction_following(),
        ]
    )
    
    result.show()


# ============================================================
# EXAMPLE 8: USING METRIC PRESETS
# ============================================================

def example_metric_presets():
    """Use pre-configured metric bundles."""
    print("=" * 60)
    print("Example 8: Metric Presets")
    print("=" * 60)
    
    # Summarization preset
    print("\nSummarization Metrics:", MetricPresets.summarization())
    
    # Agent evaluation preset
    print("\nAgent Evaluation Metrics:", MetricPresets.agent_evaluation())
    
    # QA with reference preset
    print("\nQA Metrics:", MetricPresets.qa_with_reference())
    
    # Multi-turn chat preset
    print("\nMulti-Turn Chat Metrics:", MetricPresets.multi_turn_chat())


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Vertex AI GenAI Evaluation SDK Examples")
    print("=" * 60 + "\n")
    
    # Run examples (comment out ones you don't want to run)
    try:
        example_metric_presets()  # No API calls, always works
        
        # Uncomment to run live examples (requires API credentials):
        # example_text_generation()
        # example_custom_guidelines()
        # example_computation_metrics()
        # example_trajectory_evaluation()
        # example_multi_turn()
        # example_custom_metric()
        # example_model_comparison()
        
        print("\n✅ Examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure you have:")
        print("  1. Set GOOGLE_CLOUD_PROJECT environment variable")
        print("  2. Set GOOGLE_CLOUD_LOCATION environment variable")
        print("  3. Authenticated with: gcloud auth application-default login")
