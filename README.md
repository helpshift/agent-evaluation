# Agent Evaluation Project

A sophisticated AI agent evaluation framework using the [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) and **Vertex AI GenAI Evaluation Service** for production-scale testing.

## 🚀 Quick Start

### Launch ADK Web
```bash
adk web src/agents
```
Then open http://127.0.0.1:8000

### Run ADK CLI Evaluation
```bash
adk eval src/agents src/agents/story_agent.evalset.json
```

### Run Pytest Evaluation
```bash
pytest tests/test_vertex_eval.py -v -s
```

## 📁 Project Structure

```
src/agents/
├── story_flow_agent.py        # Sophisticated Custom Agent (StoryFlow pattern)
├── sample_agent.py            # Simple Calculator Agent
├── evaluator_agent.py         # LLM-as-Judge Agent
├── orchestrator_agent.py      # Evaluation Pipeline Orchestrator
├── story_agent.evalset.json   # ADK Evalset Format
└── test_config.json           # Evaluation Criteria Config

src/evaluation/
└── vertex_ai_evaluator.py     # Vertex AI GenAI Evaluation Service

tests/
├── test_vertex_eval.py        # Vertex AI evaluation tests
├── test_story_eval.py         # Story agent evaluation
└── data/
    ├── story_eval_dataset.json   # 50-case golden dataset
    └── evaluation_results.json   # Results output
```

## 📊 Evaluation Metrics

### ADK Built-in Criteria
| Metric | Description |
|--------|-------------|
| `tool_trajectory_avg_score` | Exact match of tool call trajectory |
| `response_match_score` | ROUGE-1 similarity to reference |
| `final_response_match_v2` | LLM-judged semantic match |
| `rubric_based_tool_use_quality_v1` | LLM-judged tool usage quality |
| `hallucinations_v1` | Groundedness check |
| `safety_v1` | Safety/harmlessness check |

### Vertex AI GenAI Metrics
| Metric | Description |
|--------|-------------|
| `coherence` | Logical flow and structure |
| `fluency` | Grammar and readability |
| `groundedness` | Factual accuracy |
| `summarization_quality` | Summary effectiveness |

## 🧠 StoryFlowAgent Architecture

The StoryFlowAgent demonstrates ADK best practices:

1. **Custom Orchestration** - Implements `BaseAgent._run_async_impl`
2. **LoopAgent** - Iterative critique/revision (max 3 iterations)
3. **SequentialAgent** - Post-processing pipeline
4. **Conditional Logic** - Regenerate if tone is negative

```
StoryFlowAgent (Custom BaseAgent)
├─ StoryGenerator (LlmAgent)
├─ CriticReviserLoop (LoopAgent)
│  ├─ Critic (LlmAgent)
│  └─ Reviser (LlmAgent)
└─ PostProcessing (SequentialAgent)
   ├─ GrammarCheck (LlmAgent)
   └─ ToneCheck (LlmAgent)
```

## 🔬 Evaluation Framework

### 1. ADK CLI Evaluation
```bash
# Run with default criteria
adk eval src/agents src/agents/story_agent.evalset.json

# Run with custom config
adk eval src/agents src/agents/story_agent.evalset.json --config test_config.json
```

### 2. Vertex AI Programmatic Evaluation
```python
from src.evaluation.vertex_ai_evaluator import run_evaluation
from src.agents.story_flow_agent import root_agent

result = await run_evaluation(
    agent=root_agent,
    evalset_path="src/agents/story_agent.evalset.json",
    output_path="tests/data/vertex_eval_results.json",
    use_vertex_ai=True
)

print(f"Avg Trajectory Score: {result.avg_trajectory_score}")
print(f"Avg Coherence: {result.avg_coherence}")
print(f"Avg Groundedness: {result.avg_groundedness}")
```

### 3. Pytest Suite
```bash
# All evaluation tests
pytest tests/test_vertex_eval.py -v -s

# Trajectory tests only
pytest tests/test_vertex_eval.py::TestTrajectoryEvaluation -v

# Full pipeline integration
pytest tests/test_vertex_eval.py::TestFullPipeline -v
```

## 🔧 Setup

```bash
pip install google-adk pytest python-dotenv pandas vertexai
```

Create `.env`:
```
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```
