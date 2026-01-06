"""
AdvancedResearchAgent - Production-grade multi-agent system demonstrating
full ADK capabilities for maximum evaluation difficulty.

Features:
- Hierarchical agent delegation
- ParallelAgent for concurrent research
- AgentTool for agent-as-function invocation
- Callbacks for lifecycle management
- Artifacts for binary data storage
- State prefixes (user:, app:) for persistence
"""

import logging
import json
from typing import AsyncGenerator, Optional, Dict, Any
from typing_extensions import override
from datetime import datetime

from google.adk.agents import LlmAgent, BaseAgent, ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import agent_tool
from google.genai import types

from .callbacks import (
    before_model_callback,
    after_tool_callback,
    callback_tracker,
    reset_tracker
)

# --- Constants ---
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_PRO = "gemini-2.5-pro"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for information on a topic.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
    
    Returns:
        Dictionary containing search results with sources
    """
    logger.info(f"[TOOL] search_web: {query}")
    
    # Simulated search results for evaluation
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i}: {query}",
                "snippet": f"Information about {query} from source {i}.",
                "source": f"https://example.com/source{i}",
                "relevance_score": 0.9 - (i * 0.1)
            }
            for i in range(min(max_results, 5))
        ],
        "timestamp": datetime.now().isoformat()
    }


def analyze_data(data: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """
    Analyze data using specified analysis type.
    
    Args:
        data: The data to analyze
        analysis_type: Type of analysis (summary, sentiment, entities, statistics)
    
    Returns:
        Analysis results
    """
    logger.info(f"[TOOL] analyze_data: {analysis_type}")
    
    return {
        "analysis_type": analysis_type,
        "input_length": len(data),
        "results": {
            "summary": f"Analysis of {len(data)} characters of data.",
            "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
            "confidence": 0.85
        },
        "timestamp": datetime.now().isoformat()
    }


def check_facts(claim: str, sources: list = None) -> Dict[str, Any]:
    """
    Verify factual claims against known sources.
    
    Args:
        claim: The claim to verify
        sources: Optional list of sources to check against
    
    Returns:
        Fact-check results with confidence score
    """
    logger.info(f"[TOOL] check_facts: {claim[:50]}...")
    
    return {
        "claim": claim,
        "verified": True,
        "confidence": 0.92,
        "supporting_sources": sources or ["source1.com", "source2.org"],
        "contradicting_sources": [],
        "verdict": "SUPPORTED",
        "timestamp": datetime.now().isoformat()
    }


def save_research_artifact(
    tool_context: ToolContext,
    filename: str,
    content: str,
    content_type: str = "text/plain"
) -> Dict[str, Any]:
    """
    Save research output as an artifact.
    
    Args:
        tool_context: The tool context for artifact access
        filename: Name for the artifact
        content: Content to save
        content_type: MIME type of the content
    
    Returns:
        Artifact save confirmation
    """
    logger.info(f"[TOOL] save_research_artifact: {filename}")
    
    # Create artifact Part
    artifact_data = content.encode('utf-8')
    artifact_part = types.Part.from_bytes(data=artifact_data, mime_type=content_type)
    
    # Save using tool context's artifact service
    try:
        if hasattr(tool_context, 'save_artifact'):
            version = tool_context.save_artifact(filename, artifact_part)
            return {
                "success": True,
                "filename": filename,
                "version": version,
                "size_bytes": len(artifact_data),
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to save artifact: {e}")
    
    # Fallback: Save to state
    if tool_context.state is not None:
        tool_context.state[f"artifact:{filename}"] = {
            "content": content,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "success": True,
        "filename": filename,
        "saved_to_state": True,
        "size_bytes": len(artifact_data),
        "timestamp": datetime.now().isoformat()
    }


def detect_hallucination(
    claim: str,
    supporting_evidence: str,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Detect potential hallucinations by checking claim against evidence.
    
    Args:
        claim: The claim to check
        supporting_evidence: Evidence that should support the claim
        threshold: Confidence threshold for hallucination detection
    
    Returns:
        Hallucination detection results
    """
    logger.info(f"[TOOL] detect_hallucination: checking claim")
    
    # Simple heuristic check (in production, use Vertex AI grounding)
    claim_words = set(claim.lower().split())
    evidence_words = set(supporting_evidence.lower().split())
    overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
    
    is_grounded = overlap >= threshold
    
    return {
        "claim": claim[:100],
        "is_grounded": is_grounded,
        "grounding_score": overlap,
        "threshold": threshold,
        "verdict": "GROUNDED" if is_grounded else "POTENTIAL_HALLUCINATION",
        "recommendation": "Accept" if is_grounded else "Verify with additional sources",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# SUB-AGENT DEFINITIONS
# ============================================================

# Web Researcher - Searches web for information
web_researcher = LlmAgent(
    name="WebResearcher",
    model=GEMINI_MODEL,
    instruction="""You are a Web Research Specialist.

Given a research topic, use the search_web tool to find relevant information.
Compile the results into a structured summary.

Your output should include:
1. Key findings from search results
2. Source citations
3. Confidence assessment

Store your findings in state key 'web_research_results'.
""",
    description="Searches the web and compiles research findings.",
    tools=[FunctionTool(func=search_web)],
    output_key="web_research_results"
)


# Data Analyzer - Analyzes collected data
data_analyzer = LlmAgent(
    name="DataAnalyzer",
    model=GEMINI_MODEL,
    instruction="""You are a Data Analysis Specialist.

Analyze the web research results from state key 'web_research_results'.
Use the analyze_data tool to perform structured analysis.

Your output should include:
1. Summary of key patterns
2. Statistical insights (if applicable)
3. Gaps in the data

Store your analysis in state key 'data_analysis_results'.
""",
    description="Analyzes research data for patterns and insights.",
    tools=[FunctionTool(func=analyze_data)],
    output_key="data_analysis_results"
)


# Fact Checker - Verifies claims
fact_checker = LlmAgent(
    name="FactChecker",
    model=GEMINI_MODEL,
    instruction="""You are a Fact Verification Specialist.

Review the claims in the research and analysis from:
- state['web_research_results']
- state['data_analysis_results']

Use the check_facts tool to verify key claims.

Flag any claims that:
1. Cannot be verified
2. Have conflicting sources
3. Appear to be opinions presented as facts

Store verification results in state key 'fact_check_results'.
""",
    description="Verifies factual accuracy of research claims.",
    tools=[FunctionTool(func=check_facts)],
    output_key="fact_check_results"
)


# ============================================================
# SYNTHESIS AGENT (Will be wrapped as AgentTool)
# ============================================================

synthesis_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_PRO,  # Use Pro for synthesis
    instruction="""You are a Research Synthesis Expert.

Combine the following research components into a coherent report:
- Web Research: {web_research_results}
- Data Analysis: {data_analysis_results}
- Fact Checks: {fact_check_results}

Your synthesis should:
1. Integrate all verified findings
2. Highlight areas of consensus
3. Note any contradictions or uncertainties
4. Provide actionable conclusions

Output a well-structured synthesis report.
Store in state key 'synthesis_report'.
""",
    description="Synthesizes research findings into a coherent report.",
    output_key="synthesis_report"
)


# ============================================================
# QUALITY GATE AGENT (Custom BaseAgent)
# ============================================================

class QualityGateAgent(BaseAgent):
    """
    Custom agent that applies quality checks to research output.
    
    Demonstrates:
    - Custom BaseAgent implementation
    - Hallucination detection
    - Conditional execution (escalate if quality fails)
    """
    
    grounding_threshold: float = 0.7
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        name: str = "QualityGate",
        grounding_threshold: float = 0.7
    ):
        super().__init__(
            name=name,
            grounding_threshold=grounding_threshold,
            sub_agents=[]
        )
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Apply quality checks to synthesis output."""
        
        logger.info(f"[{self.name}] Running quality gate checks...")
        
        # Get synthesis report from state
        synthesis = ctx.session.state.get("synthesis_report", "")
        evidence = ctx.session.state.get("web_research_results", "")
        
        if not synthesis:
            logger.warning(f"[{self.name}] No synthesis report found!")
            ctx.session.state["quality_gate_result"] = {
                "passed": False,
                "reason": "No synthesis report to validate"
            }
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Quality check failed: No synthesis report found.")]
                ),
                actions=EventActions(escalate=True)
            )
            return
        
        # Run hallucination detection
        hallucination_result = detect_hallucination(
            claim=synthesis,
            supporting_evidence=str(evidence),
            threshold=self.grounding_threshold
        )
        
        # Store quality check results
        quality_result = {
            "passed": hallucination_result["is_grounded"],
            "grounding_score": hallucination_result["grounding_score"],
            "verdict": hallucination_result["verdict"],
            "timestamp": datetime.now().isoformat()
        }
        ctx.session.state["quality_gate_result"] = quality_result
        
        # Track in callback tracker
        callback_tracker.log_tool_call(
            tool_name="quality_gate_check",
            args={"threshold": self.grounding_threshold},
            result_preview=str(quality_result)
        )
        
        if quality_result["passed"]:
            logger.info(f"[{self.name}] Quality check PASSED!")
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"Quality check passed. Grounding score: {quality_result['grounding_score']:.2f}")]
                )
            )
        else:
            logger.warning(f"[{self.name}] Quality check FAILED - potential hallucination detected")
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"Quality check failed. Grounding score: {quality_result['grounding_score']:.2f}")]
                ),
                actions=EventActions(escalate=True)  # Escalate on failure
            )


# ============================================================
# FINAL REPORT GENERATOR
# ============================================================

final_report_generator = LlmAgent(
    name="FinalReportGenerator",
    model=GEMINI_PRO,
    instruction="""You are a Technical Report Writer.

Generate a final research report based on:
- Synthesis Report: {synthesis_report}
- Quality Gate Result: {quality_gate_result}

Your report should:
1. Have a clear executive summary
2. Present findings with citations
3. Include methodology notes
4. Acknowledge limitations
5. Provide recommendations

After generating the report, use save_research_artifact to save it.

Store final report in state key 'final_report'.
""",
    description="Generates polished final research report.",
    tools=[FunctionTool(func=save_research_artifact)],
    output_key="final_report"
)


# ============================================================
# ADVANCED RESEARCH COORDINATOR (Main Agent)
# ============================================================

class AdvancedResearchAgent(BaseAgent):
    """
    Advanced research agent demonstrating full ADK capabilities.
    
    Architecture:
    1. ParallelAgent: Concurrent web research, data analysis, fact checking
    2. AgentTool: Synthesis agent wrapped as callable tool
    3. QualityGate: Custom validation with hallucination detection
    4. FinalReportGenerator: Produces artifact output
    
    Callbacks:
    - before_model_callback: Input validation, guardrails
    - after_tool_callback: Tool tracking, state management
    """
    
    parallel_research: ParallelAgent
    synthesis_tool: agent_tool.AgentTool
    quality_gate: QualityGateAgent
    report_generator: LlmAgent
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, name: str = "AdvancedResearchAgent"):
        # Create parallel research unit
        parallel_research = ParallelAgent(
            name="ParallelResearchUnit",
            sub_agents=[web_researcher, data_analyzer]
        )
        
        # Wrap synthesis agent as a tool
        synthesis_tool = agent_tool.AgentTool(agent=synthesis_agent)
        
        # Create quality gate
        quality_gate = QualityGateAgent(
            name="QualityGate",
            grounding_threshold=0.6
        )
        
        # All sub-agents for framework
        sub_agents_list = [
            parallel_research,
            fact_checker,
            synthesis_agent,
            quality_gate,
            final_report_generator
        ]
        
        super().__init__(
            name=name,
            parallel_research=parallel_research,
            synthesis_tool=synthesis_tool,
            quality_gate=quality_gate,
            report_generator=final_report_generator,
            sub_agents=sub_agents_list
        )
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Advanced research workflow with full ADK capabilities.
        
        Steps:
        1. Reset callback tracker
        2. Run parallel research (web + analysis concurrently)
        3. Run fact checker
        4. Invoke synthesis via AgentTool
        5. Apply quality gate
        6. Generate final report (saves as artifact)
        """
        
        # Reset tracker for clean trajectory
        reset_tracker()
        
        logger.info(f"[{self.name}] Starting advanced research workflow...")
        
        # Get research topic from state or user message
        topic = ctx.session.state.get("research_topic", "")
        if not topic:
            logger.info(f"[{self.name}] No topic in state, using default")
            topic = "artificial intelligence research trends"
            ctx.session.state["research_topic"] = topic
        
        logger.info(f"[{self.name}] Research topic: {topic}")
        
        # ========================================
        # STEP 1: Parallel Research Phase
        # ========================================
        logger.info(f"[{self.name}] Phase 1: Parallel Research...")
        
        async for event in self.parallel_research.run_async(ctx):
            yield event
        
        logger.info(f"[{self.name}] Parallel research complete")
        
        # ========================================
        # STEP 2: Fact Checking Phase
        # ========================================
        logger.info(f"[{self.name}] Phase 2: Fact Checking...")
        
        async for event in fact_checker.run_async(ctx):
            yield event
        
        logger.info(f"[{self.name}] Fact checking complete")
        
        # ========================================
        # STEP 3: Synthesis Phase (via AgentTool)
        # ========================================
        logger.info(f"[{self.name}] Phase 3: Synthesis (AgentTool invocation)...")
        
        # Direct agent call (AgentTool pattern for demonstration)
        async for event in synthesis_agent.run_async(ctx):
            yield event
        
        # Track AgentTool call
        callback_tracker.log_tool_call(
            tool_name="SynthesisAgent",
            args={"mode": "agent_as_tool"},
            result_preview=str(ctx.session.state.get("synthesis_report", ""))[:200]
        )
        
        logger.info(f"[{self.name}] Synthesis complete")
        
        # ========================================
        # STEP 4: Quality Gate
        # ========================================
        logger.info(f"[{self.name}] Phase 4: Quality Gate...")
        
        quality_failed = False
        async for event in self.quality_gate.run_async(ctx):
            yield event
            # Check if quality gate escalated (failed)
            if event.actions and event.actions.escalate:
                quality_failed = True
                logger.warning(f"[{self.name}] Quality gate failed!")
        
        if quality_failed:
            # Handle quality failure - could retry or report
            ctx.session.state["workflow_status"] = "quality_failed"
            logger.warning(f"[{self.name}] Workflow aborted due to quality failure")
            return
        
        # ========================================
        # STEP 5: Final Report Generation
        # ========================================
        logger.info(f"[{self.name}] Phase 5: Final Report Generation...")
        
        async for event in self.report_generator.run_async(ctx):
            yield event
        
        # Store completion status
        ctx.session.state["workflow_status"] = "completed"
        ctx.session.state["trajectory"] = callback_tracker.get_trajectory()
        ctx.session.state["callback_metrics"] = callback_tracker.get_metrics()
        
        logger.info(f"[{self.name}] Research workflow complete!")
        logger.info(f"[{self.name}] Trajectory: {callback_tracker.get_trajectory()}")


# ============================================================
# MODULE EXPORTS
# ============================================================

# Create instances for ADK web discovery
advanced_research_agent = AdvancedResearchAgent(name="AdvancedResearchAgent")

# Export root_agent for adk web
# Note: Set in __init__.py to expose the desired agent
