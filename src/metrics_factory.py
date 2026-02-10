import json
import logging
import yaml
import ast
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from vertexai.preview.evaluation.metrics import PointwiseMetric, PointwiseMetricPromptTemplate

@dataclass
class MetricPromptConfig:
    metric: str
    instruction: str
    criteria: Dict[str, str]
    rating_rubric: Dict[str, str]
    input_variables: List[str]
    prompt_version: Optional[str] = None

def _read_metric_file(metric: str, instruction_dir: Path) -> Dict[str, Any]:
    for ext in ["yaml", "yml", "json"]:
        fp = instruction_dir / f"{metric}.{ext}"
        if fp.exists() and fp.is_file():
            raw = fp.read_text(encoding="utf-8")
            return json.loads(raw) if ext == "json" else yaml.safe_load(raw)
    raise RuntimeError(f"Missing instruction file for metric '{metric}' in {instruction_dir}")

def _escape_unescaped_braces(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.replace("{{", "__DBL_L__").replace("}}", "__DBL_R__")
    text = text.replace("{", "{{").replace("}", "}}")
    text = text.replace("__DBL_L__", "{{").replace("__DBL_R__", "}}")
    return text

def _preserve_placeholders(text: str, allowed_vars: List[str]) -> str:
    if not isinstance(text, str):
        return text
    for v in allowed_vars:
        text = text.replace(f"{{{{{v}}}}}", f"{{{v}}}")
    return text

def load_metric_prompt_config(metric: str, instruction_dir: Path) -> MetricPromptConfig:
    data = _read_metric_file(metric, instruction_dir)
    required = ["instruction", "criteria", "rating_rubric", "input_variables"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"{instruction_dir}/{metric} missing required keys: {missing}")

    input_vars = [str(v) for v in data["input_variables"]]
    
    instruction = _preserve_placeholders(_escape_unescaped_braces(str(data["instruction"])), input_vars)

    criteria: Dict[str, str] = {}
    for k, v in (data["criteria"] or {}).items():
        criteria[str(k)] = _preserve_placeholders(_escape_unescaped_braces(str(v)), input_vars)

    rating_rubric: Dict[str, str] = {}
    for k, v in (data["rating_rubric"] or {}).items():
        rating_rubric[str(k)] = _preserve_placeholders(_escape_unescaped_braces(str(v)), input_vars)

    return MetricPromptConfig(
        metric=metric,
        instruction=instruction,
        criteria=criteria,
        rating_rubric=rating_rubric,
        input_variables=input_vars,
        prompt_version=data.get("prompt_version"),
    )

def build_pointwise_metrics(metric_names: List[str], instruction_dir: Path) -> Dict[str, Tuple[PointwiseMetric, MetricPromptConfig]]:
    metrics: Dict[str, Tuple[PointwiseMetric, MetricPromptConfig]] = {}
    for m in metric_names:
        cfg = load_metric_prompt_config(m, instruction_dir)
        metric_obj = PointwiseMetric(
            metric=m,
            metric_prompt_template=PointwiseMetricPromptTemplate(
                instruction=cfg.instruction,
                criteria=cfg.criteria,
                rating_rubric=cfg.rating_rubric,
                input_variables=cfg.input_variables,
            ),
        )
        metrics[m] = (metric_obj, cfg)
    return metrics

def load_agent_instruction_text(instruction_dir: Path) -> str:
    candidates = [
        instruction_dir.parent / "instruction.py",
        instruction_dir / "instruction.py",
    ]
    for fp in candidates:
        if fp.exists() and fp.is_file():
            try:
                text = fp.read_text(encoding="utf-8")
                tree = ast.parse(text)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                        node.targets[0], ast.Name
                    ):
                        if node.targets[0].id == "INSTRUCTION":
                            val = ast.literal_eval(node.value)
                            if isinstance(val, str):
                                return val
            except Exception as e:
                logging.warning("Failed reading agent instruction from %s: %s", fp, e)
    return ""
