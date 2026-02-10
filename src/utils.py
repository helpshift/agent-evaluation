import datetime as dt
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None

def parse_timestamp(ts: Any) -> Optional[dt.datetime]:
    if ts is None:
        return None
    if isinstance(ts, dt.datetime):
        return ts
    if isinstance(ts, str):
        s = ts.strip()
        try:
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    return None

def as_float_seconds(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v / 1000.0 if v > 1000.0 else v
    if isinstance(x, str):
        s = x.strip().lower()
        m = re.match(r"^(\d+(\.\d+)?)(ms|s)?$", s)
        if not m:
            return None
        v = float(m.group(1))
        unit = m.group(3) or "s"
        return v / 1000.0 if unit == "ms" else v
    return None

def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    k = max(1, int(round((p / 100.0) * len(values_sorted))))
    k = min(k, len(values_sorted))
    return float(values_sorted[k - 1])

def gen_eval_id(prefix: str = "eval") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def extract_confidence_and_rationale(raw_text: Any) -> Tuple[Optional[float], str]:
    if not isinstance(raw_text, str):
        return None, "" if raw_text is None else str(raw_text)

    s = raw_text.strip()
    if not s:
        return None, ""

    try:
        obj = json.loads(s)
        conf = _coerce_float(obj.get("confidence"))
        expl = obj.get("explanation") or obj.get("reasoning") or obj.get("rationale") or s
        return conf, str(expl).strip()
    except Exception:
        pass

    start_idx = s.find('{')
    end_idx = s.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = s[start_idx : end_idx + 1]
        try:
            obj = json.loads(candidate)
            conf = _coerce_float(obj.get("confidence"))
            expl = obj.get("explanation") or obj.get("reasoning") or obj.get("rationale") or s
            return conf, str(expl).strip()
        except Exception:
            pass

    return None, s

def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None
