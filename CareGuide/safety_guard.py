# app/safety_guard.py
from __future__ import annotations
import re

_SAFETY_LEVEL = 2  # 1 (lenient) .. 3 (strict)

def set_safety_level(level: int) -> dict:
    global _SAFETY_LEVEL
    if level < 1: level = 1
    if level > 3: level = 3
    _SAFETY_LEVEL = level
    return {"status":"ok","level":_SAFETY_LEVEL}

def get_safety_level() -> dict:
    return {"level": _SAFETY_LEVEL}

# Hard guard patterns
_RX_PAT = re.compile(r"\b(\d{1,3}\s?(mg|mcg|g|ml)|tablet|pill|dos(e|age)|take\s+\d)", re.I)
_DIAG_PAT = re.compile(r"\b(you (have|are suffering from)|definitely|certainly .* (flu|covid|strep|cancer))\b", re.I)

def safety_gate(candidate_text: str) -> dict:
    """
    Blocks unsafe phrasings (doses, definitive diagnoses). Returns possibly redacted text + flags.
    """
    t = candidate_text or ""
    flags = []
    if _RX_PAT.search(t):
        flags.append("dose_redacted")
        t = _RX_PAT.sub("[redacted dose]", t)
    if _DIAG_PAT.search(t):
        flags.append("diagnosis_hedged")
        t = _DIAG_PAT.sub("it may be", t)
    return {"text": t, "flags": flags, "level": _SAFETY_LEVEL}
