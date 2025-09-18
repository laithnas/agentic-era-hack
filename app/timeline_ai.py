# app/timeline_ai.py
from __future__ import annotations
from typing import List, Dict
from .assistant_tools import timeline_list  # reuse your existing storage

def timeline_insights() -> dict:
    """
    Analyze recent timeline entries and suggest 2–3 nudges.
    """
    data = timeline_list()
    items: List[Dict] = data.get("items", [])
    tips: List[str] = []

    # Simple heuristics
    had_urgent = any(i.get("type") == "emergency_flag" for i in items)
    many_triages = sum(1 for i in items if i.get("type") == "triage") >= 3
    has_appointment = any(i.get("type") == "appointment" for i in items)

    if had_urgent:
        tips.append("Add your emergency contacts to your phone favorites.")
    if many_triages and not has_appointment:
        tips.append("Consider booking a clinic visit to get a definitive evaluation.")
    if not tips:
        tips.append("Keep notes of symptom start dates; duration helps triage quality.")
        tips.append("Collect medication names ahead of time to speed up appointments.")

    return {"count": len(items), "tips": tips[:3]}
