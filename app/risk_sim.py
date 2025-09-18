# app/risk_sim.py
from __future__ import annotations
import re

def risk_simulate(question_text: str, age_group: str = "", severity: str = "") -> dict:
    """
    Heuristic risk banding. Returns {"band":"low|moderate|high","reasons":[...],"actions":[...]}
    """
    t = (question_text or "").lower()
    reasons = []
    actions = []
    band = "low"

    red = ["chest pain","trouble breathing","shortness of breath","passed out","stroke","severe bleeding"]
    if any(x in t for x in red):
        band = "high"; reasons.append("Contains emergency symptom keyword(s).")
        actions += ["Seek urgent care now or call emergency services.", "Do not delay."]
    elif "worsen" in t or "getting worse" in t:
        band = "moderate"; reasons.append("Mentions worsening pattern.")
        actions += ["Monitor closely for red flags.", "Consider contacting a clinician within 24–48h."]
    else:
        band = "low"; reasons.append("No red-flag terms detected.")
        actions += ["Use self-care steps where appropriate.", "Reassess if symptoms persist or worsen."]

    if age_group in ("child","older adult"): 
        reasons.append(f"Age group: {age_group} may elevate risk.")
        if band == "low": band = "moderate"

    if severity == "severe" and band != "high":
        band = "moderate"; reasons.append("User-reported severity is severe.")

    return {"band": band, "reasons": reasons[:3], "actions": actions[:3]}
