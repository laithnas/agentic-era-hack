# app/risk_sim.py
from __future__ import annotations
import re

def risk_simulate(question_text: str, age_group: str = "", severity: str = "") -> dict:
    """
    Heuristic risk banding for “what-if” safety checks.

    This function applies a **very conservative, rule-based** screen to place a user’s
    free-text concern into one of three risk bands: `"low"`, `"moderate"`, or `"high"`.
    It is intentionally simple, explainable, and **not** a medical diagnosis.

    Parameters
    ----------
    question_text : str
        The user’s question or symptom concern written in free text. Example:
        "What if my cough is getting worse and I feel short of breath?"
    age_group : str, optional
        Optional age bucket such as `"child"`, `"teen"`, `"adult"`, `"older adult"`.
        Certain groups (children and older adults) up-shift risk one level when
        no emergency terms are present.
    severity : str, optional
        Optional self-reported severity (e.g., `"mild"`, `"moderate"`, `"severe"`).
        A value of `"severe"` will up-shift non-high risk to `"moderate"`.

    Returns
    -------
    dict
        A small, UI-friendly payload:

        {
          "band":    "low" | "moderate" | "high",
          "reasons": [str, ...],   # short, human-readable rationales (≤3)
          "actions": [str, ...]    # conservative next-step suggestions (≤3)
        }

    Safety & Scope
    --------------
    - This is **not** ML and does **not** infer diagnoses.
    - The rules are conservative: any explicit emergency keyword (e.g., “chest pain”)
      immediately bands to `"high"` and recommends urgent action.
    - Age and severity are treated as risk modifiers.
    - Keep outputs short to fit nicely in chat UIs and voice responses.

    Example
    -------
    >>> risk_simulate("My chest pain is getting worse when I breathe in", "adult", "moderate")
    {'band': 'high',
     'reasons': ['Contains emergency symptom keyword(s).'],
     'actions': ['Seek urgent care now or call emergency services.', 'Do not delay.']}

    Notes
    -----
    - If you expand the emergency list, ensure phrases remain **lowercase**
      and are matched as simple substrings (current approach) or add
      appropriate normalization.
    """
    # Normalize text for simple substring checks
    t = (question_text or "").lower()

    reasons = []
    actions = []
    band = "low"

    # High-signal emergency terms (keep list concise & specific)
    red = [
        "chest pain",
        "trouble breathing",
        "shortness of breath",
        "passed out",
        "stroke",
        "severe bleeding",
    ]

    # 1) Hard red flags → HIGH
    if any(x in t for x in red):
        band = "high"
        reasons.append("Contains emergency symptom keyword(s).")
        actions += [
            "Seek urgent care now or call emergency services.",
            "Do not delay.",
        ]

    # 2) Worsening trajectory → MODERATE (if not already HIGH)
    elif "worsen" in t or "getting worse" in t:
        band = "moderate"
        reasons.append("Mentions worsening pattern.")
        actions += [
            "Monitor closely for red flags.",
            "Consider contacting a clinician within 24–48h.",
        ]

    # 3) Default → LOW
    else:
        band = "low"
        reasons.append("No red-flag terms detected.")
        actions += [
            "Use self-care steps where appropriate.",
            "Reassess if symptoms persist or worsen.",
        ]

    # 4) Risk modifiers: age group (children, older adults)
    if age_group in ("child", "older adult"):
        reasons.append(f"Age group: {age_group} may elevate risk.")
        if band == "low":
            band = "moderate"

    # 5) Risk modifier: user-reported severity
    if severity == "severe" and band != "high":
        band = "moderate"
        reasons.append("User-reported severity is severe.")

    # Keep payload compact (≤3 items each for clean rendering)
    return {"band": band, "reasons": reasons[:3], "actions": actions[:3]}
