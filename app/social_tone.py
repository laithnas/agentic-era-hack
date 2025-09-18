# app/social_tone.py
from __future__ import annotations
import re
from typing import List, Dict

# In-memory tone mode (session scope)
_TONE_MODE = "neutral"  # "neutral" | "reassuring" | "concise" | "child_friendly"

def set_care_mode(mode: str) -> dict:
    """
    Set output tone for responses.
    Allowed: neutral, reassuring, concise, child_friendly
    """
    global _TONE_MODE
    m = (mode or "").strip().lower()
    if m not in ("neutral","reassuring","concise","child_friendly"):
        return {"status":"error","message":"Use: neutral, reassuring, concise, child_friendly"}
    _TONE_MODE = m
    return {"status":"ok","mode":_TONE_MODE}

def get_care_mode() -> dict:
    return {"mode": _TONE_MODE}

def sentiment_screen(text: str) -> dict:
    """
    Tiny, deterministic classifier. Returns {"sentiment": "..", "signals":[...]}
    You can call this after each user message to auto-swap tone.
    """
    t = (text or "").lower()
    sig: List[str] = []
    score = 0
    for w in ("scared","worried","anxious","panic","chest pain","shortness of breath","urgent"):
        if w in t:
            score += 2; sig.append(w)
    for w in ("confused","lost","don't know","help"):
        if w in t:
            score += 1; sig.append(w)
    for w in ("thank you","thanks","ok","fine"):
        if w in t:
            score -= 1; sig.append(w)
    label = "stressed" if score >= 2 else ("concerned" if score == 1 else "calm")
    return {"sentiment": label, "signals": sig}

def tone_enforce(text: str) -> dict:
    """
    Provide rendering hints for the current tone mode.
    The LLM can call this before composing long answers.
    """
    if _TONE_MODE == "reassuring":
        style = {
            "voice": "warm, validating, gentle pace",
            "guidelines": [
                "Acknowledge feelings once up front.",
                "Use short sentences, avoid medical jargon.",
                "Offer 1–3 concrete next steps."
            ]
        }
    elif _TONE_MODE == "concise":
        style = {
            "voice": "direct, bullet-first",
            "guidelines": [
                "Use 3–5 bullets max.",
                "Keep sentences under 14 words.",
                "No repeated info."
            ]
        }
    elif _TONE_MODE == "child_friendly":
        style = {
            "voice": "simple, friendly, explain like I’m 10",
            "guidelines": [
                "Avoid scary words.",
                "Use examples from daily life.",
                "One idea per sentence."
            ]
        }
    else:
        style = {"voice":"neutral", "guidelines":["Default style."]}

    return {"mode": _TONE_MODE, "style": style}
