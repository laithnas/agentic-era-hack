# app/conversation_extras.py
from __future__ import annotations
import math, re, uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple  # keep simple types; avoid Optional/Union in tool signatures

from .evidence import EVIDENCE

# -------------------------------------------------------
# 1) Mixed-input Intent Router (numbers OR natural text)
# -------------------------------------------------------
# Menu mapping you use in the greeting (0–7)
_MENU_INTENTS = {
    "0": "main_menu",
    "1": "triage",
    "2": "nearby",
    "3": "costs",
    "4": "whatif",
    "5": "medsx",
    "6": "book",
    "7": "intake",
}

# Natural language patterns → the same canonical intents
_INTENT_PATTERNS: List[Tuple[str, str]] = [
    (r"\btriage|symptom|diagnos|what.*(wrong|might be)\b", "triage"),
    (r"\bnearby|clinic|hospital|urgent care|doctor|find care|close to me\b", "nearby"),
    (r"\bcost|price|how much|copay|fees?\b", "costs"),
    (r"\bwhat if|worst case|risk|danger\b", "whatif"),
    (r"\bmed(ication)?s?\b|\bside[- ]?effects?\b|drug\b|pharma\b", "medsx"),
    (r"\bbook|appointment|schedule\b", "book"),
    (r"\bform|intake|check[- ]?in\b", "intake"),
    (r"\bmenu|start over|main menu\b", "main_menu"),
]

def route_user_input(text: str) -> dict:
    """
    Return {"intent": <canonical>, "matched": "...", "confidence": 0..1}
    Prefer numeric choice 0..7. Else use regex patterns.
    """
    t = (text or "").strip().lower()
    if t in _MENU_INTENTS:
        return {"intent": _MENU_INTENTS[t], "matched": t, "confidence": 1.0}

    for pat, intent in _INTENT_PATTERNS:
        if re.search(pat, t):
            return {"intent": intent, "matched": pat, "confidence": 0.8}
    return {"intent": "unknown", "matched": "", "confidence": 0.0}

# -------------------------------------------------------
# 2) Adaptive Triage Flow (tiny state machine)
# -------------------------------------------------------
_TRIAGE_QBANK = [
    {"key": "age_group", "q": "What’s your age group? (child, teen, adult, older adult)"},
    {"key": "symptoms", "q": "Please describe your main symptoms in your own words."},
    {"key": "duration", "q": "How long has this been going on? (e.g., hours, days, weeks)"},
    {"key": "severity", "q": "How severe is it? (mild, moderate, severe)"},
]

_TRIAGE_WHY = {
    "age_group": "Age affects risk thresholds and recommended settings for care.",
    "symptoms": "Symptoms guide which common conditions we should consider safely.",
    "duration": "Duration helps separate short-lived issues from those needing review.",
    "severity": "Severity helps decide if urgent care is safer.",
}

def triage_session_start() -> dict:
    """
    Create a new triage session state.
    """
    sid = "TG-" + uuid.uuid4().hex[:8].upper()
    return {
        "session_id": sid,
        "answers": {"age_group": "", "symptoms": "", "duration": "", "severity": ""},
        "pending": [q["key"] for q in _TRIAGE_QBANK],
        "complete": False,
    }

def triage_session_step(state: dict, user_text: str) -> dict:
    """
    Consume user_text to fill the next unanswered slot.
    Returns:
      {
        "ask": "next question (or empty if complete)",
        "why": "short reason",
        "state": <updated_state>,
        "complete": bool
      }
    Interrupt handling: if the user asks e.g., "book appointment", we just set a flag.
    """
    t = (user_text or "").strip()
    # Lightweight interrupt routing
    intent = route_user_input(t).get("intent")
    if intent in {"nearby", "book", "costs", "main_menu"}:
        state["interrupt_intent"] = intent
        return {"ask": "", "why": "", "state": state, "complete": False}

    # Next question
    pending = state.get("pending", [])
    if not pending:
        state["complete"] = True
        return {"ask": "", "why": "", "state": state, "complete": True}

    key = pending[0]
    # Save the answer
    state["answers"][key] = t
    pending.pop(0)
    state["pending"] = pending

    # Next question if any
    if pending:
        nxt = pending[0]
        return {
            "ask": _get_q(nxt),
            "why": _get_why(nxt),
            "state": state,
            "complete": False,
        }
    else:
        state["complete"] = True
        return {"ask": "", "why": "", "state": state, "complete": True}

def _get_q(key: str) -> str:
    for q in _TRIAGE_QBANK:
        if q["key"] == key:
            return q["q"]
    return ""

def _get_why(key: str) -> str:
    hint = _TRIAGE_WHY.get(key, "")
    return f"_Why this helps:_ {hint}" if hint else ""

# -------------------------------------------------------
# 3) Evidence toggle (show/hide)
# -------------------------------------------------------
# In-memory flag; your agent can call set/get based on user commands.
_SHOW_EVIDENCE = True

def set_evidence_visible(value: bool) -> dict:
    global _SHOW_EVIDENCE
    _SHOW_EVIDENCE = bool(value)
    return {"status": "ok", "show_evidence": _SHOW_EVIDENCE}

def get_evidence_visible() -> dict:
    return {"show_evidence": _SHOW_EVIDENCE}

# -------------------------------------------------------
# 4) Clinic UX+ (distance, open-now from Places)
# -------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points in kilometers.
    Signatures are simple floats to satisfy ADK's auto function schema.
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)

def format_place_line(
    name: str,
    address: str = "",
    rating: float = -1.0,
    phone: str = "",
    tel_url: str = "",
    website: str = "",
    google_url: str = "",
    open_now: bool = False,
    lat: float = -1.0,
    lng: float = -1.0,
    user_lat: float = -1.0,
    user_lng: float = -1.0
) -> str:
    """
    Return a single-line, user-facing string for a clinic row.

    IMPORTANT: No Optional/Union in the signature. We use sentinel defaults:
      - rating < 0   => hide rating
      - user_lat/user_lng < 0 or lat/lng < 0 => omit distance
      - empty strings for phone/tel_url/website/google_url/address => hide those parts
    """
    bits: List[str] = []

    # Name
    nm = name or "Clinic"
    bits.append(f"**{nm}**")

    # Rating
    if rating >= 0:
        bits.append(f"★{rating:.1f}")
    else:
        bits.append("★N/A")

    # Open now
    if open_now:
        bits.append("(Open now)")

    # Distance (only if both user and place coords provided)
    if (user_lat >= 0 and user_lng >= 0 and lat >= 0 and lng >= 0):
        dist = haversine_km(user_lat, user_lng, lat, lng)
        bits.append(f"~{dist} km")

    # Phone (clickable tel:)
    if phone and tel_url:
        bits.append(f"Call: [{phone}]({tel_url})")

    # Website
    if website:
        from urllib.parse import urlparse
        try:
            dom = urlparse(website).netloc.replace("www.", "")
        except Exception:
            dom = website
        bits.append(f"Website: [{dom}]({website})")

    # Google Maps
    if google_url:
        bits.append(f"[Maps]({google_url})")

    # Address (if website not present, still show address to give context)
    if address and not website:
        bits.append(address)

    return " — ".join(bits)

# -------------------------------------------------------
# 5) Medication interaction rules (tiny, high-signal)
# -------------------------------------------------------
# Very small, conservative rule set; extend as needed.
_INTERACTION_RULES: List[Tuple[str, str, str]] = [
    # (pattern A, pattern B, message)
    (r"\b(warfarin|coumadin|anticoagulant|apixaban|rivaroxaban)\b",
     r"\b(ibuprofen|naproxen|nsaid|aspirin)\b",
     "Anticoagulants + NSAIDs may increase bleeding risk — discuss with a clinician."),
    (r"\b(ssri|sertraline|fluoxetine|paroxetine|citalopram|escitalopram)\b",
     r"\b(nsaid|ibuprofen|naproxen|aspirin)\b",
     "SSRIs + NSAIDs may raise GI bleeding risk — use caution and seek advice."),
    (r"\b(ace inhibitor|lisinopril|enalapril|ramipril)\b",
     r"\b(nsaid|ibuprofen|naproxen)\b",
     "ACE inhibitors + NSAIDs can affect kidney function — monitor and seek advice."),
]

def check_drug_interactions(names: List[str]) -> List[str]:
    t = " ".join(names).lower()
    alerts: List[str] = []
    for a, b, msg in _INTERACTION_RULES:
        if re.search(a, t) and re.search(b, t):
            if msg not in alerts:
                alerts.append(msg)
    if alerts:
        EVIDENCE.add("medsx_rules", f"interactions={alerts}")
    return alerts

# -------------------------------------------------------
# 6) Visit-prep outputs (ICS, clinician handoff JSON)
# -------------------------------------------------------
def make_ics(clinic: str, dt_iso: str, title: str = "Clinic visit") -> dict:
    try:
        dt = datetime.fromisoformat(dt_iso)
    except Exception:
        return {"status": "error", "message": "Use ISO date/time, e.g., 2025-09-16T15:30:00"}
    dt_utc = dt.strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex
    ics = (
        "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//CareGuide//EN\n"
        "BEGIN:VEVENT\n"
        f"UID:{uid}\n"
        f"DTSTAMP:{dt_utc}\n"
        f"DTSTART:{dt_utc}\n"
        f"SUMMARY:{title}\n"
        f"LOCATION:{clinic}\n"
        "END:VEVENT\nEND:VCALENDAR\n"
    )

    # Create a downloadable link
    import base64, urllib.parse
    encoded = urllib.parse.quote(ics)
    download_url = f"data:text/calendar;charset=utf-8,{encoded}"

    return {
        "status": "ok",
        "filename": "clinic_visit.ics",
        "content": ics,
        "download_url": download_url
    }


def clinician_handoff_summary(case: dict) -> dict:
    """
    Make a compact, shareable JSON (no diagnosis).
    Input case example:
      {
        "age_group": "adult",
        "symptoms": "sore throat, cough",
        "duration": "2 days",
        "severity": "mild",
        "meds": ["ibuprofen"],
        "allergies": ["penicillin"],
        "watchouts_denied": ["trouble breathing", "chest pain"]
      }
    """
    out = {
        "type": "careguide_case",
        "version": "1",
        "collected_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "summary": {
            "age_group": case.get("age_group", ""),
            "symptoms": case.get("symptoms", ""),
            "duration": case.get("duration", ""),
            "severity": case.get("severity", ""),
            "medications": case.get("meds", []),
            "allergies": case.get("allergies", []),
            "watchouts_denied": case.get("watchouts_denied", []),
        },
        "notes": "No diagnosis provided. For emergencies, seek immediate care.",
    }
    return out

# -------------------------------------------------------
# 7) Tone & formatting helpers
# -------------------------------------------------------
def tone_numbered(title: str, bullets: List[str], disclaimer: bool = True) -> str:
    lines = []
    if title:
        lines.append(f"**{title}**")
    for i, b in enumerate(bullets, 1):
        lines.append(f"{i}) {b}")
    # ❌ Remove the disclaimer here, since it's already always appended in the system prompt
    return "\n".join(lines)
