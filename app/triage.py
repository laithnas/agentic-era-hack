import json, re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "conditions.json"

EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bdifficulty (breathing|inhaling)\b|\bshort(ness)? of breath\b",
    r"\bsevere bleeding\b|\bheavy bleeding\b",
    r"\bpassed out\b|\bfainted\b|\bloss of consciousness\b",
    r"\bstroke symptoms\b|\bface droop\b|\bslurred speech\b|\bweakness on (one|1) side\b",
    r"\bstiff neck\b.*\bfever\b",
    r"\b(vision loss|double vision)\b.*\bheadache\b",
]

CHRONIC_PATTERNS = [
    r"\bfor (\d+|several|many) (days|weeks|months|years)\b",
    r"\b(chronic|ongoing|persistent)\b",
    r"\b(diabetes|cancer|copd|asthma flare|heart failure)\b",
]

SYMPTOM_TERMS = [
    "fever","high fever","low-grade fever","chills","sore throat","cough","dry cough","runny nose","stuffy nose",
    "sneezing","itchy eyes","watery eyes","post-nasal drip","headache","throbbing headache","nausea","vomiting",
    "diarrhea","abdominal cramps","fatigue","pressure around head","neck tightness","aura","rash","itchy rash","redness",
    "dry patches","swelling","shortness of breath","chest pain","wheezing","photophobia","phonophobia"
]

def _normalize(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.strip().lower())

def red_flag_checker(text: str) -> str:
    t = _normalize(text)
    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, t):
            return ("⚠️ This could be a medical emergency. Please seek **immediate** medical attention "
                    "or call your local emergency number now.")
    return ""

def chronic_checker(text: str) -> str:
    t = _normalize(text)
    for pat in CHRONIC_PATTERNS:
        if re.search(pat, t):
            return ("This sounds like a chronic or complex issue. I recommend connecting with a **live clinician** "
                    "for proper evaluation. I can hand you off now if you'd like.")
    return ""

def symptom_extract(text: str) -> dict:
    t = _normalize(text)
    found = [s for s in SYMPTOM_TERMS if s in t]
    # duration
    m = re.search(r"(for|since) ([\w\s]+?)(?:\.|,|;|$)", t)
    duration = m.group(2).strip() if m else ""
    # severity
    severity = "moderate"
    if re.search(r"\b(mild|slight)\b", t): severity = "mild"
    if re.search(r"\b(severe|intense|worst)\b", t): severity = "severe"
    return {"symptoms": sorted(set(found)), "duration": duration, "severity": severity}

def _load_conditions():
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def _score(symptoms:list[str], candidate:dict) -> float:
    a = set(symptoms)
    b = set([s.lower() for s in candidate.get("symptoms", [])])
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def rules_lookup(symptoms: list[str], duration: str="", severity: str="") -> dict:
    db = _load_conditions()
    scored = sorted(
        [{"item": c, "score": _score(symptoms, c)} for c in db],
        key=lambda x: x["score"],
        reverse=True,
    )
    top = [s for s in scored if s["score"] >= 0.15][:2]
    results = []
    for s in top:
        c = s["item"]
        results.append({
            "condition": c["condition"],
            "score": round(s["score"], 2),
            "self_care": c.get("self_care", []),
            "watchouts": c.get("watchouts", [])
        })
    return {"matches": results, "duration": duration, "severity": severity}

def advice_renderer(analysis: dict, user_text: str) -> str:
    matches = analysis.get("matches", [])
    duration = analysis.get("duration","")
    severity = analysis.get("severity","")
    lines = []
    lines.append("Thanks for the details. Here’s a quick, cautious triage summary:")
    if matches:
        maybe = ", ".join([m["condition"] for m in matches])
        lines.append(f"**What it might be:** {maybe} (not a diagnosis).")
    else:
        lines.append("**What it might be:** I can't tell yet from the description.")
    steps = []
    for m in matches:
        for s in m["self_care"]:
            if s not in steps: steps.append(s)
    if steps:
        lines.append("**What you can do now:**")
        for s in steps[:6]:
            lines.append(f"- {s}")
    watch = []
    for m in matches:
        for w in m["watchouts"]:
            if w not in watch: watch.append(w)
    if watch:
        lines.append("**Watch for these:**")
        for w in watch[:6]:
            lines.append(f"- {w}")
    if duration:
        lines.append(f"_You mentioned this has been going on for **{duration}**; if it persists or worsens, seek care._")
    if severity == "severe":
        lines.append("_Given you described **severe** symptoms, consider contacting a clinician sooner._")
    lines.append("\n> _I’m a virtual assistant, not a medical professional. For emergencies or worsening symptoms, seek medical care._")
    return "\n".join(lines)

def triage_pipeline(user_text: str) -> str:
    em = red_flag_checker(user_text)
    if em: return em
    ch = chronic_checker(user_text)
    if ch: return ch
    info = symptom_extract(user_text)
    analysis = rules_lookup(
        info.get("symptoms", []),
        info.get("duration",""),
        info.get("severity","")
    )
    return advice_renderer(analysis, user_text)


