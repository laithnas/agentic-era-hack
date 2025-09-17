import json, re
from pathlib import Path
from typing import List, Dict, Any

DATA_PATH = Path(__file__).parent / "data" / "conditions.json"
SYMPTOM_KB_PATH = Path(__file__).parent / "data" / "kb_symptom_to_conditions.json"

# Only VERY serious triggers
EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\b(shortness of breath|severe trouble breathing)\b",
    r"\bsevere bleeding\b",
    r"\b(passed out|fainted|loss of consciousness)\b",
    r"\b(stroke symptoms|face droop|slurred speech|weakness on (one|1) side)\b",
]

SYMPTOM_TERMS = [
    "fever","cough","sore throat","runny nose","stuffy nose","sneezing","headache","nausea","vomiting",
    "diarrhea","abdominal cramps","fatigue","rash","itchy rash","shortness of breath","wheezing","chest pain",
    "dry mouth","xerostomia","thirst","sticky saliva","bad breath","difficulty swallowing","cracked lips"
]

def _norm(s:str)->str:
    return re.sub(r"\s+"," ",s.strip().lower())

def _disclaimer_block() -> str:
    # Exact structure: line, blank line, then "Disclaimer: ..."
    return "This is general guidance, not a medical diagnosis.\n\nDisclaimer: This is general guidance, not a medical diagnosis."

def red_flag_checker(text: str) -> str:
    t=_norm(text)
    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, t):
            msg = ("⚠️ This could be a medical emergency. Please seek **immediate** medical attention "
                   "or call your local emergency number now.")
            return f"{msg}\n\n{_disclaimer_block()}"
    return ""

def symptom_extract(text: str) -> dict:
    t=_norm(text)
    found=[s for s in SYMPTOM_TERMS if s in t]
    m=re.search(r"(for|since) ([\w\s\-]+?)(?:\.|,|;|$)", t)
    duration = m.group(2).strip() if m else ""
    severity="moderate"
    if re.search(r"\b(mild|slight)\b", t): severity="mild"
    if re.search(r"\b(severe|intense|worst)\b", t): severity="severe"
    return {"symptoms":sorted(set(found)),"duration":duration,"severity":severity}

def _load_conditions():
    return json.loads(DATA_PATH.read_text()) if DATA_PATH.exists() else []

def _load_sym_kb():
    if SYMPTOM_KB_PATH.exists():
        return json.loads(SYMPTOM_KB_PATH.read_text())
    return {}

def _score(symptoms:list[str], c:dict)->float:
    a=set(symptoms); b=set([s.lower() for s in c.get("symptoms",[])])
    return 0.0 if not a or not b else len(a & b)/len(a | b)

def rules_lookup(symptoms:list[str], duration:str="", severity:str="")->dict:
    db=_load_conditions()
    scored=sorted([{"item":c,"score":_score(symptoms,c)} for c in db], key=lambda x:x["score"], reverse=True)
    top=[s for s in scored if s["score"]>=0.15][:3]  # up to 3
    results=[{
        "condition": s["item"]["condition"],
        "score": round(s["score"],2),
        "self_care": s["item"].get("self_care",[]),
        "watchouts": s["item"].get("watchouts",[])
    } for s in top]
    return {"matches":results,"duration":duration,"severity":severity}

def kb_lookup(symptoms: List[str]) -> List[str]:
    kb=_load_sym_kb(); out=[]
    joined=" ".join(symptoms)
    for k,vals in kb.items():
        if k in joined and vals:
            for v in vals:
                if v not in out:
                    out.append(v)
    return out[:5]

def advice_renderer(analysis:dict)->str:
    m=analysis.get("matches",[])
    duration=analysis.get("duration",""); severity=analysis.get("severity","")
    lines=["**Triage summary (not a diagnosis)**"]
    lines.append(f"- What it might be: {', '.join([x['condition'] for x in m]) if m else 'We need a bit more detail to narrow this.'}")
    if m:
        steps=[]; warns=[]
        for mm in m:
            for s in mm.get("self_care",[]):
                if s not in steps: steps.append(s)
            for w in mm.get("watchouts",[]):
                if w not in warns: warns.append(w)
        if steps:
            lines.append("- What you can do now:")
            for s in steps[:6]:
                lines.append(f"  • {s}")
        if warns:
            lines.append("- Watch for:")
            for w in warns[:6]:
                lines.append(f"  • {w}")
    if duration:
        lines.append(f"- Duration noted: **{duration}** (seek care if it persists or worsens)")
    if severity=="severe":
        lines.append("- Severity noted: **severe** — consider contacting a clinician sooner")

    # Two-line disclaimer (exact structure)
    lines.append(_disclaimer_block())
    return "\n".join(lines)

def triage_pipeline(user_text: str) -> str:
    """
    Primary entry point called by the agent.
    - If very serious red flags present -> emergency message (with disclaimer).
    - Else: produce up to 3 plausible conditions (hedged) using rules + KB; render self-care and watch-outs.
    """
    # Emergency short-circuit
    em = red_flag_checker(user_text)
    if em:
        return em  # already includes disclaimer block

    # Extract
    info = symptom_extract(user_text)

    # Rules (from conditions.json)
    analysis = rules_lookup(info.get("symptoms", []), info.get("duration",""), info.get("severity",""))

    # If rules are sparse, backfill from KB to ensure we suggest at least some plausible conditions
    current = [m["condition"] for m in analysis.get("matches",[])]
    if len(current) < 3:
        kb_suggestions = [c for c in kb_lookup(info.get("symptoms", [])) if c not in current]
        for c in kb_suggestions:
            analysis.setdefault("matches", []).append({
                "condition": c,
                "score": 0.2,              # neutral placeholder score for KB matches
                "self_care": [],
                "watchouts": []
            })
            if len(analysis["matches"]) >= 3:
                break

    # Final render (includes the required disclaimers)
    return advice_renderer(analysis)
