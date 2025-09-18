# app/triage.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Any, Optional
from .evidence import EVIDENCE

DATA_PATH = Path(__file__).parent / "data" / "conditions.json"
SYMPTOM_KB_PATH = Path(__file__).parent / "data" / "kb_symptom_to_conditions.json"

# VERY serious triggers only
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

SYN = {
    "feverish": "fever",
    "dyspnea": "shortness of breath",
    "sob": "shortness of breath",
    "xerostomia": "dry mouth",
}

def _norm(s:str)->str:
    return re.sub(r"\s+"," ",s.strip().lower())

def red_flag_checker(text: str) -> str:
    t=_norm(text)
    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, t):
            return ("⚠️ This could be a medical emergency. Please seek **immediate** medical attention "
                    "or call your local emergency number now.")
    return ""

def symptom_extract(text: str) -> dict:
    t=_norm(text)
    # synonym normalize
    for k,v in SYN.items():
        t = t.replace(k, v)
    found=[s for s in SYMPTOM_TERMS if s in t]
    m=re.search(r"(for|since)\s+([\w\s\-]+?)(?:\.|,|;|$)", t)
    duration = m.group(2).strip() if m else ""
    severity="moderate"
    if re.search(r"\b(mild|slight)\b", t): severity="mild"
    if re.search(r"\b(severe|intense|worst|10\/10|10 out of 10)\b", t): severity="severe"
    return {"symptoms":sorted(set(found)),"duration":duration,"severity":severity}

def _load_conditions():
    return json.loads(DATA_PATH.read_text()) if DATA_PATH.exists() else []

def _load_sym_kb():
    if SYMPTOM_KB_PATH.exists():
        return json.loads(SYMPTOM_KB_PATH.read_text())
    return {}

def _score(symptoms:List[str], c:dict)->float:
    a=set(symptoms); b=set([s.lower() for s in c.get("symptoms",[])])
    return 0.0 if not a or not b else len(a & b)/len(a | b)

def rules_lookup(symptoms:List[str], duration:str="", severity:str="", age_group:str|None=None)->dict:
    db=_load_conditions()
    scored=sorted([{"item":c,"score":_score(symptoms,c)} for c in db], key=lambda x:x["score"], reverse=True)
    top=[s for s in scored if s["score"]>=0.15][:3]
    # light age filter (if metadata exists in JSON)
    results=[]
    for s in top:
        c=s["item"]
        ag = c.get("age")  # e.g., ["adult","teen"] if present
        if ag and age_group and age_group not in ag:
            s_score = max(0.01, s["score"]*0.8)  # downweight
        else:
            s_score = s["score"]
        results.append({
            "condition": c["condition"],
            "score": round(s_score,2),
            "self_care": c.get("self_care",[]),
            "watchouts": c.get("watchouts",[])
        })
    if results:
        EVIDENCE.add("triage_rules", f"rules matched {len(results)} candidates")
    return {"matches":results,"duration":duration,"severity":severity}

def _kb_lookup(symptoms: List[str]) -> List[str]:
    kb=_load_sym_kb(); out=[]
    joined=" ".join(symptoms)
    for k,vals in kb.items():
        if k in joined and vals:
            for v in vals:
                if v not in out: out.append(v)
    if out:
        EVIDENCE.add("triage_kb", f"kb hints: {', '.join(out[:3])}")
    return out[:5]

def advice_renderer(analysis:dict)->str:
    m=analysis.get("matches",[])
    duration=analysis.get("duration",""); severity=analysis.get("severity","")
    lines=["**Triage summary (not a diagnosis)**"]
    lines.append(f"- What it might be: {', '.join([x['condition'] for x in m]) if m else 'We need a bit more detail to narrow this.'}")
    if m:
        steps=[]; warns=[]
        for mm in m:
            steps.extend([s for s in mm["self_care"] if s not in steps])
            warns.extend([w for w in mm["watchouts"] if w not in warns])
        if steps:
            lines.append("- What you can do now:")
            for s in steps[:6]: lines.append(f"  • {s}")
        if warns:
            lines.append("- Watch for:")
            for w in warns[:6]: lines.append(f"  • {w}")
    if duration:
        lines.append(f"- Duration noted: **{duration}** (seek care if it persists/worsens)")
    if severity == "severe":
        lines.append(f"- Severity: **severe** → consider clinician sooner")

    return "\n".join(lines)

def triage_pipeline(user_text: str, age_group: Optional[str]=None) -> str:
    em = red_flag_checker(user_text)
    if em:
        return em
    info = symptom_extract(user_text)
    analysis = rules_lookup(info.get("symptoms", []), info.get("duration",""), info.get("severity",""), age_group=age_group)
    # Optional hints from KB (non-binding)
    _kb_lookup(info.get("symptoms", []))
    return advice_renderer(analysis)
