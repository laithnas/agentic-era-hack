# app/prescription_parser.py
from __future__ import annotations
import re
from typing import List, Dict

_DRUG_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z\-]{1,29})(?:\s+(?:tablet|cap(?:sule)?|syrup|solution))?\b", re.I)

# Common fillers to drop
_STOPLIST = {"take","tab","tablet","capsule","mg","mcg","ml","dose","daily","bid","tid","qid","po","prn"}

def extract_meds_from_text(file_text: str) -> dict:
    """
    Returns {"medications":[...]} parsed from raw text (PDF->text already).
    """
    t = (file_text or "").replace("\n"," ")
    cands = [m.group(1).lower() for m in _DRUG_PATTERN.finditer(t)]
    meds: List[str] = []
    for c in cands:
        if c in _STOPLIST: 
            continue
        if c not in meds:
            meds.append(c)
    return {"medications": meds[:20]}
