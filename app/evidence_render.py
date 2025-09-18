# app/evidence_render.py
from __future__ import annotations
from typing import List, Dict

def evidence_markdown(panel: dict) -> dict:
    """
    Input: panel like {"items":[{"source":"RAG","detail":"..."}, ...]}
    Output: {"markdown":"..."}
    """
    items: List[Dict] = panel.get("items", [])
    if not items:
        return {"markdown": "_(No evidence items)_"}

    lines = ["**Evidence**"]
    for it in items:
        src = it.get("source","tool")
        detail = it.get("detail","")
        lines.append(f"- **{src}:** {detail}")
    return {"markdown": "\n".join(lines)}
