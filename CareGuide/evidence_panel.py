# app/evidence_panel.py
from typing import List, Dict

class EvidenceCollector:
    def __init__(self):
        self._items: List[Dict] = []

    def add(self, kind: str, detail: str):
        self._items.append({"kind": kind, "detail": detail})

    def clear(self):
        self._items.clear()

    def render(self) -> str:
        if not self._items:
            return ""
        lines = ["\n**Evidence Panel**"]
        for it in self._items[:10]:
            lines.append(f"- {it['kind']}: {it['detail']}")
        return "\n".join(lines)

# singleton used by tools
EVIDENCE = EvidenceCollector()
