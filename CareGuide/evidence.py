# app/evidence.py
from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class EvidenceItem:
    source: str   # e.g., "dataset", "whatif_calc"
    detail: str   # short text to show
    extra: Dict[str, Any] = field(default_factory=dict)

class EvidenceLog:
    def __init__(self) -> None:
        self._items: List[EvidenceItem] = []

    def add(self, source: str, detail: str, **extra: Any) -> None:
        self._items.append(EvidenceItem(source=source, detail=detail, extra=extra))

    def snapshot(self, clear: bool = True) -> List[Dict[str, Any]]:
        out = [dict(source=i.source, detail=i.detail, **(i.extra or {})) for i in self._items]
        if clear:
            self._items.clear()
        return out

# Singleton used everywhere
EVIDENCE = EvidenceLog()
