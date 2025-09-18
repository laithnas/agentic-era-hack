# app/handoff.py
from __future__ import annotations
from typing import Dict

# Set where to route (phone, URL, or internal queue)
_HANDOFF_DEST = {"type":"phone","value":"+1-800-555-0100"}

def set_handoff_destination(kind: str, value: str) -> dict:
    """
    kind: "phone" | "url" | "queue"
    value: e.g., "+1-800-555-0100" or "https://clinic.example.com"
    """
    global _HANDOFF_DEST
    k = (kind or "").lower()
    if k not in ("phone","url","queue"):
        return {"status":"error","message":"kind must be phone|url|queue"}
    _HANDOFF_DEST = {"type":k, "value":value}
    return {"status":"ok","dest":_HANDOFF_DEST}

def request_live_handoff(reason: str, payload_json: str = "") -> dict:
    """
    Return a deterministic 'handoff ticket' the UI can render.
    """
    return {
        "status":"ok",
        "dest": _HANDOFF_DEST,
        "reason": reason,
        "payload_json": payload_json,
        "message":"A clinician will review this packet. Use the destination to continue."
    }
