# app/triage_kb_admin.py
from __future__ import annotations
import os
from importlib import import_module
from typing import Any, Optional

_ACTIVE_PROFILE = "default"

def kb_reload(gcs_uri: str) -> dict:
    """
    Hot-swap the triage KB source. Always sets TRIAGE_KB_GCS.
    If app.rag_dataset exposes hot-reload hooks, we'll use them.
    Otherwise, we just set the env var and return a note that the
    next query will load from the new URI.
    """
    uri = (gcs_uri or "").strip()
    if not uri:
        return {"status": "error", "message": "Provide a GCS URI, e.g. gs://bucket/path.csv"}

    os.environ["TRIAGE_KB_GCS"] = uri

    # Best-effort: call optional hooks if they exist
    reloaded = False
    rows_count: Optional[int] = None
    detail: Optional[str] = None

    try:
        rag = import_module("app.rag_dataset")

        # 1) Single-call convenience hook
        if hasattr(rag, "reload_from_gcs"):
            data_or_count = rag.reload_from_gcs(uri)  # type: ignore[attr-defined]
            # Try to derive a row count
            try:
                rows_count = len(data_or_count)  # dataset object (list/df)
            except Exception:
                if isinstance(data_or_count, dict) and "rows" in data_or_count:
                    rows_count = data_or_count.get("rows")  # type: ignore[assignment]
            reloaded = True

        # 2) Separate load/build/set hooks
        elif hasattr(rag, "build_index"):
            # Try various loader names
            loader = None
            for name in ("load_dataset_from_gcs", "load_from_gcs", "load"):
                if hasattr(rag, name):
                    loader = getattr(rag, name)
                    break
            data = loader(uri) if loader else None  # type: ignore[misc]
            if data is not None:
                try:
                    rows_count = len(data)
                except Exception:
                    rows_count = None
                idx = rag.build_index(data)  # type: ignore[attr-defined]
                if hasattr(rag, "set_active_index"):
                    rag.set_active_index(idx)  # type: ignore[attr-defined]
                reloaded = True

        # 3) Warmup hook (rebuilds from current env)
        elif hasattr(rag, "warmup"):
            rag.warmup()  # type: ignore[attr-defined]
            reloaded = True

    except Exception as e:
        # Keep it non-fatal: env var is still set
        detail = str(e)

    return {
        "status": "ok",
        "gcs_uri": uri,
        "reloaded": reloaded,
        "rows": rows_count,
        "note": ("env var set; rag_dataset has no hot-reload hooks"
                 if not reloaded else "hot reload invoked"),
        "detail": detail,
    }

def set_profile(name: str) -> dict:
    """
    A/B profile knob the LLM can read to adjust behavior (question style, etc.).
    """
    global _ACTIVE_PROFILE
    n = (name or "").strip().lower()
    if not n:
        return {"status": "error", "message": "Provide a profile name"}
    _ACTIVE_PROFILE = n
    return {"status": "ok", "profile": _ACTIVE_PROFILE}

def get_profile() -> dict:
    return {"profile": _ACTIVE_PROFILE}
