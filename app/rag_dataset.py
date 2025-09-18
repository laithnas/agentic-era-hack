# app/rag_dataset.py
from __future__ import annotations
import os, io, csv, time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .config import TRIAGE_KB_GCS, TRIAGE_KB_LOCAL
from .evidence import EVIDENCE

# Lazy imports (avoid heavy deps on import time)
def _np():  # numpy
    import numpy as np
    return np

def _sk_text():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer

def _sk_metrics():
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity

def _download_gcs_to_local(gcs_uri: str, local_path: str) -> bool:
    """Best-effort download; if GCS libs aren’t available, skip."""
    try:
        from google.cloud import storage
        if not gcs_uri.startswith("gs://"):
            return False
        bucket_name, blob_path = gcs_uri[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return True
    except Exception:
        return False

@dataclass
class KBRow:
    condition: str
    symptoms: str
    advice: str
    url: str|None = None

_Vectorizer = None
_MATRIX = None
_ROWS: List[KBRow] = []

def _ensure_local_csv() -> str:
    if os.path.exists(TRIAGE_KB_LOCAL):
        return TRIAGE_KB_LOCAL
    ok = _download_gcs_to_local(TRIAGE_KB_GCS, TRIAGE_KB_LOCAL)
    return TRIAGE_KB_LOCAL if ok and os.path.exists(TRIAGE_KB_LOCAL) else ""

def _load_rows() -> List[KBRow]:
    path = _ensure_local_csv()
    rows: List[KBRow] = []
    if not path:
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Flexible columns
        cond_col = next((c for c in reader.fieldnames or [] if c.lower() in ("condition","disease","name")), "condition")
        sym_col = next((c for c in reader.fieldnames or [] if c.lower() in ("symptoms","symptom","features")), "symptoms")
        adv_col = next((c for c in reader.fieldnames or [] if c.lower() in ("advice","self_care","recommendations")), "advice")
        url_col = next((c for c in reader.fieldnames or [] if c.lower() in ("url","link","source")), None)
        for r in reader:
            rows.append(KBRow(
                condition=(r.get(cond_col) or "").strip(),
                symptoms=(r.get(sym_col) or "").strip(),
                advice=(r.get(adv_col) or "").strip(),
                url=(r.get(url_col) or None) if url_col else None
            ))
    return rows

def _build_index() -> None:
    global _Vectorizer, _MATRIX, _ROWS
    if _Vectorizer is not None and _MATRIX is not None and _ROWS:
        return
    _ROWS = _load_rows()
    if not _ROWS:
        return
    texts = [f"{r.condition} | {r.symptoms} | {r.advice}" for r in _ROWS]
    TfidfVectorizer = _sk_text()
    _Vectorizer = TfidfVectorizer(min_df=1, max_features=30000, ngram_range=(1,2))
    _MATRIX = _Vectorizer.fit_transform(texts)

def rag_stats() -> Dict[str, int]:
    _build_index()
    return {"rows": len(_ROWS), "indexed": int(_MATRIX.shape[0]) if _MATRIX is not None else 0}

def rag_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    _build_index()
    if not _ROWS or _Vectorizer is None or _MATRIX is None:
        return []
    vec = _Vectorizer.transform([query])
    cosine_similarity = _sk_metrics()
    sims = cosine_similarity(vec, _MATRIX)[0]
    idx = sims.argsort()[-top_k:][::-1]
    out = []
    for i in idx:
        r = _ROWS[i]
        out.append({
            "condition": r.condition,
            "symptoms": r.symptoms,
            "advice": r.advice,
            "url": r.url,
            "score": round(float(sims[i]), 3),
        })
    EVIDENCE.add("dataset", f"{len(out)} similar cases")
    return out
