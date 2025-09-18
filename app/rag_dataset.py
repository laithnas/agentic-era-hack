# app/rag_dataset.py
from __future__ import annotations
import os, io, csv, time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .config import TRIAGE_KB_GCS, TRIAGE_KB_LOCAL
from .evidence import EVIDENCE

"""
rag_dataset.py

Purpose
-------
Lightweight, dependency-minimal Retrieval-Augmented Generation (RAG) helper
for the CareGuide agent. This module:
  - Loads a simple CSV knowledge base of (condition, symptoms, advice[, url]).
  - Builds an in-memory TF-IDF vector index (scikit-learn) on first use.
  - Exposes `rag_search(query, top_k)` to retrieve the most similar rows.
  - Logs dataset usage to the Evidence panel.

Design Notes
------------
- Imports of heavy dependencies (NumPy / scikit-learn) are deferred until first
  use via small helper functions (`_np`, `_sk_text`, `_sk_metrics`) to keep
  import time fast and errors localized.
- The knowledge base can live in:
    * Local path: `TRIAGE_KB_LOCAL` (preferred, fastest), or
    * GCS object: `TRIAGE_KB_GCS` (downloaded lazily to `TRIAGE_KB_LOCAL`).
- CSV schema is flexible: we look up common column aliases for each field.
- This is an in-process index (non-persistent). Rebuild happens on interpreter
  restart or after process redeploy.

CSV Expectations
----------------
- Columns (case-insensitive, flexible):
    condition | disease | name
    symptoms  | symptom | features
    advice    | self_care | recommendations
    url       | link | source (optional)
- Free-text fields are recommended; no strict tokenization needed.

Caveats
-------
- This is a toy/portable RAG layer — not optimized for very large datasets.
- No cross-lingual support, spelling correction, or semantic embeddings here.
- For production, consider a vector DB or server-side embeddings service.
"""


# --- Lazy imports (avoid heavy deps on import time) --------------------------
def _np():
    """Defer import of NumPy until needed."""
    import numpy as np
    return np

def _sk_text():
    """Defer import of scikit-learn's TfidfVectorizer until needed."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer

def _sk_metrics():
    """Defer import of scikit-learn's cosine_similarity until needed."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity


def _download_gcs_to_local(gcs_uri: str, local_path: str) -> bool:
    """
    Best-effort download of a GCS object to a local file.

    Parameters
    ----------
    gcs_uri : str
        GCS URI in the form gs://bucket/path/to/file.csv
    local_path : str
        Local filesystem path to save the downloaded object.

    Returns
    -------
    bool
        True if the file was downloaded successfully; False otherwise.

    Notes
    -----
    - If `google-cloud-storage` is unavailable or credentials are missing,
      this will simply return False without raising.
    """
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


# --- Data structures & globals ----------------------------------------------
@dataclass
class KBRow:
    """
    A single knowledge base entry.

    Attributes
    ----------
    condition : str
        Condition or topic label (e.g., "Strep throat").
    symptoms : str
        Free-text description of symptoms/features.
    advice : str
        Free-text self-care or general guidance text.
    url : str | None
        Optional source/reference URL for the row.
    """
    condition: str
    symptoms: str
    advice: str
    url: str|None = None


# Global singletons for the in-memory index
_Vectorizer = None   # type: Any | None
_MATRIX = None       # type: Any | None  # sparse matrix (n_samples x n_terms)
_ROWS: List[KBRow] = []  # raw KB rows


# --- Loading & indexing ------------------------------------------------------
def _ensure_local_csv() -> str:
    """
    Ensure a local CSV file exists (download from GCS if necessary).

    Returns
    -------
    str
        Local path to the CSV, or empty string if unavailable.
    """
    if os.path.exists(TRIAGE_KB_LOCAL):
        return TRIAGE_KB_LOCAL
    ok = _download_gcs_to_local(TRIAGE_KB_GCS, TRIAGE_KB_LOCAL)
    return TRIAGE_KB_LOCAL if ok and os.path.exists(TRIAGE_KB_LOCAL) else ""

def _load_rows() -> List[KBRow]:
    """
    Load KB rows from CSV, accepting flexible column names.

    Returns
    -------
    List[KBRow]
        Parsed rows; may be empty if the CSV is missing/unreadable.
    """
    path = _ensure_local_csv()
    rows: List[KBRow] = []
    if not path:
        return rows

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Resolve flexible column names with forgiving defaults.
        cond_col = next((c for c in reader.fieldnames or []
                         if c.lower() in ("condition", "disease", "name")), "condition")
        sym_col  = next((c for c in reader.fieldnames or []
                         if c.lower() in ("symptoms", "symptom", "features")), "symptoms")
        adv_col  = next((c for c in reader.fieldnames or []
                         if c.lower() in ("advice", "self_care", "recommendations")), "advice")
        url_col  = next((c for c in reader.fieldnames or []
                         if c.lower() in ("url", "link", "source")), None)

        for r in reader:
            rows.append(
                KBRow(
                    condition=(r.get(cond_col) or "").strip(),
                    symptoms=(r.get(sym_col) or "").strip(),
                    advice=(r.get(adv_col) or "").strip(),
                    url=(r.get(url_col) or None) if url_col else None,
                )
            )
    return rows

def _build_index() -> None:
    """
    Build (or no-op if already built) the TF-IDF index over KB rows.

    Side Effects
    ------------
    - Populates global `_ROWS`, `_Vectorizer`, and `_MATRIX`.

    Notes
    -----
    - This is done lazily on first call to `rag_stats`/`rag_search`.
    - Uses unigrams and bigrams for simple breadth; adjust as needed.
    """
    global _Vectorizer, _MATRIX, _ROWS
    if _Vectorizer is not None and _MATRIX is not None and _ROWS:
        return

    _ROWS = _load_rows()
    if not _ROWS:
        return

    # Concatenate fields to provide more context per row for TF-IDF.
    texts = [f"{r.condition} | {r.symptoms} | {r.advice}" for r in _ROWS]

    TfidfVectorizer = _sk_text()
    _Vectorizer = TfidfVectorizer(
        min_df=1,
        max_features=30000,
        ngram_range=(1, 2),  # unigrams + bigrams
    )
    _MATRIX = _Vectorizer.fit_transform(texts)


# --- Public API --------------------------------------------------------------
def rag_stats() -> Dict[str, int]:
    """
    Return simple stats about the in-memory RAG index.

    Returns
    -------
    dict
        {
          "rows": <int>,       # number of KB rows loaded
          "indexed": <int>     # number of rows actually indexed (0 if not built)
        }
    """
    _build_index()
    return {
        "rows": len(_ROWS),
        "indexed": int(_MATRIX.shape[0]) if _MATRIX is not None else 0
    }

def rag_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve top-k similar KB rows for a free-text query.

    Parameters
    ----------
    query : str
        Free text (e.g., user symptoms). The function will compute a TF-IDF
        vector and compare against the KB.
    top_k : int, default=3
        Number of top results to return.

    Returns
    -------
    List[dict]
        Each result is a dict with fields:
        {
          "condition": <str>,
          "symptoms":  <str>,
          "advice":    <str>,
          "url":       <str | None>,
          "score":     <float in [0,1]>   # cosine similarity (approximate)
        }

    Notes
    -----
    - If the index is not available (no CSV, empty rows, etc.), returns [].
    - Logs a compact “dataset” evidence entry with the number of hits.
    """
    _build_index()
    if not _ROWS or _Vectorizer is None or _MATRIX is None:
        return []

    # Vectorize the query and compute cosine similarity to all KB rows.
    vec = _Vectorizer.transform([query])
    cosine_similarity = _sk_metrics()
    sims = cosine_similarity(vec, _MATRIX)[0]

    # Arg-sort indices for top-k results (highest similarity first).
    idx = sims.argsort()[-top_k:][::-1]

    out: List[Dict[str, Any]] = []
    for i in idx:
        r = _ROWS[i]
        out.append({
            "condition": r.condition,
            "symptoms":  r.symptoms,
            "advice":    r.advice,
            "url":       r.url,
            "score":     round(float(sims[i]), 3),
        })

    # Record an evidence entry for transparency (do not include raw text).
    EVIDENCE.add("dataset", f"{len(out)} similar cases")
    return out
