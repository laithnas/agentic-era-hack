import os, re, io, json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# Prefer TF-IDF for hackathon; tune with env vars
_MAX_ROWS = int(os.getenv("RAG_MAX_ROWS", "120000"))
_MAX_FEATURES = int(os.getenv("RAG_MAX_FEATURES", "120000"))
_GCS_URI = os.getenv("TRIAGE_KB_GCS", "gs://lohealthcare/ai-medical-chatbot.csv")

_DF: Optional[pd.DataFrame] = None
_VEC = None
_MAT = None
_READY = False
_SOURCE = "unknown"

def _gcs_download_text(gcs_uri: str) -> str:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("TRIAGE_KB_GCS must be a gs:// URI")
    from google.cloud import storage
    bucket_name, key = gcs_uri.replace("gs://","").split("/", 1)
    client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    blob = client.bucket(bucket_name).blob(key)
    return blob.download_as_text()

def _load_df() -> pd.DataFrame:
    # Try GCS; fallback to local mounted copy if present (for dev)
    try:
        csv_text = _gcs_download_text(_GCS_URI)
        df = pd.read_csv(io.StringIO(csv_text))
        global _SOURCE; _SOURCE = _GCS_URI
        return df
    except Exception:
        local = "/mnt/data/ai-medical-chatbot.csv"
        df = pd.read_csv(local)
        global _SOURCE; _SOURCE = local
        return df

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    # Strip the leading "Q." or similar prefixes
    s = re.sub(r"^\s*(q\.|Q\.)\s*", "", s)
    return s

def _build_index():
    global _DF, _VEC, _MAT, _READY
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = _load_df()
    # Keep only needed cols; drop NAs
    cols = ["Description","Patient","Doctor"]
    df = df[cols].dropna()
    # Build text field
    df["text"] = (df["Patient"].map(_clean_text) + " " + df["Description"].map(_clean_text)).str.lower()
    # Sample to control RAM if needed
    if len(df) > _MAX_ROWS:
        df = df.sample(_MAX_ROWS, random_state=42).reset_index(drop=True)

    # Vectorize
    _VEC = TfidfVectorizer(
        max_features=_MAX_FEATURES,
        ngram_range=(1,2),
        min_df=2,
        stop_words="english"
    )
    _MAT = _VEC.fit_transform(df["text"].tolist())

    _DF = df.reset_index(drop=True)
    _READY = True

def _ensure_ready():
    if not _READY:
        _build_index()

def rag_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top-k similar cases with short doctor snippets.
    """
    _ensure_ready()
    q = _VEC.transform([query.lower()])
    # cosine similarity for tf-idf reduces to linear kernel on L2-normed vectors
    sims = (q @ _MAT.T).toarray().ravel()
    if sims.size == 0:
        return []
    k = min(top_k, sims.shape[0])
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    out = []
    for i in idx:
        row = _DF.iloc[int(i)]
        doc = row.get("Doctor", "")
        # short friendly snippet
        words = doc.split()
        snippet = " ".join(words[:60]) + ("…" if len(words) > 60 else "")
        out.append({
            "similarity": round(float(sims[int(i)]), 3),
            "description": row.get("Description",""),
            "patient": row.get("Patient",""),
            "doctor_snippet": snippet
        })
    return out

def rag_stats() -> Dict[str, Any]:
    """
    Lightweight stats for greeting (“using 120k cases”).
    """
    _ensure_ready()
    return {"rows": int(_DF.shape[0]), "source": _SOURCE, "features": _MAX_FEATURES}
