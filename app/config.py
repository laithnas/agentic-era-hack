# app/config.py
from __future__ import annotations
import os

# LLM backend
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

# Maps / Places
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# RAG dataset
TRIAGE_KB_GCS = os.getenv("TRIAGE_KB_GCS", "gs://lohealthcare/ai-medical-chatbot.csv")
TRIAGE_KB_LOCAL = os.getenv("TRIAGE_KB_LOCAL", "/tmp/ai-medical-chatbot.csv")

# Privacy / Safety toggles
PHI_ZERO_RETENTION = os.getenv("PHI_ZERO_RETENTION", "false").lower() == "true"

# Timeouts
HTTP_TIMEOUT_SECS = int(os.getenv("HTTP_TIMEOUT_SECS", "8"))

# Evidence filtering (don’t show evidence outside these flows)
EVIDENCE_ALLOWED_SOURCES = {
    "dataset",        # RAG hits during triage
    "triage_rules",   # explicit rules used
    "triage_kb",      # KB matches
    "whatif_calc",    # calculator math/logic
    "whatif_dataset", # any dataset used by what-if
    "medsx_dataset",  # side effects dataset rows
    "medsx_rules",    # rules used in meds check
}
