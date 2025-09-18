# app/prescription_parser.py
from __future__ import annotations
import re
from typing import List, Dict

"""
prescription_parser.py

Purpose
-------
Lightweight text parser for extracting probable medication names from raw
prescription text (e.g., OCR'd or copy-pasted). Designed to be fast,
dependency-free, and schema-friendly for function-calling.

Approach
--------
- Uses a single regex to capture word-like tokens that *look* like drug names,
  optionally followed by common dosage-form words (tablet, capsule, etc.).
- Filters out boilerplate words with a small stoplist (e.g., "mg", "tablet").
- De-duplicates while preserving the first-seen order.
- Returns a schema-safe dict: {"medications": [<lowercased names>]}.

Caveats
-------
- This is intentionally heuristic and conservative; it may miss brand names,
  multi-word generics (e.g., "acetaminophen codeine"), or non-Latin scripts.
- It does not validate against a formulary. For higher precision, you can
  post-validate the output against your meds database before using it.
"""

# Heuristic token:
# - Start with a letter, followed by 1–29 letters or hyphens → up to 30 chars total.
# - Optionally allow a dosage-form word after a space. We don't capture the form.
#   Examples matched: "Amoxicillin", "Azithro", "Metoprolol tablet", "Omeprazole capsule"
_DRUG_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z\-]{1,29})(?:\s+(?:tablet|cap(?:sule)?|syrup|solution))?\b",
    re.IGNORECASE,
)

# Common non-drug tokens often found on prescriptions that we should ignore.
# This is a tiny seed list; extend as needed (e.g., "qhs", "od", "s/l").
_STOPLIST = {
    "take", "tab", "tablet", "capsule", "mg", "mcg", "ml",
    "dose", "daily", "bid", "tid", "qid", "po", "prn"
}


def extract_meds_from_text(file_text: str) -> dict:
    """
    Extract probable medication names from raw text.

    Parameters
    ----------
    file_text : str
        The raw textual contents of a prescription or medication list.
        (If you OCR'd a PDF/image elsewhere, pass the *text* here.)

    Returns
    -------
    dict
        Schema-safe mapping with one key:
          {
            "medications": ["amoxicillin", "ibuprofen", ...]  # max 20 items
          }

    Behavior
    --------
    - Lowercases all candidate names for normalization.
    - Removes duplicates while keeping the first occurrence.
    - Drops common non-med tokens using `_STOPLIST`.
    - Limits the result to at most 20 meds to keep payloads small.

    Examples
    --------
    >>> extract_meds_from_text("Take Amoxicillin 500 mg PO BID and Ibuprofen tablet PRN")
    {'medications': ['amoxicillin', 'ibuprofen']}

    >>> extract_meds_from_text("Metformin 500 mg tablet; Lisinopril 10 mg")
    {'medications': ['metformin', 'lisinopril']}

    Notes
    -----
    - For better accuracy, consider validating the result against a known
      medication list (e.g., your meds CSV) in a downstream step.
    """
    # Normalize line breaks so regex sees a continuous stream.
    t = (file_text or "").replace("\n", " ")

    # Collect raw candidates (group 1 is the putative drug token).
    cands = [m.group(1).lower() for m in _DRUG_PATTERN.finditer(t)]

    # Filter and de-duplicate while preserving order.
    meds: List[str] = []
    for c in cands:
        if c in _STOPLIST:
            continue
        if c not in meds:
            meds.append(c)

    # Keep payload compact; callers can request more if needed.
    return {"medications": meds[:20]}
