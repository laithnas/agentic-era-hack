# app/evidence_render.py
from __future__ import annotations
from typing import List, Dict

"""
evidence_render.py

Purpose
-------
Render a structured Evidence panel (produced by the agent/tools layer) into a
small Markdown snippet the UI can display beneath the assistant's response.

Design
------
* Accepts a dict payload shaped like: {"items": [{"source": "...", "detail": "..."}, ...]}
* Produces a dict: {"markdown": "<string>"} so it can be used as a function tool.
* Keeps logic intentionally simple and deterministic—no external I/O or state.

Notes
-----
* We return a dict instead of a raw string to keep the function JSON-schema
  friendly for ADK automatic function calling.
* This module does not filter, sort, or redact items; upstream tools should do
  that (see `assistant_tools.evidence_snapshot`).
"""


def evidence_markdown(panel: dict) -> dict:
    """
    Convert an Evidence panel object into a Markdown string.

    Parameters
    ----------
    panel : dict
        Expected shape:
        {
          "items": [
            {"source": "rag|rules|dataset|...", "detail": "short human-friendly note"},
            ...
          ]
        }
        Missing keys are handled gracefully.

    Returns
    -------
    dict
        A mapping with a single key "markdown" that contains the rendered text.
        Example when there are items:
            {"markdown": "**Evidence**\n- **rag:** matched strep throat case\n- **rules:** fever > 102F\n"}
        Example when there are no items:
            {"markdown": "_(No evidence items)_"}

    Behavior
    --------
    - If `panel["items"]` is empty or missing, returns a neutral placeholder.
    - For each item, prints a bullet line with the bolded source and its detail.
    - Does not mutate the input `panel`.

    Examples
    --------
    >>> evidence_markdown({"items":[{"source":"rag","detail":"top-3 similar cases"},{"source":"rules","detail":"fever > 39C"}]})
    {'markdown': '**Evidence**\\n- **rag:** top-3 similar cases\\n- **rules:** fever > 39C'}

    >>> evidence_markdown({"items":[]})
    {'markdown': '_(No evidence items)_'}

    >>> evidence_markdown({})
    {'markdown': '_(No evidence items)_'}

    Safety
    ------
    This function performs no PII processing/redaction. Ensure items have been
    filtered/redacted upstream when needed.
    """
    items: List[Dict] = panel.get("items", [])
    if not items:
        return {"markdown": "_(No evidence items)_"}

    lines = ["**Evidence**"]
    for it in items:
        # Default to generic values if fields are missing to avoid exceptions.
        src = it.get("source", "tool")
        detail = it.get("detail", "")
        lines.append(f"- **{src}:** {detail}")
    return {"markdown": "\n".join(lines)}
