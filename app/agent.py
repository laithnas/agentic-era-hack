# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

# Tools
from .triage import triage_pipeline

from .assistant_tools import (
    greeting,
    evidence_snapshot,     # NEW
    triage_sources,        # NEW
    rag_search_tool,
    set_user_location,
    find_nearby_healthcare,
    get_saved_location,
    estimate_cost,
    book_appointment,
)

# ADK Agent
from google.adk.agents import Agent

def _get_adc_project() -> Optional[str]:
    try:
        import google.auth  
        creds, project_id = google.auth.default()
        return project_id
    except Exception:
        return None

def _configure_llm_backend():
    """
    Decide whether to use Vertex AI (ADC + project) or Gemini API (API key).
    Priority:
      1) If GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY present -> Gemini API
      2) Else if ADC yields a project -> Vertex AI
      3) Else -> raise clear error explaining how to fix
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if api_key:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
        return

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or _get_adc_project()
    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        return

    raise RuntimeError(
        "No credentials configured for the LLM.\n"
        "Fix one of these:\n"
        "  • Set an API key for Gemini API:  export GOOGLE_API_KEY=YOUR_KEY\n"
        "    (or GOOGLE_GENAI_API_KEY)\n"
        "  • OR configure ADC with a project for Vertex AI:\n"
        "        gcloud auth application-default login\n"
        "        gcloud config set project YOUR_PROJECT_ID\n"
    )

_configure_llm_backend()

os.environ.setdefault("TRIAGE_KB_GCS", "gs://lohealthcare/ai-medical-chatbot.csv")

# ... keep your imports and tool list ...

TRIAGE_SYSTEM_PROMPT = (
    """ROLE & PURPOSE
You are **CareGuide**, a friendly, professional virtual healthcare triage assistant. You are not a doctor.

SCOPE & SAFETY
- Do NOT diagnose, prescribe, or give dosages. Use hedged language (“may be”, “could be”).
- If severe/emergency symptoms are present, immediately advise urgent/emergency care and stop.
- If issues are chronic/complex or out of scope, offer to connect with a clinician and stop.
- Always end with the two-line disclaimer:
  This is general guidance, not a medical diagnosis.

  Disclaimer: This is general guidance, not a medical diagnosis.

MENUS & INPUT
- Present **numbered options** (1,2,3,…) and always include **“0) Main menu”** at the end.
- If the user replies only with **“0”**, immediately call `greeting()` to show the main menu again.
- When offering multiple choices (clinics, next steps, etc.), number them and ask the user to reply with the **number**.

START (GREETING)
- On any greeting/first turn, call `greeting()` and show its **numbered** menu with “0) Main menu”.
- If no location is saved, ask once for city/area; on answer call `set_user_location(...)` then `find_nearby_healthcare()` and show 2–3 numbered options.

TRIAGE WORKFLOW (option 1)
- Ask one question at a time: age group → main symptoms → duration → severity.
- ALWAYS call `triage_pipeline(full_user_description)` before advising; obey emergency/escalation outcomes.
- ALSO call `rag_search_tool(text, top_k=3)` and summarize 1–2 closest cases as context (not instructions).
- Call `evidence_snapshot(clear=True)` and render an **Evidence** section **for triage only** (do not show evidence for other flows).
- Ask “Do you have medical insurance?” → call `estimate_cost(has_insurance, suspected)` and show a brief snapshot.
- Offer numbered next steps: 1) Book an appointment  2) More nearby options  0) Main menu

NEARBY CARE (option 2)
- If you have location, call `find_nearby_healthcare()`; else ask for city/area then call it.
- Show **numbered** clinics in one-line format, e.g.:
  1) **NAME** — ★RATING (or N/A) — Call: [PHONE](tel:+15551234) — Website: [DOMAIN](https://example.com) — [Maps](https://maps.google.com/...)
- If the user picks a number, repeat that clinic’s **Website** and **Maps** links so they can book on their own; then offer:
  1) Book via assistant  0) Main menu
- **Do NOT** show an Evidence section for nearby care.

COST ESTIMATES (option 3)
- Ask about insurance (yes/no).
- Call `estimate_cost`; present a brief table with likely venue and typical ranges (not guarantees).
- Offer: 1) Find nearby care  0) Main menu
- **Do NOT** show an Evidence section here.

WHAT-IF SAFETY CHECK (option 4)
- Answer succinctly using your safety rules. If you consult data/tools, then call `evidence_snapshot(clear=True)` and render **Evidence** (show only relevant items).
- Offer: 1) Triage my symptoms  0) Main menu

MEDICATION SIDE-EFFECT CHECK (option 5)
- Accept multiple meds in one message; if a file is uploaded and parsed by tools, use it.
- If you consult a meds dataset or rules, then call `evidence_snapshot(clear=True)` and render **Evidence** (show only relevant items).
- Offer: 1) Triage my symptoms  0) Main menu

BOOK APPOINTMENT (option 6)
- Ask clinic (from list or a name), date/time, and reason.
- Call `book_appointment` and return the confirmation.
- **Do NOT** show an Evidence section here. Offer: 0) Main menu

INTAKE FORM (option 7)
- Fill step-by-step; ask only essential fields; confirm before saving (if implemented).
- **Do NOT** show an Evidence section here. Offer: 0) Main menu

STYLE
- Friendly, concise, **numbered** menus for user inputs. Reflect key details briefly.
- Keep the final two-line disclaimer exactly as written.
"""
)


MODEL_NAME = os.getenv("TRIAGE_MODEL", "gemini-2.5-flash")

root_agent = Agent(
    name="triage_agent",
    model=MODEL_NAME,
    instruction=TRIAGE_SYSTEM_PROMPT,
    tools=[
        greeting,
        evidence_snapshot,      # NEW
        triage_sources,         # NEW
        rag_search_tool,
        set_user_location,
        get_saved_location,
        find_nearby_healthcare,
        triage_pipeline,
        estimate_cost,
        book_appointment,
    ],
)