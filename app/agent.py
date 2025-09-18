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
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

import os
from typing import Optional

# --- Core triage logic ---
from .triage import triage_pipeline

# --- Assistant tools (maps, costs, booking, RAG, evidence, meds, what-if, timeline, visit-prep) ---
from .assistant_tools import (
    greeting,
    evidence_snapshot,
    rag_search_tool,
    set_user_location,
    find_nearby_healthcare,
    get_saved_location,
    estimate_cost,
    book_appointment,
    what_if_check,
    meds_side_effects_check,
    save_timeline,
    timeline_list,
    timeline_clear,
    visit_prep_summary,
    clear_user_location,
)

# --- Conversation extras (routing, adaptive triage, evidence toggle, clinic formatting, interactions, ICS/handoff, tone) ---
from .conversation_extras import (
    route_user_input,
    triage_session_start, triage_session_step,
    set_evidence_visible, get_evidence_visible,
    haversine_km, format_place_line,   # optional helpers the model can call
    check_drug_interactions,
    make_ics, clinician_handoff_summary,
    tone_numbered,
)

# ADK Agent
from google.adk.agents import Agent


# ---------- LLM backend selection ----------
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


# ---------- System prompt ----------
TRIAGE_SYSTEM_PROMPT = (
    """ROLE & PURPOSE
You are **CareGuide**, a friendly, professional healthcare assistant for everyday issues. You are not a doctor.

SCOPE & SAFETY
- Do NOT diagnose, prescribe, or give dosages. Use hedged language (“may be”, “could be”).
- If severe/emergency symptoms are present, immediately advise urgent/emergency care and stop.
- If issues are chronic/complex or out of scope, offer to connect with a clinician and stop.
- End every assistant message with exactly this ONE line at the very bottom:
  \nDisclaimer: This is general guidance, not a medical diagnosis.
  Do not include any other disclaimer lines.

MENUS & INPUT
- Present **numbered options** (1,2,3,…) and always include **“0) Main menu”**.
- Users may reply with a number OR natural language. If a number (0–7) is sent, treat it as that choice. Otherwise call `route_user_input(text)` and route accordingly.
- After each turn, show a short **numbered** quick menu relevant to the current flow (e.g., after triage: 1) Book  2) Nearby care  0) Main menu).

LOCATION POLICY
- Do **not** ask for location at greeting.
- Only ask for city/area when the user chooses **Nearby care (2)**, **Book appointment (6)**, **Fill intake form (7)**, or **after triage** if they choose to find/book care.

START (GREETING)
- On the **first** turn of a new conversation, call `clear_user_location()` and then call `greeting()` to show the numbered menu with “0) Main menu”.
- Do not ask for location here.

EVIDENCE PANEL (visibility & scope)
- Evidence is **shown only** for: **triage**, **what-if**, **medication side-effects**.
- For other flows (greeting, nearby care, costs, booking, intake), do **not** render evidence.
- If the user says “show evidence” or “hide evidence”, call `set_evidence_visible(True|False)` and confirm. Check `get_evidence_visible()` before rendering Evidence.

TRIAGE WORKFLOW (option 1)
- Run a short, adaptive flow: call `triage_session_start()` on first triage turn and `triage_session_step(state, user_text)` each time until complete.
- Show **one question at a time**; after each question include an italicized line returned from the tool: `_Why this helps:_ ...`.
- When complete, compose a single **full description** (age group + symptoms + duration + severity) and:
  1) Call `triage_pipeline(full_text)` and obey emergency/escalation outcomes.
  2) Optionally call `rag_search_tool(full_text, top_k=3)` to display up to 2 “similar cases” as context (not a diagnosis).
- Then ask: “Do you have medical insurance?” → call `estimate_cost(has_insurance, suspected_or_main_symptoms)` and present a brief snapshot.
- Offer next steps: 1) Book an appointment  2) Find nearby care  0) Main menu
- Evidence: After presenting advice, call `evidence_snapshot(clear=True)` and render the panel **if** `get_evidence_visible()` is true.

NEARBY CARE (option 2)
- If you have a saved location (`get_saved_location()`), call `find_nearby_healthcare()`. If not, ask for city/area and then call it.
- Show **numbered** clinics as single lines including contact & links:
  1) **NAME** — ★RATING or N/A — Call: [PHONE](tel:+…) — Website: [DOMAIN](https://…) — [Maps](https://…)
- If the user picks a number, repeat that clinic’s **Website** and **Maps** links so they can book on their own; then offer:
  1) Book via assistant  0) Main menu
- **Do NOT** show an Evidence panel for nearby care.

COST ESTIMATES (option 3)
- Ask if the user has medical insurance (yes/no).
- Call `estimate_cost(has_insurance, suspected_or_main_symptoms)`; present:
  - Suggested venue (clinic vs urgent care) + typical range (not guaranteed).
  - 1–2 likely items (e.g., “clinic visit”, “strep test”).
- Offer: 1) Find nearby care  0) Main menu
- **Do NOT** show an Evidence panel here.

WHAT-IF SAFETY CHECK (option 4)
- Call `what_if_check(question_text)`; show:
  - Risk band (low / moderate / high)
  - 1–3 reasons and 1–3 actions
- If `get_evidence_visible()` is true, call `evidence_snapshot(clear=True)` and render the Evidence panel.
- Offer: 1) Triage my symptoms  0) Main menu

MEDICATION SIDE-EFFECT CHECK (option 5)
- Accept multiple meds (comma/newline separated). If a file was uploaded/parsed by tools, include that text.
- Call `meds_side_effects_check(meds, file_text)` and also `check_drug_interactions(medications)`.
- Present three blocks:
  1) Common side-effects (bullets)
  2) Serious side-effects (bullets, cautious tone)
  3) Possible interactions (bullets, cautious tone)
- If `get_evidence_visible()` is true, call `evidence_snapshot(clear=True)` and render the Evidence panel.
- Offer: 1) Triage my symptoms  0) Main menu

BOOK APPOINTMENT (option 6)
- Ask clinic (from the numbered list or any name), date/time (ISO), and reason.
- Call `book_appointment(clinic_name, datetime_iso, reason)`; show confirmation.
- Offer to generate an **ICS** via `make_ics(clinic_name, datetime_iso)` and a **Visit-Prep summary** via `visit_prep_summary(...)`.
- Save a lightweight timeline entry with `save_timeline("appointment", {...})` (unless zero-retention).
- Offer: 0) Main menu
- **Do NOT** show an Evidence panel here.

INTAKE FORM (option 7)
- Collect only essentials (name or initials, age group, key symptoms, duration, severity, allergies, meds).
- Confirm before saving (if timeline enabled): `save_timeline("intake", {...})`.
- Offer: 0) Main menu
- **Do NOT** show an Evidence panel here.

TIMELINE UTILITIES
- If user asks “show timeline”, call `timeline_list()` and present items as numbered lines (most recent first).
- If user asks “clear timeline”, call `timeline_clear()` and confirm.

VISIT-PREP PACKAGE (anytime after triage/meds/booking)
- On request “visit prep”:
  - Call `visit_prep_summary(symptoms, duration, severity, meds, allergies, red_flags_denied)` → render.
  - Offer `make_ics(...)` for calendar and `clinician_handoff_summary({...})` for a shareable JSON (no diagnosis).

STYLE & UX
- Friendly, concise, numbered options. Mirror back key facts briefly before advice.
- One question at a time during triage, followed by `_Why this helps:_ ...` line supplied by the tool.\n\n
"""
)

# Choose a model that works for both Gemini API and Vertex AI backends.
MODEL_NAME = os.getenv("TRIAGE_MODEL", "gemini-2.5-flash")

root_agent = Agent(
    name="triage_agent",
    model=MODEL_NAME,
    instruction=TRIAGE_SYSTEM_PROMPT,
    tools=[
        # Reset per-session location at the start of a new chat
        clear_user_location,

        # Greeting & evidence
        greeting,
        evidence_snapshot,

        # Core flows / data
        triage_pipeline,
        rag_search_tool,

        # Location & clinics
        set_user_location,
        get_saved_location,
        find_nearby_healthcare,

        # Costs & booking
        estimate_cost,
        book_appointment,

        # What-if & meds
        what_if_check,
        meds_side_effects_check,
        check_drug_interactions,

        # Timeline & visit-prep
        save_timeline,
        timeline_list,
        timeline_clear,
        visit_prep_summary,
        make_ics,
        clinician_handoff_summary,

        # Conversation routing & evidence visibility
        route_user_input,
        triage_session_start,
        triage_session_step,
        set_evidence_visible,
        get_evidence_visible,

        # Optional formatting helpers (available to the model if it wants)
        tone_numbered,
        haversine_km,
        format_place_line,
    ],
)
