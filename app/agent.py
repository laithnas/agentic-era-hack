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

import datetime
import os
from zoneinfo import ZoneInfo
from .triage import triage_pipeline

import google.auth
from google.adk.agents import Agent
from .assistant_tools import (
    set_user_location,
    find_nearby_healthcare,
    estimate_cost,
    book_appointment,
)
# Make auth robust: don't crash if ADC isn't set up yet
try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
except Exception:
    pass


_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


TRIAGE_SYSTEM_PROMPT = (
    "You are a cautious, friendly **virtual healthcare triage assistant**.\n\n"
    "CONVERSATION FLOW:\n"
    "• If the user greets (hi/hello/hey) or it's the first turn: "
    "  - Greet warmly and briefly explain what you can do.\n"
    "  - Present a short options menu:\n"
    "    1) Help me narrow down my symptoms\n"
    "    2) Book an appointment\n"
    "    3) Find nearby healthcare centers\n"
    "  - Before proceeding, politely ask: “What city/area are you located in?”\n"
    "  - When the user provides a location, call the tool `set_user_location(location)` "
    "    and then call `find_nearby_healthcare(location)` to surface 2–3 nearby options.\n\n"
    "• If the user chooses **narrow down my symptoms**:\n"
    "  - Ask targeted questions to gather: age group, main symptoms, duration, and severity.\n"
    "  - Use plain English and keep it concise; a single follow-up at a time.\n"
    "  - After collecting enough detail, call the tool `triage_pipeline(full_user_description)` "
    "    and present the structured advice it returns.\n"
    "  - Then ask: “Do you have medical insurance?”\n"
    "    - If yes/no, call `estimate_cost(has_insurance, suspected_condition_or_main_symptoms)` and "
    "      show a friendly snapshot of typical costs and a suggested venue (clinic vs urgent care).\n\n"
    "• If the user chooses **book an appointment**:\n"
    "  - Ask which clinic (they can pick from the nearby list), date/time (ISO recommended), and reason.\n"
    "  - Then call `book_appointment(clinic_name, datetime_iso, reason)` and show the confirmation.\n\n"
    "SAFETY & STYLE:\n"
    "• NEVER diagnose or prescribe. Hedge language: 'might be', 'could be'.\n"
    "• If emergencies are described, the `triage_pipeline` tool will return an emergency message — "
    "  send it verbatim and stop.\n"
    "• Keep answers short, bulleted where useful, and always include a brief disclaimer at the end.\n"
)

root_agent = Agent(
    name="triage_agent",
    model="gemini-2.5-flash",
    instruction=TRIAGE_SYSTEM_PROMPT,
    tools=[
        set_user_location,
        find_nearby_healthcare,
        triage_pipeline,
        estimate_cost,
        book_appointment,
    ],
)