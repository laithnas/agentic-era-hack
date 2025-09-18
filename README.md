# healthcare-guy (CareGuide)

A base **ReAct Agent** built with Google’s **Agent Development Kit (ADK)**.  
Generated with [`googleCloudPlatform/agent-starter-pack`](https://github.com/GoogleCloudPlatform/agent-starter-pack) version `0.14.1`.

CareGuide is an **AI-powered medical assistant** that helps users triage symptoms, find nearby healthcare facilities, estimate costs, check medication safety, and prepare for medical visits, all within strict safety guardrails.

⚠️ **Disclaimer**: This project does not diagnose, prescribe, or replace professional medical advice. All responses are for **general guidance only**.

---


## Project Structure

This project is organized as follows:

```
healthcare-guy/
├── CareGuide/ # Core application code
│ ├── data/ # Dataset storage and related resources
│ ├── utils/ # Utility functions and helpers
│ ├── init.py
│ ├── agent.py # Main agent logic
│ ├── agent_engine_app.py # Vertex AI Agent Engine deployment logic
│ ├── assistant_tools.py # Triage, cost, booking, and meds safety tools
│ ├── config.py # Centralized configuration
│ ├── conversation_extras.py # Extra conversation helpers
│ ├── evidence.py # Evidence logging and storage
│ ├── evidence_panel.py # Evidence panel utilities
│ ├── evidence_render.py # Render evidence into Markdown
│ ├── handoff.py # Handoff logic to external systems
│ ├── i18n.py # Internationalization/localization support
│ ├── prescription_parser.py # Extract medication names from prescriptions
│ ├── rag_dataset.py # RAG dataset management (case library)
│ ├── risk_sim.py # Risk band simulation
│ ├── safety_guard.py # Rule-based safety guardrails
│ ├── schemas.py # Data models and validation schemas
│ ├── server.py # Application entrypoint (local server)
│ ├── social_tone.py # Social tone & sentiment analysis
│ ├── timeline_ai.py # Timeline tracking of interactions
│ ├── triage.py # Symptom triage pipeline
│ ├── triage_kb_admin.py # Admin tools for triage knowledge base
│ └── voice.py # Voice interface utilities
├── .cloudbuild/ # CI/CD pipeline configs for Google Cloud Build
├── deployment/ # Infrastructure & deployment scripts
├── notebooks/ # Jupyter notebooks for prototyping/evaluation
├── tests/ # Unit, integration, and load tests
├── Makefile # Build commands
├── GEMINI.md # AI-assisted development guide
└── pyproject.toml # Project dependencies and configuration
```


---

## Architecture Overview

The project follows a **modular, safety-first architecture**:

- **LLM Agent Core (`agent.py`)**  
  Built using Google’s ADK and Vertex AI (Gemini). Routes user queries, manages conversation state, and applies safety guardrails.

- **Assistant Tools (`assistant_tools.py`)**  
  Implements triage, cost estimation, booking, medication checks, and location-based care search via Google Maps API.

- **Evidence & Explainability (`evidence_render.py`, `rag_dataset.py`)**  
  Provides dataset-backed “similar case” retrieval and renders structured evidence for transparency.

- **Safety Layers (`safety_guard.py`, `risk_sim.py`)**  
  Conservative rule-based checks to block unsafe queries and assess risk categories.

- **UI & Playground**  
  Streamlit-based interface to test the agent locally (`make playground`).

### Design Patterns Used
- **ReAct Pattern** – Reasoning + tool use in one loop  
- **Tool Modularization** – Each tool (triage, cost, meds) is independently pluggable  
- **Safety-First Outputs** – Every response passes disclaimers and sentiment checks  

---

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager ([Install](https://docs.astral.sh/uv/getting-started/installation/))  
- **Google Cloud SDK**: For GCP services ([Install](https://cloud.google.com/sdk/docs/install))  
- **Terraform**: For infrastructure deployment ([Install](https://developer.hashicorp.com/terraform/downloads))  
- **make**: Build automation tool ([Install](https://www.gnu.org/software/make/))  

---

## Quick Start (Local Testing)

Install dependencies and launch the playground:

```bash
make install && make playground || true
