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

# mypy: disable-error-code="attr-defined,arg-type"

"""
Vertex AI Agent Engine bootstrap.

This module defines an `AdkApp` subclass that:
- Configures Cloud Logging and OpenTelemetry tracing (exported to Cloud Trace),
- Exposes a simple feedback ingestion endpoint,
- Deploys/updates the agent engine on Vertex AI Agent Engines,
- Persists the deployed engine ID locally for use by the Streamlit UI / clients.

Typical usage (CLI):
    python -m app.deploy_agent_engine_app \\
        --project <PROJECT_ID> \\
        --location us-central1 \\
        --agent-name healthcare-guy \\
        --requirements-file .requirements.txt \\
        --extra-packages ./app \\
        --set-env-vars TRIAGE_MODEL=gemini-2.5-flash,GOOGLE_CLOUD_PROJECT=<PROJECT_ID>

The exported function `deploy_agent_engine_app` is also callable from other scripts.
"""

import copy
import datetime
import json
import logging
import os
from typing import Any

import google.auth
import vertexai
from google.adk.artifacts import GcsArtifactService
from google.cloud import logging as google_cloud_logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, export
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from app.agent import root_agent
from app.utils.gcs import create_bucket_if_not_exists
from app.utils.tracing import CloudTraceLoggingSpanExporter
from app.utils.typing import Feedback


class AgentEngineApp(AdkApp):
    """ADK application wrapper that adds logging, tracing, and feedback intake."""

    def set_up(self) -> None:
        """Set up logging and tracing for the agent engine app.

        - Initializes Cloud Logging client and stores a structured logger on `self.logger`.
        - Configures an OpenTelemetry tracer provider with a Cloud Trace exporter.
        """
        super().set_up()

        # ----- Cloud Logging setup -----
        logging_client = google_cloud_logging.Client()
        self.logger = logging_client.logger(__name__)

        # ----- OpenTelemetry tracing → Cloud Trace -----
        provider = TracerProvider()
        processor = export.BatchSpanProcessor(
            CloudTraceLoggingSpanExporter(
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback from clients.

        Args:
            feedback: Arbitrary dict payload from the UI. Validated against `Feedback` pydantic model.

        Side effects:
            Writes a structured log entry to Cloud Logging (INFO severity).
        """
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def register_operations(self) -> dict[str, list[str]]:
        """Register callable operations exposed by the Agent Engine.

        Extends the base ADK operations with a custom `register_feedback` endpoint.

        Returns:
            A mapping from namespace to list of operation names.
        """
        operations = super().register_operations()
        operations[""] = operations[""] + ["register_feedback"]
        return operations

    def clone(self) -> "AgentEngineApp":
        """Return a deep-copied clone of this ADK application.

        This is used by the platform to create new worker replicas while preserving
        the configuration and tool wiring.

        Returns:
            AgentEngineApp: A cloned instance with the same template attributes.
        """
        template_attributes = self._tmpl_attrs
        return self.__class__(
            agent=copy.deepcopy(template_attributes["agent"]),
            enable_tracing=bool(template_attributes.get("enable_tracing", False)),
            session_service_builder=template_attributes.get("session_service_builder"),
            artifact_service_builder=template_attributes.get(
                "artifact_service_builder"
            ),
            env_vars=template_attributes.get("env_vars"),
        )


def deploy_agent_engine_app(
    project: str,
    location: str,
    agent_name: str | None = None,
    requirements_file: str = ".requirements.txt",
    extra_packages: list[str] = ["./app"],
    env_vars: dict[str, str] = {},
    service_account: str | None = None,
) -> agent_engines.AgentEngine:
    """Deploy or update the agent engine app on Vertex AI Agent Engines.

    This function:
      1. Ensures the staging and artifacts GCS buckets exist.
      2. Initializes Vertex AI for the given project/location.
      3. Builds the ADK application (`AgentEngineApp`) with a GCS artifact store.
      4. Reads Python dependencies from `requirements_file`.
      5. Creates or updates an agent engine with the given name.
      6. Persists a small JSON file (`deployment_metadata.json`) containing the engine ID.

    Args:
        project: GCP project ID to deploy into.
        location: Vertex AI region (e.g., "us-central1").
        agent_name: Display name for the agent; if an engine with this name exists, it will be updated.
        requirements_file: Path to a pip requirements file to install into the runtime.
        extra_packages: Local packages (dirs) to bundle with the deployment (e.g., `./app`).
        env_vars: Environment variables to set for the deployed runtime. Values are strings.
        service_account: Optional service account email for the runtime; defaults to project compute SA.

    Returns:
        agent_engines.AgentEngine: The created or updated remote agent engine object.

    Raises:
        FileNotFoundError: If `requirements_file` cannot be read.
        google.api_core.exceptions.GoogleAPIError: On Vertex AI API errors during create/update.
    """
    # ---- Buckets for artifacts and staging (created if missing) ----
    staging_bucket_uri = f"gs://{project}-agent-engine"
    artifacts_bucket_name = f"{project}-healthcare-guy-logs-data"

    create_bucket_if_not_exists(
        bucket_name=artifacts_bucket_name, project=project, location=location
    )
    create_bucket_if_not_exists(
        bucket_name=staging_bucket_uri, project=project, location=location
    )

    # ---- Vertex AI init ----
    vertexai.init(project=project, location=location, staging_bucket=staging_bucket_uri)

    # ---- Read requirements to pin the runtime environment ----
    with open(requirements_file) as f:
        requirements = f.read().strip().split("\n")

    # ---- Build the ADK app with a GCS artifact store ----
    agent_engine = AgentEngineApp(
        agent=root_agent,
        artifact_service_builder=lambda: GcsArtifactService(
            bucket_name=artifacts_bucket_name
        ),
    )

    # Keep the worker pool single-threaded unless you know your tools are re-entrant.
    env_vars["NUM_WORKERS"] = "1"

    # ---- Common configuration for both create and update operations ----
    agent_config = {
        "agent_engine": agent_engine,
        "display_name": agent_name,
        "description": "A base ReAct agent built with Google's Agent Development Kit (ADK)",
        "extra_packages": extra_packages,
        "env_vars": env_vars,
        "service_account": service_account,
        "requirements": requirements,
    }
    logging.info(f"Agent config: {agent_config}")

    # ---- Create or update depending on whether the name already exists ----
    existing_agents = list(agent_engines.list(filter=f"display_name={agent_name}"))
    if existing_agents:
        logging.info(f"Updating existing agent: {agent_name}")
        remote_agent = existing_agents[0].update(**agent_config)
    else:
        logging.info(f"Creating new agent: {agent_name}")
        remote_agent = agent_engines.create(**agent_config)

    # ---- Persist engine ID locally for the UI and tooling ----
    config = {
        "remote_agent_engine_id": remote_agent.resource_name,
        "deployment_timestamp": datetime.datetime.now().isoformat(),
    }
    config_file = "deployment_metadata.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Agent Engine ID written to {config_file}")

    return remote_agent


if __name__ == "__main__":
    # Simple CLI wrapper around `deploy_agent_engine_app`.
    import argparse

    parser = argparse.ArgumentParser(description="Deploy agent engine app to Vertex AI")
    parser.add_argument(
        "--project",
        default=None,
        help="GCP project ID (defaults to application default credentials)",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="GCP region (defaults to us-central1)",
    )
    parser.add_argument(
        "--agent-name",
        default="healthcare-guy",
        help="Name for the agent engine",
    )
    parser.add_argument(
        "--requirements-file",
        default=".requirements.txt",
        help="Path to requirements.txt file",
    )
    parser.add_argument(
        "--extra-packages",
        nargs="+",
        default=["./app"],
        help="Additional packages to include",
    )
    parser.add_argument(
        "--set-env-vars",
        help="Comma-separated list of environment variables in KEY=VALUE format",
    )
    parser.add_argument(
        "--service-account",
        default=None,
        help="Service account email to use for the agent engine",
    )
    args = parser.parse_args()

    # Parse environment variables passed via CLI into a dict.
    env_vars = {}
    if args.set_env_vars:
        for pair in args.set_env_vars.split(","):
            key, value = pair.split("=", 1)
            env_vars[key] = value

    # Infer project from ADC if not explicitly provided.
    if not args.project:
        _, args.project = google.auth.default()

    print(
        """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤖 DEPLOYING AGENT TO VERTEX AI AGENT ENGINE 🤖         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    )

    # Execute deployment with effective parameters.
    deploy_agent_engine_app(
        project=args.project,
        location=args.location,
        agent_name=args.agent_name,
        requirements_file=args.requirements_file,
        extra_packages=args.extra_packages,
        env_vars=env_vars,
        service_account=args.service_account,
    )
