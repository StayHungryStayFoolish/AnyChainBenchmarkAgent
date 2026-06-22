"""Safe provider/auth diagnostics for ADK workflows."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from llm.config import load_llm_config
from llm.google_auth import credential_plan

from .read_only import _tool_result


def inspect_llm_auth() -> dict[str, Any]:
    """Inspect configured LLM auth mode without reading or printing secrets."""
    config = load_llm_config()
    validation_errors = config.validate()
    plan = credential_plan(config).safe_dict() if config.provider in {"gemini", "claude"} else {}
    gcloud_path = shutil.which("gcloud") or ""
    adc_file = _local_adc_file()
    json_key_path = Path(config.google_application_credentials).expanduser() if config.google_application_credentials else None
    data: dict[str, Any] = {
        "llm": config.safe_dict(),
        "google_credential_plan": plan,
        "gcloud": {
            "available": bool(gcloud_path),
            "path": gcloud_path,
        },
        "local_adc": {
            "well_known_file_exists": adc_file.is_file(),
            "well_known_file": str(adc_file),
        },
        "service_account_file": {
            "configured": bool(config.google_application_credentials),
            "file_exists": bool(json_key_path and json_key_path.is_file()),
        },
    }
    warnings = list(validation_errors)
    next_actions = _next_actions(config.auth_mode, data, validation_errors)
    return _tool_result(
        status="ok" if not validation_errors else "blocked",
        data=data,
        warnings=warnings,
        next_actions=next_actions,
    )


def get_auth_tools() -> list:
    """Return auth diagnostic tool callables."""
    return [inspect_llm_auth]


def _local_adc_file() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home).expanduser() / "gcloud" / "application_default_credentials.json"
    return Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def _next_actions(auth_mode: str, data: dict[str, Any], validation_errors: list[str]) -> list[str]:
    if validation_errors:
        return ["update config/agent_config.sh", "run llm-config", "run provider smoke after config is complete"]
    if auth_mode == "google_adc":
        if not data["local_adc"]["well_known_file_exists"] and not data["gcloud"]["available"]:
            return ["install Google Cloud CLI", "run gcloud auth application-default login"]
        if not data["local_adc"]["well_known_file_exists"]:
            return ["run gcloud auth application-default login"]
        return ["run llm-smoke with explicit provider credentials"]
    if auth_mode == "attached_service_account":
        return ["run on GCE/GKE/Cloud Run with an attached service account", "run llm-smoke on the target host"]
    if auth_mode == "service_account_impersonation":
        return ["verify caller has serviceAccountTokenCreator permission", "run llm-smoke on the target host"]
    if auth_mode == "service_account_file":
        if not data["service_account_file"]["file_exists"]:
            return ["set GOOGLE_APPLICATION_CREDENTIALS to an existing JSON key file path"]
        return ["run llm-smoke", "prefer ADC or impersonation for enterprise usage"]
    return ["run llm-smoke"]

