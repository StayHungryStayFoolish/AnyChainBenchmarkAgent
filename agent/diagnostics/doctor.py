"""Read-only Agent readiness diagnostics."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from discovery.environment import discover_environment
from knowledge.framework_capabilities import load_framework_capabilities
from llm.config import load_agent_environment, load_llm_config
from llm.google_auth import credential_plan


def run_doctor(discovery: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a read-only readiness report for first-run Agent users."""
    environment = discovery or discover_environment()
    agent_env = load_agent_environment()
    llm_config = load_llm_config()
    llm_errors = llm_config.validate()
    google_auth = _google_auth_diagnostics(llm_config)
    capabilities = load_framework_capabilities()
    dependencies = environment.get("dependencies", {})
    required_missing = dependencies.get("missing_required", [])
    optional_missing = dependencies.get("missing_optional", [])
    warnings = list(environment.get("warnings", []))

    if llm_errors:
        warnings.append("LLM provider is not fully configured; offline dev checks still work, but ADK runtime needs a model configuration.")
    warnings.extend(google_auth.get("warnings", []))
    if required_missing:
        warnings.append("Real benchmark execution may fail until required dependencies are available.")

    readiness = _readiness(required_missing, llm_errors)
    report = {
        "status": readiness,
        "mode": "read_only",
        "environment": {
            "cloud": environment.get("cloud", {}),
            "deployment": environment.get("deployment", {}),
            "network": environment.get("network", {}),
            "disks": environment.get("disks", {}),
            "dependencies": {
                "missing_required": required_missing,
                "missing_optional": optional_missing,
            },
        },
        "llm": llm_config.safe_dict(),
        "knowledge_base": {
            "provider": agent_env.get("AGENT_KNOWLEDGE_PROVIDER", "disabled"),
            "provider_module_configured": bool(agent_env.get("AGENT_KNOWLEDGE_PROVIDER_MODULE", "")),
            "url_configured": bool(agent_env.get("AGENT_KNOWLEDGE_BASE_URL", "")),
            "auth_ref_configured": bool(agent_env.get("AGENT_KNOWLEDGE_AUTH_REF", "")),
        },
        "google_credential_plan": (
            credential_plan(llm_config).safe_dict()
            if llm_config.provider in {"gemini", "claude"} and llm_config.auth_mode != "api_key"
            else {}
        ),
        "google_auth": google_auth,
        "capabilities": {
            "chain_count": capabilities.get("chain_count"),
            "family_count": capabilities.get("family_count"),
            "unique_rpc_method_count": capabilities.get("unique_rpc_method_count"),
            "fake_node_fixture_file_count": capabilities.get("fake_node", {}).get("fixture_file_count"),
        },
        "warnings": warnings,
        "next_actions": _next_actions(required_missing, llm_errors, environment, google_auth),
    }
    return report


def format_doctor_report(report: dict[str, Any]) -> str:
    env = report.get("environment", {})
    cloud = env.get("cloud", {})
    deployment = env.get("deployment", {})
    deps = env.get("dependencies", {})
    llm = report.get("llm", {})
    kb = report.get("knowledge_base", {})
    google_auth = report.get("google_auth", {})
    capabilities = report.get("capabilities", {})
    lines = [
        "Agent doctor report.",
        f"- status: {report.get('status')}",
        f"- cloud: {cloud.get('provider', 'unknown')} / {cloud.get('platform', 'unknown')}",
        f"- deployment: {deployment.get('type', 'unknown')}",
        f"- required dependencies missing: {', '.join(deps.get('missing_required', [])) or '<none>'}",
        f"- optional dependencies missing: {', '.join(deps.get('missing_optional', [])) or '<none>'}",
        f"- LLM provider: {llm.get('provider')} / {llm.get('model')}",
        f"- LLM validation errors: {', '.join(llm.get('validation_errors', [])) or '<none>'}",
        f"- Google auth mode: {google_auth.get('auth_mode', '<not-used>')}",
        f"- gcloud available: {google_auth.get('gcloud_available', '<not-required>')}",
        f"- local ADC file exists: {google_auth.get('local_adc_file_exists', '<not-required>')}",
        f"- knowledge base: {kb.get('provider', 'disabled')}",
        (
            "- capabilities: "
            f"{capabilities.get('chain_count')} chains, "
            f"{capabilities.get('family_count')} families, "
            f"{capabilities.get('unique_rpc_method_count')} unique RPC methods, "
            f"{capabilities.get('fake_node_fixture_file_count')} fake-node fixtures"
        ),
    ]
    warnings = report.get("warnings", [])
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    next_actions = report.get("next_actions", [])
    if next_actions:
        lines.append("Next actions:")
        lines.extend(f"- {action}" for action in next_actions)
    return "\n".join(lines)


def _readiness(required_missing: list[str], llm_errors: list[str]) -> str:
    if required_missing:
        return "needs_dependencies"
    if llm_errors:
        return "ready_without_llm"
    return "ready"


def _next_actions(
    required_missing: list[str],
    llm_errors: list[str],
    environment: dict[str, Any],
    google_auth: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if required_missing:
        actions.append("Use the project Docker image or isolated dependency installer before real benchmark execution.")
    if llm_errors:
        actions.append("Configure config/agent_config.sh with a real model provider before starting the ADK Agent.")
    actions.extend(google_auth.get("next_actions", []))
    disks = environment.get("disks", {})
    if disks.get("ambiguous_candidates"):
        actions.append("Confirm ledger/accounts devices before running disk bottleneck tests.")
    actions.append("Run `run smoke` in the ADK terminal after creating a plan for a local lifecycle check.")
    return actions


def _google_auth_diagnostics(llm_config: Any) -> dict[str, Any]:
    """Return safe Google auth readiness details for Vertex-backed providers."""
    if llm_config.provider not in {"gemini", "claude"} or llm_config.auth_mode == "api_key":
        return {}

    gcloud_path = shutil.which("gcloud") or ""
    adc_file = _local_adc_file()
    service_account_file = Path(llm_config.google_application_credentials).expanduser() if llm_config.google_application_credentials else None
    data: dict[str, Any] = {
        "auth_mode": llm_config.auth_mode,
        "gcloud_available": bool(gcloud_path),
        "gcloud_path": gcloud_path,
        "local_adc_file": str(adc_file),
        "local_adc_file_exists": adc_file.is_file(),
        "service_account_file_configured": bool(service_account_file),
        "service_account_file_exists": bool(service_account_file and service_account_file.is_file()),
        "warnings": [],
        "next_actions": [],
    }

    if llm_config.auth_mode == "google_adc":
        if not data["gcloud_available"] and not data["local_adc_file_exists"]:
            data["warnings"].append("Google ADC mode requires Google Cloud CLI or an existing local ADC file.")
            data["next_actions"].append("Install Google Cloud CLI, then run `gcloud auth application-default login`.")
        elif not data["local_adc_file_exists"]:
            data["next_actions"].append("Run `gcloud auth application-default login` for Google ADC mode.")
    elif llm_config.auth_mode == "service_account_impersonation":
        if not data["gcloud_available"]:
            data["next_actions"].append("Install Google Cloud CLI if this host must create local ADC for impersonation.")
        data["next_actions"].append("Verify the caller can impersonate GOOGLE_SERVICE_ACCOUNT_EMAIL.")
    elif llm_config.auth_mode == "service_account_file" and not data["service_account_file_exists"]:
        data["warnings"].append("GOOGLE_APPLICATION_CREDENTIALS does not point to an existing JSON key file.")
        data["next_actions"].append("Set GOOGLE_APPLICATION_CREDENTIALS to an existing service-account JSON file path.")
    elif llm_config.auth_mode == "attached_service_account":
        data["next_actions"].append("Run the Agent on GCE/GKE/Cloud Run with the expected attached service account.")

    return data


def _local_adc_file() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home).expanduser() / "gcloud" / "application_default_credentials.json"
    return Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
