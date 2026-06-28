"""LLM provider and authentication configuration."""

from __future__ import annotations

import os
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SUPPORTED_LLM_PROVIDERS = {
    "gemini",
    "claude",
    "openai",
    "deepseek",
}

SUPPORTED_LLM_AUTH_MODES = {
    "api_key",
    "google_adc",
    "attached_service_account",
    "service_account_impersonation",
    "service_account_file",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_CONFIG = REPO_ROOT / "config" / "agent_config.sh"
USER_CONFIG = REPO_ROOT / "config" / "user_config.sh"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    auth_mode: str = "api_key"
    google_project: str = ""
    google_location: str = "us-central1"
    google_service_account_email: str = ""
    google_application_credentials: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_api_key_present: bool = False
    anthropic_api_key_present: bool = False
    openai_api_key_present: bool = False
    deepseek_api_key_present: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            errors.append(f"unsupported LLM_PROVIDER: {self.provider}")
        if self.auth_mode not in SUPPORTED_LLM_AUTH_MODES:
            errors.append(f"unsupported LLM_AUTH_MODE: {self.auth_mode}")
        if self.provider in {"openai", "deepseek"}:
            if self.auth_mode != "api_key":
                errors.append(f"LLM_AUTH_MODE=api_key is required for LLM_PROVIDER={self.provider}")
            if self.provider == "openai" and not self.openai_api_key_present:
                errors.append("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
            if self.provider == "deepseek" and not self.deepseek_api_key_present:
                errors.append("DEEPSEEK_API_KEY is required for LLM_PROVIDER=deepseek")
            return errors
        if self.auth_mode == "api_key":
            if self.provider == "gemini" and not self.gemini_api_key_present:
                errors.append("GEMINI_API_KEY or GOOGLE_API_KEY is required for LLM_PROVIDER=gemini with LLM_AUTH_MODE=api_key")
            if self.provider == "claude" and not self.anthropic_api_key_present:
                errors.append("ANTHROPIC_API_KEY is required for LLM_PROVIDER=claude with LLM_AUTH_MODE=api_key")
            return errors
        if self.provider in {"gemini", "claude"}:
            if not self.google_project:
                errors.append("GOOGLE_CLOUD_PROJECT is required for Gemini/`claude` on Vertex")
            if not self.google_location:
                errors.append("GOOGLE_CLOUD_LOCATION is required for Gemini/`claude` on Vertex")
            if self.auth_mode == "service_account_impersonation" and not self.google_service_account_email:
                errors.append("GOOGLE_SERVICE_ACCOUNT_EMAIL is required for service_account_impersonation")
            if self.auth_mode == "service_account_file" and not self.google_application_credentials:
                errors.append("GOOGLE_APPLICATION_CREDENTIALS is required for service_account_file")
        return errors

    def is_gemini_family(self) -> bool:
        return self.provider == "gemini" and self.model.lower().startswith("gemini")

    def google_search_eligible(self) -> tuple[bool, str]:
        """Return whether ADK google_search may be enabled for this config.

        ADK google_search is a Gemini Search Grounding tool. Google Cloud auth
        alone is not enough: `claude` on Vertex, DeepSeek, and OpenAI must not be
        advertised as google_search-capable.
        """
        if self.provider != "gemini":
            return False, "unavailable for current provider"
        if not self.model.lower().startswith("gemini"):
            return False, "unavailable for non-Gemini model"
        errors = self.validate()
        if errors:
            return False, "Gemini authentication is incomplete"
        return True, "eligible for ADK google_search"

    def safe_dict(self) -> dict[str, str | bool | list[str]]:
        return {
            "provider": self.provider,
            "model": self.model,
            "auth_mode": self.auth_mode,
            "google_project": self.google_project,
            "google_location": self.google_location,
            "google_service_account_email": self.google_service_account_email,
            "google_application_credentials_configured": bool(self.google_application_credentials),
            "gemini_api_key_configured": self.gemini_api_key_present,
            "anthropic_api_key_configured": self.anthropic_api_key_present,
            "openai_api_key_configured": self.openai_api_key_present,
            "deepseek_api_key_configured": self.deepseek_api_key_present,
            "google_search_eligible": self.google_search_eligible()[0],
            "google_search_reason": self.google_search_eligible()[1],
            "validation_errors": self.validate(),
        }


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    source = env or load_agent_environment()
    provider = source.get("LLM_PROVIDER", "gemini").strip().lower()
    default_model = "gemini-3.1-pro"
    if provider == "gemini":
        default_model = "gemini-3.1-pro"
    elif provider == "claude":
        default_model = "claude-opus-4-8"
    elif provider == "openai":
        default_model = "gpt-5.5"
    elif provider == "deepseek":
        default_model = "deepseek-v4-flash"
    default_auth_mode = "api_key"
    if provider in {"gemini", "claude"}:
        default_auth_mode = source.get("LLM_AUTH_MODE", "api_key").strip().lower()
    auth_mode = source.get("LLM_AUTH_MODE", default_auth_mode).strip().lower()
    google_project = source.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if provider in {"gemini", "claude"} and auth_mode in {"google_adc", "attached_service_account", "service_account_impersonation", "service_account_file"}:
        google_project = google_project or _infer_google_project(source)
    return LLMConfig(
        provider=provider,
        model=source.get("LLM_MODEL", default_model).strip(),
        auth_mode=auth_mode,
        google_project=google_project,
        google_location=source.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip(),
        google_service_account_email=source.get("GOOGLE_SERVICE_ACCOUNT_EMAIL", "").strip(),
        google_application_credentials=source.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
        gemini_api_key=source.get("GEMINI_API_KEY", source.get("GOOGLE_API_KEY", "")).strip(),
        anthropic_api_key=source.get("ANTHROPIC_API_KEY", "").strip(),
        openai_api_key=source.get("OPENAI_API_KEY", "").strip(),
        deepseek_api_key=source.get("DEEPSEEK_API_KEY", "").strip(),
        gemini_api_key_present=bool(source.get("GEMINI_API_KEY", "") or source.get("GOOGLE_API_KEY", "")),
        anthropic_api_key_present=bool(source.get("ANTHROPIC_API_KEY", "")),
        openai_api_key_present=bool(source.get("OPENAI_API_KEY", "")),
        deepseek_api_key_present=bool(source.get("DEEPSEEK_API_KEY", "")),
    )


def _infer_google_project(source: Mapping[str, str]) -> str:
    for key in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT_ID", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        value = source.get(key, "").strip()
        if value:
            return value
    adc_file = source.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    candidates = []
    if adc_file:
        candidates.append(Path(adc_file).expanduser())
    cloud_sdk_config = source.get("CLOUDSDK_CONFIG", "").strip()
    if cloud_sdk_config:
        candidates.append(Path(cloud_sdk_config).expanduser() / "application_default_credentials.json")
    candidates.append(Path.home() / ".config" / "gcloud" / "application_default_credentials.json")
    for path in candidates:
        project = _read_project_from_adc(path)
        if project:
            return project
    return _read_project_from_gcloud()


def _read_project_from_adc(path: Path) -> str:
    try:
        if not path.is_file():
            return ""
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for key in ("quota_project_id", "project_id"):
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return ""


def _read_project_from_gcloud() -> str:
    try:
        completed = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    value = completed.stdout.strip()
    if not value or value == "(unset)":
        return ""
    return value


def load_agent_environment() -> Mapping[str, str]:
    """Load persistent Agent config while preserving process env overrides."""
    config_files = [path for path in (AGENT_CONFIG, USER_CONFIG) if path.is_file()]
    if not config_files:
        return os.environ
    source_lines = "; ".join(f"source {str(path)!r}" for path in config_files)
    command = f"set -a; {source_lines}; env -0"
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=REPO_ROOT,
            text=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
            env=os.environ.copy(),
        )
    except Exception:
        return os.environ
    if completed.returncode != 0:
        return os.environ
    loaded: dict[str, str] = {}
    for item in completed.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        loaded[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return loaded or os.environ
