"""LLM provider and authentication configuration."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SUPPORTED_LLM_PROVIDERS = {
    "fake",
    "gemini",
    "claude",
    "openai",
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
    gemini_api_key_present: bool = False
    anthropic_api_key_present: bool = False
    openai_api_key_present: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            errors.append(f"unsupported LLM_PROVIDER: {self.provider}")
        if self.auth_mode not in SUPPORTED_LLM_AUTH_MODES:
            errors.append(f"unsupported LLM_AUTH_MODE: {self.auth_mode}")
        if self.provider == "fake":
            return errors
        if self.provider == "openai":
            if self.auth_mode != "api_key":
                errors.append("LLM_AUTH_MODE=api_key is required for LLM_PROVIDER=openai")
            if not self.openai_api_key_present:
                errors.append("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
            return errors
        if self.auth_mode == "api_key":
            if self.provider == "gemini" and not self.gemini_api_key_present:
                errors.append("GEMINI_API_KEY or GOOGLE_API_KEY is required for LLM_PROVIDER=gemini with LLM_AUTH_MODE=api_key")
            if self.provider == "claude" and not self.anthropic_api_key_present:
                errors.append("ANTHROPIC_API_KEY is required for LLM_PROVIDER=claude with LLM_AUTH_MODE=api_key")
            return errors
        if self.provider in {"gemini", "claude"}:
            if not self.google_project:
                errors.append("GOOGLE_CLOUD_PROJECT is required for Gemini/Claude on Vertex")
            if not self.google_location:
                errors.append("GOOGLE_CLOUD_LOCATION is required for Gemini/Claude on Vertex")
            if self.auth_mode == "service_account_impersonation" and not self.google_service_account_email:
                errors.append("GOOGLE_SERVICE_ACCOUNT_EMAIL is required for service_account_impersonation")
            if self.auth_mode == "service_account_file" and not self.google_application_credentials:
                errors.append("GOOGLE_APPLICATION_CREDENTIALS is required for service_account_file")
        return errors

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
            "validation_errors": self.validate(),
        }


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    source = env or load_agent_environment()
    provider = source.get("LLM_PROVIDER", "fake").strip().lower()
    default_model = "fake"
    if provider == "gemini":
        default_model = "gemini-3.1-pro"
    elif provider == "claude":
        default_model = "claude-opus-4-8"
    elif provider == "openai":
        default_model = "gpt-5.5"
    default_auth_mode = "api_key"
    if provider in {"gemini", "claude"}:
        default_auth_mode = source.get("LLM_AUTH_MODE", "api_key").strip().lower()
    return LLMConfig(
        provider=provider,
        model=source.get("LLM_MODEL", default_model).strip(),
        auth_mode=source.get("LLM_AUTH_MODE", default_auth_mode).strip().lower(),
        google_project=source.get("GOOGLE_CLOUD_PROJECT", "").strip(),
        google_location=source.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip(),
        google_service_account_email=source.get("GOOGLE_SERVICE_ACCOUNT_EMAIL", "").strip(),
        google_application_credentials=source.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
        gemini_api_key=source.get("GEMINI_API_KEY", source.get("GOOGLE_API_KEY", "")).strip(),
        anthropic_api_key=source.get("ANTHROPIC_API_KEY", "").strip(),
        openai_api_key=source.get("OPENAI_API_KEY", "").strip(),
        gemini_api_key_present=bool(source.get("GEMINI_API_KEY", "") or source.get("GOOGLE_API_KEY", "")),
        anthropic_api_key_present=bool(source.get("ANTHROPIC_API_KEY", "")),
        openai_api_key_present=bool(source.get("OPENAI_API_KEY", "")),
    )


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
