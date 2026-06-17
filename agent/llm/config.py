"""LLM provider and authentication configuration."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SUPPORTED_LLM_PROVIDERS = {
    "fake",
    "openai",
    "vertex_gemini_openai",
    "vertex_claude",
}

SUPPORTED_GOOGLE_AUTH_MODES = {
    "adc",
    "attached_service_account",
    "service_account_impersonation",
    "service_account_file",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_CONFIG = REPO_ROOT / "config" / "user_config.sh"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    google_project: str = ""
    google_location: str = "us-central1"
    google_auth_mode: str = "adc"
    google_service_account_email: str = ""
    google_application_credentials: str = ""
    openai_api_key_present: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            errors.append(f"unsupported LLM_PROVIDER: {self.provider}")
        if self.provider.startswith("vertex_"):
            if not self.google_project:
                errors.append("GOOGLE_CLOUD_PROJECT is required for Vertex providers")
            if not self.google_location:
                errors.append("GOOGLE_CLOUD_LOCATION is required for Vertex providers")
            if self.google_auth_mode not in SUPPORTED_GOOGLE_AUTH_MODES:
                errors.append(f"unsupported GOOGLE_AUTH_MODE: {self.google_auth_mode}")
            if self.google_auth_mode == "service_account_impersonation" and not self.google_service_account_email:
                errors.append("GOOGLE_SERVICE_ACCOUNT_EMAIL is required for service_account_impersonation")
            if self.google_auth_mode == "service_account_file" and not self.google_application_credentials:
                errors.append("GOOGLE_APPLICATION_CREDENTIALS is required for service_account_file")
        if self.provider == "openai" and not self.openai_api_key_present:
            errors.append("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        return errors

    def safe_dict(self) -> dict[str, str | bool | list[str]]:
        return {
            "provider": self.provider,
            "model": self.model,
            "google_project": self.google_project,
            "google_location": self.google_location,
            "google_auth_mode": self.google_auth_mode,
            "google_service_account_email": self.google_service_account_email,
            "google_application_credentials_configured": bool(self.google_application_credentials),
            "openai_api_key_configured": self.openai_api_key_present,
            "validation_errors": self.validate(),
        }


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    source = env or _load_user_config_env()
    provider = source.get("LLM_PROVIDER", "fake").strip().lower()
    default_model = "fake"
    if provider == "vertex_gemini_openai":
        default_model = "gemini-2.5-pro"
    elif provider == "vertex_claude":
        default_model = "claude-3-7-sonnet@20250219"
    elif provider == "openai":
        default_model = "gpt-4.1"
    return LLMConfig(
        provider=provider,
        model=source.get("LLM_MODEL", default_model).strip(),
        google_project=source.get("GOOGLE_CLOUD_PROJECT", "").strip(),
        google_location=source.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip(),
        google_auth_mode=source.get("GOOGLE_AUTH_MODE", "adc").strip().lower(),
        google_service_account_email=source.get("GOOGLE_SERVICE_ACCOUNT_EMAIL", "").strip(),
        google_application_credentials=source.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
        openai_api_key_present=bool(source.get("OPENAI_API_KEY", "")),
    )


def _load_user_config_env() -> Mapping[str, str]:
    """Load exported config/user_config.sh values while preserving process env overrides."""
    if not USER_CONFIG.is_file():
        return os.environ
    command = f"set -a; source {str(USER_CONFIG)!r}; env -0"
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
