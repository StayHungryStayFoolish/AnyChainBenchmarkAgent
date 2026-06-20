"""Google authentication helpers for Vertex-backed Agent providers."""

from __future__ import annotations

from dataclasses import dataclass

from llm.config import LLMConfig


CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass(frozen=True)
class GoogleCredentialPlan:
    mode: str
    project: str
    location: str
    service_account_email: str = ""
    credentials_file_configured: bool = False

    def safe_dict(self) -> dict[str, str | bool]:
        return {
            "mode": self.mode,
            "project": self.project,
            "location": self.location,
            "service_account_email": self.service_account_email,
            "credentials_file_configured": self.credentials_file_configured,
        }


def credential_plan(config: LLMConfig) -> GoogleCredentialPlan:
    return GoogleCredentialPlan(
        mode=config.auth_mode,
        project=config.google_project,
        location=config.google_location,
        service_account_email=config.google_service_account_email,
        credentials_file_configured=bool(config.google_application_credentials),
    )


def get_google_access_token(config: LLMConfig) -> str:
    """Return an OAuth access token using ADC, impersonation, or key-file auth."""
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "google-auth is required for Vertex LLM providers. Install it in an isolated "
            "environment or use the project Docker image."
        ) from exc

    scopes = [CLOUD_PLATFORM_SCOPE]
    if config.auth_mode == "service_account_file":
        try:
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("google-auth service account support is unavailable") from exc
        credentials = service_account.Credentials.from_service_account_file(
            config.google_application_credentials,
            scopes=scopes,
        )
    else:
        credentials, _ = google.auth.default(scopes=scopes)
        if config.auth_mode == "service_account_impersonation":
            try:
                from google.auth import impersonated_credentials
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise RuntimeError("google-auth impersonated credentials support is unavailable") from exc
            credentials = impersonated_credentials.Credentials(
                source_credentials=credentials,
                target_principal=config.google_service_account_email,
                target_scopes=scopes,
                lifetime=3600,
            )

    credentials.refresh(Request())
    return credentials.token
