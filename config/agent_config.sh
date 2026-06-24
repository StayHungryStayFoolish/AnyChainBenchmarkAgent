#!/usr/bin/env bash
# =====================================================================
# AnyChain Benchmark Agent Configuration
# =====================================================================
# This file configures the Agent control plane only.
# Benchmark engine defaults remain in config/user_config.sh.
# =====================================================================

# ----- LLM Provider -----
# Required for AI-assisted planning:
#   1. Set LLM_PROVIDER to one real provider.
#   2. Set LLM_MODEL to a model available in that provider.
#   3. Configure that provider's auth variables below.
#
# gemini: Gemini API key or Gemini on Vertex AI.
# claude: Anthropic API key or Claude partner models on Vertex AI.
# openai: OpenAI API.
LLM_PROVIDER="${LLM_PROVIDER:-gemini}"

# Model name for the selected provider.
# Examples: gemini-3.1-pro-preview, claude-opus-4-8, gpt-5.5.
LLM_MODEL="${LLM_MODEL:-gemini-3.1-pro-preview}"

# Authentication mode for the selected provider.
# api_key: GEMINI_API_KEY/GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.
# google_adc: local Application Default Credentials for Vertex AI.
# attached_service_account: use the VM/GKE attached service account.
# service_account_impersonation: impersonate GOOGLE_SERVICE_ACCOUNT_EMAIL.
# service_account_file: use GOOGLE_APPLICATION_CREDENTIALS JSON file.
LLM_AUTH_MODE="${LLM_AUTH_MODE:-google_adc}"

# Required when LLM_PROVIDER is gemini or claude and LLM_AUTH_MODE is not api_key.
# Google Cloud project that contains the Vertex AI endpoint.
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-}"

# Required when LLM_PROVIDER is gemini or claude and LLM_AUTH_MODE is not api_key.
# Vertex AI location/region.
# Examples: global for Gemini, global for some Claude partner models.
GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# Required only when LLM_AUTH_MODE=service_account_impersonation.
# Target service account email for service_account_impersonation.
# This avoids downloading JSON keys in enterprise environments.
GOOGLE_SERVICE_ACCOUNT_EMAIL="${GOOGLE_SERVICE_ACCOUNT_EMAIL:-}"

# Required only when LLM_AUTH_MODE=service_account_file.
# Local service-account JSON key file.
# Prefer ADC, attached service accounts, or impersonation in enterprise usage.
GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"

# Required when LLM_PROVIDER=gemini and LLM_AUTH_MODE=api_key.
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

# Required when LLM_PROVIDER=claude and LLM_AUTH_MODE=api_key.
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Required only when LLM_PROVIDER=openai.
# OpenAI API key.
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ----- Enterprise Knowledge Base Integration -----
# disabled: use only repository-local capabilities and docs.
# noop: explicitly use the built-in no-op provider contract.
# http: call a generic enterprise KB/RAG HTTP service.
# custom: load an enterprise adapter module from AGENT_KNOWLEDGE_PROVIDER_MODULE.
AGENT_KNOWLEDGE_PROVIDER="${AGENT_KNOWLEDGE_PROVIDER:-disabled}"

# Optional module path for enterprise knowledge provider adapters.
# Example: my_company.anychain_kb:Provider
AGENT_KNOWLEDGE_PROVIDER_MODULE="${AGENT_KNOWLEDGE_PROVIDER_MODULE:-}"

# Required when AGENT_KNOWLEDGE_PROVIDER=http.
# Expected base URL for the generic KB contract.
AGENT_KNOWLEDGE_BASE_URL="${AGENT_KNOWLEDGE_BASE_URL:-}"

# Optional secret reference or token for enterprise KB adapters.
# Do not commit real tokens to git.
AGENT_KNOWLEDGE_AUTH_REF="${AGENT_KNOWLEDGE_AUTH_REF:-}"

# ----- Optional Job Notification -----
# Disabled by default. When set, the Agent posts job status JSON to this URL
# when the job status is listed in AGENT_NOTIFY_ON.
AGENT_NOTIFY_WEBHOOK_URL="${AGENT_NOTIFY_WEBHOOK_URL:-}"
AGENT_NOTIFY_ON="${AGENT_NOTIFY_ON:-completed,failed}"

export LLM_PROVIDER LLM_MODEL LLM_AUTH_MODE
export GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION GOOGLE_SERVICE_ACCOUNT_EMAIL GOOGLE_APPLICATION_CREDENTIALS
export GEMINI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY
export AGENT_KNOWLEDGE_PROVIDER AGENT_KNOWLEDGE_PROVIDER_MODULE AGENT_KNOWLEDGE_BASE_URL AGENT_KNOWLEDGE_AUTH_REF
export AGENT_NOTIFY_WEBHOOK_URL AGENT_NOTIFY_ON
