#!/usr/bin/env bash
# =====================================================================
# AnyChain Benchmark Agent Configuration
# =====================================================================
# This file configures the Agent control plane only.
# Benchmark engine defaults remain in config/user_config.sh.
# =====================================================================

# ----- LLM Provider -----
# fake: deterministic/offline mode. No credentials required.
# vertex_gemini_openai: Gemini on Vertex AI through the OpenAI-compatible API.
# vertex_claude: Claude partner models on Vertex AI.
# openai: OpenAI API.
LLM_PROVIDER="${LLM_PROVIDER:-fake}"

# Model name for the selected provider.
# Examples: fake, gemini-2.5-pro, claude-3-7-sonnet@20250219, gpt-4.1.
LLM_MODEL="${LLM_MODEL:-fake}"

# Google auth mode for Vertex providers.
# adc: local Application Default Credentials.
# attached_service_account: use the VM/GKE attached service account.
# service_account_impersonation: impersonate GOOGLE_SERVICE_ACCOUNT_EMAIL.
# service_account_file: use GOOGLE_APPLICATION_CREDENTIALS JSON file.
GOOGLE_AUTH_MODE="${GOOGLE_AUTH_MODE:-adc}"

# Google Cloud project that contains the Vertex AI endpoint.
# Required only for vertex_gemini_openai or vertex_claude with --use-llm.
GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-}"

# Vertex AI location.
# Examples: us-central1 for Gemini, us-east5 for some Claude partner models.
GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"

# Target service account email for service_account_impersonation.
# This avoids downloading JSON keys in enterprise environments.
GOOGLE_SERVICE_ACCOUNT_EMAIL="${GOOGLE_SERVICE_ACCOUNT_EMAIL:-}"

# Optional local service-account JSON key fallback.
# Prefer ADC, attached service accounts, or impersonation in enterprise usage.
GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"

# OpenAI API key.
# Required only when LLM_PROVIDER=openai.
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ----- Agent Chat Memory -----
# Approximate context window used by deterministic auto-compaction.
AGENT_CONTEXT_WINDOW_TOKENS="${AGENT_CONTEXT_WINDOW_TOKENS:-1000000}"

# Auto-compact when estimated tokens reach context_window * ratio.
AGENT_COMPACT_TRIGGER_RATIO="${AGENT_COMPACT_TRIGGER_RATIO:-0.7}"

# Also compact after this many terminal chat turns.
AGENT_COMPACT_TURN_THRESHOLD="${AGENT_COMPACT_TURN_THRESHOLD:-40}"

# Number of recent raw turns kept after compaction.
AGENT_COMPACT_KEEP_RECENT_TURNS="${AGENT_COMPACT_KEEP_RECENT_TURNS:-8}"

# ----- Enterprise Knowledge Base Integration -----
# disabled: use only repository-local capabilities and docs.
# noop: explicitly use the built-in no-op provider contract.
# custom: load an enterprise adapter in a future integration package.
AGENT_KNOWLEDGE_PROVIDER="${AGENT_KNOWLEDGE_PROVIDER:-disabled}"

# Optional module path for enterprise knowledge provider adapters.
# Example: my_company.anychain_kb:Provider
AGENT_KNOWLEDGE_PROVIDER_MODULE="${AGENT_KNOWLEDGE_PROVIDER_MODULE:-}"

# Optional endpoint for enterprise KB/RAG service adapters.
AGENT_KNOWLEDGE_BASE_URL="${AGENT_KNOWLEDGE_BASE_URL:-}"

# Optional secret reference or token for enterprise KB adapters.
# Do not commit real tokens to git.
AGENT_KNOWLEDGE_AUTH_REF="${AGENT_KNOWLEDGE_AUTH_REF:-}"

# ----- Default Agent Output -----
# Default directory for terminal chat state when no --output-dir is passed.
AGENT_OUTPUT_DIR="${AGENT_OUTPUT_DIR:-.agent/chat}"

export LLM_PROVIDER LLM_MODEL
export GOOGLE_AUTH_MODE GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION GOOGLE_SERVICE_ACCOUNT_EMAIL GOOGLE_APPLICATION_CREDENTIALS
export OPENAI_API_KEY
export AGENT_CONTEXT_WINDOW_TOKENS AGENT_COMPACT_TRIGGER_RATIO AGENT_COMPACT_TURN_THRESHOLD AGENT_COMPACT_KEEP_RECENT_TURNS
export AGENT_KNOWLEDGE_PROVIDER AGENT_KNOWLEDGE_PROVIDER_MODULE AGENT_KNOWLEDGE_BASE_URL AGENT_KNOWLEDGE_AUTH_REF
export AGENT_OUTPUT_DIR
