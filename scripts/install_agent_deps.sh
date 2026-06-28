#!/usr/bin/env bash
# =====================================================================
# AnyChain Benchmark Agent — Agent runtime dependency installer
# =====================================================================
# Installs or audits dependencies needed by the human-facing Agent runtime:
#   1. Google ADK in an isolated Python 3.10+ virtual environment
#   2. Optional Google Cloud CLI for local ADC / impersonation bootstrap
#
# This script does NOT install benchmark engine dependencies. Use:
#   bash scripts/install_deps.sh
#
# Usage:
#   bash scripts/install_agent_deps.sh --check
#   bash scripts/install_agent_deps.sh --yes
#   bash scripts/install_agent_deps.sh --yes --with-gcloud
#   bash scripts/install_agent_deps.sh --yes --no-sudo
#   bash scripts/install_agent_deps.sh --adk-venv .venv-adk
#
# Exit codes:
#   0  — success (or --check passed)
#   1  — install failed
#   2  — unsupported host / missing prerequisites
#   3  — --check found missing dependencies
# =====================================================================

set -euo pipefail

MODE="interactive"   # interactive | yes | check
SKIP_ADK=0
WITH_GCLOUD=0
SKIP_SUDO=0
ADK_VENV=".venv-adk"
PYTHON_BIN=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="/tmp/install_agent_deps_$(date +%Y%m%d_%H%M%S).log"

if [[ -t 1 ]]; then
    C_RED=$'\033[0;31m'
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_BLUE=$'\033[0;34m'
    C_BOLD=$'\033[1m'
    C_RESET=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""; C_RESET=""
fi

log()    { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }
info()   { log "${C_BLUE}[INFO]${C_RESET}  $*"; }
ok()     { log "${C_GREEN}[ OK ]${C_RESET}  $*"; }
warn()   { log "${C_YELLOW}[WARN]${C_RESET}  $*"; }
err()    { log "${C_RED}[FAIL]${C_RESET}  $*" >&2; }
step()   { log ""; log "${C_BOLD}=== $* ===${C_RESET}"; }

usage() {
    sed -n '/^# Usage:/,/^# Exit codes:/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y) MODE="yes" ;;
        --check) MODE="check" ;;
        --skip-adk) SKIP_ADK=1 ;;
        --with-gcloud) WITH_GCLOUD=1 ;;
        --no-sudo) SKIP_SUDO=1 ;;
        --adk-venv)
            shift
            [[ $# -gt 0 ]] || { err "--adk-venv requires a path"; exit 2; }
            ADK_VENV="$1"
            ;;
        --python-bin)
            shift
            [[ $# -gt 0 ]] || { err "--python-bin requires a command/path"; exit 2; }
            PYTHON_BIN="$1"
            ;;
        --help|-h) usage ;;
        *) err "Unknown flag: $1 (try --help)"; exit 2 ;;
    esac
    shift
done

confirm() {
    local prompt="$1"
    case "$MODE" in
        yes)   info "(--yes) auto-confirming: $prompt"; return 0 ;;
        check) return 1 ;;
        *)
            read -r -p "$(printf '%s%s [y/N] %s' "$C_YELLOW" "$prompt" "$C_RESET")" reply
            [[ "$reply" =~ ^[Yy]$ ]]
            ;;
    esac
}

detect_distro() {
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "${ID:-unknown}"
        return
    fi
    if command -v apt-get >/dev/null 2>&1; then echo "debian"; return; fi
    if command -v dnf >/dev/null 2>&1; then echo "fedora"; return; fi
    if command -v yum >/dev/null 2>&1; then echo "rhel"; return; fi
    echo "unknown"
}

select_python() {
    if [[ -n "$PYTHON_BIN" ]]; then
        echo "$PYTHON_BIN"
        return
    fi
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && python_is_310 "$candidate"; then
            command -v "$candidate"
            return
        fi
    done
    echo ""
}

python_is_310() {
    local bin="$1"
    "$bin" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

adk_ready() {
    local venv_dir="$1"
    [[ -x "$venv_dir/bin/python" ]] || return 1
    [[ -x "$venv_dir/bin/adk" ]] || return 1
    "$venv_dir/bin/python" - <<'PY' >/dev/null 2>&1
import google.adk
PY
}

install_adk() {
    local python="$1" venv_dir="$2"
    if [[ ! -x "$venv_dir/bin/python" ]]; then
        if [[ -e "$venv_dir" ]]; then
            warn "Removing incomplete ADK venv at $venv_dir"
            rm -rf "$venv_dir"
        fi
        "$python" -m venv "$venv_dir"
    fi
    if [[ ! -x "$venv_dir/bin/python" ]]; then
        err "Python venv creation failed: $venv_dir/bin/python was not created"
        return 1
    fi
    "$venv_dir/bin/python" -m pip install --upgrade pip
    "$venv_dir/bin/python" -m pip install -r "$REPO_ROOT/requirements-adk.txt"
}

install_gcloud_apt() {
    sudo apt-get update -qq
    sudo apt-get install -y ca-certificates gnupg curl
    if [[ ! -f /usr/share/keyrings/cloud.google.gpg ]]; then
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
            | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    fi
    if [[ ! -f /etc/apt/sources.list.d/google-cloud-sdk.list ]] \
        || ! grep -q "packages.cloud.google.com/apt cloud-sdk main" /etc/apt/sources.list.d/google-cloud-sdk.list; then
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
            | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
    fi
    sudo apt-get update -qq
    sudo CLOUDSDK_SKIP_PY_COMPILATION=1 apt-get install -y google-cloud-cli
}

install_gcloud_rhel() {
    local arch baseurl pkg_manager
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64) baseurl="https://packages.cloud.google.com/yum/repos/cloud-sdk-el9-x86_64" ;;
        aarch64|arm64) baseurl="https://packages.cloud.google.com/yum/repos/cloud-sdk-el9-aarch64" ;;
        *) err "Unsupported architecture for Google Cloud CLI package repo: $arch"; return 2 ;;
    esac
    pkg_manager="dnf"
    command -v dnf >/dev/null 2>&1 || pkg_manager="yum"
    sudo tee /etc/yum.repos.d/google-cloud-sdk.repo >/dev/null <<EOF
[google-cloud-cli]
name=Google Cloud CLI
baseurl=${baseurl}
enabled=1
gpgcheck=1
repo_gpgcheck=0
gpgkey=https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF
    if [[ "$pkg_manager" == "dnf" ]]; then
        sudo dnf install -y libxcrypt-compat || true
        sudo dnf install -y google-cloud-cli
    else
        sudo yum install -y google-cloud-cli
    fi
}

step "Step 1/3 — Check Agent Python and Google ADK"
MISSING=()
ADK_VENV_ABS="$REPO_ROOT/$ADK_VENV"
if [[ "$ADK_VENV" = /* ]]; then
    ADK_VENV_ABS="$ADK_VENV"
fi

if [[ "$SKIP_ADK" == "1" ]]; then
    info "Skipping ADK check/install (--skip-adk)"
else
    PYTHON="$(select_python)"
    if [[ -z "$PYTHON" ]]; then
        warn "Python 3.10+ — MISSING"
        MISSING+=("agent:python3.10+")
    elif adk_ready "$ADK_VENV_ABS"; then
        ok "Google ADK available in $ADK_VENV_ABS"
    else
        warn "Google ADK venv — MISSING or incomplete at $ADK_VENV_ABS"
        MISSING+=("agent:google-adk")
        if [[ "$MODE" != "check" ]]; then
            info "Will create/use isolated venv: $ADK_VENV_ABS"
            info "Will install: $REPO_ROOT/requirements-adk.txt"
            if confirm "Install Google ADK into isolated venv now?"; then
                install_adk "$PYTHON" "$ADK_VENV_ABS" 2>&1 | tee -a "$LOG_FILE"
                if adk_ready "$ADK_VENV_ABS"; then
                    ok "Google ADK installed in $ADK_VENV_ABS"
                else
                    err "Google ADK install did not produce a working adk CLI"
                    exit 1
                fi
            else
                warn "Skipped ADK install (user declined)"
            fi
        fi
    fi
fi

step "Step 2/3 — Check Google Cloud CLI"
if command -v gcloud >/dev/null 2>&1; then
    ok "gcloud available: $(command -v gcloud)"
elif [[ "$WITH_GCLOUD" == "1" ]]; then
    warn "gcloud — MISSING"
    MISSING+=("agent:gcloud")
    if [[ "$MODE" != "check" ]]; then
        if [[ "$SKIP_SUDO" == "1" ]]; then
            warn "--no-sudo: cannot install Google Cloud CLI package repositories"
        else
            DISTRO="$(detect_distro)"
            info "Detected distro: $DISTRO"
            if confirm "Install Google Cloud CLI now using the official package repository?"; then
                case "$DISTRO" in
                    ubuntu|debian) install_gcloud_apt 2>&1 | tee -a "$LOG_FILE" ;;
                    rhel|centos|rocky|almalinux|amzn|fedora) install_gcloud_rhel 2>&1 | tee -a "$LOG_FILE" ;;
                    *)
                        err "Unsupported distro for automated gcloud install: $DISTRO"
                        err "Install Google Cloud CLI manually or use attached_service_account/service_account_file auth."
                        exit 2
                        ;;
                esac
                command -v gcloud >/dev/null 2>&1 || { err "gcloud still not found after install"; exit 1; }
                ok "gcloud installed: $(command -v gcloud)"
            else
                warn "Skipped gcloud install (user declined)"
            fi
        fi
    fi
else
    info "gcloud not found; skipping install because --with-gcloud was not provided"
fi

step "Step 3/3 — Summary"
STILL_MISSING=()
if [[ "$SKIP_ADK" != "1" ]] && ! adk_ready "$ADK_VENV_ABS"; then
    STILL_MISSING+=("agent:google-adk")
fi
if [[ "$WITH_GCLOUD" == "1" ]] && ! command -v gcloud >/dev/null 2>&1; then
    STILL_MISSING+=("agent:gcloud")
fi

if [[ ${#STILL_MISSING[@]} -eq 0 ]]; then
    ok "Agent dependencies satisfied."
    log "Activate ADK venv with:"
    log "    source $ADK_VENV_ABS/bin/activate"
    log "Then run:"
    log "    ./bin/anychain-agent"
    log "Log: $LOG_FILE"
    exit 0
fi

if [[ "$MODE" == "check" ]]; then
    warn "--check: ${#STILL_MISSING[@]} Agent dependency/dependencies missing:"
    for dep in "${STILL_MISSING[@]}"; do warn "    - $dep"; done
    log "Log: $LOG_FILE"
    exit 3
fi

err "${#STILL_MISSING[@]} Agent dependency/dependencies still missing:"
for dep in "${STILL_MISSING[@]}"; do err "    - $dep"; done
log "Log: $LOG_FILE"
exit 1
