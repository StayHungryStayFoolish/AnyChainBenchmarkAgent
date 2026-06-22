#!/usr/bin/env bash
# =====================================================================
# blockchain-node-benchmark — dependency installer
# =====================================================================
# Installs everything the framework needs to run on a bare VM/EC2/GCE:
#   1. System packages  (sysstat bc jq net-tools procps python3-pip python-venv-support)
#   2. Python packages  (from requirements.txt)
#   3. vegeta           (HTTP load generator, pinned version)
#
# What this script does NOT do:
#   - Does NOT edit config/* (you configure your RPC endpoint yourself)
#   - Does NOT run any benchmark
#   - Does NOT touch .git or any source files
#   - Does NOT modify system Python packages by default
#   - Does NOT install Agent runtime dependencies
#
# For Agent runtime dependencies, including Google ADK and optional gcloud:
#   bash scripts/install_agent_deps.sh --yes
#   bash scripts/install_agent_deps.sh --yes --with-gcloud
#
# After this script finishes, you still run:
#   ./blockchain_node_benchmark.sh --quick
#
# Usage:
#   bash scripts/install_deps.sh              # interactive (asks before each step)
#   bash scripts/install_deps.sh --yes        # non-interactive (CI / Docker)
#   bash scripts/install_deps.sh --check      # audit-only, no changes made
#   bash scripts/install_deps.sh --no-vegeta  # skip vegeta (e.g. K8s collector pod)
#   bash scripts/install_deps.sh --no-sudo    # skip steps that require sudo
#   bash scripts/install_deps.sh --system-python  # allow pip --break-system-packages when required
#   bash scripts/install_deps.sh --help
#
# Supported distros: Ubuntu, Debian, RHEL, CentOS, Rocky, AlmaLinux,
#                    Amazon Linux, Fedora, Alpine
#
# Exit codes:
#   0  — success (or --check passed with all deps present)
#   1  — failure during install
#   2  — unsupported distro / missing prerequisites
#   3  — --check found missing deps (non-fatal, informational)
# =====================================================================

set -euo pipefail

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
readonly VEGETA_VERSION="v12.12.0"
readonly VEGETA_BIN_DIR="${HOME}/bin"
# Official SHA256 from https://github.com/tsenart/vegeta/releases/tag/v12.12.0
# (linux amd64 tarball). Update both VEGETA_VERSION and the sums below in lockstep.
readonly VEGETA_SHA256_AMD64="e7ce26c8fa4b9e1a3668aa7f82a4d77fca6a6d955f8dd5e843816115cc568450"
readonly VEGETA_SHA256_ARM64="b531e93c02727bc84938f3762910fcd3b676cc75d73715960f3cde9c02f5293c"

readonly SYSTEM_PACKAGES=(sysstat bc jq net-tools procps python3-pip python-venv-support)

readonly LOG_FILE="/tmp/install_deps_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------
# Globals (modified by flag parser)
# ---------------------------------------------------------------------
MODE="interactive"   # interactive | yes | check
SKIP_VEGETA=0
SKIP_SUDO=0
ALLOW_SYSTEM_PYTHON=0

# ---------------------------------------------------------------------
# Colors (auto-disabled if not a TTY)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------
usage() {
    sed -n '/^# Usage:/,/^# Exit codes:/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y)        MODE="yes" ;;
        --check)         MODE="check" ;;
        --no-vegeta)     SKIP_VEGETA=1 ;;
        --no-sudo)       SKIP_SUDO=1 ;;
        --system-python) ALLOW_SYSTEM_PYTHON=1 ;;
        --help|-h)       usage ;;
        *) err "Unknown flag: $1 (try --help)"; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------
# Prompt helper (respects --yes and --check)
# ---------------------------------------------------------------------
confirm() {
    local prompt="$1"
    case "$MODE" in
        yes)   info "(--yes) auto-confirming: $prompt"; return 0 ;;
        check) return 1 ;;  # check mode never installs
        *)
            read -r -p "$(printf '%s%s [y/N] %s' "$C_YELLOW" "$prompt" "$C_RESET")" reply
            [[ "$reply" =~ ^[Yy]$ ]]
            ;;
    esac
}

# ---------------------------------------------------------------------
# Distro detection
# ---------------------------------------------------------------------
detect_distro() {
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "${ID:-unknown}"
        return
    fi
    if command -v apt-get >/dev/null 2>&1; then echo "debian"; return; fi
    if command -v dnf     >/dev/null 2>&1; then echo "fedora"; return; fi
    if command -v yum     >/dev/null 2>&1; then echo "rhel";   return; fi
    if command -v apk     >/dev/null 2>&1; then echo "alpine"; return; fi
    echo "unknown"
}

# Return the apt/yum/dnf/apk package name for a given canonical name on the
# given distro. Most names are identical across distros; the table below
# only encodes the exceptions.
distro_pkg_name() {
    local distro="$1" canonical="$2"
    case "$distro:$canonical" in
        alpine:net-tools)  echo "net-tools" ;;
        alpine:procps)     echo "procps" ;;
        alpine:sysstat)    echo "sysstat" ;;
        alpine:python3-pip) echo "py3-pip" ;;
        ubuntu:python-venv-support|debian:python-venv-support) echo "python3-venv" ;;
        alpine:python-venv-support) echo "py3-virtualenv" ;;
        rhel:python-venv-support|centos:python-venv-support|rocky:python-venv-support|almalinux:python-venv-support|amzn:python-venv-support|fedora:python-venv-support)
            echo "python3-virtualenv" ;;
        # RHEL family ships net-tools under same name; procps-ng instead of procps
        rhel:procps|centos:procps|rocky:procps|almalinux:procps|amzn:procps|fedora:procps)
            echo "procps-ng" ;;
        *)
            echo "$canonical" ;;
    esac
}

pkg_manager_install_cmd() {
    local distro="$1"
    case "$distro" in
        ubuntu|debian) echo "apt-get install -y" ;;
        rhel|centos|rocky|almalinux|amzn) command -v dnf >/dev/null 2>&1 && echo "dnf install -y" || echo "yum install -y" ;;
        fedora)        echo "dnf install -y" ;;
        alpine)        echo "apk add --no-cache" ;;
        *)             echo "" ;;
    esac
}

pkg_manager_refresh_cmd() {
    local distro="$1"
    case "$distro" in
        ubuntu|debian) echo "apt-get update -qq" ;;
        rhel|centos|rocky|almalinux|amzn) command -v dnf >/dev/null 2>&1 && echo "dnf makecache -q" || echo "yum makecache -q" ;;
        fedora)        echo "dnf makecache -q" ;;
        alpine)        echo "apk update -q" ;;
        *)             echo "" ;;
    esac
}

# ---------------------------------------------------------------------
# Step 1: distro detection
# ---------------------------------------------------------------------
step "Step 1/4 — Detect operating system"
DISTRO="$(detect_distro)"
INSTALL_CMD="$(pkg_manager_install_cmd "$DISTRO")"
REFRESH_CMD="$(pkg_manager_refresh_cmd "$DISTRO")"

if [[ -z "$INSTALL_CMD" ]]; then
    err "Unsupported distro: $DISTRO"
    err "Manual install required. See README Quick Start section."
    exit 2
fi
ok "Detected: ${C_BOLD}${DISTRO}${C_RESET}  (package manager: ${INSTALL_CMD%% *})"

# ---------------------------------------------------------------------
# Step 2: system packages
# ---------------------------------------------------------------------
step "Step 2/4 — Check system packages"

declare -a MISSING_PKGS=()
declare -A PKG_BIN_MAP=(
    [sysstat]="iostat"
    [bc]="bc"
    [jq]="jq"
    [net-tools]="netstat"
    [procps]="ps"
    [python3-pip]="pip3"
    [python-venv-support]="python3"
)

venv_support_available() {
    python3 -m venv --help >/dev/null 2>&1 || python3 -m virtualenv --help >/dev/null 2>&1
}

create_project_venv() {
    local venv_dir="$1"
    if python3 -m venv --help >/dev/null 2>&1; then
        python3 -m venv "$venv_dir"
    elif python3 -m virtualenv --help >/dev/null 2>&1; then
        python3 -m virtualenv "$venv_dir"
    else
        return 1
    fi
}

for canonical in "${SYSTEM_PACKAGES[@]}"; do
    bin="${PKG_BIN_MAP[$canonical]:-$canonical}"
    if [[ "$canonical" == "python-venv-support" ]]; then
        if venv_support_available; then
            ok "  $canonical (python3 -m venv or virtualenv) — installed"
        else
            warn "  $canonical (python3 -m venv or virtualenv) — MISSING"
            MISSING_PKGS+=("$(distro_pkg_name "$DISTRO" "$canonical")")
        fi
    elif command -v "$bin" >/dev/null 2>&1; then
        ok "  $canonical (provides $bin) — installed"
    else
        warn "  $canonical (provides $bin) — MISSING"
        MISSING_PKGS+=("$(distro_pkg_name "$DISTRO" "$canonical")")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    if [[ "$MODE" == "check" ]]; then
        warn "  --check: ${#MISSING_PKGS[@]} system package(s) missing: ${MISSING_PKGS[*]}"
    elif [[ "$SKIP_SUDO" == "1" ]]; then
        warn "  --no-sudo: skipping system package install"
        warn "  You must manually run: sudo $INSTALL_CMD ${MISSING_PKGS[*]}"
    else
        info "Will run (requires sudo):"
        info "    sudo $REFRESH_CMD"
        info "    sudo $INSTALL_CMD ${MISSING_PKGS[*]}"
        if confirm "Install missing system packages now?"; then
            # shellcheck disable=SC2086
            sudo $REFRESH_CMD 2>&1 | tee -a "$LOG_FILE"
            # shellcheck disable=SC2086
            sudo $INSTALL_CMD "${MISSING_PKGS[@]}" 2>&1 | tee -a "$LOG_FILE"
            ok "System packages installed"
        else
            warn "Skipped system package install (user declined)"
        fi
    fi
else
    ok "All system packages present"
fi

# ---------------------------------------------------------------------
# Step 3: Python packages from requirements.txt
# ---------------------------------------------------------------------
step "Step 3/4 — Check Python packages"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${REPO_ROOT}/requirements.txt"

if [[ ! -r "$REQ_FILE" ]]; then
    err "requirements.txt not found at: $REQ_FILE"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found. Install python3 first (e.g. sudo $INSTALL_CMD python3)"
    exit 2
fi

PYTHON_CHECK_BIN="python3"
PROJECT_VENV_DIR="${REPO_ROOT}/.venv"
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_CHECK_BIN="${VIRTUAL_ENV}/bin/python"
    info "Python checks will use active virtualenv: $VIRTUAL_ENV"
elif [[ -x "${PROJECT_VENV_DIR}/bin/python" ]]; then
    PYTHON_CHECK_BIN="${PROJECT_VENV_DIR}/bin/python"
    info "Python checks will use project virtualenv: $PROJECT_VENV_DIR"
fi

# Map distribution name (in requirements.txt) → import name (for `import X` check).
# Most are identical; the table below encodes the exceptions.
py_import_name() {
    case "$1" in
        scikit-learn) echo "sklearn" ;;
        *)            echo "$1" ;;
    esac
}

declare -a MISSING_PY=()
# Parse requirements.txt: strip comments, blank lines, version specifiers.
# Robust against `pkg>=1.0`, `pkg==1.0,<2.0`, `pkg ; python_version>='3.8'`.
while IFS= read -r line; do
    pkg="${line%%[<>=;!~ ]*}"   # cut at first version/marker char
    pkg="${pkg// /}"            # strip whitespace
    [[ -z "$pkg" ]] && continue
    [[ "$pkg" == \#* ]] && continue
    import_name="$(py_import_name "$pkg")"
    if "$PYTHON_CHECK_BIN" -c "import $import_name" >/dev/null 2>&1; then
        ok "  $pkg (import $import_name) — installed"
    else
        warn "  $pkg (import $import_name) — MISSING"
        MISSING_PY+=("$pkg")
    fi
done < <(grep -v '^[[:space:]]*#' "$REQ_FILE" | grep -v '^[[:space:]]*$')

if [[ ${#MISSING_PY[@]} -gt 0 ]]; then
    if [[ "$MODE" == "check" ]]; then
        warn "  --check: ${#MISSING_PY[@]} Python package(s) missing: ${MISSING_PY[*]}"
    else
        # Detect PEP 668 (Debian 12+, Ubuntu 23.04+ mark system Python as externally-managed).
        # Note: [[ -f /path/glob* ]] does NOT expand globs inside [[ ]], so we use
        # `compgen -G` or a `ls` fallback to detect the marker file.
        PIP_ARGS=("install" "-r" "$REQ_FILE")
        PIP_CMD="pip3"
        SKIP_PY_INSTALL=0
        in_venv=0
        if [[ -n "${VIRTUAL_ENV:-}" ]]; then
            in_venv=1
        elif ! "$PYTHON_CHECK_BIN" -c "import sys; sys.exit(0 if sys.prefix == sys.base_prefix else 1)" 2>/dev/null; then
            # python3 reports we're inside a venv even though $VIRTUAL_ENV is unset
            in_venv=1
        fi
        pep668=0
        if compgen -G "/usr/lib/python3*/EXTERNALLY-MANAGED" >/dev/null 2>&1; then
            pep668=1
        fi
        if [[ "$in_venv" == "1" ]]; then
            info "Active virtualenv detected (${VIRTUAL_ENV:-auto-detected}). Will install into it."
            PIP_CMD="${PYTHON_CHECK_BIN}"
            PIP_ARGS=("-m" "pip" "install" "-r" "$REQ_FILE")
        elif [[ "$pep668" == "1" ]]; then
            if [[ "$ALLOW_SYSTEM_PYTHON" == "1" ]]; then
                warn "Detected PEP 668 (externally-managed Python)."
                warn "--system-python was provided, so pip will use --user --break-system-packages."
                PIP_ARGS=("install" "--user" "--break-system-packages" "-r" "$REQ_FILE")
            else
                VENV_DIR="$PROJECT_VENV_DIR"
                info "Detected PEP 668 (externally-managed Python)."
                info "To avoid changing system Python packages, the installer will create/use:"
                info "    $VENV_DIR"
                info "After installation, activate it with:"
                info "    source .venv/bin/activate"
                if [[ ! -d "$VENV_DIR" ]]; then
                    if confirm "Create project virtualenv at $VENV_DIR?"; then
                        create_project_venv "$VENV_DIR" 2>&1 | tee -a "$LOG_FILE"
                    else
                        warn "Skipped Python package install (user declined virtualenv creation)"
                        SKIP_PY_INSTALL=1
                    fi
                fi
                PIP_CMD="${VENV_DIR}/bin/pip"
                PYTHON_CHECK_BIN="${VENV_DIR}/bin/python"
            fi
        else
            info "No active virtualenv detected. Will install to user site with --user."
            PIP_ARGS=("install" "--user" "-r" "$REQ_FILE")
        fi
        if [[ "$SKIP_PY_INSTALL" == "0" ]]; then
            info "Will run: $PIP_CMD ${PIP_ARGS[*]}"
            if confirm "Install missing Python packages now?"; then
                "$PIP_CMD" "${PIP_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
                ok "Python packages installed"
            else
                warn "Skipped Python package install (user declined)"
            fi
        fi
    fi
else
    ok "All Python packages present"
fi

# ---------------------------------------------------------------------
# Step 4: vegeta
# ---------------------------------------------------------------------
step "Step 4/4 — Check vegeta ${VEGETA_VERSION}"

if [[ "$SKIP_VEGETA" == "1" ]]; then
    info "Skipping vegeta install (--no-vegeta)"
elif command -v vegeta >/dev/null 2>&1; then
    INSTALLED_VER="$(vegeta -version 2>&1 | head -1 || true)"
    ok "vegeta already installed (on PATH): $INSTALLED_VER"
    info "  (script does not upgrade existing binaries — remove first if you want a different version)"
elif [[ -x "$VEGETA_BIN_DIR/vegeta" ]]; then
    INSTALLED_VER="$("$VEGETA_BIN_DIR/vegeta" -version 2>&1 | head -1 || true)"
    ok "vegeta already installed at $VEGETA_BIN_DIR/vegeta: $INSTALLED_VER"
    if [[ ":$PATH:" != *":$VEGETA_BIN_DIR:"* ]]; then
        warn "  $VEGETA_BIN_DIR is NOT on your PATH. Add to ~/.bashrc:"
        warn "      export PATH=\"\$HOME/bin:\$PATH\""
    fi
else
    case "$(uname -m)" in
        x86_64|amd64) ARCH="amd64"; SHA256="$VEGETA_SHA256_AMD64" ;;
        aarch64|arm64) ARCH="arm64"; SHA256="$VEGETA_SHA256_ARM64" ;;
        *) err "Unsupported architecture: $(uname -m). vegeta releases only ship amd64/arm64."; exit 2 ;;
    esac

    URL="https://github.com/tsenart/vegeta/releases/download/${VEGETA_VERSION}/vegeta_${VEGETA_VERSION#v}_linux_${ARCH}.tar.gz"
    info "Will download:  $URL"
    info "Will verify:    sha256 == $SHA256"
    info "Will install to: ${VEGETA_BIN_DIR}/vegeta"

    if [[ "$MODE" == "check" ]]; then
        warn "  --check: vegeta missing"
    elif confirm "Download and install vegeta ${VEGETA_VERSION} to ${VEGETA_BIN_DIR}?"; then
        TMPDIR="$(mktemp -d)"
        trap 'rm -rf "$TMPDIR"' EXIT
        info "Downloading..."
        curl -fsSL -o "$TMPDIR/vegeta.tar.gz" "$URL"
        info "Verifying sha256..."
        ACTUAL_SHA="$(sha256sum "$TMPDIR/vegeta.tar.gz" | awk '{print $1}')"
        if [[ "$ACTUAL_SHA" != "$SHA256" ]]; then
            err "SHA256 mismatch!"
            err "  expected: $SHA256"
            err "  actual:   $ACTUAL_SHA"
            err "ABORTING — possible tampering or wrong version pinned."
            exit 1
        fi
        ok "SHA256 verified"
        tar -xzf "$TMPDIR/vegeta.tar.gz" -C "$TMPDIR"
        mkdir -p "$VEGETA_BIN_DIR"
        mv "$TMPDIR/vegeta" "$VEGETA_BIN_DIR/vegeta"
        chmod +x "$VEGETA_BIN_DIR/vegeta"
        ok "vegeta installed to $VEGETA_BIN_DIR/vegeta"

        # PATH hint
        if [[ ":$PATH:" != *":$VEGETA_BIN_DIR:"* ]]; then
            warn "  $VEGETA_BIN_DIR is NOT on your PATH."
            warn "  Add this to your ~/.bashrc (or ~/.zshrc):"
            warn "      export PATH=\"\$HOME/bin:\$PATH\""
            warn "  Then run: source ~/.bashrc"
        fi
    else
        warn "Skipped vegeta install (user declined)"
    fi
fi

# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------
step "Summary"

# Re-check everything for the final report
declare -a STILL_MISSING=()

for canonical in "${SYSTEM_PACKAGES[@]}"; do
    bin="${PKG_BIN_MAP[$canonical]:-$canonical}"
    if [[ "$canonical" == "python-venv-support" ]]; then
        venv_support_available || STILL_MISSING+=("system:$canonical")
    else
        command -v "$bin" >/dev/null 2>&1 || STILL_MISSING+=("system:$canonical")
    fi
done

while IFS= read -r line; do
    pkg="${line%%[<>=;!~ ]*}"; pkg="${pkg// /}"
    [[ -z "$pkg" || "$pkg" == \#* ]] && continue
    import_name="$(py_import_name "$pkg")"
    "$PYTHON_CHECK_BIN" -c "import $import_name" >/dev/null 2>&1 || STILL_MISSING+=("python:$pkg")
done < <(grep -v '^[[:space:]]*#' "$REQ_FILE" | grep -v '^[[:space:]]*$')

if [[ "$SKIP_VEGETA" != "1" ]]; then
    # Check both PATH (existing install) AND ~/bin (this script's install location).
    # We don't auto-modify PATH (would require touching ~/.bashrc), but a freshly
    # installed binary at $VEGETA_BIN_DIR/vegeta still counts as "installed".
    if ! command -v vegeta >/dev/null 2>&1 && [[ ! -x "$VEGETA_BIN_DIR/vegeta" ]]; then
        STILL_MISSING+=("binary:vegeta")
    fi
fi

log ""
if [[ ${#STILL_MISSING[@]} -eq 0 ]]; then
    ok "All dependencies satisfied. You can now run:"
    log "    ./blockchain_node_benchmark.sh --quick"
    log ""
    log "Log: $LOG_FILE"
    exit 0
else
    if [[ "$MODE" == "check" ]]; then
        warn "--check: ${#STILL_MISSING[@]} dep(s) missing:"
        for m in "${STILL_MISSING[@]}"; do warn "    - $m"; done
        log ""
        log "Run without --check to install them."
        log "Log: $LOG_FILE"
        exit 3
    else
        err "${#STILL_MISSING[@]} dep(s) still missing after install:"
        for m in "${STILL_MISSING[@]}"; do err "    - $m"; done
        log ""
        log "Log: $LOG_FILE"
        exit 1
    fi
fi
