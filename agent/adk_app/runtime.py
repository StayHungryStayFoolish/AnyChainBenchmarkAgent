"""Development bridge to the official Google ADK CLI.

This module is not the product terminal entrypoint. It remains available for
developer diagnostics that explicitly need ``adk run`` behavior.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ADK_AGENT_DIR = REPO_ROOT / "agent" / "adk_app"


def run_adk_cli(argv: list[str] | None = None) -> int:
    """Run the official ADK CLI for the AnyChain agent package."""
    args = _parse_args(argv)
    adk_bin = str(Path(args.adk_bin)) if Path(args.adk_bin).is_file() else shutil.which(args.adk_bin)
    if not adk_bin:
        _print_missing_adk(args.adk_bin)
        return 2

    command = [adk_bin, "run", str(args.agent_dir)]
    if args.adk_arg:
        command.extend(args.adk_arg)

    input_text = None
    if args.prompt:
        input_text = args.prompt.rstrip() + "\nexit\n"

    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            input=input_text,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        _print_missing_adk(args.adk_bin)
        return 2
    return completed.returncode


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AnyChain ADK app through the official Google ADK CLI for diagnostics.",
    )
    parser.add_argument(
        "--prompt",
        help="Send one prompt to the ADK CLI through stdin, then exit.",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(ADK_AGENT_DIR),
        help="ADK agent directory. Defaults to agent/adk_app.",
    )
    parser.add_argument(
        "--adk-bin",
        default=_default_adk_bin(),
        help="ADK CLI executable name or path.",
    )
    parser.add_argument(
        "adk_arg",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after `adk run <agent-dir>`.",
    )
    return parser.parse_args(argv)


def _default_adk_bin() -> str:
    project_adk = REPO_ROOT / ".venv-adk" / "bin" / "adk"
    if project_adk.is_file():
        return str(project_adk)
    return "adk"


def _print_missing_adk(adk_bin: str) -> None:
    message = f"""Google ADK CLI was not found: {adk_bin}

AnyChain Benchmark Agent uses its own product terminal entrypoint and uses
Google ADK underneath for Agent runtime capabilities. Install ADK in an
isolated Python 3.10+ environment, then run:

  bash scripts/install_agent_deps.sh --yes
  ./bin/anychain-agent

Offline contract checks that do not require model credentials are still
available for development:

  python3 agent/cli.py adk-eval
  python3 agent/cli.py adk-status

"""
    sys.stderr.write(message)


if __name__ == "__main__":
    raise SystemExit(run_adk_cli())
