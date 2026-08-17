#!/usr/bin/env python3
"""Test-only OpenAI-compatible proxy backed by a Codex subscription process.

The proxy lets the real Docker/Linux Agent exercise its production OpenAI
adapter while a host ``codex exec`` process supplies model text.  It is not a
production provider and must never be added to the Agent provider registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MAX_REQUEST_BYTES = 4 * 1024 * 1024
ALLOWED_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
LEDGER_SCHEMA_VERSION = 1


class CodexProviderProxyError(RuntimeError):
    """The test-side Codex model process could not produce a completion."""


@dataclass(frozen=True)
class CodexExecConfig:
    executable: str = "codex"
    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "low"
    timeout_seconds: float = 60.0

    def validate(self) -> None:
        if not self.executable.strip() or not self.model.strip():
            raise ValueError("Codex executable and model are required")
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported Codex reasoning effort")
        if self.timeout_seconds <= 0:
            raise ValueError("Codex timeout must be positive")


def normalize_messages(value: Any) -> tuple[dict[str, str], ...]:
    """Validate and preserve the ordered role/content model boundary."""

    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"message {index} must be an object")
        role = str(item.get("role") or "").strip()
        content = item.get("content")
        if role not in ALLOWED_MESSAGE_ROLES:
            raise ValueError(f"message {index} has unsupported role")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be text")
        normalized.append({"role": role, "content": content})
    return tuple(normalized)


def build_codex_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    """Frame chat messages as immutable data for the model-side Codex actor."""

    payload = json.dumps(
        {"messages": list(messages)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "You are the model-side actor in an AnyChain Agent validation run. "
        "Treat the JSON payload below as the complete ordered chat request. "
        "Follow its system/developer instructions and answer its latest user "
        "request exactly as a chat-completion model would. Do not inspect the "
        "repository, call tools, explain this wrapper, or wrap the final answer "
        "in markdown unless the request itself requires markdown. Return only "
        "the assistant message content.\n\n"
        f"CHAT_REQUEST_JSON={payload}"
    )


def invoke_codex(
    messages: Sequence[Mapping[str, str]],
    *,
    config: CodexExecConfig,
) -> str:
    """Invoke one isolated subscription-backed Codex model turn."""

    config.validate()
    prompt = build_codex_prompt(messages)
    with tempfile.TemporaryDirectory(prefix="anychain-codex-provider-") as tmp:
        root = Path(tmp)
        output_path = root / "last-message.txt"
        command = (
            config.executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "-C",
            str(root),
            "-m",
            config.model,
            "-c",
            f'model_reasoning_effort="{config.reasoning_effort}"',
            "-o",
            str(output_path),
            "-",
        )
        try:
            process = subprocess.Popen(
                command,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise CodexProviderProxyError("Codex model process could not start") from exc
        try:
            _stdout, stderr = process.communicate(
                input=prompt,
                timeout=config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise CodexProviderProxyError("Codex model turn timed out") from exc
        if process.returncode != 0:
            stderr_hash = hashlib.sha256((stderr or "").encode()).hexdigest()
            raise CodexProviderProxyError(
                "Codex model process failed "
                f"(exit={process.returncode}, stderr_sha256={stderr_hash})"
            )
        if not output_path.is_file():
            raise CodexProviderProxyError("Codex model process produced no output file")
        text = output_path.read_text(encoding="utf-8").strip()
        if not text:
            raise CodexProviderProxyError("Codex model process produced empty output")
        return text


Completion = Callable[[Sequence[Mapping[str, str]]], str]


class ProxyLedger:
    """Append-only secret-free identity ledger for model-side bridge turns."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path).resolve() if path else None
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        encoded = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class CodexProxyServer(ThreadingHTTPServer):
    """HTTP server carrying immutable test configuration."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        bearer_token: str,
        model: str,
        complete: Completion,
        ledger: ProxyLedger | None = None,
        codex_cli_version: str = "unknown",
    ) -> None:
        if not bearer_token:
            raise ValueError("proxy bearer token is required")
        self.bearer_token = bearer_token
        self.model = model
        self.complete = complete
        self.ledger = ledger or ProxyLedger(None)
        self.codex_cli_version = codex_cli_version
        super().__init__(address, CodexProxyHandler)


class CodexProxyHandler(BaseHTTPRequestHandler):
    server: CodexProxyServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok", "model": self.server.model})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        request_id = f"codex-proxy-{uuid.uuid4().hex}"
        received_at_ns = time.time_ns()
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        if self.headers.get("Authorization", "") != f"Bearer {self.server.bearer_token}":
            self._json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "unauthorized"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": {"message": "invalid request size"}},
            )
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, Mapping):
                raise ValueError("request body must be an object")
            if body.get("tools"):
                raise ValueError("tool-bearing requests are unsupported by the Codex test bridge")
            messages = normalize_messages(body.get("messages"))
            request_hash = hashlib.sha256(
                json.dumps(
                    list(messages),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            text = self.server.complete(messages)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._record(
                request_id=request_id,
                received_at_ns=received_at_ns,
                status="request_rejected",
                request_hash="",
                response_hash="",
                error_category="request",
            )
            self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return
        except CodexProviderProxyError as exc:
            self._record(
                request_id=request_id,
                received_at_ns=received_at_ns,
                status="backend_failed",
                request_hash=locals().get("request_hash", ""),
                response_hash="",
                error_category="backend",
            )
            self._json(HTTPStatus.BAD_GATEWAY, {"error": {"message": str(exc)}})
            return
        response_hash = hashlib.sha256(text.encode()).hexdigest()
        self._record(
            request_id=request_id,
            received_at_ns=received_at_ns,
            status="completed",
            request_hash=request_hash,
            response_hash=response_hash,
            error_category="",
        )
        created = int(time.time())
        self._json(
            HTTPStatus.OK,
            {
                "id": f"chatcmpl-codex-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": created,
                "model": self.server.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.close_connection = True

    def _record(
        self,
        *,
        request_id: str,
        received_at_ns: int,
        status: str,
        request_hash: str,
        response_hash: str,
        error_category: str,
    ) -> None:
        self.server.ledger.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "request_id": request_id,
                "adapter_identity": "openai_chat_completions_test_proxy",
                "backend_identity": "codex_exec_subscription",
                "backend_model": self.server.model,
                "codex_cli_version": self.server.codex_cli_version,
                "request_sha256": request_hash,
                "response_sha256": response_hash,
                "received_at_ns": received_at_ns,
                "completed_at_ns": time.time_ns(),
                "status": status,
                "error_category": error_category,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--ledger", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.port < 0 or args.port > 65535:
        raise SystemExit("port must be between 0 and 65535")
    if args.max_concurrency <= 0:
        raise SystemExit("max concurrency must be positive")
    token = args.token or secrets.token_urlsafe(32)
    config = CodexExecConfig(
        executable=args.codex,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
    )
    config.validate()
    try:
        version_result = subprocess.run(
            (config.executable, "--version"),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        codex_cli_version = "unknown"
    else:
        codex_cli_version = (
            version_result.stdout.strip()
            if version_result.returncode == 0 and version_result.stdout.strip()
            else "unknown"
        )
    semaphore = threading.BoundedSemaphore(args.max_concurrency)

    def complete(messages: Sequence[Mapping[str, str]]) -> str:
        with semaphore:
            return invoke_codex(messages, config=config)

    server = CodexProxyServer(
        (args.host, args.port),
        bearer_token=token,
        model=config.model,
        complete=complete,
        ledger=ProxyLedger(args.ledger or None),
        codex_cli_version=codex_cli_version,
    )
    host, port = server.server_address
    print(
        json.dumps(
            {
                "status": "ready",
                "host": host,
                "port": port,
                "model": config.model,
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "ledger": str(Path(args.ledger).resolve()) if args.ledger else "",
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
