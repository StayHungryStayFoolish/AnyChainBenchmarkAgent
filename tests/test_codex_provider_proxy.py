from __future__ import annotations

import json
import signal
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error as urlerror
from urllib import request as urlrequest

from tests.agent_live.codex_provider_proxy import (
    CodexExecConfig,
    CodexProviderProxyError,
    CodexProxyServer,
    ProxyLedger,
    build_codex_prompt,
    invoke_codex,
    normalize_messages,
)


class CodexProviderProxyTests(unittest.TestCase):
    def test_messages_preserve_order_roles_and_unicode(self) -> None:
        messages = normalize_messages(
            [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "解释 BSC MGas/s"},
            ]
        )
        self.assertEqual(messages[1]["content"], "解释 BSC MGas/s")
        payload = json.loads(build_codex_prompt(messages).split("CHAT_REQUEST_JSON=", 1)[1])
        self.assertEqual(payload["messages"], list(messages))

    def test_messages_reject_invalid_shape(self) -> None:
        for value in (None, [], "text", [{"role": "unknown", "content": "x"}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_messages(value)
        with self.assertRaises(ValueError):
            normalize_messages([{"role": "user", "content": ["not", "text"]}])

    @patch("tests.agent_live.codex_provider_proxy.subprocess.Popen")
    def test_invoke_codex_returns_only_final_message(self, popen) -> None:
        process = popen.return_value
        process.returncode = 0

        def complete(*, input, timeout):
            command = popen.call_args.args[0]
            output = Path(command[command.index("-o") + 1])
            output.write_text('{"actions":[]}', encoding="utf-8")
            return "", "ignored diagnostic"

        process.communicate.side_effect = complete
        result = invoke_codex(
            ({"role": "user", "content": "route this"},),
            config=CodexExecConfig(executable="codex-test", model="gpt-test"),
        )
        self.assertEqual(result, '{"actions":[]}')
        command = popen.call_args.args[0]
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-rules", command)
        self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.DEVNULL)

    @patch("tests.agent_live.codex_provider_proxy.subprocess.Popen")
    def test_invoke_codex_fails_closed(self, popen) -> None:
        process = popen.return_value
        process.returncode = 2
        process.communicate.return_value = ("", "secret detail")
        with self.assertRaisesRegex(CodexProviderProxyError, "stderr_sha256"):
            invoke_codex(
                ({"role": "user", "content": "x"},),
                config=CodexExecConfig(),
            )

    @patch("tests.agent_live.codex_provider_proxy.os.killpg")
    @patch("tests.agent_live.codex_provider_proxy.subprocess.Popen")
    def test_invoke_codex_timeout_is_typed(self, popen, killpg) -> None:
        process = popen.return_value
        process.pid = 1234
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(("codex",), 1),
            ("", ""),
        ]
        with self.assertRaisesRegex(CodexProviderProxyError, "timed out"):
            invoke_codex(
                ({"role": "user", "content": "x"},),
                config=CodexExecConfig(),
            )
        killpg.assert_called_once_with(1234, signal.SIGKILL)

    def test_http_contract_auth_and_completion(self) -> None:
        observed = []

        def complete(messages):
            observed.append(messages)
            return '{"kind":"consultation"}'

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            server = CodexProxyServer(
                ("127.0.0.1", 0),
                bearer_token="test-token",
                model="gpt-test",
                complete=complete,
                ledger=ProxyLedger(ledger_path),
                codex_cli_version="codex-cli test",
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
                body = json.dumps(
                    {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}
                ).encode()
                unauthorized = urlrequest.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urlerror.HTTPError) as raised:
                    urlrequest.urlopen(unauthorized, timeout=2)
                self.assertEqual(raised.exception.code, 401)

                authorized = urlrequest.Request(
                    url,
                    data=body,
                    headers={
                        "Authorization": "Bearer test-token",
                        "Content-Type": "application/json",
                    },
                )
                with urlrequest.urlopen(authorized, timeout=2) as response:
                    payload = json.load(response)
                self.assertEqual(
                    payload["choices"][0]["message"]["content"],
                    '{"kind":"consultation"}',
                )
                self.assertEqual(observed[0][0]["content"], "hi")
                records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
                self.assertEqual(records[-1]["status"], "completed")
                self.assertEqual(records[-1]["backend_identity"], "codex_exec_subscription")
                self.assertEqual(len(records[-1]["request_sha256"]), 64)
                self.assertNotIn("hi", ledger_path.read_text())
                self.assertNotIn("test-token", ledger_path.read_text())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
