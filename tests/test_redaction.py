"""Shared persistence-boundary redaction contracts."""

from __future__ import annotations

import unittest

from agent.utils.redaction import redact, secret_values


class RedactionTest(unittest.TestCase):
    def test_secret_reference_must_be_the_complete_durable_value(self) -> None:
        from agent.harness.secret_refs import (
            normalize_state_secret_binding,
            raw_secret_paths_in_state,
        )
        from agent.harness.state import new_state

        mixed = (
            "semantic-secret:opaque "
            "Authorization: Bearer raw-token-123456"
        )
        state = new_state("mixed-secret-reference")
        state["confirmed_config"]["RPC_API_KEY"] = mixed

        self.assertEqual(
            raw_secret_paths_in_state(state),
            ("confirmed_config.RPC_API_KEY",),
        )
        with self.assertRaisesRegex(
            ValueError,
            "state secret binding is invalid",
        ):
            normalize_state_secret_binding({
                "reference": mixed,
                "scope_id": "turn:1",
                "atom_id": "input-secret-1",
                "value_hash": "a" * 64,
            })

    def test_websocket_endpoint_tokens_are_redacted(self) -> None:
        secret = "abcdefghijklmnopqrstuvwxyz123456"
        value = f"wss://rpc.example/ws/{secret}?api_key={secret}"
        redacted = str(redact(value))
        self.assertNotIn(secret, redacted)
        self.assertIn("wss://rpc.example/ws/***REDACTED***", redacted)

    def test_structured_secret_keys_are_detected(self) -> None:
        payload = (
            '{"api_key":"short-value",'
            '"headers":{"X-Api-Key":"another-value"}}'
        )
        self.assertEqual(
            set(secret_values(payload)),
            {"short-value", "another-value"},
        )

    def test_url_tokens_are_redacted_before_trailing_punctuation(self) -> None:
        secret = "abcdefghijklmnopqrstuvwxyz123456"
        value = f"Use https://rpc.example/{secret}, then continue."
        redacted = str(redact(value))
        self.assertNotIn(secret, redacted)
        self.assertIn("https://rpc.example/***REDACTED***,", redacted)

    def test_query_and_fragment_credentials_are_redacted(self) -> None:
        secret = "abcdefghijklmnopqrstuvwxyz123456"
        redacted = str(redact(
            f"https://rpc.example/path?api_key={secret}&cursor={secret}#{secret}"
        ))
        self.assertNotIn(secret, redacted)
        self.assertIn("REDACTED", redacted)

    def test_tuple_content_uses_the_same_recursive_boundary(self) -> None:
        secret = "abcdefghijklmnopqrstuvwxyz123456"
        value = (f"Authorization: Bearer {secret}",)
        self.assertEqual(redact(value), ("Authorization: Bearer ***REDACTED***",))

    def test_basic_authorization_secret_is_detected_and_redacted(self) -> None:
        credential = "dXNlcjpwYXNz"
        value = (
            "curl -H 'Authorization: Basic "
            f"{credential}' https://rpc.example/rpc"
        )
        self.assertEqual(secret_values(value), (credential,))
        self.assertNotIn(credential, str(redact(value)))


if __name__ == "__main__":
    unittest.main()
