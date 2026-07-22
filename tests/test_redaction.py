"""Shared persistence-boundary redaction contracts."""

from __future__ import annotations

import unittest

from agent.utils.redaction import redact


class RedactionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
