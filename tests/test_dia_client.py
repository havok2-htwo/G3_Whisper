from __future__ import annotations

import asyncio
import io
import unittest
from unittest import mock

import httpx

from backend import genesis_whisper_server_dia_client as dia_client


def _diarize_with_response(response: httpx.Response) -> dia_client.DiaClientError:
    transport = httpx.MockTransport(lambda _request: response)
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    with (
        mock.patch.object(dia_client, "_effective_config", return_value=("http://dia:7864", "")),
        mock.patch.object(dia_client.httpx, "AsyncClient", side_effect=client_factory),
    ):
        try:
            asyncio.run(dia_client.diarize_v2(io.BytesIO(b"RIFF"), "a.wav", "audio/wav"))
        except dia_client.DiaClientError as exc:
            return exc
    raise AssertionError("DiaClientError expected")


class DiaClientErrorDetailTests(unittest.TestCase):
    def test_client_error_carries_dia_reason(self) -> None:
        exc = _diarize_with_response(
            httpx.Response(400, json={"detail": "Konnte Audiodatei nicht verarbeiten: kaputt"})
        )
        self.assertEqual(exc.code, "DIA_UPSTREAM_ERROR")
        self.assertFalse(exc.retryable)
        self.assertIn("HTTP 400", exc.message)
        self.assertIn("Grund: Konnte Audiodatei nicht verarbeiten: kaputt", exc.message)

    def test_server_error_carries_shortened_reason_and_stays_retryable(self) -> None:
        exc = _diarize_with_response(httpx.Response(500, json={"detail": "x" * 1000}))
        self.assertTrue(exc.retryable)
        self.assertIn("HTTP 500", exc.message)
        self.assertIn("Grund: " + "x" * 300, exc.message)
        self.assertNotIn("x" * 301, exc.message)

    def test_non_json_error_body_keeps_the_plain_message(self) -> None:
        exc = _diarize_with_response(httpx.Response(502, text="<html>bad gateway</html>"))
        self.assertEqual(exc.message, "DIA-Server meldete HTTP 502.")

    def test_auth_failure_does_not_echo_upstream_body(self) -> None:
        exc = _diarize_with_response(httpx.Response(401, json={"detail": "Invalid API key"}))
        self.assertEqual(exc.code, "DIA_AUTH_FAILED")
        self.assertNotIn("Invalid API key", exc.message)


if __name__ == "__main__":
    unittest.main()
