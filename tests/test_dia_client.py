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


class DiaUnreachableHintTests(unittest.TestCase):
    def test_localhost_inside_a_container_points_to_the_compose_name(self) -> None:
        with mock.patch.object(dia_client, "_running_in_container", return_value=True):
            hint = dia_client.dia_unreachable_hint("http://localhost:7864")
            self.assertIn("http://dia:7864", hint)
            self.assertIn("host.docker.internal", hint)
            self.assertIn("http://dia:7864", dia_client.dia_unreachable_hint("http://127.0.0.1:7864", german=True))

    def test_localhost_outside_a_container_needs_no_hint(self) -> None:
        with mock.patch.object(dia_client, "_running_in_container", return_value=False):
            self.assertEqual(dia_client.dia_unreachable_hint("http://localhost:7864"), "")

    def test_compose_name_hint_and_other_hosts(self) -> None:
        self.assertIn("Compose", dia_client.dia_unreachable_hint("http://dia:7864"))
        self.assertEqual(dia_client.dia_unreachable_hint("http://10.0.0.5:7864"), "")

    def test_unreachable_upstream_message_names_url_and_hint(self) -> None:
        def refuse(_request):
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(refuse)
        real_client = httpx.AsyncClient
        with (
            mock.patch.object(dia_client, "_effective_config", return_value=("http://localhost:7864", "")),
            mock.patch.object(dia_client.httpx, "AsyncClient", side_effect=lambda **kw: real_client(transport=transport, **kw)),
            mock.patch.object(dia_client, "_running_in_container", return_value=True),
        ):
            with self.assertRaises(dia_client.DiaClientError) as caught:
                asyncio.run(dia_client.diarize_v2(io.BytesIO(b"RIFF"), "a.wav", "audio/wav"))
        self.assertIn("http://localhost:7864", caught.exception.message)
        self.assertIn("http://dia:7864", caught.exception.message)


if __name__ == "__main__":
    unittest.main()
