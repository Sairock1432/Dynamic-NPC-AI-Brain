import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from google.genai import _api_client, errors
import tenacity

from app import main
from app.memory import database
from app.npcs import get_npc
from app.services import gemini_service


class GeminiErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_patcher = patch.object(
            database,
            "DATABASE_PATH",
            Path(self.temp_dir.name) / "state.db",
        )
        self.database_patcher.start()
        database.init_db()
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        self.database_patcher.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _fake_generation(response):
        return SimpleNamespace(
            models=SimpleNamespace(generate_content=Mock(return_value=response))
        )

    def test_gemini_success_returns_response_without_changing_request_shape(self) -> None:
        fake_client = self._fake_generation(
            SimpleNamespace(function_calls=[], text="Welcome to the village.")
        )
        game_state = database.get_player_state("session-a")
        with patch.object(gemini_service, "_get_client", return_value=fake_client):
            reply = gemini_service.generate_npc_reply(
                "Hello",
                [],
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=game_state,
                inventory=[],
            )

        self.assertEqual(reply, "Welcome to the village.")
        fake_client.models.generate_content.assert_called_once()
        request = fake_client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], gemini_service.MODEL_ID)
        self.assertEqual(len(request["contents"]), 1)

    def test_server_error_is_preserved_as_cause_and_not_retried_by_app(self) -> None:
        upstream_error = errors.ServerError(
            503,
            {"error": {"message": "temporarily unavailable", "status": "UNAVAILABLE"}},
        )
        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=Mock(side_effect=upstream_error))
        )
        with patch.object(gemini_service, "_get_client", return_value=fake_client):
            with self.assertRaises(gemini_service.GeminiServiceError) as raised:
                gemini_service.generate_npc_reply(
                    "Hello",
                    [],
                    npc=get_npc("ravi"),
                    session_id="session-a",
                    game_state=database.get_player_state("session-a"),
                    inventory=[],
                )

        self.assertIs(raised.exception.__cause__, upstream_error)
        fake_client.models.generate_content.assert_called_once()

    def test_read_timeout_is_preserved_as_cause_and_not_retried_by_app(self) -> None:
        timeout_error = httpx.ReadTimeout("upstream read timed out")
        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=Mock(side_effect=timeout_error))
        )
        with patch.object(gemini_service, "_get_client", return_value=fake_client):
            with self.assertRaises(gemini_service.GeminiServiceError) as raised:
                gemini_service.generate_npc_reply(
                    "Hello",
                    [],
                    npc=get_npc("ravi"),
                    session_id="session-a",
                    game_state=database.get_player_state("session-a"),
                    inventory=[],
                )

        self.assertIs(raised.exception.__cause__, timeout_error)
        fake_client.models.generate_content.assert_called_once()

    def test_sdk_retry_policy_is_bounded_and_retries_server_error_and_timeout(self) -> None:
        self.assertEqual(gemini_service.GEMINI_TIMEOUT_MS, 30_000)
        retry_options = gemini_service.GEMINI_RETRY_OPTIONS
        self.assertEqual(retry_options.attempts, 2)
        self.assertEqual(retry_options.http_status_codes, [408, 500, 502, 503, 504])

        retry_kwargs = _api_client.retry_args(retry_options)
        retry_kwargs["wait"] = tenacity.wait_none()
        retry = tenacity.Retrying(**retry_kwargs)

        server_calls = []
        def server_error_once():
            server_calls.append(True)
            if len(server_calls) == 1:
                raise errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}})
            return "recovered"

        self.assertEqual(retry(server_error_once), "recovered")
        self.assertEqual(len(server_calls), 2)

        timeout_calls = []
        def timeout_once():
            timeout_calls.append(True)
            if len(timeout_calls) == 1:
                raise httpx.ReadTimeout("temporary timeout")
            return "recovered"

        self.assertEqual(retry(timeout_once), "recovered")
        self.assertEqual(len(timeout_calls), 2)

        failed_calls = []
        def always_timeout():
            failed_calls.append(True)
            raise httpx.ReadTimeout("timeout persists")

        with self.assertRaises(httpx.ReadTimeout):
            retry(always_timeout)
        self.assertEqual(len(failed_calls), 2)

    def test_chat_maps_upstream_server_and_timeout_errors_to_generic_502(self) -> None:
        failures = (
            errors.ServerError(503, {"error": {"message": "private upstream detail"}}),
            httpx.ReadTimeout("private timeout detail"),
        )
        for failure in failures:
            with self.subTest(error=type(failure).__name__):
                service_error = gemini_service.GeminiServiceError("failed")
                service_error.__cause__ = failure
                with (
                    patch.object(main, "search_memories", return_value=[]),
                    patch.object(main, "generate_npc_reply", side_effect=service_error),
                ):
                    with self.assertRaises(HTTPException) as raised:
                        main.chat(main.ChatRequest(
                            session_id="session-a",
                            npc_id="ravi",
                            message="A quiet ordinary message.",
                        ))

                self.assertEqual(raised.exception.status_code, 502)
                self.assertNotIn("private", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()