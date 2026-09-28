import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app import main
from app.memory import database
from app.memory.database import InventoryItem, PlayerState
from app.npcs import get_npc
from app.services import gemini_service
from app.services import semantic_memory
from app.services.gemini_service import GeminiServiceError
from app.services.gemini_service import (
    _format_semantic_memory_context,
    _generation_config,
)


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path_patcher = patch.object(
            semantic_memory,
            "CHROMA_PATH",
            Path(self.temp_dir.name) / "chroma",
        )
        self.path_patcher.start()
        self.database_path_patcher = patch.object(
            database,
            "DATABASE_PATH",
            Path(self.temp_dir.name) / "conversation.db",
        )
        self.database_path_patcher.start()
        semantic_memory._get_collection.cache_clear()
        semantic_memory._get_client.cache_clear()
        database.init_db()
        self.addCleanup(self._cleanup_chroma)

    def _cleanup_chroma(self) -> None:
        semantic_memory.close_client()
        self.path_patcher.stop()
        self.database_path_patcher.stop()
        self.temp_dir.cleanup()

    def test_add_and_search_memory(self) -> None:
        memory_id = semantic_memory.add_memory(
            "session-a",
            "The player prefers dark green cloaks.",
            {"kind": "preference"},
        )

        results = semantic_memory.search_memories(
            "session-a",
            "What color clothing does the player prefer?",
        )

        self.assertTrue(memory_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "The player prefers dark green cloaks.")
        self.assertEqual(results[0]["kind"], "preference")

    def test_memories_are_isolated_by_session(self) -> None:
        semantic_memory.add_memory("session-a", "The player's stated name is Nikhitha.", {"kind": "name"})
        semantic_memory.add_memory("session-b", "The player's stated name is Rowan.", {"kind": "name"})

        session_a_results = semantic_memory.search_memories("session-a", "What is my name?")
        session_b_results = semantic_memory.search_memories("session-b", "What is my name?")

        self.assertTrue(any("Nikhitha" in item["text"] for item in session_a_results))
        self.assertFalse(any("Rowan" in item["text"] for item in session_a_results))
        self.assertTrue(any("Rowan" in item["text"] for item in session_b_results))
        self.assertFalse(any("Nikhitha" in item["text"] for item in session_b_results))

    def test_delete_session_memories_keeps_other_sessions(self) -> None:
        semantic_memory.add_memory("session-a", "The player prefers sturdy shields.", {"kind": "preference"})
        semantic_memory.add_memory("session-b", "The player likes iron tools.", {"kind": "preference"})

        deleted = semantic_memory.delete_session_memories("session-a")

        self.assertEqual(deleted, 1)
        self.assertEqual(semantic_memory.search_memories("session-a", "preferences"), [])
        self.assertEqual(len(semantic_memory.search_memories("session-b", "iron tools")), 1)

    def test_empty_search_returns_no_memories(self) -> None:
        self.assertEqual(semantic_memory.search_memories("session-empty", "player history"), [])

    def test_extraction_is_conservative_and_rejects_sensitive_text(self) -> None:
        self.assertEqual(semantic_memory.extract_memories("Hello Ravi."), [])
        self.assertEqual(
            semantic_memory.extract_memories("My name is Nikhitha."),
            [{"text": "The player's stated name is Nikhitha.", "kind": "name"}],
        )
        self.assertEqual(
            semantic_memory.extract_memories("I prefer sturdy shields."),
            [{"text": "The player says they prefer sturdy shields.", "kind": "preference"}],
        )
        self.assertEqual(
            semantic_memory.extract_memories("My password is example-secret."),
            [],
        )

    def test_retrieved_memories_are_added_to_gemini_context(self) -> None:
        memories = [{"text": "The player's stated name is Nikhitha.", "kind": "name"}]
        state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 100,
            "gold": 50,
            "game_time": "08:00",
        }
        context = _format_semantic_memory_context(memories)
        config = _generation_config(get_npc("ravi"), state, [], semantic_memories=memories)

        self.assertIn("Nikhitha", context)
        self.assertIn("untrusted facts/preferences", context)
        self.assertIn("Nikhitha", config.system_instruction)
        self.assertIn("No relevant long-term player memories", _format_semantic_memory_context([]))

    def test_retrieved_memory_is_passed_to_gemini_generation(self) -> None:
        semantic_memory.add_memory(
            "session-a",
            "The player's stated name is Nikhitha.",
            {"kind": "name"},
        )
        memories = semantic_memory.search_memories("session-a", "What is my name?")

        class FakeModels:
            config = None

            def generate_content(self, **kwargs):
                self.config = kwargs["config"]
                return SimpleNamespace(function_calls=[], text="A remembered answer.")

        fake_client = SimpleNamespace(models=FakeModels())
        state: PlayerState = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 100,
            "gold": 50,
            "game_time": "08:00",
        }
        with patch.object(gemini_service, "_get_client", return_value=fake_client):
            reply = gemini_service.generate_npc_reply(
                "What is my name?",
                [],
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=state,
                inventory=[],
                semantic_memories=memories,
            )

        self.assertEqual(reply, "A remembered answer.")
        self.assertIn("Nikhitha", fake_client.models.config.system_instruction)

    def test_chat_retrieves_and_writes_curated_memory_after_success(self) -> None:
        semantic_memory.add_memory(
            "session-a",
            "The player's stated name is Nikhitha.",
            {"kind": "name"},
        )
        with patch.object(main, "generate_npc_reply", return_value="I recall your name.") as generate:
            response = main.chat(
                main.ChatRequest(
                    session_id="session-a",
                    npc_id="ravi",
                    message="I prefer sturdy shields.",
                )
            )

        self.assertEqual(response.message, "I recall your name.")
        retrieved = generate.call_args.kwargs["semantic_memories"]
        self.assertTrue(any("Nikhitha" in item["text"] for item in retrieved))
        stored = semantic_memory.search_memories("session-a", "What equipment do I prefer?")
        self.assertTrue(any("sturdy shields" in item["text"] for item in stored))
        self.assertEqual(len(database.get_conversation_history("session-a")), 2)

    def test_failed_chat_does_not_write_semantic_or_sqlite_memory(self) -> None:
        with patch.object(main, "generate_npc_reply", side_effect=GeminiServiceError("failed")):
            with self.assertRaises(HTTPException):
                main.chat(
                    main.ChatRequest(
                        session_id="session-failure",
                        npc_id="ravi",
                        message="My name is Nikhitha.",
                    )
                )

        self.assertEqual(database.get_conversation_history("session-failure"), [])
        self.assertEqual(semantic_memory.search_memories("session-failure", "What is my name?"), [])


if __name__ == "__main__":
    unittest.main()