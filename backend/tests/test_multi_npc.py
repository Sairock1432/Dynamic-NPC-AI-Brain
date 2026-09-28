import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.memory import database
from app.npcs import NPC_CONFIGS, UnknownNPCError, get_npc
from app.services import semantic_memory
from app.services.semantic_memory import COLLECTION_NAME


class MultiNPCArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_patcher = patch.object(database, "DATABASE_PATH", root / "memory.db")
        self.chroma_patcher = patch.object(semantic_memory, "CHROMA_PATH", root / "chroma")
        self.database_patcher.start()
        self.chroma_patcher.start()
        semantic_memory._get_collection.cache_clear()
        semantic_memory._get_client.cache_clear()
        database.init_db()
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        semantic_memory.close_client()
        self.chroma_patcher.stop()
        self.database_patcher.stop()
        self.temp_dir.cleanup()

    def test_npc_registry_and_unknown_lookup(self) -> None:
        self.assertEqual(set(NPC_CONFIGS), {"ravi", "meera"})
        self.assertEqual(get_npc("ravi").role, "Blacksmith")
        self.assertEqual(get_npc("meera").role, "Healer")
        self.assertIn("blacksmith", get_npc("ravi").system_instruction.lower())
        self.assertIn("healer", get_npc("meera").system_instruction.lower())
        with self.assertRaises(UnknownNPCError):
            get_npc("unknown")

    def test_existing_sqlite_messages_migrate_to_ravi_without_data_loss(self) -> None:
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        with closing(sqlite3.connect(legacy_path)) as connection:
            connection.execute(
                """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                ("legacy-session", "user", "Old Ravi conversation"),
            )
            connection.commit()

        with patch.object(database, "DATABASE_PATH", legacy_path):
            database.init_db()
            self.assertEqual(
                database.get_conversation_history("legacy-session", npc_id="ravi"),
                [{"role": "user", "content": "Old Ravi conversation"}],
            )
            self.assertEqual(
                database.get_conversation_history("legacy-session", npc_id="meera"),
                [],
            )
            with closing(sqlite3.connect(legacy_path)) as connection:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
            self.assertIn("npc_id", columns)

    def test_conversation_and_semantic_memory_isolate_same_session_by_npc(self) -> None:
        database.save_message("session-a", "user", "Ravi-only turn", npc_id="ravi")
        database.save_message("session-a", "user", "Meera-only turn", npc_id="meera")
        semantic_memory.add_memory(
            "session-a", "The player's stated name is Rowan.", {"kind": "name"}, npc_id="ravi"
        )
        semantic_memory.add_memory(
            "session-a", "The player's stated name is Meera-Guest.", {"kind": "name"}, npc_id="meera"
        )
        semantic_memory.add_memory(
            "session-b", "The player's stated name is Other-Guest.", {"kind": "name"}, npc_id="ravi"
        )

        ravi_history = database.get_conversation_history("session-a", npc_id="ravi")
        meera_history = database.get_conversation_history("session-a", npc_id="meera")
        ravi_a = semantic_memory.search_memories("session-a", "What is my name?", npc_id="ravi")
        meera_a = semantic_memory.search_memories("session-a", "What is my name?", npc_id="meera")
        ravi_b = semantic_memory.search_memories("session-b", "What is my name?", npc_id="ravi")

        self.assertEqual(ravi_history[0]["content"], "Ravi-only turn")
        self.assertEqual(meera_history[0]["content"], "Meera-only turn")
        self.assertTrue(any("Rowan" in item["text"] for item in ravi_a))
        self.assertFalse(any("Meera-Guest" in item["text"] for item in ravi_a))
        self.assertTrue(any("Meera-Guest" in item["text"] for item in meera_a))
        self.assertFalse(any("Rowan" in item["text"] for item in meera_a))
        self.assertTrue(any("Other-Guest" in item["text"] for item in ravi_b))
        self.assertFalse(any("Rowan" in item["text"] for item in ravi_b))
        self.assertEqual(semantic_memory.delete_session_memories("session-a", npc_id="ravi"), 1)
        self.assertEqual(semantic_memory.search_memories("session-a", "What is my name?", npc_id="ravi"), [])
        self.assertTrue(
            semantic_memory.search_memories("session-a", "What is my name?", npc_id="meera")
        )

    def test_legacy_chroma_memory_is_additively_assigned_to_ravi(self) -> None:
        collection = semantic_memory._get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=["legacy-memory"],
            documents=["The player's stated name is Legacy Rowan."],
            metadatas=[{"session_id": "legacy-session", "kind": "name"}],
        )
        semantic_memory._get_collection.cache_clear()

        ravi_results = semantic_memory.search_memories(
            "legacy-session",
            "What is my name?",
            npc_id="ravi",
        )
        meera_results = semantic_memory.search_memories(
            "legacy-session",
            "What is my name?",
            npc_id="meera",
        )

        self.assertTrue(any("Legacy Rowan" in item["text"] for item in ravi_results))
        self.assertEqual(meera_results, [])

    def test_chat_api_switching_npcs_restores_each_history_and_rejects_unknown(self) -> None:
        calls = []

        def generate(message, history, *, npc, **kwargs):
            calls.append({"npc_id": npc.id, "history": list(history), "memories": kwargs["semantic_memories"]})
            return f"I am {npc.name}."

        with patch.object(main, "generate_npc_reply", side_effect=generate):
            with TestClient(main.app) as client:
                ravi_first = client.post(
                    "/chat",
                    json={"session_id": "session-a", "npc_id": "ravi", "message": "My name is Rowan."},
                )
                meera = client.post(
                    "/chat",
                    json={"session_id": "session-a", "npc_id": "meera", "message": "What is my name?"},
                )
                ravi_again = client.post(
                    "/chat",
                    json={"session_id": "session-a", "npc_id": "ravi", "message": "What is my name?"},
                )
                unknown = client.post(
                    "/chat",
                    json={"session_id": "session-a", "npc_id": "ravi-extra", "message": "Hello"},
                )
                missing_npc_id = client.post(
                    "/chat",
                    json={"session_id": "session-a", "message": "Hello"},
                )
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/game-state/session-a").status_code, 200)
                self.assertEqual(client.get("/inventory/session-a").status_code, 200)

        self.assertEqual(ravi_first.status_code, 200)
        self.assertEqual(meera.status_code, 200)
        self.assertEqual(ravi_again.status_code, 200)
        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(missing_npc_id.status_code, 422)
        self.assertEqual(calls[0]["npc_id"], "ravi")
        self.assertEqual(calls[1]["npc_id"], "meera")
        self.assertEqual(calls[1]["history"], [])
        self.assertEqual(calls[1]["memories"], [])
        self.assertEqual(calls[2]["npc_id"], "ravi")
        self.assertEqual(len(calls[2]["history"]), 2)
        self.assertTrue(any("Rowan" in item["text"] for item in calls[2]["memories"]))
        self.assertEqual(len(database.get_conversation_history("session-a", npc_id="ravi")), 4)
        self.assertEqual(len(database.get_conversation_history("session-a", npc_id="meera")), 2)
        self.assertEqual(calls[0]["npc_id"], get_npc("ravi").id)

    def test_history_api_returns_empty_and_session_scoped_npc_summaries(self) -> None:
        with TestClient(main.app) as client:
            empty = client.get("/history/session-empty")
            self.assertEqual(empty.status_code, 200)
            self.assertEqual(empty.json(), [])

            database.save_message("session-a", "user", "First Ravi question", npc_id="ravi")
            database.save_message("session-a", "assistant", "Ravi answer", npc_id="ravi")
            database.save_message("session-a", "user", "Meera question preview", npc_id="meera")
            database.save_message("session-a", "assistant", "Meera answer", npc_id="meera")
            database.save_message("session-b", "user", "Private other session", npc_id="ravi")

            session_a = client.get("/history/session-a")
            session_b = client.get("/history/session-b")

        self.assertEqual(session_a.status_code, 200)
        self.assertEqual(session_b.status_code, 200)
        summaries_a = session_a.json()
        self.assertEqual([item["npc_id"] for item in summaries_a], ["meera", "ravi"])
        meera_summary = summaries_a[0]
        self.assertEqual(meera_summary["conversation_id"], "meera")
        self.assertEqual(meera_summary["npc_name"], "Meera")
        self.assertEqual(meera_summary["title"], "Conversation with Meera")
        self.assertEqual(meera_summary["message_count"], 2)
        self.assertIn("Meera question preview", meera_summary["preview"])
        self.assertTrue(meera_summary["started_at"])
        self.assertTrue(meera_summary["updated_at"])
        self.assertEqual([item["npc_id"] for item in session_b.json()], ["ravi"])
        self.assertNotIn("Private other session", str(summaries_a))

    def test_history_api_loads_full_conversation_in_chronological_order(self) -> None:
        for index in range(24):
            database.save_message(
                "session-a",
                "user" if index % 2 == 0 else "assistant",
                f"Stored turn {index}",
                npc_id="ravi",
            )
        database.save_message("session-a", "user", "Meera-only content", npc_id="meera")

        with TestClient(main.app) as client:
            ravi = client.get("/history/session-a/ravi")
            meera = client.get("/history/session-a/meera")
            other_session = client.get("/history/session-b/ravi")
            unknown_npc = client.get("/history/session-a/unknown")
            invalid_session = client.get("/history/bad%20session")

        self.assertEqual(ravi.status_code, 200)
        self.assertEqual(ravi.json()["conversation_id"], "ravi")
        self.assertEqual(ravi.json()["npc_name"], "Ravi")
        self.assertEqual(len(ravi.json()["messages"]), 24)
        self.assertEqual(
            [message["content"] for message in ravi.json()["messages"]],
            [f"Stored turn {index}" for index in range(24)],
        )
        self.assertEqual(ravi.json()["messages"][0]["role"], "user")
        self.assertTrue(all(message["created_at"] for message in ravi.json()["messages"]))
        self.assertEqual([message["content"] for message in meera.json()["messages"]], [
            "Meera-only content"
        ])
        self.assertEqual(other_session.status_code, 404)
        self.assertEqual(unknown_npc.status_code, 400)
        self.assertIn(invalid_session.status_code, (400, 422))


if __name__ == "__main__":
    unittest.main()
