import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import main
from app.memory import database
from app.npcs import get_npc
from app.services import gemini_service, npc_communication, npc_plan_executor
from app.services.language_support import (
    LANGUAGE_INSTRUCTIONS,
    language_instruction,
    normalize_language,
)
from app.services.npc_decision import NPCDialogueDecision
from app.services.npc_planner import create_player_plan
from app.services.npc_relationship_progression import get_npc_learning_memories


class LanguageSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_patcher = patch.object(
            database,
            "DATABASE_PATH",
            Path(self.temp_dir.name) / "state.db",
        )
        self.database_patcher.start()
        database.init_db()
        database.create_default_player_state("session-a")
        self.memory_patcher = patch(
            "app.services.npc_relationship_progression.get_npc_learning_memories",
            return_value=[],
        )
        self.search_patcher = patch.object(main, "search_memories", return_value=[])
        self.remember_patcher = patch.object(main, "remember_user_facts", return_value=0)
        self.learn_patcher = patch.object(main, "learn_from_player_interaction", return_value=0)
        for test_patch in (
            self.memory_patcher,
            self.search_patcher,
            self.remember_patcher,
            self.learn_patcher,
        ):
            test_patch.start()
            self.addCleanup(test_patch.stop)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        self.database_patcher.stop()
        self.temp_dir.cleanup()

    def test_chat_request_accepts_supported_languages_and_defaults_invalid_to_english(self) -> None:
        for language in ("en", "te", "hi", "te-IN", "hi_IN"):
            with self.subTest(language=language):
                request = main.ChatRequest(
                    session_id="session-a",
                    npc_id="ravi",
                    message="Hello",
                    language=language,
                )
                self.assertEqual(request.language, normalize_language(language))
        self.assertEqual(
            main.ChatRequest(
                session_id="session-a", npc_id="ravi", message="Hello"
            ).language,
            "en",
        )
        self.assertEqual(
            main.ChatRequest(
                session_id="session-a", npc_id="ravi", message="Hello", language="fr"
            ).language,
            "en",
        )

    def test_all_language_instructions_are_fixed_and_allowlisted(self) -> None:
        self.assertEqual(set(LANGUAGE_INSTRUCTIONS), {"en", "te", "hi"})
        self.assertIn("English", language_instruction("en"))
        self.assertIn("Telugu", language_instruction("te"))
        self.assertIn("Hindi", language_instruction("hi"))
        self.assertEqual(language_instruction("unsupported"), language_instruction("en"))

    def test_generation_config_includes_only_selected_language_instruction(self) -> None:
        state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 100,
            "gold": 50,
            "game_time": "08:00",
        }
        for language in ("en", "te", "hi"):
            with self.subTest(language=language):
                config = gemini_service._generation_config(
                    get_npc("ravi"), state, [], language=language
                )
                self.assertIn(LANGUAGE_INSTRUCTIONS[language], config.system_instruction)
        fallback = gemini_service._generation_config(
            get_npc("ravi"), state, [], language="xx"
        )
        self.assertIn(LANGUAGE_INSTRUCTIONS["en"], fallback.system_instruction)

    def test_relationship_aware_narration_uses_selected_language(self) -> None:
        fake_models = SimpleNamespace(
            generate_content=Mock(
                return_value=SimpleNamespace(
                    function_calls=[],
                    text="రవి మిమ్మల్ని ఆహ్వానిస్తున్నాడు.",
                )
            )
        )
        state = database.get_player_state("session-a")
        decision = NPCDialogueDecision("session-a", "ravi", "neutral_greeting")
        with patch.object(
            gemini_service,
            "_get_client",
            return_value=SimpleNamespace(models=fake_models),
        ):
            reply = gemini_service.generate_npc_reply(
                "నమస్కారం Ravi",
                [],
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=state,
                inventory=[],
                dialogue_decision=decision,
                language="te",
            )

        prompt = fake_models.generate_content.call_args.kwargs["config"].system_instruction
        self.assertEqual(reply, "రవి మిమ్మల్ని ఆహ్వానిస్తున్నాడు.")
        self.assertIn(LANGUAGE_INSTRUCTIONS["te"], prompt)
        self.assertNotIn("Hindi", prompt)

    def test_changing_language_keeps_session_history_and_relationship_state(self) -> None:
        source_message_id = database.save_message(
            "session-a", "user", "Thanks, that helped.", npc_id="ravi"
        )
        database.apply_npc_player_relationship_delta(
            "session-a",
            "ravi",
            20,
            source_message_id=source_message_id,
            initial_score=20,
        )
        generated = []
        memory_queries = []

        def reply(message, history, *, language, npc, **kwargs):
            generated.append({
                "message": message,
                "history": list(history),
                "language": language,
                "npc": npc.id,
                "semantic_memories": kwargs["semantic_memories"],
            })
            return {"en": "Hello, traveler.", "te": "నమస్కారం ప్రయాణికుడా."}[language]

        def search_scoped_memories(session_id, query, *, npc_id):
            memory_queries.append((session_id, npc_id))
            return [{"text": "The player prefers swords.", "kind": "preference"}]

        with (
            patch.object(main, "generate_npc_reply", side_effect=reply),
            patch.object(main, "search_memories", side_effect=search_scoped_memories),
        ):
            english = main.chat(main.ChatRequest(
                session_id="session-a", npc_id="ravi", message="Hello Ravi.", language="en"
            ))
            telugu = main.chat(main.ChatRequest(
                session_id="session-a", npc_id="ravi", message="మీరు ఎలా ఉన్నారు?", language="te"
            ))

        state_after = database.get_npc_player_relationship_state("session-a", "ravi")
        self.assertEqual(english.message, "Hello, traveler.")
        self.assertEqual(telugu.message, "నమస్కారం ప్రయాణికుడా.")
        self.assertEqual([call["language"] for call in generated], ["en", "te"])
        self.assertEqual([call["npc"] for call in generated], ["ravi", "ravi"])
        self.assertEqual(memory_queries, [("session-a", "ravi"), ("session-a", "ravi")])
        self.assertEqual(generated[1]["semantic_memories"][0]["text"], "The player prefers swords.")
        self.assertEqual(len(generated[1]["history"]), 3)
        self.assertEqual(state_after["trust_score"], 40)
        self.assertEqual(
            len(database.get_conversation_history("session-a", npc_id="ravi")),
            5,
        )
        self.assertIsNone(database.get_npc_player_relationship_state("session-a", "meera"))

    def test_peer_consultation_receives_selected_language(self) -> None:
        state = database.get_player_state("session-a")
        with (
            patch.object(npc_communication, "get_player_state", return_value=state),
            patch.object(npc_communication, "get_inventory", return_value=[]),
            patch(
                "app.services.gemini_service.generate_peer_npc_reply",
                return_value="విల్లో బెరడు ఉపయోగించండి.",
            ) as peer_reply,
        ):
            response = npc_communication.request_npc_message(
                "ravi", "meera", "ఏ ఔషధం ఉపయోగపడుతుంది?", "session-a", language="te"
            )

        self.assertTrue(response["success"])
        self.assertEqual(peer_reply.call_args.kwargs["language"], "te")

    def test_planner_executor_passes_language_to_existing_consultation(self) -> None:
        plan = create_player_plan("ravi", "ASK_HEALING")
        state = database.get_player_state("session-a")
        with patch.object(
            npc_plan_executor,
            "request_npc_message",
            return_value={"success": True, "from_npc": "Meera", "message": "సలహా."},
        ) as request_message:
            result = npc_plan_executor.execute_npc_plan(
                "session-a",
                plan,
                game_state=state,
                inventory=[],
                language="hi",
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(request_message.call_args.kwargs["language"], "hi")


if __name__ == "__main__":
    unittest.main()