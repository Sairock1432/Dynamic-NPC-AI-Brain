import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import main
from app.memory import database
from app.npcs import get_npc
from app.services import gemini_service, semantic_memory
from app.services.npc_decision import (
    NPCDecisionError,
    NPCDialogueDecision,
    is_relationship_dialogue_candidate,
    select_npc_dialogue,
)
from app.services.npc_relationship_progression import (
    NPCRelationshipError,
    RelationshipEvaluation,
)
from app.services.npc_communication import MAX_NPC_MESSAGES_PER_REQUEST


class NPCDialogueDecisionTests(unittest.TestCase):
    def _evaluation(
        self,
        stage: str,
        *,
        session_id: str = "session-a",
        npc_id: str = "ravi",
        events: tuple[str, ...] = (),
    ) -> RelationshipEvaluation:
        return RelationshipEvaluation(session_id, npc_id, stage, (), events)

    def _select(
        self,
        stage: str,
        message: str,
        *,
        intent: str | None = None,
        events: tuple[str, ...] = (),
    ):
        npc = get_npc("ravi")
        return select_npc_dialogue(
            npc,
            self._evaluation(stage, events=events),
            session_id="session-a",
            message=message,
            intent=intent,
        )

    def test_greeting_selection_is_deterministic_for_every_stage(self) -> None:
        for stage, expected in (
            ("NEUTRAL", "neutral_greeting"),
            ("FRIENDLY", "friendly_greeting"),
            ("TRUSTING", "trusting_greeting"),
        ):
            with self.subTest(stage=stage):
                selected = self._select(stage, "Hello, Ravi!")
                self.assertEqual(selected.decision, expected)
                self.assertIsNone(selected.topic)
                self.assertFalse(hasattr(selected, "stage"))

    def test_previous_help_acknowledgment_requires_validated_evidence(self) -> None:
        without_evidence = self._select("FRIENDLY", "Thanks again.")
        with_evidence = self._select(
            "FRIENDLY",
            "Thanks again.",
            events=("successful_help",),
        )

        self.assertIsNone(without_evidence)
        self.assertEqual(with_evidence.decision, "acknowledge_previous_help")

    def test_help_and_information_decisions_follow_allowlisted_intent_and_stage(self) -> None:
        self.assertEqual(
            self._select("NEUTRAL", "Can you help me?", intent="ASK_GENERAL_HELP").decision,
            "normal_response",
        )
        self.assertEqual(
            self._select("FRIENDLY", "Can you help me?", intent="ASK_GENERAL_HELP").decision,
            "offer_contextual_help",
        )
        assistance = self._select(
            "TRUSTING",
            "Can you help with healing?",
            intent="ASK_HEALING",
        )
        self.assertEqual(assistance.decision, "request_assistance")
        information = self._select(
            "NEUTRAL",
            "Tell me about swords.",
            intent="ASK_SWORD",
        )
        self.assertEqual((information.decision, information.topic), (
            "provide_validated_information", "swords"
        ))

    def test_ambiguous_and_unsupported_text_stays_on_normal_path(self) -> None:
        message = "Tell me a story about the village."
        self.assertFalse(is_relationship_dialogue_candidate(message, None))
        self.assertIsNone(self._select("FRIENDLY", message))

    def test_invalid_stage_and_decision_scope_are_rejected(self) -> None:
        with self.assertRaises(NPCRelationshipError):
            self._evaluation("UNKNOWN")
        with self.assertRaises(NPCDecisionError):
            NPCDialogueDecision("session-a", "ravi", "set_relationship_stage")
        with self.assertRaises(NPCDecisionError):
            NPCDialogueDecision("session-a", "meera", "provide_validated_information", "swords")
        with self.assertRaises(NPCDecisionError):
            select_npc_dialogue(
                get_npc("ravi"),
                self._evaluation("FRIENDLY", session_id="session-b"),
                session_id="session-a",
                message="Hello Ravi.",
                intent=None,
            )
        with self.assertRaises(NPCDecisionError):
            select_npc_dialogue(
                get_npc("ravi"),
                self._evaluation("FRIENDLY", npc_id="meera"),
                session_id="session-a",
                message="Hello Ravi.",
                intent=None,
            )


class NPCDialogueChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_patcher = patch.object(database, "DATABASE_PATH", root / "state.db")
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

    @staticmethod
    def _capture_reply(calls):
        def generate(message, history, **kwargs):
            calls.append({"message": message, "history": history, **kwargs})
            return "A brief in-character reply."

        return generate

    def test_chat_relationship_context_isolated_by_session_and_npc(self) -> None:
        semantic_memory.add_learning_memory(
            "session-a",
            "The player has accepted help from Ravi.",
            "PLAYER_ACCEPTED_HELP",
            npc_id="ravi",
        )
        semantic_memory.add_memory(
            "session-a",
            "The player's stated name is MeeraPrivateToken.",
            {"kind": "name"},
            npc_id="meera",
        )
        calls = []
        with patch.object(main, "generate_npc_reply", side_effect=self._capture_reply(calls)):
            first = main.chat(main.ChatRequest(session_id="session-a", npc_id="ravi", message="Hello Ravi."))
            second = main.chat(main.ChatRequest(session_id="session-b", npc_id="ravi", message="Hello Ravi."))
            third = main.chat(main.ChatRequest(session_id="session-a", npc_id="meera", message="Hello Meera."))

        self.assertEqual((first.message, second.message, third.message), (
            "A brief in-character reply.",
            "A brief in-character reply.",
            "A brief in-character reply.",
        ))
        self.assertEqual(calls[0]["dialogue_decision"].decision, "friendly_greeting")
        self.assertEqual(calls[1]["dialogue_decision"].decision, "neutral_greeting")
        self.assertEqual(calls[2]["dialogue_decision"].decision, "neutral_greeting")
        self.assertTrue(all(call["history"] == [] for call in calls))
        self.assertTrue(all(call["semantic_memories"] == [] for call in calls))
        self.assertNotIn(
            "MeeraPrivateToken",
            str(calls[0]),
        )

    def test_explicit_game_actions_and_ambiguous_text_keep_normal_chat_path(self) -> None:
        retrieved = [{"text": "Ravi scoped memory", "kind": "fact"}]
        calls = []
        with (
            patch.object(main, "search_memories", return_value=retrieved),
            patch.object(main, "evaluate_player_relationship") as evaluate,
            patch.object(main, "generate_npc_reply", side_effect=self._capture_reply(calls)),
        ):
            main.chat(main.ChatRequest(session_id="session-a", npc_id="ravi", message="Buy 1 iron."))
            main.chat(main.ChatRequest(
                session_id="session-a",
                npc_id="ravi",
                message="Tell me a story about the village.",
            ))

        evaluate.assert_not_called()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["dialogue_decision"] is None for call in calls))
        self.assertEqual(calls[0]["semantic_memories"], retrieved)
        self.assertEqual(calls[1]["semantic_memories"], retrieved)
        self.assertEqual(MAX_NPC_MESSAGES_PER_REQUEST, 1)

    def test_relationship_aware_planner_request_uses_existing_executor(self) -> None:
        state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 100,
            "gold": 50,
            "game_time": "08:00",
        }
        relationship = RelationshipEvaluation("session-a", "ravi", "FRIENDLY", (), ())
        calls = []
        with (
            patch.object(main, "get_conversation_history", return_value=[]),
            patch.object(main, "get_player_state", return_value=state),
            patch.object(main, "get_inventory", return_value=[]),
            patch.object(main, "search_memories", return_value=[{"text": "private", "kind": "fact"}]),
            patch.object(main, "evaluate_player_relationship", return_value=relationship),
            patch.object(main, "execute_npc_plan", return_value={
                "status": "completed", "completed_steps": 3, "total_steps": 3,
            }) as execute,
            patch.object(main, "generate_npc_reply", side_effect=self._capture_reply(calls)),
            patch.object(main, "save_message"),
            patch.object(main, "remember_user_facts", return_value=0),
            patch.object(main, "learn_from_player_interaction", return_value=0),
        ):
            response = main.chat(main.ChatRequest(
                session_id="session-a",
                npc_id="ravi",
                message="I want to know about swords.",
            ))

        self.assertEqual(response.message, "A brief in-character reply.")
        execute.assert_called_once()
        self.assertEqual(calls[0]["dialogue_decision"].decision, "provide_validated_information")
        self.assertEqual(calls[0]["dialogue_decision"].topic, "swords")
        self.assertTrue(calls[0]["validated_plan_completed"])
        self.assertIsNone(calls[0]["plan_context"])
        self.assertEqual(calls[0]["semantic_memories"], [])
        self.assertEqual(calls[0]["history"], [])

    def test_dialogue_selection_does_not_mutate_game_state(self) -> None:
        before = database.get_player_state("session-a")
        database.add_inventory_item("session-a", "herb", 2)
        before_inventory = database.get_inventory("session-a")
        with patch.object(main, "generate_npc_reply", return_value="Welcome, traveler."):
            response = main.chat(
                main.ChatRequest(session_id="session-a", npc_id="ravi", message="Hello Ravi.")
            )

        self.assertEqual(response.model_dump(), {"message": "Welcome, traveler."})
        self.assertEqual(database.get_player_state("session-a"), before)
        self.assertEqual(database.get_inventory("session-a"), before_inventory)

    def test_gemini_receives_only_validated_dialogue_context_and_has_no_tools(self) -> None:
        fake_models = SimpleNamespace(
            generate_content=Mock(
                return_value=SimpleNamespace(
                    function_calls=[],
                    text="Ravi welcomes the traveler warmly.",
                )
            )
        )
        decision = NPCDialogueDecision("session-a", "ravi", "friendly_greeting")
        history = [{"role": "user", "content": "HISTORY_PRIVATE_SENTINEL"}]
        memories = [{"text": "CHROMA_PRIVATE_SENTINEL", "kind": "fact"}]
        state = {
            "session_id": "session-a",
            "location": "STATE_PRIVATE_SENTINEL",
            "health": 51,
            "gold": 73,
            "game_time": "09:30",
        }
        inventory = [{"item_name": "INVENTORY_PRIVATE_SENTINEL", "quantity": 1}]

        with patch.object(
            gemini_service,
            "_get_client",
            return_value=SimpleNamespace(models=fake_models),
        ):
            reply = gemini_service.generate_npc_reply(
                "Hello Ravi.",
                history,
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=state,
                inventory=inventory,
                semantic_memories=memories,
                dialogue_decision=decision,
            )

        request = fake_models.generate_content.call_args.kwargs
        context = request["config"].system_instruction + " ".join(
            part.text
            for content in request["contents"]
            for part in content.parts
            if part.text
        )
        self.assertEqual(reply, "Ravi welcomes the traveler warmly.")
        self.assertEqual(request["model"], gemini_service.MODEL_ID)
        self.assertIsNone(request["config"].tools)
        self.assertIn("friendly_greeting", context)
        for private_value in (
            "HISTORY_PRIVATE_SENTINEL",
            "CHROMA_PRIVATE_SENTINEL",
            "STATE_PRIVATE_SENTINEL",
            "INVENTORY_PRIVATE_SENTINEL",
            "TRUSTING",
            "FRIENDLY",
            "NEUTRAL",
        ):
            self.assertNotIn(private_value, context)

    def test_gemini_tool_request_cannot_mutate_state_in_dialogue_mode(self) -> None:
        fake_models = SimpleNamespace(
            generate_content=Mock(
                return_value=SimpleNamespace(
                    function_calls=[SimpleNamespace(name="heal_player", args={"amount": 50})],
                    text=None,
                )
            )
        )
        with (
            patch.object(
                gemini_service, "_get_client", return_value=SimpleNamespace(models=fake_models)
            ),
            patch.object(gemini_service, "execute_game_action") as execute_action,
            patch.object(database, "apply_healing") as apply_healing,
        ):
            with self.assertRaises(gemini_service.GeminiServiceError):
                gemini_service.generate_npc_reply(
                    "Hello Ravi.",
                    [],
                    npc=get_npc("ravi"),
                    session_id="session-a",
                    game_state=database.get_player_state("session-a"),
                    inventory=[],
                    dialogue_decision=NPCDialogueDecision(
                        "session-a", "ravi", "neutral_greeting"
                    ),
                )

        execute_action.assert_not_called()
        apply_healing.assert_not_called()


if __name__ == "__main__":
    unittest.main()