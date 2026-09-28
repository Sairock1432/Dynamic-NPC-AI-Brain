import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app import main
from app.memory import database
from app.npcs import get_npc
from app.services import npc_behavior, npc_learning, semantic_memory
from app.services.npc_decision import decide_npc_action
from app.services.npc_learning import (
    MAX_NPC_LEARNING_MEMORIES_PER_INTERACTION,
    NPCLearningError,
    learn_from_player_interaction,
    transition_relationship_stage,
    validate_memory_category,
)
from app.services.npc_communication import request_npc_message


class NPCLearningTests(unittest.TestCase):
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

    def test_memory_categories_are_allowlisted_and_sensitive_text_is_rejected(self) -> None:
        allowed = {
            "PLAYER_PREFERENCE",
            "PLAYER_INTEREST",
            "NPC_INTERACTION_PATTERN",
            "PLAYER_ACCEPTED_HELP",
        }
        self.assertEqual(semantic_memory.MEMORY_CATEGORIES, allowed)
        for category in allowed:
            self.assertEqual(validate_memory_category(category), category)
        with self.assertRaises(NPCLearningError):
            validate_memory_category("PRIVATE_SECRET")

        self.assertEqual(learn_from_player_interaction("session-a", "ravi", "My password is hidden."), 0)
        self.assertEqual(
            learn_from_player_interaction(
                "session-a", "ravi", "I prefer swords; my password is hidden."
            ),
            0,
        )
        self.assertEqual(
            learn_from_player_interaction(
                "session-a", "ravi", "I prefer swords, and my home address is 12 Oak Lane."
            ),
            0,
        )
        self.assertEqual(
            learn_from_player_interaction(
                "session-a", "ravi", "I prefer swords, and my religious beliefs are private."
            ),
            0,
        )
        with self.assertRaises(ValueError):
            semantic_memory.add_learning_memory(
                "session-a", "Arbitrary user-controlled memory.", "PLAYER_PREFERENCE", npc_id="ravi"
            )
        self.assertEqual(semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi"), [])

    def test_ravi_and_meera_learn_only_their_own_safe_preferences(self) -> None:
        self.assertEqual(learn_from_player_interaction("session-a", "ravi", "I prefer swords."), 1)
        self.assertEqual(learn_from_player_interaction("session-a", "meera", "I usually use herbs."), 1)
        ravi_memories = semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")
        meera_memories = semantic_memory.get_npc_learning_memories("session-a", npc_id="meera")

        self.assertEqual(ravi_memories[0]["category"], "PLAYER_PREFERENCE")
        self.assertIn("swords", ravi_memories[0]["text"])
        self.assertEqual(meera_memories[0]["category"], "PLAYER_PREFERENCE")
        self.assertIn("herbs", meera_memories[0]["text"])
        self.assertFalse(any("herbs" in item["text"] for item in ravi_memories))
        self.assertFalse(any("swords" in item["text"] for item in meera_memories))

    def test_learning_isolated_by_session_and_npc(self) -> None:
        learn_from_player_interaction("session-a", "ravi", "I prefer swords.")
        learn_from_player_interaction("session-a", "meera", "I prefer herbs.")
        learn_from_player_interaction("session-b", "ravi", "I prefer shields.")

        self.assertIn(
            "swords",
            semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")[0]["text"],
        )
        self.assertIn(
            "herbs",
            semantic_memory.get_npc_learning_memories("session-a", npc_id="meera")[0]["text"],
        )
        self.assertIn(
            "shields",
            semantic_memory.get_npc_learning_memories("session-b", npc_id="ravi")[0]["text"],
        )
        self.assertEqual(semantic_memory.get_npc_learning_memories("session-b", npc_id="meera"), [])

    def test_duplicate_prevention_upgrades_existing_fact_and_caps_per_interaction(self) -> None:
        semantic_memory.remember_user_facts("session-a", "I prefer swords.", npc_id="ravi")
        self.assertEqual(learn_from_player_interaction("session-a", "ravi", "I prefer swords."), 1)
        self.assertEqual(learn_from_player_interaction("session-a", "ravi", "I prefer swords."), 0)
        memories = semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")
        self.assertEqual(len(memories), 1)
        self.assertEqual(MAX_NPC_LEARNING_MEMORIES_PER_INTERACTION, 1)

        self.assertEqual(
            learn_from_player_interaction(
                "session-a", "ravi", "I prefer shields and I am interested in metal."
            ),
            1,
        )
        self.assertLessEqual(
            len(semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")),
            2,
        )

    def test_total_learning_memory_count_is_bounded(self) -> None:
        with patch.object(semantic_memory, "MAX_NPC_LEARNING_MEMORIES", 2):
            self.assertTrue(
                semantic_memory.add_learning_memory(
                    "session-a", "The player says they prefer swords.", "PLAYER_PREFERENCE", npc_id="ravi"
                )
            )
            self.assertTrue(
                semantic_memory.add_learning_memory(
                    "session-a", "The player is interested in shields.", "PLAYER_INTEREST", npc_id="ravi"
                )
            )
            self.assertFalse(
                semantic_memory.add_learning_memory(
                    "session-a", "The player says they prefer herbs.", "PLAYER_PREFERENCE", npc_id="ravi"
                )
            )
        self.assertEqual(len(semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")), 2)

    def test_explicit_accepted_help_and_repeated_questions_are_categorized(self) -> None:
        self.assertEqual(
            learn_from_player_interaction("session-a", "meera", "Thanks, that helped."),
            1,
        )
        accepted = semantic_memory.get_npc_learning_memories("session-a", npc_id="meera")[0]
        self.assertEqual(accepted["category"], "PLAYER_ACCEPTED_HELP")

        database.save_message("session-a", "user", "Tell me about metal?")
        database.save_message("session-a", "user", "How does iron work?")
        self.assertEqual(
            learn_from_player_interaction("session-a", "ravi", "Tell me more about metal?"),
            1,
        )
        pattern = semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")[0]
        self.assertEqual(pattern["category"], "NPC_INTERACTION_PATTERN")
        self.assertIn("frequently asks Ravi about metal", pattern["text"])

    def test_learning_runs_only_after_successful_chat(self) -> None:
        with patch.object(main, "generate_npc_reply", return_value="I can help with that."):
            main.chat(
                main.ChatRequest(
                    session_id="session-a", npc_id="ravi", message="I prefer swords."
                )
            )
        self.assertEqual(len(semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")), 1)

        with (
            patch.object(main, "generate_npc_reply", side_effect=main.GeminiServiceError("failed")),
            self.assertRaises(HTTPException),
        ):
            main.chat(
                main.ChatRequest(
                    session_id="session-failed", npc_id="ravi", message="I prefer shields."
                )
            )
        self.assertEqual(
            semantic_memory.get_npc_learning_memories("session-failed", npc_id="ravi"),
            [],
        )

    def test_learning_selects_only_allowlisted_decisions_and_does_not_mutate_state(self) -> None:
        learn_from_player_interaction("session-a", "ravi", "I usually buy swords.")
        learn_from_player_interaction("session-a", "meera", "I prefer herbs.")
        ravi_memory = semantic_memory.get_npc_learning_memories("session-a", npc_id="ravi")
        meera_memory = semantic_memory.get_npc_learning_memories("session-a", npc_id="meera")
        state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 82,
            "gold": 50,
            "game_time": "08:00",
        }
        ravi_decision = decide_npc_action(
            get_npc("ravi"), state, [], event_name="morning", schedule_location="At the forge",
            learning_memories=ravi_memory,
        )
        meera_decision = decide_npc_action(
            get_npc("meera"), state, [], event_name="morning", schedule_location="At the healer hut",
            learning_memories=meera_memory,
        )

        self.assertEqual((ravi_decision.decision, ravi_decision.learning_topic), (
            "offer_sword_information", "swords"
        ))
        self.assertEqual((meera_decision.decision, meera_decision.learning_topic), (
            "offer_herb_information", "herbs"
        ))
        self.assertEqual(state, {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 82,
            "gold": 50,
            "game_time": "08:00",
        })

    def test_event_and_communication_do_not_receive_another_npcs_private_memory(self) -> None:
        learn_from_player_interaction("session-a", "ravi", "I prefer swords.")
        learn_from_player_interaction("session-a", "meera", "I prefer herbs.")
        state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 100,
            "gold": 50,
            "game_time": "08:00",
        }
        inventory = []
        with (
            patch.object(npc_behavior, "get_player_state", return_value=state),
            patch.object(npc_behavior, "get_inventory", return_value=inventory),
            patch.object(npc_behavior, "generate_npc_event_reaction", return_value="Ravi offers sword advice.") as react,
        ):
            result = npc_behavior.dispatch_world_event("session-a", "morning")

        ravi_call = next(call for call in react.call_args_list if call.kwargs["npc"].id == "ravi")
        self.assertEqual(ravi_call.kwargs["decision"].decision, "offer_sword_information")
        meera_call = next(call for call in react.call_args_list if call.kwargs["npc"].id == "meera")
        self.assertEqual(meera_call.kwargs["decision"].decision, "offer_herb_information")
        self.assertEqual(result["action_count"], 2)

        with patch(
            "app.services.gemini_service.generate_peer_npc_reply", return_value="Use a cool cloth."
        ) as peer_reply:
            request_npc_message("ravi", "meera", "What care should I suggest?", "session-a")
        self.assertNotIn("semantic_memories", peer_reply.call_args.kwargs)
        self.assertNotIn("conversation_history", peer_reply.call_args.kwargs)
        self.assertNotIn("swords", peer_reply.call_args.kwargs["message"])

    def test_relationship_transitions_are_finite_and_validated(self) -> None:
        self.assertEqual(transition_relationship_stage("NEUTRAL", "successful_help"), "FRIENDLY")
        self.assertEqual(
            transition_relationship_stage("FRIENDLY", "repeated_positive_interaction"),
            "TRUSTING",
        )
        self.assertEqual(
            transition_relationship_stage("NEUTRAL", "repeated_positive_interaction"),
            "FRIENDLY",
        )
        with self.assertRaises(NPCLearningError):
            transition_relationship_stage("TRUSTING", "set_trust_to_999")
        with self.assertRaises(NPCLearningError):
            transition_relationship_stage("UNKNOWN", "successful_help")
        with self.assertRaises(NPCLearningError):
            transition_relationship_stage([], "successful_help")


if __name__ == "__main__":
    unittest.main()
