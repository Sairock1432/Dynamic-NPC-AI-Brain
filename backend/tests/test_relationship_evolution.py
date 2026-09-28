import tempfile
import sqlite3
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from app import main
from app.memory import database
from app.npcs import get_npc
from app.services import gemini_service, npc_relationship_progression
from app.services.npc_decision import select_npc_dialogue
from app.services.npc_relationship_progression import (
    FRIENDLY_RELATIONSHIP_THRESHOLD,
    INITIAL_RELATIONSHIP_SCORE,
    NPCRelationshipError,
    TRUSTING_RELATIONSHIP_THRESHOLD,
    apply_player_relationship_events,
    classify_player_relationship_events,
    evaluate_player_relationship,
)


class RelationshipEvolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database_patcher = patch.object(database, "DATABASE_PATH", root / "state.db")
        self.database_patcher.start()
        database.init_db()
        database.create_default_player_state("session-a")
        self.memory_patcher = patch.object(
            npc_relationship_progression,
            "get_npc_learning_memories",
            return_value=[],
        )
        self.search_patcher = patch.object(main, "search_memories", return_value=[])
        self.remember_patcher = patch.object(main, "remember_user_facts", return_value=0)
        self.learning_patcher = patch.object(main, "learn_from_player_interaction", return_value=0)
        for test_patch in (
            self.memory_patcher,
            self.search_patcher,
            self.remember_patcher,
            self.learning_patcher,
        ):
            test_patch.start()
            self.addCleanup(test_patch.stop)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        self.database_patcher.stop()
        self.temp_dir.cleanup()

    def _apply_event(
        self,
        event: str,
        *,
        session_id: str = "session-a",
        npc_id: str = "ravi",
        message: str = "An interaction occurred.",
    ) -> int:
        database.create_default_player_state(session_id)
        relationship = evaluate_player_relationship(session_id, npc_id)
        message_id = database.save_message(session_id, "user", message, npc_id=npc_id)
        return apply_player_relationship_events(
            session_id,
            npc_id,
            (event,),
            source_message_id=message_id,
            initial_score=relationship.trust_score,
        )

    def test_initial_relationship_is_neutral_and_not_persisted_until_an_event(self) -> None:
        relationship = evaluate_player_relationship("session-a", "ravi")

        self.assertEqual(relationship.trust_score, INITIAL_RELATIONSHIP_SCORE)
        self.assertEqual(relationship.stage, "NEUTRAL")
        self.assertIsNone(database.get_npc_player_relationship_state("session-a", "ravi"))

    def test_positive_player_interaction_increases_persisted_relationship(self) -> None:
        events = classify_player_relationship_events("Thanks, that helped.")
        self.assertEqual(events, ("successful_help",))
        score = self._apply_event(events[0], message="Thanks, that helped.")
        relationship = evaluate_player_relationship("session-a", "ravi")

        self.assertEqual(score, FRIENDLY_RELATIONSHIP_THRESHOLD)
        self.assertEqual(relationship.stage, "FRIENDLY")
        self.assertEqual(relationship.trust_score, score)

    def test_negative_interaction_decreases_score_and_can_regress_stage(self) -> None:
        self._apply_event("successful_help", message="Thanks, that helped.")
        self._apply_event("successful_help", message="Your advice helped, thanks.")
        previous_score = evaluate_player_relationship("session-a", "ravi").trust_score

        negative_events = classify_player_relationship_events("You are useless.")
        self.assertEqual(negative_events, ("disrespectful_interaction",))
        database.create_default_player_state("session-a")
        message_id = database.save_message(
            "session-a", "user", "You are useless.", npc_id="ravi"
        )
        lowered_score = apply_player_relationship_events(
            "session-a",
            "ravi",
            negative_events,
            source_message_id=message_id,
            initial_score=previous_score,
        )

        self.assertLess(lowered_score, previous_score)
        self.assertEqual(evaluate_player_relationship("session-a", "ravi").stage, "NEUTRAL")

    def test_thresholds_progress_and_regress_relationship_stages(self) -> None:
        self._apply_event("successful_help", message="First thanks.")
        friendly = evaluate_player_relationship("session-a", "ravi")
        self._apply_event("successful_help", message="Second thanks.")
        still_friendly = evaluate_player_relationship("session-a", "ravi")
        self._apply_event("successful_help", message="Third thanks.")
        trusting = evaluate_player_relationship("session-a", "ravi")

        self.assertEqual(friendly.trust_score, FRIENDLY_RELATIONSHIP_THRESHOLD)
        self.assertEqual(friendly.stage, "FRIENDLY")
        self.assertEqual(still_friendly.stage, "FRIENDLY")
        self.assertEqual(trusting.trust_score, TRUSTING_RELATIONSHIP_THRESHOLD)
        self.assertEqual(trusting.stage, "TRUSTING")

        self._apply_event("disrespectful_interaction", message="I hate you.")
        self.assertEqual(evaluate_player_relationship("session-a", "ravi").stage, "FRIENDLY")

    def test_repeated_message_replay_is_idempotent_but_a_new_turn_counts(self) -> None:
        message_id = database.save_message(
            "session-a", "user", "Thanks, that helped.", npc_id="ravi"
        )
        starting_score = evaluate_player_relationship("session-a", "ravi").trust_score
        first = apply_player_relationship_events(
            "session-a", "ravi", ("successful_help",),
            source_message_id=message_id, initial_score=starting_score,
        )
        replay = apply_player_relationship_events(
            "session-a", "ravi", ("successful_help",),
            source_message_id=message_id, initial_score=starting_score,
        )
        next_message_id = database.save_message(
            "session-a", "user", "Thanks, that helped.", npc_id="ravi"
        )
        next_turn = apply_player_relationship_events(
            "session-a", "ravi", ("successful_help",),
            source_message_id=next_message_id, initial_score=starting_score,
        )

        self.assertEqual(first, replay)
        self.assertGreater(next_turn, replay)

    def test_relationship_persists_across_database_connections(self) -> None:
        self._apply_event("gifted_item", message="I give you a gift.")
        saved = database.get_npc_player_relationship_state("session-a", "ravi")
        loaded = evaluate_player_relationship("session-a", "ravi")

        self.assertIsNotNone(saved)
        self.assertEqual(saved["trust_score"], 40)
        self.assertEqual(loaded.trust_score, 40)
        self.assertEqual(loaded.stage, "FRIENDLY")

    def test_relationship_state_is_isolated_by_npc_and_session(self) -> None:
        self._apply_event("successful_help", message="Thanks, that helped.")

        ravi_a = evaluate_player_relationship("session-a", "ravi")
        meera_a = evaluate_player_relationship("session-a", "meera")
        ravi_b = evaluate_player_relationship("session-b", "ravi")

        self.assertEqual(ravi_a.stage, "FRIENDLY")
        self.assertEqual(meera_a.stage, "NEUTRAL")
        self.assertEqual(ravi_b.stage, "NEUTRAL")
        self.assertIsNone(database.get_npc_player_relationship_state("session-a", "meera"))
        self.assertIsNone(database.get_npc_player_relationship_state("session-b", "ravi"))

    def test_invalid_events_scope_and_cross_npc_message_are_rejected(self) -> None:
        ravi_message_id = database.save_message(
            "session-a", "user", "Thanks, that helped.", npc_id="ravi"
        )
        with self.assertRaises(NPCRelationshipError):
            apply_player_relationship_events(
                "session-a", "ravi", ("set_trust_to_100",),
                source_message_id=ravi_message_id, initial_score=20,
            )
        with self.assertRaises(NPCRelationshipError):
            apply_player_relationship_events(
                "session-a", "ravi", ["successful_help"],
                source_message_id=ravi_message_id, initial_score=20,
            )
        with self.assertRaises(NPCRelationshipError):
            apply_player_relationship_events(
                "session-a", "unknown", ("successful_help",),
                source_message_id=ravi_message_id, initial_score=20,
            )

        meera_message_id = database.save_message(
            "session-a", "user", "Thanks, that helped.", npc_id="meera"
        )
        with self.assertRaises(NPCRelationshipError):
            apply_player_relationship_events(
                "session-a", "ravi", ("successful_help",),
                source_message_id=meera_message_id, initial_score=20,
            )
        self.assertIsNone(database.get_npc_player_relationship_state("session-a", "ravi"))

    def test_existing_step_18_learning_seeds_dialogue_stage(self) -> None:
        validated_memory = {
            "text": "The player has accepted help from Ravi.",
            "kind": "fact",
            "category": "PLAYER_ACCEPTED_HELP",
        }
        with patch.object(
            npc_relationship_progression,
            "get_npc_learning_memories",
            return_value=[validated_memory],
        ):
            relationship = evaluate_player_relationship("session-a", "ravi")
        decision = select_npc_dialogue(
            get_npc("ravi"),
            relationship,
            session_id="session-a",
            message="Hello Ravi.",
            intent=None,
        )

        self.assertEqual(relationship.stage, "FRIENDLY")
        self.assertEqual(decision.decision, "friendly_greeting")

    def test_persisted_score_drives_step_18_dialogue_and_regresses(self) -> None:
        self._apply_event("successful_help", message="First thanks.")
        self._apply_event("successful_help", message="Second thanks.")
        self._apply_event("successful_help", message="Third thanks.")
        trusting = evaluate_player_relationship("session-a", "ravi")
        trusting_greeting = select_npc_dialogue(
            get_npc("ravi"), trusting,
            session_id="session-a", message="Hello Ravi.", intent=None,
        )
        self._apply_event("disrespectful_interaction", message="I hate you.")
        friendly = evaluate_player_relationship("session-a", "ravi")
        friendly_greeting = select_npc_dialogue(
            get_npc("ravi"), friendly,
            session_id="session-a", message="Hello Ravi.", intent=None,
        )

        self.assertEqual(trusting_greeting.decision, "trusting_greeting")
        self.assertEqual(friendly.stage, "FRIENDLY")
        self.assertEqual(friendly_greeting.decision, "friendly_greeting")

    def test_relationship_updates_do_not_mutate_player_state_or_inventory(self) -> None:
        database.update_player_state("session-a", health=67, gold=31, game_time="10:15")
        database.add_inventory_item("session-a", "iron", 3)
        state_before = database.get_player_state("session-a")
        inventory_before = database.get_inventory("session-a")

        self._apply_event("successful_help", message="Thanks, that helped.")

        self.assertEqual(database.get_player_state("session-a"), state_before)
        self.assertEqual(database.get_inventory("session-a"), inventory_before)

    def test_chat_persists_only_classified_successful_interactions(self) -> None:
        with patch.object(main, "generate_npc_reply", return_value="I am glad I could help."):
            main.chat(main.ChatRequest(
                session_id="session-a", npc_id="ravi", message="Thanks, that helped."
            ))
            main.chat(main.ChatRequest(
                session_id="session-a", npc_id="ravi", message="You are useless."
            ))

        relationship = evaluate_player_relationship("session-a", "ravi")
        self.assertEqual(relationship.trust_score, 15)
        self.assertEqual(relationship.stage, "NEUTRAL")
        self.assertEqual(
            database.get_npc_player_relationship_state("session-a", "ravi")[
                "last_processed_message_id"
            ],
            3,
        )

    def test_relationship_storage_failure_does_not_fail_a_saved_chat(self) -> None:
        with (
            patch.object(main, "generate_npc_reply", return_value="I hear you."),
            patch.object(
                main,
                "apply_player_relationship_events",
                side_effect=sqlite3.OperationalError("database is temporarily locked"),
            ),
        ):
            response = main.chat(main.ChatRequest(
                session_id="session-a",
                npc_id="ravi",
                message="Thanks, that helped.",
            ))

        self.assertEqual(response.message, "I hear you.")
        self.assertEqual(
            len(database.get_conversation_history("session-a", npc_id="ravi")),
            2,
        )

    def test_successful_planner_help_persists_for_the_selected_npc(self) -> None:
        state = database.get_player_state("session-a")
        with (
            patch.object(main, "execute_npc_plan", return_value={
                "status": "completed", "completed_steps": 3, "total_steps": 3,
            }) as execute,
            patch.object(main, "generate_npc_reply", return_value="I can help with that.") ,
        ):
            response = main.chat(main.ChatRequest(
                session_id="session-a",
                npc_id="ravi",
                message="Can you help me?",
            ))

        self.assertEqual(response.message, "I can help with that.")
        execute.assert_called_once()
        relationship = evaluate_player_relationship("session-a", "ravi")
        self.assertEqual(relationship.trust_score, 40)
        self.assertEqual(relationship.stage, "FRIENDLY")
        self.assertEqual(database.get_player_state("session-a"), state)

    def test_only_successful_existing_game_actions_emit_relationship_events(self) -> None:
        callback_events = []
        state = database.get_player_state("session-a")
        inventory = database.get_inventory("session-a")
        tool_response = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    name="give_item",
                    args={"item_name": "iron", "quantity": 1},
                )
            ],
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="give item")],
                    )
                )
            ],
            text=None,
        )
        final_response = SimpleNamespace(function_calls=[], text="Thank you for the iron.")
        fake_models = SimpleNamespace(
            generate_content=Mock(side_effect=[tool_response, final_response])
        )

        with (
            patch.object(
                gemini_service,
                "_get_client",
                return_value=SimpleNamespace(models=fake_models),
            ),
            patch.object(
                gemini_service,
                "execute_game_action",
                return_value={"success": True, "message": "Gave 1 iron."},
            ),
        ):
            reply = gemini_service.generate_npc_reply(
                "Give iron.",
                [],
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=state,
                inventory=inventory,
                relationship_event_callback=callback_events.append,
            )

        self.assertEqual(reply, "Thank you for the iron.")
        self.assertEqual(callback_events, ["gifted_item"])

        callback_events.clear()
        fake_models.generate_content.side_effect = [tool_response, final_response]
        with (
            patch.object(
                gemini_service,
                "_get_client",
                return_value=SimpleNamespace(models=fake_models),
            ),
            patch.object(
                gemini_service,
                "execute_game_action",
                return_value={"success": False, "message": "You do not have that item."},
            ),
        ):
            gemini_service.generate_npc_reply(
                "Give iron.",
                [],
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=state,
                inventory=inventory,
                relationship_event_callback=callback_events.append,
            )
        self.assertEqual(callback_events, [])

    def test_chat_persists_a_successful_gift_for_the_selected_npc(self) -> None:
        def reply_with_successful_gift(message, history, **kwargs):
            kwargs["relationship_event_callback"]("gifted_item")
            return "Thank you for the iron."

        with patch.object(
            main,
            "generate_npc_reply",
            side_effect=reply_with_successful_gift,
        ):
            response = main.chat(main.ChatRequest(
                session_id="session-a",
                npc_id="ravi",
                message="Give 1 iron.",
            ))

        relationship = evaluate_player_relationship("session-a", "ravi")
        self.assertEqual(response.message, "Thank you for the iron.")
        self.assertEqual(relationship.trust_score, 40)
        self.assertEqual(relationship.stage, "FRIENDLY")
        self.assertIsNone(database.get_npc_player_relationship_state("session-a", "meera"))

    def test_positive_signal_classifier_does_not_promote_unrelated_text(self) -> None:
        self.assertEqual(classify_player_relationship_events("Hello Ravi."), ())
        self.assertEqual(classify_player_relationship_events("I like swords."), ())
        self.assertEqual(
            classify_player_relationship_events("Thanks again, that helped."),
            ("successful_help",),
        )


if __name__ == "__main__":
    unittest.main()