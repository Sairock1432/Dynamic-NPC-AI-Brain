import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.memory import database
from app.npcs import get_npc
from app.services import semantic_memory
from app.services.gemini_service import generate_npc_event_reaction
from app.services.npc_decision import decide_npc_action, select_npc_dialogue
from app.services.npc_behavior import LOW_HEALTH_THRESHOLD, dispatch_world_event
from app.services.npc_relationship_progression import evaluate_player_relationship
from app.services.npc_schedules import get_npc_schedule_location


@unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY is not configured")
class LiveGeminiToolCallingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.database_path_patch = patch.object(
            database,
            "DATABASE_PATH",
            Path(cls.temp_dir.name) / "live-gemini-test.db",
        )
        cls.database_path_patch.start()
        cls.chroma_path_patch = patch.object(
            semantic_memory,
            "CHROMA_PATH",
            Path(cls.temp_dir.name) / "chroma",
        )
        cls.chroma_path_patch.start()
        semantic_memory._get_collection.cache_clear()
        semantic_memory._get_client.cache_clear()
        database.init_db()

    @classmethod
    def tearDownClass(cls) -> None:
        semantic_memory.close_client()
        cls.chroma_path_patch.stop()
        cls.database_path_patch.stop()
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        self.session_id = f"session-live-{uuid.uuid4().hex}"
        database.create_default_player_state(self.session_id)

    def test_buy_iron_tool_round_trip_and_persistence(self) -> None:
        with TestClient(app) as client:
            with self.assertLogs("app.services.gemini_service", level="INFO") as logs:
                response = client.post(
                    "/chat",
                    json={
                        "session_id": self.session_id,
                        "npc_id": "ravi",
                        "message": "Buy 1 iron.",
                    },
                )

        self.assertEqual(response.status_code, 200)
        final_message = response.json()["message"]
        self.assertTrue(final_message.strip())
        self.assertIn("iron", final_message.lower())

        state = database.get_player_state(self.session_id)
        inventory = database.get_inventory(self.session_id)
        history = database.get_conversation_history(self.session_id)
        self.assertEqual(state["gold"], 40)
        self.assertEqual(inventory, [{"item_name": "iron", "quantity": 1}])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "Buy 1 iron.")
        self.assertEqual(history[1]["content"], final_message)

        log_output = "\n".join(logs.output)
        self.assertIn("Gemini requested allowlisted game action: buy_item", log_output)
        self.assertIn("Executed allowlisted game action: buy_item success=True", log_output)
        self.assertIn("Returned game action result to Gemini: buy_item", log_output)
        self.assertNotIn("Gemini follow-up failed after a game action", log_output, log_output)

    def test_insufficient_gold_tool_result_is_returned_without_state_change(self) -> None:
        database.update_player_state(self.session_id, gold=20)

        with TestClient(app) as client:
            with self.assertLogs("app.services.gemini_service", level="INFO") as logs:
                response = client.post(
                    "/chat",
                    json={
                        "session_id": self.session_id,
                        "npc_id": "ravi",
                        "message": "Buy 100 swords.",
                    },
                )

        self.assertEqual(response.status_code, 200)
        final_message = response.json()["message"]
        self.assertTrue(final_message.strip())
        self.assertNotIn("bought 100 swords", final_message.lower())
        self.assertEqual(database.get_player_state(self.session_id)["gold"], 20)
        self.assertEqual(database.get_inventory(self.session_id), [])

        history = database.get_conversation_history(self.session_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["content"], final_message)
        log_output = "\n".join(logs.output)
        self.assertIn("Gemini requested allowlisted game action: buy_item", log_output)
        self.assertIn("Executed allowlisted game action: buy_item success=False", log_output)
        self.assertIn("Returned game action result to Gemini: buy_item", log_output)
        self.assertNotIn("Gemini follow-up failed after a game action", log_output, log_output)

    def test_live_learning_personalizes_bounded_events_without_memory_leakage(self) -> None:
        database.update_player_state(
            self.session_id,
            game_time="08:00",
        )
        with TestClient(app) as client:
            ravi_preference = client.post(
                "/chat",
                json={
                    "session_id": self.session_id,
                    "npc_id": "ravi",
                    "message": (
                        "Hello Ravi, who are you? I want to know about swords. "
                        "My name is Rowan. I prefer swords."
                    ),
                },
            )
            meera_preference = client.post(
                "/chat",
                json={
                    "session_id": self.session_id,
                    "npc_id": "meera",
                    "message": (
                        "Hello Meera, who are you? Please tell me about herbs. "
                        "I prefer herbs. Do you know my name?"
                    ),
                },
            )

        self.assertEqual(ravi_preference.status_code, 200)
        self.assertEqual(meera_preference.status_code, 200)
        ravi_reply = ravi_preference.json()["message"].lower()
        meera_reply = meera_preference.json()["message"].lower()
        self.assertTrue(any(word in ravi_reply for word in ("blacksmith", "forge", "anvil")))
        self.assertTrue(any(word in meera_reply for word in ("healer", "healing", "herb")))
        self.assertIn("sword", ravi_reply)
        self.assertIn("herb", meera_reply)
        self.assertNotIn("rowan", meera_reply)
        ravi_memories = semantic_memory.get_npc_learning_memories(
            self.session_id, npc_id="ravi"
        )
        meera_memories = semantic_memory.get_npc_learning_memories(
            self.session_id, npc_id="meera"
        )
        self.assertTrue(any("swords" in memory["text"] for memory in ravi_memories))
        self.assertFalse(any("herbs" in memory["text"] for memory in ravi_memories))
        self.assertTrue(any("herbs" in memory["text"] for memory in meera_memories))
        self.assertFalse(any("swords" in memory["text"] for memory in meera_memories))

        morning = dispatch_world_event(self.session_id, "morning")
        decisions = {item["npc_id"]: item for item in morning["actions"]}
        self.assertEqual(decisions["ravi"]["decision"], "offer_sword_information")
        self.assertEqual(decisions["meera"]["decision"], "offer_herb_information")
        self.assertEqual(decisions["ravi"]["movement"]["destination"], "forge")
        self.assertEqual(decisions["meera"]["movement"]["destination"], "healer_hut")
        self.assertIn("forge", decisions["ravi"]["reaction"].lower())
        self.assertIn("sword", decisions["ravi"]["reaction"].lower())
        self.assertIn("herb", decisions["meera"]["reaction"].lower())
        self.assertLessEqual(morning["action_count"], 2)

        state = database.get_player_state(self.session_id)
        self.assertEqual(state["health"], 100)
        self.assertEqual(state["gold"], 50)
        self.assertEqual(state["location"], "Mystery Village")
        self.assertEqual(state["game_time"], "08:00")
        self.assertEqual(database.get_inventory(self.session_id), [])

    def test_live_gemini_narrates_backend_selected_trusting_relationship_decision(self) -> None:
        semantic_memory.add_learning_memory(
            self.session_id,
            "The player has accepted help from Ravi.",
            "PLAYER_ACCEPTED_HELP",
            npc_id="ravi",
        )
        semantic_memory.add_learning_memory(
            self.session_id,
            "The player frequently asks Ravi about metal.",
            "NPC_INTERACTION_PATTERN",
            npc_id="ravi",
        )
        database.save_message(self.session_id, "user", "Thanks, that helped.", npc_id="ravi")
        database.save_message(self.session_id, "assistant", "Glad I could help.", npc_id="ravi")
        database.save_message(
            self.session_id,
            "user",
            "Your advice helped, thanks.",
            npc_id="ravi",
        )
        npc = get_npc("ravi")
        state = database.get_player_state(self.session_id)
        inventory = database.get_inventory(self.session_id)
        relationship = evaluate_player_relationship(self.session_id, npc.id)
        decision = decide_npc_action(
            npc,
            state,
            inventory,
            event_name="morning",
            schedule_location=get_npc_schedule_location(npc.id, state["game_time"]),
            relationship_evaluation=relationship,
        )

        self.assertEqual(relationship.stage, "TRUSTING")
        self.assertEqual(decision.relationship_decision, "acknowledge_trust")
        response = generate_npc_event_reaction(
            npc=npc,
            event_name="morning",
            behavior="Open the blacksmith shop and prepare the forge.",
            schedule_location=decision.schedule_location,
            game_state=state,
            inventory=inventory,
            decision=decision,
        )

        self.assertTrue(response.strip())
        self.assertEqual(decision.relationship_decision, "acknowledge_trust")

    def test_live_chat_narrates_same_npc_across_relationship_stages_without_leaks(self) -> None:
        semantic_memory.add_memory(
            self.session_id,
            "The player's stated name is MeeraPrivateLiveToken.",
            {"kind": "name"},
            npc_id="meera",
        )
        npc = get_npc("ravi")
        greeting = "Hello Ravi."
        decisions = []
        responses = []

        neutral = evaluate_player_relationship(self.session_id, npc.id)
        neutral_decision = select_npc_dialogue(
            npc,
            neutral,
            session_id=self.session_id,
            message=greeting,
            intent=None,
        )
        decisions.append(neutral_decision.decision)
        with TestClient(app) as client:
            neutral_response = client.post(
                "/chat",
                json={"session_id": self.session_id, "npc_id": "ravi", "message": greeting},
            )
        self.assertEqual(neutral_response.status_code, 200)
        responses.append(neutral_response.json()["message"])

        semantic_memory.add_learning_memory(
            self.session_id,
            "The player has accepted help from Ravi.",
            "PLAYER_ACCEPTED_HELP",
            npc_id="ravi",
        )
        friendly = evaluate_player_relationship(self.session_id, npc.id)
        friendly_decision = select_npc_dialogue(
            npc,
            friendly,
            session_id=self.session_id,
            message=greeting,
            intent=None,
        )
        decisions.append(friendly_decision.decision)
        with TestClient(app) as client:
            friendly_response = client.post(
                "/chat",
                json={"session_id": self.session_id, "npc_id": "ravi", "message": greeting},
            )
        self.assertEqual(friendly_response.status_code, 200)
        responses.append(friendly_response.json()["message"])

        database.save_message(self.session_id, "user", "Thanks, that helped.", npc_id="ravi")
        database.save_message(self.session_id, "user", "Your advice helped, thanks.", npc_id="ravi")
        trusting = evaluate_player_relationship(self.session_id, npc.id)
        trusting_decision = select_npc_dialogue(
            npc,
            trusting,
            session_id=self.session_id,
            message=greeting,
            intent=None,
        )
        decisions.append(trusting_decision.decision)
        with TestClient(app) as client:
            trusting_response = client.post(
                "/chat",
                json={"session_id": self.session_id, "npc_id": "ravi", "message": greeting},
            )
        self.assertEqual(trusting_response.status_code, 200)
        responses.append(trusting_response.json()["message"])

        self.assertEqual(
            decisions,
            ["neutral_greeting", "friendly_greeting", "trusting_greeting"],
        )
        for response in responses:
            self.assertTrue(response.strip())
            response_text = response.casefold()
            self.assertNotIn("neutral_greeting", response_text)
            self.assertNotIn("friendly_greeting", response_text)
            self.assertNotIn("trusting_greeting", response_text)
            self.assertNotIn("meeraprivatelivetoken", response_text)


if __name__ == "__main__":
    unittest.main()