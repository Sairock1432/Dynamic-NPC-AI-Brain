import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.memory import database
from app.npcs import get_npc
from app.services import gemini_service, npc_behavior, npc_movement, npc_plan_executor, semantic_memory
from app.services.npc_decision import NPCDecisionError, decide_npc_action
from app.services.npc_relationship_progression import evaluate_player_relationship
from app.services.npc_behavior import (
    LOW_HEALTH_THRESHOLD,
    MAX_AUTONOMOUS_ACTIONS_PER_EVENT,
    NPCBehaviorError,
    dispatch_world_event,
    get_npc_schedule_location,
)
from app.services.world_events import (
    WORLD_EVENTS,
    UnknownWorldEventError,
    WorldEvent,
    get_world_event,
)


class NPCBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 100,
            "gold": 50,
            "game_time": "08:00",
        }
        self.inventory = [{"item_name": "herb", "quantity": 2}]

    def _dispatch_with_reactions(self, event_name: str, state: dict | None = None):
        selected_state = state or self.state
        with (
            patch.object(npc_behavior, "get_player_state", return_value=selected_state),
            patch.object(npc_behavior, "get_inventory", return_value=self.inventory),
            patch.object(npc_behavior, "get_npc_learning_memories", return_value=[]),
            patch.object(
                npc_behavior,
                "generate_npc_event_reaction",
                side_effect=lambda **kwargs: f"{kwargs['npc'].name} reacts.",
            ) as generate,
        ):
            result = dispatch_world_event("session-a", event_name)
        return result, generate

    def test_event_registry_accepts_only_predefined_names(self) -> None:
        self.assertEqual(
            set(WORLD_EVENTS),
            {"morning", "afternoon", "evening", "player_low_health", "village_event"},
        )
        self.assertEqual(get_world_event("morning").event_id, "morning")
        for event_name in ("", " ", "unknown", "morning;exec"):
            with self.subTest(event_name=event_name), self.assertRaises(UnknownWorldEventError):
                get_world_event(event_name)
        with patch.object(npc_behavior, "get_player_state") as get_state:
            with self.assertRaises(UnknownWorldEventError):
                dispatch_world_event("session-a", "")
        get_state.assert_not_called()
        with self.assertRaises(NPCBehaviorError):
            dispatch_world_event("invalid session", "morning")

    def test_schedules_use_existing_game_time(self) -> None:
        self.assertEqual(get_npc_schedule_location("ravi", "08:00"), "At the forge")
        self.assertEqual(get_npc_schedule_location("meera", "08:00"), "At the healer hut")
        self.assertEqual(get_npc_schedule_location("ravi", "13:00"), "In the village")
        self.assertEqual(get_npc_schedule_location("meera", "13:00"), "In the village")
        self.assertEqual(get_npc_schedule_location("ravi", "19:00"), "At home after closing")
        self.assertEqual(get_npc_schedule_location("meera", "19:00"), "At the healer hut")
        with self.assertRaises(NPCBehaviorError):
            get_npc_schedule_location("ravi", "not game time")

    def test_morning_afternoon_and_evening_events_return_deterministic_schedules(self) -> None:
        expected = (
            ("morning", "08:00", "At the forge", "At the healer hut", "forge", "healer_hut"),
            ("afternoon", "13:00", "In the village", "In the village", "market", "market"),
            ("evening", "19:00", "At home after closing", "At the healer hut", "village_square", "healer_hut"),
        )
        for (
            event_name,
            game_time,
            ravi_location,
            meera_location,
            ravi_destination,
            meera_destination,
        ) in expected:
            with self.subTest(event_name=event_name):
                state = {**self.state, "game_time": game_time}
                result, generate = self._dispatch_with_reactions(event_name, state)
                activities = {action["npc_id"]: action for action in result["actions"]}
                self.assertEqual(result["world_time"], game_time)
                self.assertEqual(result["action_count"], 2)
                self.assertLessEqual(result["action_count"], MAX_AUTONOMOUS_ACTIONS_PER_EVENT)
                self.assertEqual(activities["ravi"]["schedule_location"], ravi_location)
                self.assertEqual(activities["meera"]["schedule_location"], meera_location)
                self.assertEqual(activities["ravi"]["movement"]["destination"], ravi_destination)
                self.assertEqual(activities["meera"]["movement"]["destination"], meera_destination)
                self.assertEqual(generate.call_count, 2)

    def test_village_event_is_allowlisted_and_bounded(self) -> None:
        result, generate = self._dispatch_with_reactions("village_event")
        self.assertEqual(result["status"], "completed")
        self.assertEqual({item["npc_id"] for item in result["actions"]}, {"ravi", "meera"})
        self.assertEqual(result["action_count"], 2)
        self.assertLessEqual(result["action_count"], MAX_AUTONOMOUS_ACTIONS_PER_EVENT)
        self.assertEqual(generate.call_count, 2)

    def test_low_health_selects_meera_without_mutating_game_state_or_reading_memories(self) -> None:
        low_health_state = {**self.state, "health": LOW_HEALTH_THRESHOLD - 1}
        with (
            patch.object(npc_behavior, "get_player_state", return_value=low_health_state),
            patch.object(npc_behavior, "get_inventory", return_value=self.inventory),
            patch.object(npc_behavior, "get_npc_learning_memories", return_value=[]),
            patch.object(
                npc_behavior,
                "generate_npc_event_reaction",
                return_value="Meera offers calm help.",
            ) as generate,
            patch.object(
                npc_plan_executor,
                "request_npc_message",
                return_value={"success": True, "from_npc": "Meera", "message": "Offer care."},
            ) as request_help,
            patch.object(database, "get_conversation_history") as history_read,
            patch.object(semantic_memory, "search_memories") as memory_search,
            patch.object(database, "apply_healing") as heal,
            patch.object(database, "purchase_inventory_item") as purchase,
            patch.object(database, "remove_inventory_item_and_get_remaining") as give,
            patch.object(database, "add_inventory_item") as add_item,
            patch.object(database, "clear_inventory") as clear_inventory,
            patch.object(database, "update_player_state") as update_state,
            patch.object(semantic_memory, "remember_user_facts") as store_memory,
        ):
            result = dispatch_world_event("session-a", "player_low_health")

        self.assertEqual(result["action_count"], 2)
        actions = {item["npc_id"]: item for item in result["actions"]}
        self.assertEqual(actions["ravi"]["decision"], "request_help_from_npc")
        self.assertEqual(actions["ravi"]["consultation"]["from_npc"], "Meera")
        self.assertEqual(actions["meera"]["decision"], "offer_help")
        self.assertEqual(actions["meera"]["reaction"], "Meera offers calm help.")
        self.assertEqual(low_health_state["health"], LOW_HEALTH_THRESHOLD - 1)
        self.assertEqual(low_health_state["gold"], 50)
        self.assertEqual(self.inventory, [{"item_name": "herb", "quantity": 2}])
        self.assertEqual(generate.call_count, 2)
        narrated_decisions = {
            call.kwargs["npc"].id: call.kwargs["decision"].relationship_decision
            for call in generate.call_args_list
        }
        self.assertEqual(narrated_decisions["ravi"], "request_assistance")
        self.assertEqual(narrated_decisions["meera"], "offer_help")
        self.assertFalse(
            any("relationship_stage" in action or "relationship_decision" in action for action in actions.values())
        )
        request_help.assert_called_once_with(
            "ravi",
            "meera",
            "A traveler needs care. What practical guidance may I share?",
            "session-a",
        )
        history_read.assert_not_called()
        memory_search.assert_not_called()
        heal.assert_not_called()
        purchase.assert_not_called()
        give.assert_not_called()
        add_item.assert_not_called()
        clear_inventory.assert_not_called()
        update_state.assert_not_called()
        store_memory.assert_not_called()

    def test_health_at_or_above_threshold_does_not_trigger_healer(self) -> None:
        state = {**self.state, "health": LOW_HEALTH_THRESHOLD}
        result, generate = self._dispatch_with_reactions("player_low_health", state)
        self.assertEqual(result["status"], "not_triggered")
        self.assertEqual(result["actions"], [])
        generate.assert_not_called()

    def test_event_state_is_isolated_by_session(self) -> None:
        states = {
            "session-healthy": {**self.state, "session_id": "session-healthy", "health": 80},
            "session-injured": {**self.state, "session_id": "session-injured", "health": 20},
        }
        with (
            patch.object(npc_behavior, "get_player_state", side_effect=states.get),
            patch.object(npc_behavior, "get_inventory", return_value=self.inventory),
            patch.object(npc_behavior, "get_npc_learning_memories", return_value=[]),
            patch.object(npc_plan_executor, "request_npc_message", return_value={"success": True}),
            patch.object(npc_behavior, "generate_npc_event_reaction", return_value="Meera helps."),
        ):
            healthy = dispatch_world_event("session-healthy", "player_low_health")
            injured = dispatch_world_event("session-injured", "player_low_health")

        self.assertEqual(healthy["status"], "not_triggered")
        self.assertEqual(healthy["actions"], [])
        self.assertEqual(injured["action_count"], 2)
        self.assertEqual(injured["session_id"], "session-injured")

    def test_event_reuses_at_most_one_npc_consultation(self) -> None:
        state = {**self.state, "health": LOW_HEALTH_THRESHOLD - 1}
        duplicated_ravi_event = WorldEvent(
            event_id="player_low_health",
            description="A traveler needs care.",
            npc_behaviors=(
                ("ravi", "Ask Meera for advice."),
                ("ravi", "Ask Meera for advice again."),
            ),
        )
        with (
            patch.object(npc_behavior, "get_world_event", return_value=duplicated_ravi_event),
            patch.object(npc_behavior, "get_player_state", return_value=state),
            patch.object(npc_behavior, "get_inventory", return_value=self.inventory),
            patch.object(npc_behavior, "get_npc_learning_memories", return_value=[]),
            patch.object(
                npc_plan_executor,
                "request_npc_message",
                return_value={"success": True, "from_npc": "Meera", "message": "Advice."},
            ) as request_help,
        ):
            result = dispatch_world_event("session-a", "player_low_health")

        self.assertEqual(result["action_count"], 2)
        self.assertEqual(request_help.call_count, 1)
        self.assertTrue(result["actions"][0]["consultation"]["success"])
        self.assertFalse(result["actions"][1]["consultation"]["success"])

    def test_event_rejects_duplicate_movement_for_same_npc(self) -> None:
        event = WorldEvent(
            event_id="morning",
            description="Morning begins.",
            npc_behaviors=(("ravi", "Open the forge."), ("ravi", "Prepare the forge.")),
        )
        state = {**self.state, "game_time": "08:00"}
        movement = {
            "status": "completed",
            "npc_id": "ravi",
            "previous_location": "village_square",
            "destination": "forge",
            "world_time": "08:00",
            "moved": True,
        }
        with (
            patch.object(npc_behavior, "get_world_event", return_value=event),
            patch.object(npc_behavior, "get_player_state", return_value=state),
            patch.object(npc_behavior, "get_inventory", return_value=self.inventory),
            patch.object(npc_behavior, "get_npc_learning_memories", return_value=[]),
            patch.object(npc_plan_executor, "move_npc_for_event", return_value=movement) as move,
            patch.object(npc_behavior, "generate_npc_event_reaction", return_value="At the forge."),
        ):
            result = dispatch_world_event("session-a", "morning")

        self.assertEqual(move.call_count, 1)
        self.assertEqual(result["action_count"], 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["actions"][0]["movement"]["destination"], "forge")
        self.assertEqual(result["actions"][1]["plan_result"]["status"], "failed")
        self.assertEqual(result["actions"][1]["plan_result"]["failed_step"], "move_to_location")

    def test_action_budget_is_enforced_and_event_does_not_recurse(self) -> None:
        with patch.object(npc_behavior, "MAX_AUTONOMOUS_ACTIONS_PER_EVENT", 1):
            result, generate = self._dispatch_with_reactions("morning")
        self.assertEqual(result["action_count"], 1)
        self.assertEqual(generate.call_count, 1)
        self.assertLessEqual(result["action_count"], MAX_AUTONOMOUS_ACTIONS_PER_EVENT)

    def test_gemini_event_reaction_is_a_single_tool_free_call(self) -> None:
        fake_models = SimpleNamespace(
            generate_content=Mock(
                return_value=SimpleNamespace(
                    function_calls=[],
                    text="Ravi opens the forge and welcomes the morning.",
                )
            )
        )
        with patch.object(
            gemini_service, "_get_client", return_value=SimpleNamespace(models=fake_models)
        ):
            movement = npc_movement.move_npc_for_event(
                "session-a", "ravi", "morning", game_time="08:00"
            )
            reaction = gemini_service.generate_npc_event_reaction(
                npc=get_npc("ravi"),
                event_name="morning",
                behavior=dict(WORLD_EVENTS["morning"].npc_behaviors)["ravi"],
                schedule_location="At the forge",
                game_state=self.state,
                inventory=self.inventory,
                task_result={
                    "status": "completed",
                    "completed_steps": 2,
                    "total_steps": 2,
                    "failed_step": None,
                },
                movement_result=movement,
            )

        self.assertIn("opens the forge", reaction)
        self.assertEqual(fake_models.generate_content.call_count, 1)
        request = fake_models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.5-flash-lite")
        self.assertIsNone(request["config"].tools)
        self.assertIn("At the forge", request["config"].system_instruction)
        self.assertIn("Health: 100", request["config"].system_instruction)
        self.assertIn("completed_steps=2", request["config"].system_instruction)
        self.assertIn("Forge", request["config"].system_instruction)
        self.assertNotIn("task_id", request["config"].system_instruction)
        self.assertNotIn("identify_player_need", request["config"].system_instruction)
        self.assertNotIn("Rowan", request["config"].system_instruction)

    def test_gemini_receives_only_the_validated_relationship_decision(self) -> None:
        fake_models = SimpleNamespace(
            generate_content=Mock(
                return_value=SimpleNamespace(
                    function_calls=[],
                    text="Ravi greets the traveler warmly before returning to his work.",
                )
            )
        )
        relationship = evaluate_player_relationship(
            "session-a",
            "ravi",
            interaction_events=("successful_help",),
            learning_memories=[],
        )
        decision = decide_npc_action(
            get_npc("ravi"),
            self.state,
            self.inventory,
            event_name="morning",
            schedule_location="At the forge",
            relationship_evaluation=relationship,
        )
        self.assertEqual(decision.relationship_decision, "greet_warmly")

        with patch.object(
            gemini_service, "_get_client", return_value=SimpleNamespace(models=fake_models)
        ):
            response = gemini_service.generate_npc_event_reaction(
                npc=get_npc("ravi"),
                event_name="morning",
                behavior=dict(WORLD_EVENTS["morning"].npc_behaviors)["ravi"],
                schedule_location="At the forge",
                game_state=self.state,
                inventory=self.inventory,
                decision=decision,
            )

        prompt = fake_models.generate_content.call_args.kwargs["config"].system_instruction
        self.assertIn("Backend-selected social decision: greet_warmly", prompt)
        self.assertNotIn("FRIENDLY", prompt)
        self.assertNotIn("TRUSTING", prompt)
        self.assertNotIn("PLAYER_ACCEPTED_HELP", prompt)
        self.assertIsNone(fake_models.generate_content.call_args.kwargs["config"].tools)
        self.assertTrue(response)

        with self.assertRaises(NPCDecisionError):
            replace(decision, relationship_decision="set_relationship_stage")

    def test_gemini_rejects_fabricated_movement_destination(self) -> None:
        movement = {
            "status": "completed",
            "npc_id": "ravi",
            "previous_location": "village_square",
            "destination": "forest_edge",
            "world_time": "08:00",
            "moved": True,
        }
        with self.assertRaises(gemini_service.GeminiServiceError):
            gemini_service.generate_npc_event_reaction(
                npc=get_npc("ravi"),
                event_name="morning",
                behavior=dict(WORLD_EVENTS["morning"].npc_behaviors)["ravi"],
                schedule_location="At the forge",
                game_state=self.state,
                inventory=self.inventory,
                movement_result=movement,
            )


if __name__ == "__main__":
    unittest.main()
