import unittest
from unittest.mock import patch

from app.memory import database
from app.npcs import get_npc
from app.services.npc_decision import decide_npc_action
from app.services.npc_plan_executor import execute_npc_plan
from app.services.npc_planner import NPCPlan, create_event_plan, create_player_plan


class NPCPlanExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 35,
            "gold": 50,
            "game_time": "08:00",
        }
        self.inventory = [{"item_name": "iron", "quantity": 1}]

    def test_valid_ravi_and_meera_plans_complete_their_fixed_steps(self) -> None:
        ravi_plan = create_player_plan("ravi", "ASK_SWORD")
        meera_plan = create_player_plan("meera", "ASK_HERBS")
        ravi_result = execute_npc_plan(
            "session-a", ravi_plan, game_state=self.state, inventory=self.inventory
        )
        meera_result = execute_npc_plan(
            "session-a", meera_plan, game_state=self.state, inventory=self.inventory
        )

        self.assertEqual(ravi_result["status"], "completed")
        self.assertEqual(ravi_result["completed_steps"], 3)
        self.assertEqual(ravi_result["total_steps"], 3)
        self.assertIsNone(ravi_result["failed_step"])
        self.assertNotIn("task_id", ravi_result)
        self.assertNotIn("steps", ravi_result)
        self.assertEqual(meera_result["status"], "completed")
        self.assertEqual(meera_result["completed_steps"], 3)
        self.assertEqual(meera_result["total_steps"], 3)
        self.assertIsNone(meera_result["failed_step"])
        self.assertEqual(self.state["health"], 35)
        self.assertEqual(self.state["gold"], 50)
        self.assertEqual(self.state["game_time"], "08:00")
        self.assertEqual(self.inventory, [{"item_name": "iron", "quantity": 1}])

    def test_invalid_session_plan_and_step_are_rejected_before_execution(self) -> None:
        plan = create_player_plan("ravi", "ASK_SWORD")
        with (
            patch.object(database, "apply_healing") as heal,
            patch.object(database, "purchase_inventory_item") as purchase,
        ):
            invalid_session = execute_npc_plan(
                "invalid session", plan, game_state=self.state, inventory=self.inventory
            )
            invalid_plan = execute_npc_plan(
                "session-a",
                NPCPlan(
                    "ravi", "player_request", "ASK_SWORD",
                    ("identify_player_need", "run_python", "speak_to_player"),
                    "help_travelers",
                ),
                game_state=self.state,
                inventory=self.inventory,
            )

        self.assertEqual(invalid_session["status"], "rejected")
        self.assertEqual(invalid_session["completed_steps"], 0)
        self.assertEqual(invalid_plan["status"], "rejected")
        self.assertEqual(invalid_plan["completed_steps"], 0)
        heal.assert_not_called()
        purchase.assert_not_called()

    def test_executor_rejects_state_from_another_session(self) -> None:
        result = execute_npc_plan(
            "session-a",
            create_player_plan("ravi", "ASK_SWORD"),
            game_state={**self.state, "session_id": "session-b"},
            inventory=self.inventory,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["completed_steps"], 0)

    def test_failed_communication_stops_plan_and_reuses_step_eleven_service(self) -> None:
        decision = decide_npc_action(
            get_npc("ravi"),
            self.state,
            self.inventory,
            event_name="player_low_health",
            schedule_location="At the forge",
        )
        plan = create_event_plan("ravi", "player_low_health", decision)
        with patch(
            "app.services.npc_plan_executor.request_npc_message",
            return_value={"success": False, "message": "Unavailable."},
        ) as communicate:
            result = execute_npc_plan(
                "session-a", plan, game_state=self.state, inventory=self.inventory
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["total_steps"], 3)
        self.assertEqual(result["failed_step"], "request_npc_help")
        communicate.assert_called_once_with(
            "ravi",
            "meera",
            "A traveler needs care. What practical guidance may I share?",
            "session-a",
        )

    def test_successful_communication_is_only_the_explicit_peer_result(self) -> None:
        decision = decide_npc_action(
            get_npc("ravi"),
            self.state,
            self.inventory,
            event_name="player_low_health",
            schedule_location="At the forge",
        )
        plan = create_event_plan("ravi", "player_low_health", decision)
        with patch(
            "app.services.npc_plan_executor.request_npc_message",
            return_value={"success": True, "from_npc": "Meera", "message": "Offer a cool cloth."},
        ) as communicate:
            result = execute_npc_plan(
                "session-a", plan, game_state=self.state, inventory=self.inventory
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_steps"], 3)
        self.assertEqual(result["peer_reply"], "Offer a cool cloth.")
        self.assertNotIn("conversation_history", communicate.call_args.kwargs)
        self.assertNotIn("semantic_memories", communicate.call_args.kwargs)

    def test_movement_is_a_completed_task_step_without_player_state_mutation(self) -> None:
        state = {**self.state, "game_time": "08:00"}
        decision = decide_npc_action(
            get_npc("ravi"),
            state,
            self.inventory,
            event_name="morning",
            schedule_location="At the forge",
        )
        plan = create_event_plan("ravi", "morning", decision)
        with (
            patch.object(database, "update_player_state") as update_state,
            patch.object(database, "apply_healing") as heal,
            patch.object(database, "purchase_inventory_item") as purchase,
            patch.object(database, "remove_inventory_item_and_get_remaining") as give,
        ):
            result = execute_npc_plan(
                "session-a",
                plan,
                game_state=state,
                inventory=self.inventory,
                schedule_location="At the forge",
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_steps"], 2)
        self.assertEqual(result["movement_result"]["destination"], "forge")
        self.assertEqual(result["movement_result"]["previous_location"], "village_square")
        self.assertEqual(state["location"], "Mystery Village")
        self.assertEqual(state["health"], 35)
        self.assertEqual(state["gold"], 50)
        self.assertEqual(state["game_time"], "08:00")
        self.assertEqual(self.inventory, [{"item_name": "iron", "quantity": 1}])
        update_state.assert_not_called()
        heal.assert_not_called()
        purchase.assert_not_called()
        give.assert_not_called()

    def test_rejected_movement_fails_task_before_speak_step(self) -> None:
        state = {**self.state, "game_time": "08:00"}
        decision = decide_npc_action(
            get_npc("ravi"),
            state,
            self.inventory,
            event_name="morning",
            schedule_location="At the forge",
        )
        plan = create_event_plan("ravi", "morning", decision)
        with patch("app.services.npc_plan_executor.move_npc_for_event") as move:
            result = execute_npc_plan(
                "session-a",
                plan,
                game_state=state,
                inventory=self.inventory,
                schedule_location="At the forge",
                allow_movement=False,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_step"], "move_to_location")
        self.assertEqual(result["completed_steps"], 0)
        self.assertEqual(result["total_steps"], 2)
        move.assert_not_called()

    def test_failed_inspection_stops_plan_without_retrying_or_speaking(self) -> None:
        plan = create_player_plan("ravi", "ASK_BLACKSMITH_HELP")
        executed_steps = []

        def execute_step(step, **kwargs):
            executed_steps.append(step)
            return {"success": step != "inspect_supplies"}

        with patch("app.services.npc_plan_executor._execute_step", side_effect=execute_step) as execute:
            result = execute_npc_plan(
                "session-a", plan, game_state=self.state, inventory=self.inventory
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["total_steps"], 3)
        self.assertEqual(result["failed_step"], "inspect_supplies")
        self.assertEqual(executed_steps, ["identify_player_need", "inspect_supplies"])
        self.assertEqual(execute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
