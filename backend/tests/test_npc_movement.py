import unittest
from unittest.mock import patch

from app.memory import database
from app.services import semantic_memory
from app.services.npc_movement import (
    ALLOWED_MOVEMENT_TRANSITIONS,
    MAX_MOVEMENTS_PER_NPC_PER_EVENT,
    MAX_NPC_MOVEMENTS_PER_EVENT,
    NPCMovementError,
    VALID_NPC_LOCATIONS,
    move_npc_for_event,
    validate_npc_location,
)


class NPCMovementTests(unittest.TestCase):
    def test_morning_movement_uses_configured_forge_and_healer_destinations(self) -> None:
        ravi = move_npc_for_event(
            "session-a", "ravi", "morning", game_time="08:00", current_location="village_square"
        )
        meera = move_npc_for_event(
            "session-a", "meera", "morning", game_time="08:00", current_location="village_square"
        )
        self.assertEqual(ravi["destination"], "forge")
        self.assertEqual(meera["destination"], "healer_hut")
        self.assertTrue(ravi["moved"])
        self.assertTrue(meera["moved"])

    def test_afternoon_and_evening_destinations_follow_existing_game_time(self) -> None:
        ravi_market = move_npc_for_event(
            "session-a", "ravi", "afternoon", game_time="13:00", current_location="village_square"
        )
        meera_market = move_npc_for_event(
            "session-a", "meera", "afternoon", game_time="13:00", current_location="village_square"
        )
        ravi_forge = move_npc_for_event(
            "session-a", "ravi", "afternoon", game_time="15:00", current_location="village_square"
        )
        meera_hut = move_npc_for_event(
            "session-a", "meera", "afternoon", game_time="15:00", current_location="village_square"
        )
        ravi_evening = move_npc_for_event(
            "session-a", "ravi", "evening", game_time="19:00"
        )
        meera_evening = move_npc_for_event(
            "session-a", "meera", "evening", game_time="19:00", current_location="village_square"
        )

        self.assertEqual(ravi_market["destination"], "market")
        self.assertEqual(meera_market["destination"], "market")
        self.assertEqual(ravi_forge["destination"], "forge")
        self.assertEqual(meera_hut["destination"], "healer_hut")
        self.assertEqual(ravi_evening["destination"], "village_square")
        self.assertEqual(ravi_evening["previous_location"], "forge")
        self.assertTrue(ravi_evening["moved"])
        self.assertEqual(meera_evening["destination"], "healer_hut")

    def test_low_health_moves_only_meera_to_the_symbolic_player_location(self) -> None:
        result = move_npc_for_event(
            "session-a", "meera", "player_low_health", game_time="08:00"
        )
        self.assertEqual(result["destination"], "player_location")
        with self.assertRaises(NPCMovementError):
            move_npc_for_event("session-a", "ravi", "player_low_health", game_time="08:00")

    def test_unknown_npc_location_event_session_and_time_are_rejected(self) -> None:
        with self.assertRaises(NPCMovementError):
            move_npc_for_event("session-a", "stranger", "morning", game_time="08:00")
        with self.assertRaises(NPCMovementError):
            move_npc_for_event("session-a", "ravi", "unknown", game_time="08:00")
        with self.assertRaises(NPCMovementError):
            move_npc_for_event("invalid session", "ravi", "morning", game_time="08:00")
        with self.assertRaises(NPCMovementError):
            move_npc_for_event(
                "session-a", "ravi", "morning", game_time="13:00", current_location="village_square"
            )
        for location in ("", "unknown", "0,0", "coordinates: 1,2"):
            with self.subTest(location=location), self.assertRaises(NPCMovementError):
                validate_npc_location(location)

    def test_invalid_directed_transition_is_rejected_without_state_writes(self) -> None:
        with (
            patch.object(database, "_connect") as connect,
            patch.object(database, "update_player_state") as update_state,
            patch.object(database, "apply_healing") as heal,
            patch.object(semantic_memory, "_get_collection") as collection,
        ):
            with self.assertRaises(NPCMovementError):
                move_npc_for_event(
                    "session-a",
                    "ravi",
                    "morning",
                    game_time="08:00",
                    current_location="forest_edge",
                )

        connect.assert_not_called()
        update_state.assert_not_called()
        heal.assert_not_called()
        collection.assert_not_called()

    def test_location_and_movement_budgets_are_fixed(self) -> None:
        self.assertEqual(
            set(VALID_NPC_LOCATIONS),
            {"village_square", "forge", "healer_hut", "market", "forest_edge", "player_location"},
        )
        self.assertEqual(MAX_MOVEMENTS_PER_NPC_PER_EVENT, 1)
        self.assertEqual(MAX_NPC_MOVEMENTS_PER_EVENT, 2)
        self.assertIn("forge", ALLOWED_MOVEMENT_TRANSITIONS["village_square"])
        self.assertNotIn("forge", ALLOWED_MOVEMENT_TRANSITIONS["forest_edge"])


if __name__ == "__main__":
    unittest.main()
