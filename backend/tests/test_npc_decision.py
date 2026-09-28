import unittest
from dataclasses import replace

from app.npcs import get_npc, get_npc_relationships
from app.services.npc_decision import (
    ALLOWED_NPC_DECISIONS,
    LOW_HEALTH_THRESHOLD,
    MAX_NPC_DECISIONS_PER_EVENT,
    NPCDecision,
    NPCDecisionError,
    decide_npc_action,
    validate_npc_decision,
)
from app.services.npc_goals import NPCGoalError, get_npc_goal, get_npc_goals


class NPCDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 100,
            "gold": 50,
            "game_time": "08:00",
        }

    def test_goals_are_centralized_with_deterministic_priorities(self) -> None:
        ravi_goals = get_npc_goals("ravi")
        meera_goals = get_npc_goals("meera")
        self.assertEqual(ravi_goals[0].goal_id, "maintain_forge")
        self.assertEqual(ravi_goals[0].priority, 1)
        self.assertEqual(ravi_goals[1].goal_id, "help_travelers")
        self.assertEqual(ravi_goals[2].goal_id, "maintain_material_supplies")
        self.assertEqual(meera_goals[0].goal_id, "protect_villagers_health")
        self.assertEqual(meera_goals[1].goal_id, "help_injured_travelers")
        self.assertEqual(meera_goals[2].goal_id, "maintain_medicine_supplies")
        self.assertEqual(get_npc_goal("meera", "protect_villagers_health"), meera_goals[0])
        with self.assertRaises(NPCGoalError):
            get_npc_goals("stranger")
        with self.assertRaises(NPCGoalError):
            get_npc_goal("ravi", "invent-a-goal")

    def test_relationship_metadata_is_fixed_and_cooperative(self) -> None:
        ravi_relationship = get_npc_relationships("ravi")[0]
        meera_relationship = get_npc_relationships("meera")[0]
        self.assertEqual((ravi_relationship.npc_id, ravi_relationship.trust), ("meera", 5))
        self.assertEqual(ravi_relationship.cooperation, 5)
        self.assertEqual((meera_relationship.npc_id, meera_relationship.trust), ("ravi", 5))
        self.assertEqual(meera_relationship.cooperation, 5)

    def test_morning_decisions_follow_goals_and_schedules(self) -> None:
        ravi = decide_npc_action(
            get_npc("ravi"),
            self.state,
            [],
            event_name="morning",
            schedule_location="At the forge",
        )
        meera = decide_npc_action(
            get_npc("meera"),
            self.state,
            [],
            event_name="morning",
            schedule_location="At the healer hut",
        )
        self.assertEqual((ravi.decision, ravi.goal_id, ravi.schedule_location), (
            "stay_at_location", "maintain_forge", "At the forge"
        ))
        self.assertEqual((meera.decision, meera.goal_id, meera.schedule_location), (
            "stay_at_location", "maintain_medicine_supplies", "At the healer hut"
        ))

    def test_morning_adapts_only_to_categorized_allowlisted_learning(self) -> None:
        ravi_with_sword_preference = decide_npc_action(
            get_npc("ravi"),
            self.state,
            [],
            event_name="morning",
            schedule_location="At the forge",
            learning_memories=[
                {
                    "text": "The player says they prefer swords.",
                    "kind": "preference",
                    "category": "PLAYER_PREFERENCE",
                }
            ],
        )
        ravi_with_uncategorized_fact = decide_npc_action(
            get_npc("ravi"),
            self.state,
            [],
            event_name="morning",
            schedule_location="At the forge",
            learning_memories=[
                {"text": "The player says they prefer swords.", "kind": "preference"}
            ],
        )
        meera_with_herb_interest = decide_npc_action(
            get_npc("meera"),
            self.state,
            [],
            event_name="morning",
            schedule_location="At the healer hut",
            learning_memories=[
                {
                    "text": "The player is interested in herbs.",
                    "kind": "decision",
                    "category": "PLAYER_INTEREST",
                }
            ],
        )

        self.assertEqual(ravi_with_sword_preference.decision, "offer_sword_information")
        self.assertEqual(ravi_with_sword_preference.learning_topic, "swords")
        self.assertEqual(ravi_with_uncategorized_fact.decision, "stay_at_location")
        self.assertEqual(meera_with_herb_interest.decision, "offer_herb_information")
        self.assertEqual(meera_with_herb_interest.learning_topic, "herbs")

    def test_low_health_meera_offers_help_and_ravi_uses_configured_trust(self) -> None:
        state = {**self.state, "health": LOW_HEALTH_THRESHOLD - 1}
        meera = decide_npc_action(
            get_npc("meera"),
            state,
            [],
            event_name="player_low_health",
            schedule_location="At the healer hut",
        )
        ravi = decide_npc_action(
            get_npc("ravi"),
            state,
            [],
            event_name="player_low_health",
            schedule_location="At the forge",
        )
        self.assertEqual(meera.decision, "offer_help")
        self.assertEqual(meera.goal_id, "protect_villagers_health")
        self.assertEqual(ravi.decision, "request_help_from_npc")
        self.assertEqual(ravi.target_npc_id, "meera")
        self.assertEqual(ravi.relationship_trust, 5)
        self.assertEqual(ravi.relationship_cooperation, 5)
        self.assertTrue(ravi.message)
        self.assertIsNone(
            decide_npc_action(
                get_npc("meera"),
                self.state,
                [],
                event_name="player_low_health",
                schedule_location="At the healer hut",
            )
        )

    def test_decisions_are_allowlisted_and_reject_invalid_npc_or_event(self) -> None:
        for action in ALLOWED_NPC_DECISIONS:
            self.assertEqual(validate_npc_decision(action), action)
        with self.assertRaises(NPCDecisionError):
            validate_npc_decision("update_game_state")
        with self.assertRaises(NPCDecisionError):
            NPCDecision("ravi", "run_code", "maintain_forge", "Goal", 1, "At the forge")
        with self.assertRaises(NPCDecisionError):
            decide_npc_action(
                replace(get_npc("ravi"), id="stranger"),
                self.state,
                [],
                event_name="morning",
                schedule_location="At the forge",
            )
        with self.assertRaises(NPCDecisionError):
            decide_npc_action(
                get_npc("ravi"),
                self.state,
                [],
                event_name="unlisted_event",
                schedule_location="At the forge",
            )
        self.assertEqual(MAX_NPC_DECISIONS_PER_EVENT, 2)


if __name__ == "__main__":
    unittest.main()
