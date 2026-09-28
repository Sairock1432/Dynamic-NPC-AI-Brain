import unittest
from unittest.mock import patch

from app import main
from app.npcs import get_npc
from app.services.npc_decision import decide_npc_action
from app.services.npc_relationship_progression import RelationshipEvaluation
from app.services.npc_planner import (
    ALLOWED_INTENTS,
    ALLOWED_PLAN_STEPS,
    MAX_PLAN_STEPS,
    NPCPlan,
    NPCPlanError,
    classify_player_intent,
    create_event_plan,
    create_player_plan,
    validate_intent,
    validate_npc_plan,
)


class NPCPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 30,
            "gold": 50,
            "game_time": "08:00",
        }

    def test_validated_player_intents_are_fixed_and_npc_scoped(self) -> None:
        self.assertEqual(classify_player_intent("ravi", "I want to know about swords."), "ASK_SWORD")
        self.assertEqual(classify_player_intent("ravi", "Can you help me with the forge?"), "ASK_BLACKSMITH_HELP")
        self.assertEqual(classify_player_intent("meera", "I need help with herbs."), "ASK_HERBS")
        self.assertEqual(classify_player_intent("meera", "Can you explain healing?"), "ASK_HEALING")
        self.assertEqual(classify_player_intent("meera", "Can you help me?"), "ASK_GENERAL_HELP")
        self.assertIsNone(classify_player_intent("ravi", "My name is Rowan."))
        self.assertIsNone(classify_player_intent("meera", "Tell me about swords."))
        self.assertIsNone(classify_player_intent("ravi", "Buy a sword."))
        self.assertIsNone(classify_player_intent("ravi", "Can you buy me a sword?"))
        with self.assertRaises(NPCPlanError):
            classify_player_intent("stranger", "Tell me about swords.")

    def test_ravi_and_meera_plans_use_fixed_ordered_steps(self) -> None:
        ravi = create_player_plan("ravi", "ASK_SWORD")
        meera = create_player_plan("meera", "ASK_HERBS")
        self.assertEqual(ravi.steps, (
            "identify_player_need", "provide_sword_information", "speak_to_player"
        ))
        self.assertEqual(meera.steps, (
            "identify_player_need", "provide_herb_information", "speak_to_player"
        ))
        self.assertEqual(ravi.goal_id, "help_travelers")
        self.assertEqual(meera.goal_id, "help_injured_travelers")
        self.assertEqual(len(ravi.steps), MAX_PLAN_STEPS)
        self.assertTrue(set(ravi.steps).issubset(ALLOWED_PLAN_STEPS))

    def test_unknown_intents_and_cross_npc_capabilities_are_rejected(self) -> None:
        self.assertEqual(len(ALLOWED_INTENTS), 5)
        for intent in ("", "RUN_CODE", "ASK_PRIVATE_DATA"):
            with self.subTest(intent=intent), self.assertRaises(NPCPlanError):
                validate_intent(intent)
        with self.assertRaises(NPCPlanError):
            create_player_plan("stranger", "ASK_SWORD")
        with self.assertRaises(NPCPlanError):
            create_player_plan("meera", "ASK_SWORD")
        with self.assertRaises(NPCPlanError):
            create_player_plan("ravi", "ASK_HERBS")

    def test_invalid_and_zero_step_plans_are_rejected(self) -> None:
        valid = create_player_plan("ravi", "ASK_SWORD")
        invalid_cases = (
            NPCPlan(
                "ravi", "player_request", "ASK_SWORD", (), "help_travelers"
            ),
            NPCPlan(
                "ravi", "player_request", "ASK_SWORD",
                ("identify_player_need", "run_python", "speak_to_player"),
                "help_travelers",
            ),
            NPCPlan(
                "ravi", "player_request", "ASK_SWORD",
                ("speak_to_player", "provide_sword_information", "identify_player_need"),
                "help_travelers",
            ),
            NPCPlan(
                "meera", "player_request", "ASK_SWORD", valid.steps, "help_travelers"
            ),
            NPCPlan(
                "ravi", "player_request", "ASK_SWORD",
                (*valid.steps, "speak_to_player"), "help_travelers"
            ),
        )
        for plan in invalid_cases:
            with self.subTest(plan=plan), self.assertRaises(NPCPlanError):
                validate_npc_plan(plan)

    def test_event_plans_require_an_existing_decision(self) -> None:
        decision = decide_npc_action(
            get_npc("ravi"),
            self.state,
            [],
            event_name="player_low_health",
            schedule_location="At the forge",
        )
        plan = create_event_plan("ravi", "player_low_health", decision)
        self.assertEqual(plan.source, "world_event")
        self.assertEqual(plan.steps, (
            "identify_player_need", "request_npc_help", "speak_to_player"
        ))
        with self.assertRaises(NPCPlanError):
            create_event_plan("meera", "player_low_health", decision)
        with self.assertRaises(NPCPlanError):
            create_event_plan("ravi", "unknown_event", decision)

    def test_validated_learning_can_select_only_the_matching_information_plan(self) -> None:
        ravi_memory = [{
            "text": "The player says they prefer swords.",
            "kind": "preference",
            "category": "PLAYER_PREFERENCE",
        }]
        decision = decide_npc_action(
            get_npc("ravi"),
            {**self.state, "health": 100},
            [],
            event_name="morning",
            schedule_location="At the forge",
            learning_memories=ravi_memory,
        )
        plan = create_event_plan("ravi", "morning", decision)
        self.assertEqual(plan.decision, "offer_sword_information")
        self.assertIn("provide_sword_information", plan.steps)

    def test_chat_executes_backend_plan_before_final_gemini_response(self) -> None:
        state = {**self.state, "health": 100}
        with (
            patch.object(main, "get_conversation_history", return_value=[]),
            patch.object(main, "get_player_state", return_value=state),
            patch.object(main, "get_inventory", return_value=[]),
            patch.object(main, "search_memories", return_value=[]),
            patch.object(
                main,
                "evaluate_player_relationship",
                return_value=RelationshipEvaluation("session-a", "ravi", "NEUTRAL", (), ()),
            ),
            patch.object(
                main,
                "execute_npc_plan",
                return_value={
                    "status": "completed",
                    "completed_steps": 3,
                    "total_steps": 3,
                },
            ) as execute,
            patch.object(main, "generate_npc_reply", return_value="Ravi explains sword craft.") as generate,
            patch.object(main, "save_message"),
            patch.object(main, "remember_user_facts", return_value=0),
            patch.object(main, "learn_from_player_interaction", return_value=0),
        ):
            response = main.chat(
                main.ChatRequest(
                    session_id="session-a",
                    npc_id="ravi",
                    message="I want to know about swords.",
                )
            )

        self.assertEqual(response.message, "Ravi explains sword craft.")
        plan = execute.call_args.args[1]
        self.assertEqual(plan.intent, "ASK_SWORD")
        self.assertEqual(plan.steps[-1], "speak_to_player")
        self.assertEqual(generate.call_args.kwargs["dialogue_decision"].decision, "provide_validated_information")
        self.assertEqual(generate.call_args.kwargs["dialogue_decision"].topic, "swords")
        self.assertTrue(generate.call_args.kwargs["validated_plan_completed"])
        self.assertIsNone(generate.call_args.kwargs["plan_context"])

    def test_unclassified_player_text_does_not_create_or_execute_a_plan(self) -> None:
        state = {**self.state, "health": 100}
        with (
            patch.object(main, "get_conversation_history", return_value=[]),
            patch.object(main, "get_player_state", return_value=state),
            patch.object(main, "get_inventory", return_value=[]),
            patch.object(main, "search_memories", return_value=[]),
            patch.object(main, "evaluate_player_relationship") as evaluate_relationship,
            patch.object(main, "create_player_plan") as create_plan,
            patch.object(main, "execute_npc_plan") as execute,
            patch.object(main, "generate_npc_reply", return_value="Welcome, traveler.") as generate,
            patch.object(main, "save_message"),
            patch.object(main, "remember_user_facts", return_value=0),
            patch.object(main, "learn_from_player_interaction", return_value=0),
        ):
            response = main.chat(
                main.ChatRequest(
                    session_id="session-a",
                    npc_id="ravi",
                    message="Tell me a story about the village.",
                )
            )

        self.assertEqual(response.message, "Welcome, traveler.")
        create_plan.assert_not_called()
        execute.assert_not_called()
        evaluate_relationship.assert_not_called()
        self.assertIsNone(generate.call_args.kwargs["dialogue_decision"])

    def test_ravi_healing_plan_passes_only_the_validated_peer_reply(self) -> None:
        state = {**self.state, "health": 100}
        peer_response = {"npc_name": "Meera", "message": "Offer a cool cloth."}
        with (
            patch.object(main, "get_conversation_history", return_value=[]),
            patch.object(main, "get_player_state", return_value=state),
            patch.object(main, "get_inventory", return_value=[]),
            patch.object(main, "search_memories", return_value=[]),
            patch.object(
                main,
                "evaluate_player_relationship",
                return_value=RelationshipEvaluation("session-a", "ravi", "FRIENDLY", (), ()),
            ),
            patch.object(
                main,
                "execute_npc_plan",
                return_value={
                    "status": "completed",
                    "completed_steps": 3,
                    "total_steps": 3,
                    "communication_count": 1,
                    "peer_response": peer_response,
                },
            ) as execute,
            patch.object(main, "generate_npc_reply", return_value="Meera suggests a cool cloth.") as generate,
            patch.object(main, "save_message"),
            patch.object(main, "remember_user_facts", return_value=0),
            patch.object(main, "learn_from_player_interaction", return_value=0),
        ):
            response = main.chat(
                main.ChatRequest(
                    session_id="session-a",
                    npc_id="ravi",
                    message="I need help with healing.",
                )
            )

        self.assertEqual(response.message, "Meera suggests a cool cloth.")
        self.assertIn("request_npc_help", execute.call_args.args[1].steps)
        self.assertEqual(generate.call_args.kwargs["supplemental_peer_response"], peer_response)
        self.assertEqual(generate.call_args.kwargs["initial_npc_message_count"], 1)


if __name__ == "__main__":
    unittest.main()
