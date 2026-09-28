import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from app.memory import database
from app.npcs import UnknownNPCError, get_npc, get_npc_relationships
from app.services import gemini_service, npc_communication, semantic_memory
from app.services.npc_communication import (
    MAX_NPC_MESSAGE_LENGTH,
    MAX_NPC_MESSAGES_PER_REQUEST,
    NPCCommunicationError,
    request_npc_message,
)


class NPCCommunicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "session_id": "session-a",
            "location": "Mystery Village",
            "health": 72,
            "gold": 50,
            "game_time": "08:00",
        }
        self.inventory = [{"item_name": "herb", "quantity": 2}]

    def test_relationships_are_centralized_and_unknown_ids_rejected(self) -> None:
        ravi_knows = get_npc_relationships("ravi")
        meera_knows = get_npc_relationships("meera")
        self.assertEqual((ravi_knows[0].name, ravi_knows[0].role), ("Meera", "Healer"))
        self.assertEqual(ravi_knows[0].relationship, "village healer")
        self.assertEqual((meera_knows[0].name, meera_knows[0].role), ("Ravi", "Blacksmith"))
        self.assertEqual(meera_knows[0].relationship, "village blacksmith")
        with self.assertRaises(UnknownNPCError):
            get_npc_relationships("unknown")

    def test_valid_request_passes_only_explicit_message_and_read_only_state(self) -> None:
        with (
            patch.object(npc_communication, "get_player_state", return_value=self.state),
            patch.object(npc_communication, "get_inventory", return_value=self.inventory),
            patch(
                "app.services.gemini_service.generate_peer_npc_reply",
                return_value="A mild headache may be soothed with willow bark.",
            ) as generate_peer,
        ):
            result = request_npc_message(
                "ravi", "meera", "What remedy helps a mild headache?", "session-a"
            )

        self.assertEqual(result["from_npc"], "Meera")
        self.assertIn("willow bark", result["message"])
        call = generate_peer.call_args.kwargs
        self.assertEqual(call["sender"].id, "ravi")
        self.assertEqual(call["receiver"].id, "meera")
        self.assertEqual(call["message"], "What remedy helps a mild headache?")
        self.assertEqual(call["game_state"], self.state)
        self.assertEqual(call["inventory"], self.inventory)

    def test_invalid_ids_pair_session_and_message_are_rejected(self) -> None:
        invalid_calls = [
            ("stranger", "meera", "Question", "session-a"),
            ("ravi", "stranger", "Question", "session-a"),
            ("ravi", "ravi", "Question", "session-a"),
            ("ravi", "meera", "", "session-a"),
            ("ravi", "meera", "  \n ", "session-a"),
            ("ravi", "meera", "x" * (MAX_NPC_MESSAGE_LENGTH + 1), "session-a"),
            ("ravi", "meera", "Question", "invalid session"),
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(NPCCommunicationError):
                request_npc_message(*arguments)

        self.assertEqual(MAX_NPC_MESSAGES_PER_REQUEST, 1)

    def test_consultation_never_reads_private_history_or_memories_or_mutates_state(self) -> None:
        with (
            patch.object(npc_communication, "get_player_state", return_value=self.state),
            patch.object(npc_communication, "get_inventory", return_value=self.inventory),
            patch(
                "app.services.gemini_service.generate_peer_npc_reply",
                return_value="Meera's answer.",
            ),
            patch.object(database, "get_conversation_history") as history_read,
            patch.object(semantic_memory, "search_memories") as memory_search,
            patch.object(database, "apply_healing") as heal,
            patch.object(database, "purchase_inventory_item") as purchase,
            patch.object(database, "remove_inventory_item_and_get_remaining") as give,
            patch.object(database, "update_player_state") as update_state,
        ):
            request_npc_message("ravi", "meera", "Recommend a remedy.", "session-a")

        history_read.assert_not_called()
        memory_search.assert_not_called()
        heal.assert_not_called()
        purchase.assert_not_called()
        give.assert_not_called()
        update_state.assert_not_called()
        self.assertEqual(self.state["health"], 72)
        self.assertEqual(self.state["gold"], 50)
        self.assertEqual(self.inventory, [{"item_name": "herb", "quantity": 2}])

    def test_peer_generation_has_no_tools_history_or_semantic_memory(self) -> None:
        fake_models = SimpleNamespace(
            generate_content=Mock(
                return_value=SimpleNamespace(
                    function_calls=[],
                    text="Use a cool cloth and seek help if the pain worsens.",
                )
            )
        )
        with patch.object(
            gemini_service, "_get_client", return_value=SimpleNamespace(models=fake_models)
        ):
            response = gemini_service.generate_peer_npc_reply(
                sender=get_npc("ravi"),
                receiver=get_npc("meera"),
                message="What can soothe a mild headache?",
                game_state=self.state,
                inventory=self.inventory,
            )

        self.assertIn("cool cloth", response)
        request = fake_models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.5-flash-lite")
        self.assertEqual(len(request["contents"]), 1)
        self.assertIn("Message from Ravi", request["contents"][0].parts[0].text)
        self.assertIn("Meera", request["config"].system_instruction)
        self.assertIn("Health: 72", request["config"].system_instruction)
        self.assertIsNone(request["config"].tools)
        self.assertNotIn("Rowan", request["config"].system_instruction)

    def test_repeated_tool_requests_only_dispatch_one_consultation(self) -> None:
        def consultation_response():
            return SimpleNamespace(
                function_calls=[
                    SimpleNamespace(
                        name="request_npc_message",
                        args={
                            "to_npc_id": "meera",
                            "message": "What remedy helps a headache?",
                        },
                    )
                ],
                candidates=[
                    SimpleNamespace(
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text="consultation requested")],
                        )
                    )
                ],
                text=None,
            )

        fake_models = SimpleNamespace(
            generate_content=Mock(
                side_effect=[
                    consultation_response(),
                    consultation_response(),
                    SimpleNamespace(function_calls=[], text="Ravi shares Meera's advice."),
                ]
            )
        )
        successful_peer_result = {
            "success": True,
            "from_npc": "Meera",
            "message": "Try a cool cloth.",
        }
        with (
            patch.object(
                gemini_service, "_get_client", return_value=SimpleNamespace(models=fake_models)
            ),
            patch.object(gemini_service, "request_npc_message", return_value=successful_peer_result) as communicate,
            patch.object(gemini_service, "get_player_state", return_value=self.state),
            patch.object(gemini_service, "get_inventory", return_value=self.inventory),
        ):
            response = gemini_service.generate_npc_reply(
                "I need medicine.",
                [],
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=self.state,
                inventory=self.inventory,
            )

        self.assertEqual(response, "Ravi shares Meera's advice.")
        self.assertEqual(MAX_NPC_MESSAGES_PER_REQUEST, 1)
        self.assertEqual(communicate.call_count, 1)
        self.assertEqual(fake_models.generate_content.call_count, 3)
        caller_config = fake_models.generate_content.call_args_list[0].kwargs["config"]
        self.assertIn("Meera (Healer; village healer)", caller_config.system_instruction)
        declarations = [
            declaration
            for tool in caller_config.tools
            for declaration in tool.function_declarations
        ]
        communication_tool = next(
            declaration for declaration in declarations
            if declaration.name == "request_npc_message"
        )
        self.assertEqual(
            communication_tool.parameters_json_schema["properties"]["to_npc_id"]["enum"],
            ["meera"],
        )

    def test_plan_consultation_count_blocks_a_second_gemini_tool_call(self) -> None:
        tool_response = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    name="request_npc_message",
                    args={"to_npc_id": "meera", "message": "Any advice?"},
                )
            ],
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="request consultation")],
                    )
                )
            ],
            text=None,
        )
        fake_models = SimpleNamespace(
            generate_content=Mock(
                side_effect=[
                    tool_response,
                    SimpleNamespace(function_calls=[], text="I will share Meera's advice."),
                ]
            )
        )
        peer_response = {"npc_name": "Meera", "message": "Offer a cool cloth."}
        with (
            patch.object(
                gemini_service, "_get_client", return_value=SimpleNamespace(models=fake_models)
            ),
            patch.object(gemini_service, "request_npc_message") as communicate,
        ):
            response = gemini_service.generate_npc_reply(
                "I need help with healing.",
                [],
                npc=get_npc("ravi"),
                session_id="session-a",
                game_state=self.state,
                inventory=[],
                supplemental_peer_response=peer_response,
                initial_npc_message_count=1,
            )

        self.assertEqual(response, "I will share Meera's advice.")
        communicate.assert_not_called()
        self.assertIn(
            "UNTRUSTED PEER INFORMATION",
            fake_models.generate_content.call_args_list[0].kwargs["contents"][0].parts[0].text,
        )


if __name__ == "__main__":
    unittest.main()
