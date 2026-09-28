import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.memory import database
from app.services.game_actions import (
    ALLOWED_GAME_ACTIONS,
    buy_item,
    execute_game_action,
    give_item,
    heal_player,
)
from app.services.gemini_service import (
    GAME_ACTION_TOOL,
    _explicit_action_for_message,
)


class GameActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "test.db"
        self.path_patcher = patch.object(database, "DATABASE_PATH", self.database_path)
        self.path_patcher.start()
        self.addCleanup(self.path_patcher.stop)
        self.addCleanup(self.temp_dir.cleanup)
        database.init_db()

    def test_buy_item_successfully_deducts_gold_and_adds_inventory(self) -> None:
        result = buy_item("session-a", "sword", 1)

        self.assertTrue(result["success"])
        self.assertEqual(result["changes"]["gold"], 20)
        self.assertEqual(
            database.get_inventory("session-a"),
            [{"item_name": "sword", "quantity": 1}],
        )

    def test_insufficient_gold_does_not_partially_purchase(self) -> None:
        database.update_player_state("session-a", gold=20)

        result = buy_item("session-a", "sword", 1)

        self.assertFalse(result["success"])
        self.assertEqual(database.get_player_state("session-a")["gold"], 20)
        self.assertEqual(database.get_inventory("session-a"), [])

    def test_buy_item_rejects_unknown_item_and_invalid_quantities(self) -> None:
        self.assertFalse(buy_item("session-a", "dragon", 1)["success"])
        self.assertFalse(
            execute_game_action(
                "session-a",
                "buy_item",
                {"item_name": "sword", "quantity": 1, "price": 1},
            )["success"]
        )
        for quantity in (0, -1, True, 1.5, "2"):
            with self.subTest(quantity=quantity):
                self.assertFalse(buy_item("session-a", "sword", quantity)["success"])
        self.assertEqual(database.get_player_state("session-a")["gold"], 50)
        self.assertEqual(database.get_inventory("session-a"), [])

    def test_give_item_removes_only_owned_quantity(self) -> None:
        database.add_inventory_item("session-a", "iron", 5)

        result = give_item("session-a", "iron", 2)

        self.assertTrue(result["success"])
        self.assertEqual(database.get_inventory("session-a"), [{"item_name": "iron", "quantity": 3}])
        failed = give_item("session-a", "iron", 10)
        self.assertFalse(failed["success"])
        self.assertEqual(database.get_inventory("session-a"), [{"item_name": "iron", "quantity": 3}])

    def test_healing_caps_health_at_one_hundred(self) -> None:
        database.update_player_state("session-a", health=70)
        self.assertEqual(heal_player("session-a", 20)["changes"]["health"], 90)
        database.update_player_state("session-a", health=90)
        self.assertEqual(heal_player("session-a", 30)["changes"]["health"], 100)
        self.assertEqual(database.get_player_state("session-a")["health"], 100)
        self.assertFalse(heal_player("session-a", True)["success"])

    def test_actions_are_isolated_by_session(self) -> None:
        self.assertTrue(buy_item("session-a", "sword", 1)["success"])

        state_b = database.get_player_state("session-b")
        inventory_b = database.get_inventory("session-b")

        self.assertEqual(state_b["gold"], 50)
        self.assertEqual(state_b["health"], 100)
        self.assertEqual(state_b["location"], "Mystery Village")
        self.assertEqual(inventory_b, [])

    def test_purchase_rolls_back_if_inventory_insert_fails(self) -> None:
        with closing(database._connect()) as connection:
            connection.execute(
                """
                CREATE TRIGGER fail_inventory_insert
                BEFORE INSERT ON inventory
                BEGIN
                    SELECT RAISE(ABORT, 'test failure');
                END
                """
            )
            connection.commit()
        result = buy_item("session-a", "sword", 1)

        self.assertFalse(result["success"])
        self.assertEqual(database.get_player_state("session-a")["gold"], 50)
        self.assertEqual(database.get_inventory("session-a"), [])

    def test_unknown_tool_is_rejected_without_state_changes(self) -> None:
        result = execute_game_action("session-a", "delete_player_state", {})

        self.assertFalse(result["success"])
        self.assertEqual(result["action"], "unknown")
        self.assertEqual(database.get_player_state("session-a")["gold"], 50)
        self.assertEqual(database.get_inventory("session-a"), [])
        self.assertEqual(ALLOWED_GAME_ACTIONS, ("buy_item", "give_item", "heal_player"))

    def test_explicit_action_routing_and_tool_allowlist(self) -> None:
        self.assertEqual(_explicit_action_for_message("Buy 100 swords."), "buy_item")
        self.assertEqual(_explicit_action_for_message("Give iron."), "give_item")
        self.assertEqual(_explicit_action_for_message("Heal me."), "heal_player")
        self.assertIsNone(_explicit_action_for_message("How much gold do I have?"))
        names = [declaration.name for declaration in GAME_ACTION_TOOL.function_declarations]
        self.assertEqual(names, ["buy_item", "give_item", "heal_player"])
        buy_properties = GAME_ACTION_TOOL.function_declarations[0].parameters_json_schema[
            "properties"
        ]
        self.assertNotIn("price", buy_properties)

    def test_step_five_conversation_memory_remains_available(self) -> None:
        database.save_message("session-a", "user", "Hello Ravi")
        database.save_message("session-a", "assistant", "Welcome to the village.")

        self.assertEqual(
            database.get_conversation_history("session-a"),
            [
                {"role": "user", "content": "Hello Ravi"},
                {"role": "assistant", "content": "Welcome to the village."},
            ],
        )


if __name__ == "__main__":
    unittest.main()