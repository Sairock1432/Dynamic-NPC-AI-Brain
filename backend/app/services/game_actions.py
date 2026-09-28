import logging
import sqlite3
from typing import Any

from app.memory import database

logger = logging.getLogger(__name__)

SHOP_ITEMS = {
    "sword": 30,
    "shield": 25,
    "iron": 10,
    "wood": 5,
}

ALLOWED_GAME_ACTIONS = ("buy_item", "give_item", "heal_player")


def _failure(action: str, message: str) -> dict[str, Any]:
    return {"success": False, "action": action, "message": message}


def _valid_quantity(quantity: object) -> bool:
    return isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0


def buy_item(session_id: str, item_name: str, quantity: int) -> dict[str, Any]:
    action = "buy_item"
    if not isinstance(item_name, str):
        return _failure(action, "Choose an available shop item.")
    item_name = item_name.strip().lower()
    if item_name not in SHOP_ITEMS:
        return _failure(action, "That item is not available in the shop.")
    if not _valid_quantity(quantity):
        return _failure(action, "Quantity must be a positive integer.")

    try:
        database._validate_session_id(session_id)
        result = database.purchase_inventory_item(
            session_id,
            item_name,
            quantity,
            SHOP_ITEMS[item_name] * quantity,
        )
    except ValueError:
        return _failure(action, "The purchase request is invalid.")
    except (sqlite3.Error, OverflowError) as error:
        logger.error("buy_item transaction failed (%s).", type(error).__name__)
        return _failure(action, "The purchase could not be completed.")

    if result is None:
        return _failure(action, "Not enough gold.")

    gold, item_quantity = result
    unit_price = SHOP_ITEMS[item_name]
    item_label = item_name if quantity == 1 else f"{item_name}s"
    return {
        "success": True,
        "action": action,
        "message": f"Bought {quantity} {item_label} for {unit_price * quantity} gold.",
        "changes": {
            "gold": gold,
            "inventory": [{"item_name": item_name, "quantity": item_quantity}],
        },
    }


def give_item(session_id: str, item_name: str, quantity: int) -> dict[str, Any]:
    action = "give_item"
    if not isinstance(item_name, str) or not item_name.strip():
        return _failure(action, "Choose an item to give.")
    item_name = item_name.strip()
    if not _valid_quantity(quantity):
        return _failure(action, "Quantity must be a positive integer.")

    try:
        database._validate_session_id(session_id)
        remaining = database.remove_inventory_item_and_get_remaining(
            session_id,
            item_name,
            quantity,
        )
        if remaining is None:
            return _failure(action, "You do not have enough of that item.")
    except ValueError:
        return _failure(action, "The item transfer request is invalid.")
    except sqlite3.Error as error:
        logger.error("give_item transaction failed (%s).", type(error).__name__)
        return _failure(action, "The item could not be given.")

    item_label = item_name if quantity == 1 else f"{item_name}s"
    return {
        "success": True,
        "action": action,
        "message": f"Gave {quantity} {item_label}.",
        "changes": {"inventory": [{"item_name": item_name, "quantity": remaining}]},
    }


def heal_player(session_id: str, amount: int) -> dict[str, Any]:
    action = "heal_player"
    if not _valid_quantity(amount):
        return _failure(action, "Healing amount must be a positive integer.")

    try:
        database._validate_session_id(session_id)
        health = database.apply_healing(session_id, amount)
    except ValueError:
        return _failure(action, "The healing request is invalid.")
    except (sqlite3.Error, OverflowError) as error:
        logger.error("heal_player transaction failed (%s).", type(error).__name__)
        return _failure(action, "Healing could not be completed.")

    return {
        "success": True,
        "action": action,
        "message": f"You were healed. Your health is now {health}.",
        "changes": {"health": health},
    }


def execute_game_action(
    session_id: str,
    action_name: str,
    arguments: object,
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return _failure(action_name, "The action arguments are invalid.")

    if action_name == "buy_item":
        if set(arguments) != {"item_name", "quantity"}:
            return _failure(action_name, "The purchase arguments are invalid.")
        return buy_item(session_id, arguments["item_name"], arguments["quantity"])

    if action_name == "give_item":
        if set(arguments) != {"item_name", "quantity"}:
            return _failure(action_name, "The item transfer arguments are invalid.")
        return give_item(session_id, arguments["item_name"], arguments["quantity"])

    if action_name == "heal_player":
        if set(arguments) != {"amount"}:
            return _failure(action_name, "The healing arguments are invalid.")
        return heal_player(session_id, arguments["amount"])

    return _failure("unknown", "That game action is not allowed.")