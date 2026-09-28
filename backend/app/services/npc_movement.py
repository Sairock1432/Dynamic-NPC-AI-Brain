from types import MappingProxyType
from typing import Mapping

from app.memory.database import _validate_session_id
from app.npcs import UnknownNPCError, get_npc
from app.services.npc_schedules import (
    NPCScheduleError,
    game_time_minutes,
    get_npc_schedule_location,
)
from app.services.world_events import UnknownWorldEventError, get_world_event

VALID_NPC_LOCATIONS: Mapping[str, str] = MappingProxyType(
    {
        "village_square": "Village Square",
        "forge": "Forge",
        "healer_hut": "Healer Hut",
        "market": "Market",
        "forest_edge": "Forest Edge",
        "player_location": "Player Location",
    }
)
NPC_START_LOCATION = "village_square"
MAX_NPC_MOVEMENTS_PER_EVENT = 2
MAX_MOVEMENTS_PER_NPC_PER_EVENT = 1
EVENT_TIME_WINDOWS = MappingProxyType(
    {
        "morning": (8 * 60, 12 * 60),
        "afternoon": (12 * 60, 18 * 60),
        "evening": (18 * 60, 24 * 60),
    }
)

SCHEDULE_LOCATION_IDS: Mapping[str, str] = MappingProxyType(
    {
        "At home before opening": "village_square",
        "At the forge": "forge",
        "In the village": "market",
        "At home after closing": "village_square",
        "At the healer hut": "healer_hut",
    }
)

EVENT_DESTINATIONS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "morning": MappingProxyType({"ravi": "forge", "meera": "healer_hut"}),
        "afternoon": MappingProxyType({"ravi": "market", "meera": "market"}),
        "evening": MappingProxyType({"ravi": "village_square", "meera": "healer_hut"}),
        "village_event": MappingProxyType(
            {"ravi": "village_square", "meera": "village_square"}
        ),
        "player_low_health": MappingProxyType({"meera": "player_location"}),
    }
)
EVENT_ORIGINS: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        ("morning", "ravi"): "village_square",
        ("morning", "meera"): "village_square",
        ("afternoon", "ravi"): "forge",
        ("afternoon", "meera"): "healer_hut",
        ("evening", "ravi"): "forge",
        ("evening", "meera"): "healer_hut",
    }
)

ALLOWED_MOVEMENT_TRANSITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "village_square": frozenset(VALID_NPC_LOCATIONS) - {"village_square"},
        "forge": frozenset({"village_square", "market", "player_location"}),
        "healer_hut": frozenset({"village_square", "market", "player_location"}),
        "market": frozenset(VALID_NPC_LOCATIONS) - {"market"},
        "forest_edge": frozenset({"market", "village_square"}),
        "player_location": frozenset(
            {"village_square", "forge", "healer_hut", "market"}
        ),
    }
)


class NPCMovementError(ValueError):
    """Raised when an NPC movement request is invalid or disallowed."""


def validate_npc_location(location: str) -> str:
    if not isinstance(location, str) or location not in VALID_NPC_LOCATIONS:
        raise NPCMovementError("NPC location is not allowed.")
    return location


def move_npc_for_event(
    session_id: str,
    npc_id: str,
    event_name: str,
    *,
    game_time: str,
    current_location: str | None = None,
) -> dict[str, str | bool]:
    try:
        _validate_session_id(session_id)
    except ValueError:
        raise NPCMovementError("Invalid session ID.") from None
    try:
        npc = get_npc(npc_id)
    except UnknownNPCError:
        raise NPCMovementError("Unknown NPC ID.") from None
    try:
        event = get_world_event(event_name)
    except UnknownWorldEventError:
        raise NPCMovementError("Unknown world event.") from None

    try:
        current_time = game_time_minutes(game_time)
    except NPCScheduleError as error:
        raise NPCMovementError("Game time is invalid.") from error
    time_window = EVENT_TIME_WINDOWS.get(event.event_id)
    if time_window is not None and not time_window[0] <= current_time < time_window[1]:
        raise NPCMovementError("World event does not match the current schedule time.")

    try:
        schedule_location = get_npc_schedule_location(npc.id, game_time)
    except NPCScheduleError as error:
        raise NPCMovementError("NPC schedule or game time is invalid.") from error

    if current_location is None:
        current_location = EVENT_ORIGINS.get((event.event_id, npc.id))
        if current_location is None:
            current_location = SCHEDULE_LOCATION_IDS.get(schedule_location)
        current_location = current_location or NPC_START_LOCATION
    current_location = validate_npc_location(current_location)

    destination = EVENT_DESTINATIONS.get(event.event_id, {}).get(npc.id)
    if destination is None:
        raise NPCMovementError("No movement is configured for this NPC and event.")
    validate_npc_location(destination)

    if event.event_id in {"morning", "evening"}:
        scheduled_destination = SCHEDULE_LOCATION_IDS.get(schedule_location or "")
        if scheduled_destination != destination:
            raise NPCMovementError("Destination does not match the configured schedule.")
    elif event.event_id == "afternoon":
        scheduled_destination = SCHEDULE_LOCATION_IDS.get(schedule_location or "")
        allowed_for_npc = {"ravi": {"market", "forge"}, "meera": {"market", "healer_hut"}}
        if scheduled_destination not in allowed_for_npc[npc.id]:
            raise NPCMovementError("Schedule does not permit this afternoon destination.")
        destination = scheduled_destination

    if destination == current_location:
        return {
            "status": "completed",
            "npc_id": npc.id,
            "previous_location": current_location,
            "destination": destination,
            "world_time": game_time,
            "moved": False,
        }
    if destination not in ALLOWED_MOVEMENT_TRANSITIONS[current_location]:
        raise NPCMovementError("Movement transition is not allowed.")

    return {
        "status": "completed",
        "npc_id": npc.id,
        "previous_location": current_location,
        "destination": destination,
        "world_time": game_time,
        "moved": True,
    }


def validate_movement_result(
    npc_id: str,
    event_name: str,
    game_time: str,
    result: dict[str, str | bool],
) -> dict[str, str | bool]:
    expected_keys = {
        "status",
        "npc_id",
        "previous_location",
        "destination",
        "world_time",
        "moved",
    }
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise NPCMovementError("Movement result is invalid.")
    if (
        result["status"] != "completed"
        or result["npc_id"] != npc_id
        or result["world_time"] != game_time
        or not isinstance(result["moved"], bool)
    ):
        raise NPCMovementError("Movement result does not match the current event.")
    previous_location = validate_npc_location(result["previous_location"])
    expected = move_npc_for_event(
        "npc-movement-validation",
        npc_id,
        event_name,
        game_time=game_time,
        current_location=previous_location,
    )
    if result != expected:
        raise NPCMovementError("Movement result does not match backend movement rules.")
    return result
