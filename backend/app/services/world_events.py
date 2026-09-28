from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class WorldEvent:
    event_id: str
    description: str
    npc_behaviors: tuple[tuple[str, str], ...]


class UnknownWorldEventError(ValueError):
    """Raised when an event is not in the world-event registry."""


WORLD_EVENTS: Mapping[str, WorldEvent] = MappingProxyType(
    {
        "morning": WorldEvent(
            event_id="morning",
            description="Morning begins in the village.",
            npc_behaviors=(
                ("ravi", "Open the blacksmith shop and prepare the forge."),
                ("meera", "Prepare healing supplies for the day."),
            ),
        ),
        "afternoon": WorldEvent(
            event_id="afternoon",
            description="The village settles into the afternoon.",
            npc_behaviors=(
                ("ravi", "Continue blacksmith work and help villagers as needed."),
                ("meera", "Attend to villagers and organize remedies."),
            ),
        ),
        "evening": WorldEvent(
            event_id="evening",
            description="Evening falls over the village.",
            npc_behaviors=(
                ("ravi", "Close the forge and wind down for the evening."),
                ("meera", "Finish tending patients and organize healing supplies."),
            ),
        ),
        "player_low_health": WorldEvent(
            event_id="player_low_health",
            description="The traveler is in poor health and may need care.",
            npc_behaviors=(
                ("ravi", "Ask Meera for practical advice to support the unwell traveler."),
                (
                    "meera",
                    "Offer calm, practical support and invite the traveler to the healer's hut. "
                    "Do not change the traveler's health.",
                ),
            ),
        ),
        "village_event": WorldEvent(
            event_id="village_event",
            description="A gathering is taking place in the village square.",
            npc_behaviors=(
                ("ravi", "Join the village gathering when his schedule allows."),
                ("meera", "Join the village gathering when her schedule allows."),
            ),
        ),
    }
)


def get_world_event(event_name: str) -> WorldEvent:
    if not isinstance(event_name, str) or not event_name.strip():
        raise UnknownWorldEventError("World event name must not be empty.")
    event = WORLD_EVENTS.get(event_name.strip())
    if event is None:
        raise UnknownWorldEventError("Unknown world event.")
    return event
