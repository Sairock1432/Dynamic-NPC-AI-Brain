import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from app.npcs import UnknownNPCError, get_npc

CLOCK_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class ScheduleBlock:
    start: str
    end: str
    location: str


NPC_SCHEDULES: Mapping[str, tuple[ScheduleBlock, ...]] = MappingProxyType(
    {
        "ravi": (
            ScheduleBlock("00:00", "08:00", "At home before opening"),
            ScheduleBlock("08:00", "12:00", "At the forge"),
            ScheduleBlock("12:00", "14:00", "In the village"),
            ScheduleBlock("14:00", "18:00", "At the forge"),
            ScheduleBlock("18:00", "24:00", "At home after closing"),
        ),
        "meera": (
            ScheduleBlock("00:00", "08:00", "At the healer hut"),
            ScheduleBlock("08:00", "12:00", "At the healer hut"),
            ScheduleBlock("12:00", "14:00", "In the village"),
            ScheduleBlock("14:00", "18:00", "At the healer hut"),
            ScheduleBlock("18:00", "24:00", "At the healer hut"),
        ),
    }
)


class NPCScheduleError(ValueError):
    """Raised when an NPC schedule or game-time value is invalid."""


def _clock_minutes(value: str, *, allow_end_of_day: bool = False) -> int:
    if allow_end_of_day and value == "24:00":
        return 24 * 60
    if not isinstance(value, str) or not CLOCK_PATTERN.fullmatch(value):
        raise NPCScheduleError("Game time must use the HH:MM format.")
    hours, minutes = (int(part) for part in value.split(":"))
    return hours * 60 + minutes


def game_time_minutes(value: str) -> int:
    return _clock_minutes(value)


def get_npc_schedule_location(npc_id: str, game_time: str) -> str:
    try:
        get_npc(npc_id)
    except UnknownNPCError:
        raise NPCScheduleError("Unknown NPC ID.") from None

    current_minutes = _clock_minutes(game_time)
    schedule = NPC_SCHEDULES.get(npc_id)
    if schedule is None:
        raise NPCScheduleError("No schedule is configured for this NPC.")

    for block in schedule:
        start = _clock_minutes(block.start)
        end = _clock_minutes(block.end, allow_end_of_day=True)
        if start <= current_minutes < end:
            return block.location
    raise NPCScheduleError("No schedule block matches the current game time.")
