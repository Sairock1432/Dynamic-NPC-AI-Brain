from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from app.npcs import UnknownNPCError, get_npc


@dataclass(frozen=True)
class NPCGoal:
    goal_id: str
    description: str
    priority: int


class NPCGoalError(ValueError):
    """Raised when an NPC goal is not configured."""


NPC_GOALS: Mapping[str, tuple[NPCGoal, ...]] = MappingProxyType(
    {
        "ravi": (
            NPCGoal("maintain_forge", "Protect and maintain the village forge.", 1),
            NPCGoal("help_travelers", "Help travelers with blacksmith-related needs.", 2),
            NPCGoal("maintain_material_supplies", "Maintain useful metal and wood supplies.", 3),
        ),
        "meera": (
            NPCGoal("protect_villagers_health", "Protect the health of villagers.", 1),
            NPCGoal("help_injured_travelers", "Help injured travelers with practical care.", 2),
            NPCGoal("maintain_medicine_supplies", "Maintain medicine and herb supplies.", 3),
        ),
    }
)


def get_npc_goals(npc_id: str) -> tuple[NPCGoal, ...]:
    try:
        get_npc(npc_id)
    except UnknownNPCError:
        raise NPCGoalError("Unknown NPC ID.") from None
    return NPC_GOALS.get(npc_id, ())


def get_npc_goal(npc_id: str, goal_id: str) -> NPCGoal:
    for goal in get_npc_goals(npc_id):
        if goal.goal_id == goal_id:
            return goal
    raise NPCGoalError("Unknown goal for NPC.")
