import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from app.npcs import UnknownNPCError, get_npc
from app.services.npc_decision import ALLOWED_NPC_DECISIONS, NPCDecision
from app.services.npc_goals import NPCGoalError, get_npc_goal
from app.services.world_events import UnknownWorldEventError, get_world_event

MAX_PLAN_STEPS = 3
ALLOWED_INTENTS = frozenset(
    {
        "ASK_SWORD",
        "ASK_HERBS",
        "ASK_HEALING",
        "ASK_BLACKSMITH_HELP",
        "ASK_GENERAL_HELP",
    }
)
ALLOWED_PLAN_STEPS = frozenset(
    {
        "identify_player_need",
        "inspect_supplies",
        "offer_help",
        "provide_sword_information",
        "provide_herb_information",
        "request_npc_help",
        "move_to_location",
        "speak_to_player",
        "stay_at_location",
    }
)


class NPCPlanError(ValueError):
    """Raised when an intent or plan is invalid or unsupported."""


@dataclass(frozen=True)
class PlanTemplate:
    goal_id: str
    steps: tuple[str, ...]


@dataclass(frozen=True)
class NPCPlan:
    npc_id: str
    source: Literal["player_request", "world_event"]
    intent: str
    steps: tuple[str, ...]
    goal_id: str
    event_name: str | None = None
    decision: str | None = None


PLAYER_PLAN_TEMPLATES: Mapping[tuple[str, str], PlanTemplate] = MappingProxyType(
    {
        ("ravi", "ASK_SWORD"): PlanTemplate(
            "help_travelers",
            ("identify_player_need", "provide_sword_information", "speak_to_player"),
        ),
        ("ravi", "ASK_BLACKSMITH_HELP"): PlanTemplate(
            "help_travelers",
            ("identify_player_need", "inspect_supplies", "speak_to_player"),
        ),
        ("ravi", "ASK_HEALING"): PlanTemplate(
            "help_travelers",
            ("identify_player_need", "request_npc_help", "speak_to_player"),
        ),
        ("ravi", "ASK_GENERAL_HELP"): PlanTemplate(
            "help_travelers",
            ("identify_player_need", "offer_help", "speak_to_player"),
        ),
        ("meera", "ASK_HERBS"): PlanTemplate(
            "help_injured_travelers",
            ("identify_player_need", "provide_herb_information", "speak_to_player"),
        ),
        ("meera", "ASK_HEALING"): PlanTemplate(
            "help_injured_travelers",
            ("identify_player_need", "offer_help", "speak_to_player"),
        ),
        ("meera", "ASK_GENERAL_HELP"): PlanTemplate(
            "help_injured_travelers",
            ("identify_player_need", "offer_help", "speak_to_player"),
        ),
    }
)

EVENT_PLAN_TEMPLATES: Mapping[tuple[str, str, str], PlanTemplate] = MappingProxyType(
    {
        ("morning", "ravi", "stay_at_location"): PlanTemplate(
            "maintain_forge", ("move_to_location", "speak_to_player")
        ),
        ("morning", "ravi", "offer_sword_information"): PlanTemplate(
            "help_travelers",
            ("move_to_location", "provide_sword_information", "speak_to_player"),
        ),
        ("morning", "meera", "stay_at_location"): PlanTemplate(
            "maintain_medicine_supplies", ("move_to_location", "speak_to_player")
        ),
        ("morning", "meera", "offer_herb_information"): PlanTemplate(
            "help_injured_travelers",
            ("move_to_location", "provide_herb_information", "speak_to_player"),
        ),
        ("afternoon", "ravi", "stay_at_location"): PlanTemplate(
            "help_travelers", ("move_to_location", "speak_to_player")
        ),
        ("afternoon", "meera", "stay_at_location"): PlanTemplate(
            "protect_villagers_health", ("move_to_location", "speak_to_player")
        ),
        ("evening", "ravi", "stay_at_location"): PlanTemplate(
            "maintain_forge", ("move_to_location", "speak_to_player")
        ),
        ("evening", "meera", "stay_at_location"): PlanTemplate(
            "maintain_medicine_supplies", ("move_to_location", "speak_to_player")
        ),
        ("player_low_health", "ravi", "request_help_from_npc"): PlanTemplate(
            "help_travelers",
            ("identify_player_need", "request_npc_help", "speak_to_player"),
        ),
        ("player_low_health", "meera", "offer_help"): PlanTemplate(
            "protect_villagers_health",
            ("move_to_location", "offer_help", "speak_to_player"),
        ),
        ("village_event", "ravi", "speak_to_player"): PlanTemplate(
            "help_travelers", ("move_to_location", "speak_to_player")
        ),
        ("village_event", "meera", "speak_to_player"): PlanTemplate(
            "help_injured_travelers", ("move_to_location", "speak_to_player")
        ),
    }
)

REQUEST_PATTERN = re.compile(
    r"\b(?:tell me|explain|teach me|learn about|know about|need help|help me|"
    r"can you|could you|what|how|recommend|advice)\b",
    re.IGNORECASE,
)
MUTATION_REQUEST_PATTERN = re.compile(
    r"\b(?:buy|purchase|give|hand over|donate|heal|restore)\b",
    re.IGNORECASE,
)


def validate_intent(intent: str) -> str:
    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise NPCPlanError("Intent is not allowed.")
    return intent


def classify_player_intent(npc_id: str, message: str) -> str | None:
    try:
        get_npc(npc_id)
    except UnknownNPCError:
        raise NPCPlanError("Unknown NPC ID.") from None
    if not isinstance(message, str) or not message.strip() or MUTATION_REQUEST_PATTERN.search(message):
        return None
    if not REQUEST_PATTERN.search(message):
        return None

    text = message.casefold()
    if re.search(r"\b(?:sword|swords)\b", text):
        intent = "ASK_SWORD"
    elif re.search(r"\b(?:herb|herbs)\b", text):
        intent = "ASK_HERBS"
    elif re.search(r"\b(?:heal|healing|medicine|treatment)\b", text):
        intent = "ASK_HEALING"
    elif re.search(r"\b(?:blacksmith|forge|metal|iron|wood|repair)\b", text):
        intent = "ASK_BLACKSMITH_HELP"
    elif re.search(r"\b(?:help|assist|support)\b", text):
        intent = "ASK_GENERAL_HELP"
    else:
        return None

    return intent if (npc_id, intent) in PLAYER_PLAN_TEMPLATES else None


def create_player_plan(npc_id: str, intent: str) -> NPCPlan:
    try:
        get_npc(npc_id)
    except UnknownNPCError:
        raise NPCPlanError("Unknown NPC ID.") from None
    validate_intent(intent)
    template = PLAYER_PLAN_TEMPLATES.get((npc_id, intent))
    if template is None:
        raise NPCPlanError("This NPC is not configured for that intent.")
    try:
        goal = get_npc_goal(npc_id, template.goal_id)
    except NPCGoalError as error:
        raise NPCPlanError("Plan goal is not configured.") from error
    plan = NPCPlan(
        npc_id=npc_id,
        source="player_request",
        intent=intent,
        steps=template.steps,
        goal_id=goal.goal_id,
    )
    return validate_npc_plan(plan)


def create_event_plan(npc_id: str, event_name: str, decision: NPCDecision) -> NPCPlan:
    try:
        get_npc(npc_id)
        event = get_world_event(event_name)
    except UnknownNPCError:
        raise NPCPlanError("Unknown NPC ID.") from None
    except UnknownWorldEventError:
        raise NPCPlanError("Unknown world event.") from None
    if not isinstance(decision, NPCDecision) or decision.npc_id != npc_id:
        raise NPCPlanError("Decision does not match the planned NPC.")
    if decision.decision not in ALLOWED_NPC_DECISIONS:
        raise NPCPlanError("Decision is not allowed.")
    template = EVENT_PLAN_TEMPLATES.get((event.event_id, npc_id, decision.decision))
    if template is None or template.goal_id != decision.goal_id:
        raise NPCPlanError("No plan is configured for this event decision.")
    plan = NPCPlan(
        npc_id=npc_id,
        source="world_event",
        intent=decision.decision,
        steps=template.steps,
        goal_id=decision.goal_id,
        event_name=event.event_id,
        decision=decision.decision,
    )
    return validate_npc_plan(plan)


def validate_npc_plan(plan: NPCPlan) -> NPCPlan:
    if not isinstance(plan, NPCPlan):
        raise NPCPlanError("Plan structure is invalid.")
    try:
        get_npc(plan.npc_id)
    except UnknownNPCError:
        raise NPCPlanError("Unknown NPC ID.") from None
    if not isinstance(plan.steps, tuple) or not plan.steps:
        raise NPCPlanError("Plan must contain at least one step.")
    if len(plan.steps) > MAX_PLAN_STEPS:
        raise NPCPlanError("Plan exceeds the maximum step count.")
    if any(not isinstance(step, str) or step not in ALLOWED_PLAN_STEPS for step in plan.steps):
        raise NPCPlanError("Plan contains an unknown step.")

    if plan.source == "player_request":
        validate_intent(plan.intent)
        template = PLAYER_PLAN_TEMPLATES.get((plan.npc_id, plan.intent))
    elif plan.source == "world_event":
        try:
            event = get_world_event(plan.event_name or "")
        except UnknownWorldEventError:
            raise NPCPlanError("Unknown world event.") from None
        if plan.intent not in ALLOWED_NPC_DECISIONS or plan.decision != plan.intent:
            raise NPCPlanError("Event plan decision is not allowed.")
        template = EVENT_PLAN_TEMPLATES.get((event.event_id, plan.npc_id, plan.decision))
    else:
        raise NPCPlanError("Plan source is not allowed.")

    if template is None or plan.steps != template.steps or plan.goal_id != template.goal_id:
        raise NPCPlanError("Plan steps or ordering do not match backend configuration.")
    try:
        goal = get_npc_goal(plan.npc_id, plan.goal_id)
    except NPCGoalError as error:
        raise NPCPlanError("Plan goal is not configured.") from error
    if goal.goal_id != plan.goal_id:
        raise NPCPlanError("Plan goal is invalid.")
    return plan
