from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping

from app.memory.database import _validate_session_id
from app.memory.database import InventoryItem, PlayerState
from app.npcs import (
    NPCConfig,
    NPCRelationship,
    UnknownNPCError,
    get_npc,
    get_npc_relationships,
)
from app.services.npc_goals import NPCGoal, NPCGoalError, get_npc_goal
from app.services.npc_relationship_progression import (
    NPCRelationshipError,
    RelationshipEvaluation,
    NPC_RELEVANT_TOPICS,
    select_relationship_decision,
    validate_relationship_decision,
)
from app.services.semantic_memory import RetrievedMemory
from app.services.world_events import UnknownWorldEventError, get_world_event

MAX_NPC_DECISIONS_PER_EVENT = 2
LOW_HEALTH_THRESHOLD = 40
ALLOWED_NPC_DIALOGUE_DECISIONS = frozenset(
    {
        "neutral_greeting",
        "friendly_greeting",
        "trusting_greeting",
        "acknowledge_previous_help",
        "offer_contextual_help",
        "provide_validated_information",
        "request_assistance",
        "normal_response",
    }
)
NPC_DIALOGUE_INTENTS = frozenset(
    {
        "ASK_SWORD",
        "ASK_HERBS",
        "ASK_HEALING",
        "ASK_BLACKSMITH_HELP",
        "ASK_GENERAL_HELP",
    }
)
NPC_DIALOGUE_INFORMATION_TOPICS = MappingProxyType(
    {"ASK_SWORD": "swords", "ASK_HERBS": "herbs"}
)
NPC_DIALOGUE_HELP_INTENTS = frozenset(
    {"ASK_HEALING", "ASK_BLACKSMITH_HELP", "ASK_GENERAL_HELP"}
)
GREETING_PATTERN = re.compile(
    r"\s*(?:hi|hello|hey|greetings|good morning|good afternoon|good evening)"
    r"(?:[,! ]+(?:ravi|meera|friend|traveler))?[.!?\s]*\s*",
    re.IGNORECASE,
)
PREVIOUS_HELP_ACKNOWLEDGMENT_PATTERN = re.compile(
    r"\b(?:thanks again|thank you again|you helped me before|your help last time|"
    r"remember when you helped me)\b",
    re.IGNORECASE,
)
ALLOWED_NPC_DECISIONS = frozenset(
    {
        "stay_at_location",
        "speak_to_player",
        "request_help_from_npc",
        "offer_help",
        "inspect_supplies",
        "offer_sword_information",
        "offer_herb_information",
    }
)

EVENT_DECISION_RULES: Mapping[str, Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "morning": MappingProxyType(
            {
                "ravi": ("stay_at_location", "maintain_forge"),
                "meera": ("stay_at_location", "maintain_medicine_supplies"),
            }
        ),
        "afternoon": MappingProxyType(
            {
                "ravi": ("stay_at_location", "help_travelers"),
                "meera": ("stay_at_location", "protect_villagers_health"),
            }
        ),
        "evening": MappingProxyType(
            {
                "ravi": ("stay_at_location", "maintain_forge"),
                "meera": ("stay_at_location", "maintain_medicine_supplies"),
            }
        ),
        "player_low_health": MappingProxyType(
            {
                "ravi": ("request_help_from_npc", "help_travelers"),
                "meera": ("offer_help", "protect_villagers_health"),
            }
        ),
        "village_event": MappingProxyType(
            {
                "ravi": ("speak_to_player", "help_travelers"),
                "meera": ("speak_to_player", "help_injured_travelers"),
            }
        ),
    }
)


class NPCDecisionError(ValueError):
    """Raised when an NPC decision is not allowed or configured."""


@dataclass(frozen=True)
class NPCDialogueDecision:
    session_id: str
    npc_id: str
    decision: str
    topic: str | None = None

    def __post_init__(self) -> None:
        try:
            _validate_session_id(self.session_id)
            get_npc(self.npc_id)
            validate_npc_dialogue_decision(self.npc_id, self.decision, self.topic)
        except (ValueError, UnknownNPCError) as error:
            raise NPCDecisionError("NPC dialogue decision is invalid.") from error


def validate_npc_dialogue_decision(
    npc_id: str,
    decision: str,
    topic: str | None = None,
) -> str:
    try:
        npc = get_npc(npc_id)
    except UnknownNPCError as error:
        raise NPCDecisionError("Unknown NPC ID.") from error
    if not isinstance(decision, str) or decision not in ALLOWED_NPC_DIALOGUE_DECISIONS:
        raise NPCDecisionError("NPC dialogue decision is not allowed.")
    if decision == "provide_validated_information":
        if not isinstance(topic, str) or topic not in NPC_RELEVANT_TOPICS[npc.id]:
            raise NPCDecisionError("NPC dialogue topic is not configured.")
    elif topic is not None:
        raise NPCDecisionError("NPC dialogue topic is not allowed for this decision.")
    if decision == "request_assistance" and not any(
        relationship.role.casefold() == "healer"
        and relationship.trust >= 4
        and relationship.cooperation >= 4
        for relationship in get_npc_relationships(npc.id)
    ):
        raise NPCDecisionError("Trusted assistance is not configured for this NPC.")
    return decision


def is_relationship_dialogue_candidate(message: str, intent: str | None) -> bool:
    if intent is not None:
        return isinstance(intent, str) and intent in NPC_DIALOGUE_INTENTS
    if not isinstance(message, str):
        return False
    return bool(
        GREETING_PATTERN.fullmatch(message)
        or PREVIOUS_HELP_ACKNOWLEDGMENT_PATTERN.search(message)
    )


def select_npc_dialogue(
    npc: NPCConfig,
    relationship: RelationshipEvaluation,
    *,
    session_id: str,
    message: str,
    intent: str | None,
) -> NPCDialogueDecision | None:
    try:
        configured_npc = get_npc(npc.id)
    except (AttributeError, UnknownNPCError):
        raise NPCDecisionError("Unknown NPC ID.") from None
    if configured_npc != npc:
        raise NPCDecisionError("NPC configuration is not trusted.")
    if not isinstance(relationship, RelationshipEvaluation):
        raise NPCDecisionError("Relationship context is invalid.")
    if relationship.npc_id != npc.id:
        raise NPCDecisionError("Relationship context does not match this NPC.")
    if relationship.session_id != session_id:
        raise NPCDecisionError("Relationship context does not match this session.")
    if (
        not isinstance(relationship.stage, str)
        or relationship.stage not in {"NEUTRAL", "FRIENDLY", "TRUSTING"}
    ):
        raise NPCDecisionError("Relationship stage is not allowed.")
    if intent is not None and intent not in NPC_DIALOGUE_INTENTS:
        raise NPCDecisionError("NPC dialogue intent is not allowed.")

    stage = relationship.stage
    if intent in NPC_DIALOGUE_INFORMATION_TOPICS:
        topic = NPC_DIALOGUE_INFORMATION_TOPICS[intent]
        if topic not in NPC_RELEVANT_TOPICS[npc.id]:
            raise NPCDecisionError("NPC dialogue topic is not configured.")
        decision, selected_topic = "provide_validated_information", topic
    elif intent in NPC_DIALOGUE_HELP_INTENTS:
        if stage == "NEUTRAL":
            decision, selected_topic = "normal_response", None
        elif intent == "ASK_HEALING" and any(
            relationship_config.role.casefold() == "healer"
            and relationship_config.trust >= 4
            and relationship_config.cooperation >= 4
            for relationship_config in get_npc_relationships(npc.id)
        ):
            decision, selected_topic = "request_assistance", None
        else:
            decision, selected_topic = "offer_contextual_help", None
    elif (
        PREVIOUS_HELP_ACKNOWLEDGMENT_PATTERN.search(message)
        and "successful_help" in relationship.applied_events
        and stage != "NEUTRAL"
    ):
        decision, selected_topic = "acknowledge_previous_help", None
    elif GREETING_PATTERN.fullmatch(message):
        decision = {
            "NEUTRAL": "neutral_greeting",
            "FRIENDLY": "friendly_greeting",
            "TRUSTING": "trusting_greeting",
        }[stage]
        selected_topic = None
    else:
        return None

    return NPCDialogueDecision(
        session_id=relationship.session_id,
        npc_id=npc.id,
        decision=decision,
        topic=selected_topic,
    )


def validate_npc_decision(decision: str) -> str:
    if not isinstance(decision, str) or decision not in ALLOWED_NPC_DECISIONS:
        raise NPCDecisionError("NPC decision is not allowed.")
    return decision


@dataclass(frozen=True)
class NPCDecision:
    npc_id: str
    decision: str
    goal_id: str
    goal_description: str
    goal_priority: int
    schedule_location: str
    target_npc_id: str | None = None
    message: str | None = None
    relationship_trust: int | None = None
    relationship_cooperation: int | None = None
    learning_topic: str | None = None
    relationship_decision: str | None = None
    relationship_topic: str | None = None

    def __post_init__(self) -> None:
        validate_npc_decision(self.decision)
        expected_topic = {
            "offer_sword_information": "swords",
            "offer_herb_information": "herbs",
        }.get(self.decision)
        if expected_topic is not None and self.learning_topic != expected_topic:
            raise NPCDecisionError("Decision does not match its validated learning topic.")
        if expected_topic is None and self.learning_topic is not None:
            raise NPCDecisionError("Learning topic is not allowed for this decision.")
        try:
            validate_relationship_decision(
                self.npc_id,
                self.relationship_decision,
                self.relationship_topic,
            )
        except NPCRelationshipError as error:
            raise NPCDecisionError(str(error)) from error


def _has_learning_topic(
    memories: list[RetrievedMemory],
    *,
    topic: str,
) -> bool:
    allowed_categories = {
        "PLAYER_PREFERENCE",
        "PLAYER_INTEREST",
        "NPC_INTERACTION_PATTERN",
    }
    return any(
        memory.get("category") in allowed_categories
        and topic in memory.get("text", "").casefold()
        for memory in memories
    )


def _trusted_help_relationship(
    relationships: tuple[NPCRelationship, ...],
) -> NPCRelationship:
    for relationship in relationships:
        if (
            relationship.role.casefold() == "healer"
            and relationship.trust >= 4
            and relationship.cooperation >= 4
        ):
            return relationship
    raise NPCDecisionError("No trusted helper is configured for this NPC.")


def decide_npc_action(
    npc: NPCConfig,
    game_state: PlayerState,
    inventory: list[InventoryItem],
    *,
    event_name: str | None,
    schedule_location: str,
    learning_memories: list[RetrievedMemory] | None = None,
    relationship_evaluation: RelationshipEvaluation | None = None,
) -> NPCDecision | None:
    try:
        configured_npc = get_npc(npc.id)
    except (AttributeError, UnknownNPCError):
        raise NPCDecisionError("Unknown NPC ID.") from None
    if configured_npc != npc:
        raise NPCDecisionError("NPC configuration is not trusted.")
    if relationship_evaluation is not None and (
        relationship_evaluation.npc_id != npc.id
        or relationship_evaluation.session_id != game_state.get("session_id")
    ):
        raise NPCDecisionError("Relationship context does not match this NPC and session.")

    if event_name is None:
        rule = ("speak_to_player", "help_travelers" if npc.id == "ravi" else "help_injured_travelers")
    else:
        try:
            event = get_world_event(event_name)
        except UnknownWorldEventError:
            raise NPCDecisionError("Unknown world event.") from None
        if event.event_id == "player_low_health" and game_state["health"] >= LOW_HEALTH_THRESHOLD:
            return None
        if npc.id not in {event_npc_id for event_npc_id, _ in event.npc_behaviors}:
            return None
        rule = EVENT_DECISION_RULES.get(event.event_id, {}).get(npc.id)
        if rule is None:
            raise NPCDecisionError("No decision is configured for this event and NPC.")

    decision_name, goal_id = rule
    try:
        goal: NPCGoal = get_npc_goal(npc.id, goal_id)
    except NPCGoalError as error:
        raise NPCDecisionError("NPC goal configuration is invalid.") from error

    decision_data: dict[str, str | int | None] = {
        "target_npc_id": None,
        "message": None,
        "relationship_trust": None,
        "relationship_cooperation": None,
        "learning_topic": None,
        "relationship_decision": None,
        "relationship_topic": None,
    }
    if event_name == "morning" and npc.id == "ravi" and _has_learning_topic(
        learning_memories or [], topic="swords"
    ):
        decision_name, goal_id = "offer_sword_information", "help_travelers"
        decision_data["learning_topic"] = "swords"
    elif event_name == "morning" and npc.id == "meera" and _has_learning_topic(
        learning_memories or [], topic="herb"
    ):
        decision_name, goal_id = "offer_herb_information", "help_injured_travelers"
        decision_data["learning_topic"] = "herbs"
    try:
        goal = get_npc_goal(npc.id, goal_id)
    except NPCGoalError as error:
        raise NPCDecisionError("NPC goal configuration is invalid.") from error
    if decision_name == "request_help_from_npc":
        relationship = _trusted_help_relationship(get_npc_relationships(npc.id))
        decision_data.update(
            {
                "target_npc_id": relationship.npc_id,
                "message": "A traveler is unwell. What practical support should I offer?",
                "relationship_trust": relationship.trust,
                "relationship_cooperation": relationship.cooperation,
            }
        )
    if relationship_evaluation is not None:
        try:
            social_decision, relationship_topic = select_relationship_decision(
                relationship_evaluation,
                event_name=event_name,
            )
        except NPCRelationshipError as error:
            raise NPCDecisionError("Relationship context is invalid.") from error
        decision_data["relationship_decision"] = social_decision
        decision_data["relationship_topic"] = relationship_topic

    return NPCDecision(
        npc_id=npc.id,
        decision=decision_name,
        goal_id=goal.goal_id,
        goal_description=goal.description,
        goal_priority=goal.priority,
        schedule_location=schedule_location,
        **decision_data,
    )
