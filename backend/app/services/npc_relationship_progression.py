import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from app.memory.database import (
    MAX_HISTORY_MESSAGES,
    _validate_session_id,
    apply_npc_player_relationship_delta,
    get_conversation_history,
    get_npc_player_relationship_state,
)
from app.npcs import NPC_CONFIGS, UnknownNPCError, get_npc, get_npc_relationships
from app.services.semantic_memory import RetrievedMemory, get_npc_learning_memories
from app.services.world_events import UnknownWorldEventError, get_world_event

RELATIONSHIP_STAGES = frozenset({"NEUTRAL", "FRIENDLY", "TRUSTING"})
RELATIONSHIP_EVENTS = frozenset(
    {"successful_help", "repeated_positive_interaction", "successful_npc_communication"}
)
INITIAL_RELATIONSHIP_SCORE = 20
FRIENDLY_RELATIONSHIP_THRESHOLD = 40
TRUSTING_RELATIONSHIP_THRESHOLD = 80
RELATIONSHIP_EVENT_DELTAS: Mapping[str, int] = MappingProxyType(
    {
        "successful_help": 20,
        "repeated_positive_interaction": 15,
        "successful_npc_communication": 10,
        "gifted_item": 20,
        "disrespectful_interaction": -25,
    }
)
ALLOWED_RELATIONSHIP_DECISIONS = frozenset(
    {
        "greet_warmly",
        "offer_help",
        "share_basic_information",
        "request_assistance",
        "acknowledge_trust",
    }
)
RELATIONSHIP_TRANSITIONS: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        ("NEUTRAL", "successful_help"): "FRIENDLY",
        ("NEUTRAL", "repeated_positive_interaction"): "FRIENDLY",
        ("NEUTRAL", "successful_npc_communication"): "FRIENDLY",
        ("FRIENDLY", "successful_help"): "FRIENDLY",
        ("FRIENDLY", "repeated_positive_interaction"): "TRUSTING",
        ("FRIENDLY", "successful_npc_communication"): "TRUSTING",
        ("TRUSTING", "successful_help"): "TRUSTING",
        ("TRUSTING", "repeated_positive_interaction"): "TRUSTING",
        ("TRUSTING", "successful_npc_communication"): "TRUSTING",
    }
)

NPC_RELEVANT_TOPICS = MappingProxyType(
    {
        npc_id: frozenset(npc.relationship_topics)
        for npc_id, npc in NPC_CONFIGS.items()
    }
)
PREFERENCE_TOPIC_ORDER = ("swords", "shields", "armor", "metal", "iron", "wood", "herbs")
SAFE_TOPICS = frozenset(PREFERENCE_TOPIC_ORDER)
PREFERENCE_FACT_PATTERN = re.compile(
    r"The player says they prefer (swords|shields|armor|metal|iron|wood|herbs)\.", re.I
)
INTEREST_FACT_PATTERN = re.compile(
    r"The player is interested in (swords|shields|armor|metal|iron|wood|herbs)\.", re.I
)
POSITIVE_INTERACTION_PATTERN = re.compile(
    r"\b(?:thanks[, ]+(?:that helped|you|your advice)|that helped me|"
    r"your advice helped|that worked,? thank you|thanks again|thank you again)\b",
    re.I,
)
NEGATIVE_INTERACTION_PATTERN = re.compile(
    r"\b(?:i hate you|you(?:'re| are) (?:useless|worthless|stupid)|shut up)\b",
    re.I,
)


class NPCRelationshipError(ValueError):
    """Raised when relationship progression or a social decision is invalid."""


@dataclass(frozen=True)
class RelationshipEvaluation:
    session_id: str
    npc_id: str
    stage: str
    relevant_preferences: tuple[str, ...]
    applied_events: tuple[str, ...]
    trust_score: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not isinstance(self.npc_id, str):
            raise NPCRelationshipError("Relationship scope is invalid.")
        try:
            _validate_session_id(self.session_id)
            npc = get_npc(self.npc_id)
            get_npc_relationships(npc.id)
        except (ValueError, UnknownNPCError) as error:
            raise NPCRelationshipError("Relationship scope is invalid.") from error
        if not isinstance(self.stage, str) or self.stage not in RELATIONSHIP_STAGES:
            raise NPCRelationshipError("Relationship stage is not allowed.")
        if self.trust_score is None:
            object.__setattr__(self, "trust_score", _score_floor_for_stage(self.stage))
        elif (
            isinstance(self.trust_score, bool)
            or not isinstance(self.trust_score, int)
            or not 0 <= self.trust_score <= 100
            or relationship_stage_for_score(self.trust_score) != self.stage
        ):
            raise NPCRelationshipError("Relationship trust score is invalid.")
        if (
            not isinstance(self.relevant_preferences, tuple)
            or any(
                not isinstance(topic, str) or topic not in NPC_RELEVANT_TOPICS[npc.id]
                for topic in self.relevant_preferences
            )
        ):
            raise NPCRelationshipError("Relationship preferences are not allowed.")
        if (
            not isinstance(self.applied_events, tuple)
            or any(
                not isinstance(event, str) or event not in RELATIONSHIP_EVENTS
                for event in self.applied_events
            )
        ):
            raise NPCRelationshipError("Relationship events are not allowed.")


def transition_relationship_stage(current_stage: str, interaction_event: str) -> str:
    if not isinstance(current_stage, str) or current_stage not in RELATIONSHIP_STAGES:
        raise NPCRelationshipError("Relationship stage is not allowed.")
    if not isinstance(interaction_event, str) or interaction_event not in RELATIONSHIP_EVENTS:
        raise NPCRelationshipError("Relationship event is not allowed.")
    next_stage = RELATIONSHIP_TRANSITIONS.get((current_stage, interaction_event))
    if next_stage is None:
        raise NPCRelationshipError("Relationship transition is not allowed.")
    return next_stage


def relationship_stage_for_score(trust_score: int) -> str:
    if (
        isinstance(trust_score, bool)
        or not isinstance(trust_score, int)
        or not 0 <= trust_score <= 100
    ):
        raise NPCRelationshipError("Relationship trust score is invalid.")
    if trust_score >= TRUSTING_RELATIONSHIP_THRESHOLD:
        return "TRUSTING"
    if trust_score >= FRIENDLY_RELATIONSHIP_THRESHOLD:
        return "FRIENDLY"
    return "NEUTRAL"


def _score_floor_for_stage(stage: str) -> int:
    return {
        "NEUTRAL": INITIAL_RELATIONSHIP_SCORE,
        "FRIENDLY": FRIENDLY_RELATIONSHIP_THRESHOLD,
        "TRUSTING": TRUSTING_RELATIONSHIP_THRESHOLD,
    }[stage]


def classify_player_relationship_events(message: str) -> tuple[str, ...]:
    if not isinstance(message, str) or not message.strip():
        return ()
    if NEGATIVE_INTERACTION_PATTERN.search(message):
        return ("disrespectful_interaction",)
    if POSITIVE_INTERACTION_PATTERN.search(message):
        return ("successful_help",)
    return ()


def apply_player_relationship_events(
    session_id: str,
    npc_id: str,
    events: tuple[str, ...],
    *,
    source_message_id: int,
    initial_score: int,
) -> int:
    try:
        _validate_session_id(session_id)
        npc = get_npc(npc_id)
    except (ValueError, UnknownNPCError) as error:
        raise NPCRelationshipError("Relationship scope is invalid.") from error
    if not isinstance(events, tuple) or not events:
        raise NPCRelationshipError("Relationship events are invalid.")
    if any(not isinstance(event, str) or event not in RELATIONSHIP_EVENT_DELTAS for event in events):
        raise NPCRelationshipError("Relationship event is not allowed.")
    delta = max(
        -100,
        min(100, sum(RELATIONSHIP_EVENT_DELTAS[event] for event in events)),
    )
    try:
        return apply_npc_player_relationship_delta(
            session_id,
            npc.id,
            delta,
            source_message_id=source_message_id,
            initial_score=initial_score,
        )
    except ValueError as error:
        raise NPCRelationshipError("Relationship event could not be applied.") from error


def _validated_memory_signals(
    npc_id: str,
    memories: list[RetrievedMemory],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    npc = get_npc(npc_id)
    events: set[str] = set()
    preferences: set[str] = set()

    for memory in memories:
        if not isinstance(memory, dict):
            continue
        text = memory.get("text")
        category = memory.get("category")
        if not isinstance(text, str) or not isinstance(category, str):
            continue

        if category == "PLAYER_ACCEPTED_HELP":
            if text.casefold() == f"the player has accepted help from {npc.name}.".casefold():
                events.add("successful_help")
            continue

        if category == "PLAYER_PREFERENCE":
            match = PREFERENCE_FACT_PATTERN.fullmatch(text)
        elif category == "PLAYER_INTEREST":
            match = INTEREST_FACT_PATTERN.fullmatch(text)
        else:
            continue
        if match:
            topic = match.group(1).casefold()
            if topic in SAFE_TOPICS and topic in NPC_RELEVANT_TOPICS[npc_id]:
                preferences.add(topic)

    ordered_events = tuple(
        event
        for event in ("successful_help", "repeated_positive_interaction")
        if event in events
    )
    topic_order = tuple(topic for topic in PREFERENCE_TOPIC_ORDER if topic in preferences)
    return ordered_events, topic_order


def evaluate_player_relationship(
    session_id: str,
    npc_id: str,
    *,
    interaction_events: tuple[str, ...] = (),
    learning_memories: list[RetrievedMemory] | None = None,
) -> RelationshipEvaluation:
    try:
        _validate_session_id(session_id)
        npc = get_npc(npc_id)
        get_npc_relationships(npc.id)
    except (ValueError, UnknownNPCError) as error:
        raise NPCRelationshipError("Relationship scope is invalid.") from error

    if not isinstance(interaction_events, tuple):
        raise NPCRelationshipError("Relationship events must be a tuple.")
    if any(
        not isinstance(event, str) or event not in RELATIONSHIP_EVENTS
        for event in interaction_events
    ):
        raise NPCRelationshipError("Relationship event is not allowed.")

    memories = (
        get_npc_learning_memories(session_id, npc_id=npc.id)
        if learning_memories is None
        else learning_memories
    )
    if not isinstance(memories, list):
        raise NPCRelationshipError("Relationship learning context is invalid.")

    memory_events, preferences = _validated_memory_signals(npc.id, memories)
    try:
        history = get_conversation_history(
            session_id,
            limit=MAX_HISTORY_MESSAGES,
            npc_id=npc.id,
        )
    except Exception as error:
        raise NPCRelationshipError("Relationship history is unavailable.") from error
    positive_interactions = sum(
        1
        for message in history
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and POSITIVE_INTERACTION_PATTERN.search(message["content"])
    )
    event_signals = set(memory_events)
    if positive_interactions:
        event_signals.add("successful_help")
    if positive_interactions >= 2:
        event_signals.add("repeated_positive_interaction")
    applied_events = tuple(
        event
        for event in ("successful_help", "repeated_positive_interaction")
        if event in event_signals
    ) + interaction_events
    applied_events = tuple(dict.fromkeys(applied_events))
    legacy_stage = "NEUTRAL"
    for event in applied_events:
        legacy_stage = transition_relationship_stage(legacy_stage, event)

    stored_state = get_npc_player_relationship_state(session_id, npc.id)
    if stored_state is None:
        trust_score = _score_floor_for_stage(legacy_stage)
        for event in interaction_events:
            trust_score = max(
                0,
                min(100, trust_score + RELATIONSHIP_EVENT_DELTAS[event]),
            )
    else:
        trust_score = stored_state["trust_score"]
        for event in interaction_events:
            trust_score = max(
                0,
                min(100, trust_score + RELATIONSHIP_EVENT_DELTAS[event]),
            )
    stage = relationship_stage_for_score(trust_score)

    return RelationshipEvaluation(
        session_id=session_id,
        npc_id=npc.id,
        stage=stage,
        relevant_preferences=preferences,
        applied_events=applied_events,
        trust_score=trust_score,
    )


def select_relationship_decision(
    evaluation: RelationshipEvaluation,
    *,
    event_name: str | None,
) -> tuple[str | None, str | None]:
    if not isinstance(evaluation, RelationshipEvaluation):
        raise NPCRelationshipError("Relationship evaluation is invalid.")
    if event_name is not None:
        if not isinstance(event_name, str):
            raise NPCRelationshipError("World event is not allowed.")
        try:
            get_world_event(event_name)
        except UnknownWorldEventError as error:
            raise NPCRelationshipError("World event is not allowed.") from error

    if evaluation.stage == "TRUSTING":
        decision, topic = "acknowledge_trust", None
    elif evaluation.relevant_preferences:
        decision, topic = "share_basic_information", evaluation.relevant_preferences[0]
    elif evaluation.stage == "FRIENDLY" and event_name == "player_low_health":
        has_trusted_healer = any(
            relationship.role.casefold() == "healer"
            and relationship.trust >= 4
            and relationship.cooperation >= 4
            for relationship in get_npc_relationships(evaluation.npc_id)
        )
        decision = "request_assistance" if has_trusted_healer else "offer_help"
        topic = None
    elif evaluation.stage == "FRIENDLY":
        decision, topic = "greet_warmly", None
    elif event_name == "player_low_health" and evaluation.npc_id == "meera":
        decision, topic = "offer_help", None
    else:
        return None, None

    validate_relationship_decision(evaluation.npc_id, decision, topic)
    return decision, topic


def validate_relationship_decision(
    npc_id: str,
    decision: str | None,
    topic: str | None,
) -> str | None:
    try:
        npc = get_npc(npc_id)
        get_npc_relationships(npc.id)
    except UnknownNPCError as error:
        raise NPCRelationshipError("Relationship NPC is not configured.") from error

    if decision is None:
        if topic is not None:
            raise NPCRelationshipError("Relationship topic has no selected decision.")
        return None
    if not isinstance(decision, str) or decision not in ALLOWED_RELATIONSHIP_DECISIONS:
        raise NPCRelationshipError("Relationship decision is not allowed.")
    if decision == "share_basic_information":
        if not isinstance(topic, str) or topic not in NPC_RELEVANT_TOPICS[npc.id]:
            raise NPCRelationshipError("Relationship topic is not relevant to this NPC.")
    elif topic is not None:
        raise NPCRelationshipError("Relationship topic is not allowed for this decision.")
    if decision == "request_assistance" and not any(
        relationship.role.casefold() == "healer"
        and relationship.trust >= 4
        and relationship.cooperation >= 4
        for relationship in get_npc_relationships(npc.id)
    ):
        raise NPCRelationshipError("Trusted assistance is not configured for this NPC.")
    return decision