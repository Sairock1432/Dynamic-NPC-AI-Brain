import re

from app.memory.database import MAX_HISTORY_MESSAGES, _validate_session_id, get_conversation_history
from app.npcs import get_npc
from app.services.semantic_memory import (
    EMAIL_PATTERN,
    MEMORY_CATEGORIES,
    PHONE_PATTERN,
    SENSITIVE_MEMORY_TOPIC_PATTERN,
    SENSITIVE_PATTERN,
    add_learning_memory,
    extract_memories,
)
from app.services.npc_relationship_progression import (
    NPCRelationshipError,
    RELATIONSHIP_EVENTS,
    RELATIONSHIP_STAGES,
    transition_relationship_stage as _transition_relationship_stage,
)

MAX_NPC_LEARNING_MEMORIES_PER_INTERACTION = 1

SAFE_TOPICS = {
    "sword": "swords",
    "swords": "swords",
    "shield": "shields",
    "shields": "shields",
    "armor": "armor",
    "metal": "metal",
    "iron": "iron",
    "wood": "wood",
    "herb": "herbs",
    "herbs": "herbs",
}

SAFE_PREFERENCE_PATTERN = re.compile(
    r"\bI\s+(?:(?:usually|often|generally)\s+(?:buy|use)|prefer|like|love)\s+"
    r"(?:a|an|the)?\s*(swords?|shields?|armor|metal|iron|wood|herbs?)\b",
    re.IGNORECASE,
)
SAFE_INTEREST_PATTERN = re.compile(
    r"\bI\s+(?:am interested in|want to learn about|am curious about)\s+"
    r"(?:a|an|the)?\s*(swords?|shields?|armor|metal|iron|wood|herbs?)\b",
    re.IGNORECASE,
)
ACCEPTED_HELP_PATTERN = re.compile(
    r"\b(?:thanks[, ]+(?:that helped|you|your advice)|that helped me|"
    r"your advice helped|that worked,? thank you)\b",
    re.IGNORECASE,
)
QUESTION_PATTERN = re.compile(
    r"\b(?:what|how|where|when|why|tell me|explain|can you|could you)\b",
    re.IGNORECASE,
)
TOPIC_PATTERNS = {
    "ravi": (("metal", re.compile(r"\b(?:metal|iron|forge|blacksmith)\b", re.I)),
             ("weapons", re.compile(r"\b(?:swords?|shields?|weapons?|armor)\b", re.I))),
    "meera": (("herbs", re.compile(r"\b(?:herbs?|plants?)\b", re.I)),
               ("healing", re.compile(r"\b(?:healing|medicine|remedies|health)\b", re.I))),
}

PRIVATE_IDENTIFYING_PATTERN = re.compile(
    r"\b(?:my (?:home )?address|passport number|driver'?s license|date of birth|"
    r"born on|social security|ssn)\b",
    re.IGNORECASE,
)


class NPCLearningError(ValueError):
    """Raised when a learning category or relationship transition is invalid."""


def validate_memory_category(category: str) -> str:
    if not isinstance(category, str) or category not in MEMORY_CATEGORIES:
        raise NPCLearningError("Memory category is not allowed.")
    return category


def transition_relationship_stage(current_stage: str, interaction_event: str) -> str:
    try:
        return _transition_relationship_stage(current_stage, interaction_event)
    except NPCRelationshipError as error:
        raise NPCLearningError(str(error)) from error


def _contains_sensitive_input(message: str) -> bool:
    return bool(
        EMAIL_PATTERN.search(message)
        or PHONE_PATTERN.search(message)
        or SENSITIVE_PATTERN.search(message)
        or SENSITIVE_MEMORY_TOPIC_PATTERN.search(message)
        or PRIVATE_IDENTIFYING_PATTERN.search(message)
    )


def _safe_topic(text: str) -> str | None:
    return SAFE_TOPICS.get(text.casefold())


def _preference_candidate(message: str) -> tuple[str, str] | None:
    match = SAFE_PREFERENCE_PATTERN.search(message)
    if not match:
        return None

    topic = _safe_topic(match.group(1))
    if topic is None:
        return None
    return "PLAYER_PREFERENCE", f"The player says they prefer {topic}."


def _interest_candidate(message: str) -> tuple[str, str] | None:
    match = SAFE_INTEREST_PATTERN.search(message)
    if not match:
        return None

    topic = _safe_topic(match.group(1))
    if topic is None:
        return None
    return "PLAYER_INTEREST", f"The player is interested in {topic}."


def _interaction_pattern_candidate(
    session_id: str,
    npc_id: str,
    user_message: str,
) -> tuple[str, str] | None:
    if not QUESTION_PATTERN.search(user_message):
        return None

    history = get_conversation_history(
        session_id,
        limit=MAX_HISTORY_MESSAGES,
        npc_id=npc_id,
    )
    user_messages = [item["content"] for item in history if item["role"] == "user"]
    for topic, pattern in TOPIC_PATTERNS[npc_id]:
        matching_questions = sum(
            1
            for previous_message in user_messages
            if QUESTION_PATTERN.search(previous_message) and pattern.search(previous_message)
        )
        if matching_questions >= 2 and pattern.search(user_message):
            npc_name = get_npc(npc_id).name
            return (
                "NPC_INTERACTION_PATTERN",
                f"The player frequently asks {npc_name} about {topic}.",
            )
    return None


def _learning_candidate(
    session_id: str,
    npc_id: str,
    user_message: str,
) -> tuple[str, str] | None:
    if _contains_sensitive_input(user_message):
        return None

    existing_extractions = extract_memories(user_message)
    explicit_preference = _preference_candidate(user_message)
    if explicit_preference:
        # Preserve the existing conservative extraction signal for ordinary preferences.
        if any(memory["kind"] == "preference" for memory in existing_extractions) or re.search(
            r"\b(?:usually|often|generally)\s+(?:buy|use)\b", user_message, re.I
        ):
            return explicit_preference

    interest = _interest_candidate(user_message)
    if interest:
        return interest

    if ACCEPTED_HELP_PATTERN.search(user_message):
        npc_name = get_npc(npc_id).name
        return "PLAYER_ACCEPTED_HELP", f"The player has accepted help from {npc_name}."

    return _interaction_pattern_candidate(session_id, npc_id, user_message)


def learn_from_player_interaction(
    session_id: str,
    npc_id: str,
    user_message: str,
) -> int:
    _validate_session_id(session_id)
    get_npc(npc_id)
    if not isinstance(user_message, str) or not user_message.strip():
        return 0

    candidate = _learning_candidate(session_id, npc_id, user_message.strip())
    if candidate is None:
        return 0

    category, text = candidate
    validate_memory_category(category)
    return int(
        add_learning_memory(
            session_id,
            text,
            category,
            npc_id=npc_id,
        )
    )
