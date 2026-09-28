import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import chromadb

from app.memory.database import _validate_session_id
from app.npcs import get_npc

CHROMA_PATH = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "mystery_village_player_memories"
MAX_MEMORY_RESULTS = 3
MAX_MEMORY_LENGTH = 240
MAX_NPC_LEARNING_MEMORIES = 20
MEMORY_CATEGORY_KINDS = {
    "PLAYER_PREFERENCE": "preference",
    "PLAYER_INTEREST": "decision",
    "NPC_INTERACTION_PATTERN": "fact",
    "PLAYER_ACCEPTED_HELP": "fact",
}
MEMORY_CATEGORIES = frozenset(MEMORY_CATEGORY_KINDS)
LEARNING_FACT_PATTERNS = {
    "PLAYER_PREFERENCE": re.compile(
        r"The player says they prefer (swords|shields|armor|metal|iron|wood|herbs)\.",
        re.IGNORECASE,
    ),
    "PLAYER_INTEREST": re.compile(
        r"The player is interested in (swords|shields|armor|metal|iron|wood|herbs)\.",
        re.IGNORECASE,
    ),
    "NPC_INTERACTION_PATTERN": re.compile(
        r"The player frequently asks (Ravi|Meera) about (metal|weapons|herbs|healing)\.",
        re.IGNORECASE,
    ),
    "PLAYER_ACCEPTED_HELP": re.compile(
        r"The player has accepted help from (Ravi|Meera)\.",
        re.IGNORECASE,
    ),
}
SENSITIVE_MEMORY_TOPIC_PATTERN = re.compile(
    r"\b(?:religion|religious|politic(?:s|al)?|race|ethnicity|sexual orientation|"
    r"health condition|diagnosis|disease|income|salary|bank account|credit card|"
    r"home address|date of birth|social security)\b",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d(). -]{7,}\d)")
SENSITIVE_PATTERN = re.compile(
    r"\b(?:api[\s_-]?key|password|passphrase|authorization|secret|bearer|"
    r"access token|refresh token|environment variable)\b",
    re.IGNORECASE,
)
NAME_PATTERN = re.compile(
    r"\bmy name is\s+([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})",
    re.IGNORECASE,
)
PREFERENCE_PATTERN = re.compile(
    r"\bI\s+(like|love|prefer|dislike|hate)\s+([^.!?\r\n]{2,120})",
    re.IGNORECASE,
)
ITEM_DECISION_PATTERN = re.compile(
    r"\bI\s+(?:want|need|plan to|decided to)\s+(?:to buy\s+)?"
    r"(?:a|an|the)?\s*(sword|shield|armor|iron|wood)\b",
    re.IGNORECASE,
)


class ExtractedMemory(TypedDict):
    text: str
    kind: str


class RetrievedMemory(TypedDict, total=False):
    text: str
    kind: str
    category: str


@lru_cache(maxsize=1)
def _get_client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


@lru_cache(maxsize=1)
def _get_collection():
    collection = _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    _migrate_legacy_memories_to_ravi(collection)
    return collection


def _migrate_legacy_memories_to_ravi(collection) -> None:
    legacy = collection.get(include=["metadatas"])
    ids_to_update = []
    metadata_to_update = []
    for memory_id, metadata in zip(legacy.get("ids", []), legacy.get("metadatas", [])):
        if metadata and metadata.get("session_id") and not metadata.get("npc_id"):
            ids_to_update.append(memory_id)
            metadata_to_update.append({**metadata, "npc_id": "ravi"})
    if ids_to_update:
        collection.update(ids=ids_to_update, metadatas=metadata_to_update)


def close_client() -> None:
    if _get_client.cache_info().currsize:
        _get_client().close()
    _get_collection.cache_clear()
    _get_client.cache_clear()


def _validate_memory_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Memory text must be a string.")
    text = text.strip()
    if not text or len(text) > MAX_MEMORY_LENGTH:
        raise ValueError("Memory text is empty or too long.")
    if EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text) or SENSITIVE_PATTERN.search(text):
        raise ValueError("Memory text contains sensitive information.")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
        raise ValueError("Memory text contains unsupported control characters.")
    return text


def add_memory(
    session_id: str,
    text: str,
    metadata: dict[str, str] | None = None,
    *,
    npc_id: str = "ravi",
) -> str:
    _validate_session_id(session_id)
    get_npc(npc_id)
    text = _validate_memory_text(text)
    metadata = metadata or {}
    memory_kind = metadata.get("kind", "fact")
    if memory_kind not in {"name", "preference", "decision", "fact"}:
        memory_kind = "fact"
    memory_category = metadata.get("category")
    if memory_category is not None:
        if not isinstance(memory_category, str):
            raise ValueError("Memory category is not allowed.")
        expected_kind = MEMORY_CATEGORY_KINDS.get(memory_category)
        if expected_kind is None or memory_kind != expected_kind:
            raise ValueError("Memory category is not allowed for this memory kind.")

    memory_id = str(uuid.uuid4())
    memory_metadata = {
        "session_id": session_id,
        "npc_id": npc_id,
        "kind": memory_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if memory_category is not None:
        memory_metadata["category"] = memory_category
    _get_collection().upsert(
        ids=[memory_id],
        documents=[text],
        metadatas=[memory_metadata],
    )
    return memory_id


def _normalize_memory_fact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def add_learning_memory(
    session_id: str,
    text: str,
    category: str,
    *,
    npc_id: str,
) -> bool:
    _validate_session_id(session_id)
    get_npc(npc_id)
    text = _validate_memory_text(text)
    if not isinstance(category, str):
        raise ValueError("Memory category is not allowed.")
    kind = MEMORY_CATEGORY_KINDS.get(category)
    if kind is None:
        raise ValueError("Memory category is not allowed.")
    fact_match = LEARNING_FACT_PATTERNS[category].fullmatch(text)
    if fact_match is None:
        raise ValueError("Learning memory is not a validated low-risk fact.")
    if category in {"NPC_INTERACTION_PATTERN", "PLAYER_ACCEPTED_HELP"}:
        if fact_match.group(1).casefold() != get_npc(npc_id).name.casefold():
            raise ValueError("Learning memory does not match its NPC scope.")

    collection = _get_collection()
    existing = collection.get(
        where={"$and": [{"session_id": session_id}, {"npc_id": npc_id}]},
        include=["documents", "metadatas"],
    )
    documents = existing.get("documents", [])
    metadatas = existing.get("metadatas", [])
    ids = existing.get("ids", [])
    normalized = _normalize_memory_fact(text)
    learning_count = 0
    for memory_id, document, metadata in zip(ids, documents, metadatas):
        if not metadata:
            continue
        existing_category = metadata.get("category")
        if existing_category in MEMORY_CATEGORIES:
            learning_count += 1
        if document and _normalize_memory_fact(document) == normalized:
            if existing_category == category:
                return False
            if existing_category is None:
                collection.update(
                    ids=[memory_id],
                    metadatas=[{**metadata, "category": category, "kind": kind}],
                )
                return True
            return False

    if learning_count >= MAX_NPC_LEARNING_MEMORIES:
        return False

    memory_id = str(uuid.uuid4())
    collection.upsert(
        ids=[memory_id],
        documents=[text],
        metadatas=[
            {
                "session_id": session_id,
                "npc_id": npc_id,
                "kind": kind,
                "category": category,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    return True


def get_npc_learning_memories(session_id: str, *, npc_id: str) -> list[RetrievedMemory]:
    _validate_session_id(session_id)
    get_npc(npc_id)
    result = _get_collection().get(
        where={"$and": [{"session_id": session_id}, {"npc_id": npc_id}]},
        include=["documents", "metadatas"],
    )
    memories = []
    for document, metadata in zip(
        result.get("documents", []), result.get("metadatas", [])
    ):
        if (
            document
            and metadata
            and metadata.get("session_id") == session_id
            and metadata.get("npc_id") == npc_id
            and metadata.get("category") in MEMORY_CATEGORIES
        ):
            memories.append(
                {
                    "text": document,
                    "kind": metadata.get("kind", "fact"),
                    "category": metadata["category"],
                }
            )
    return memories[:MAX_NPC_LEARNING_MEMORIES]


def search_memories(
    session_id: str,
    query: str,
    limit: int = MAX_MEMORY_RESULTS,
    *,
    npc_id: str = "ravi",
) -> list[RetrievedMemory]:
    _validate_session_id(session_id)
    get_npc(npc_id)
    if not isinstance(query, str) or not query.strip():
        return []
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_MEMORY_RESULTS:
        raise ValueError("Memory search limit must be between 1 and 3.")

    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_texts=[query.strip()],
        n_results=min(limit, total),
        where={
            "$and": [
                {"session_id": session_id},
                {"npc_id": npc_id},
            ]
        },
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return [
        {
            "text": document,
            "kind": metadata.get("kind", "fact"),
            **({"category": metadata["category"]} if metadata.get("category") else {}),
        }
        for document, metadata in zip(documents, metadatas)
        if (
            document
            and metadata
            and metadata.get("session_id") == session_id
            and metadata.get("npc_id") == npc_id
        )
    ]


def delete_session_memories(session_id: str, *, npc_id: str = "ravi") -> int:
    _validate_session_id(session_id)
    get_npc(npc_id)
    collection = _get_collection()
    where = {"$and": [{"session_id": session_id}, {"npc_id": npc_id}]}
    matching = collection.get(where=where, include=[])
    ids = matching.get("ids", [])
    if ids:
        collection.delete(ids=ids, where=where)
    return len(ids)


def extract_memories(user_message: str) -> list[ExtractedMemory]:
    if not isinstance(user_message, str):
        return []

    extracted: list[ExtractedMemory] = []
    name_match = NAME_PATTERN.search(user_message)
    if name_match:
        name = " ".join(part.capitalize() for part in name_match.group(1).split())
        extracted.append({"text": f"The player's stated name is {name}.", "kind": "name"})

    preference_match = PREFERENCE_PATTERN.search(user_message)
    if preference_match and not SENSITIVE_MEMORY_TOPIC_PATTERN.search(preference_match.group(2)):
        verb = preference_match.group(1).lower()
        preference = preference_match.group(2).strip()
        extracted.append(
            {
                "text": f"The player says they {verb} {preference}.",
                "kind": "preference",
            }
        )

    decision_match = ITEM_DECISION_PATTERN.search(user_message)
    if decision_match:
        item = decision_match.group(1).lower()
        extracted.append(
            {"text": f"The player expressed interest in {item}.", "kind": "decision"}
        )

    safe_memories = []
    for memory in extracted:
        try:
            _validate_memory_text(memory["text"])
        except ValueError:
            continue
        if memory not in safe_memories:
            safe_memories.append(memory)
    return safe_memories


def remember_user_facts(
    session_id: str,
    user_message: str,
    *,
    npc_id: str = "ravi",
) -> int:
    memories = extract_memories(user_message)
    stored_count = 0
    for memory in memories:
        existing = _get_collection().get(
            where={"$and": [{"session_id": session_id}, {"npc_id": npc_id}]},
            include=["documents"],
        )
        normalized = _normalize_memory_fact(memory["text"])
        if any(
            document and _normalize_memory_fact(document) == normalized
            for document in existing.get("documents", [])
        ):
            continue
        add_memory(
            session_id,
            memory["text"],
            {"kind": memory["kind"]},
            npc_id=npc_id,
        )
        stored_count += 1
    return stored_count