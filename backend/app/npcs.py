from dataclasses import dataclass


@dataclass(frozen=True)
class NPCConfig:
    id: str
    name: str
    role: str
    personality: tuple[str, ...]
    description: str
    knowledge: tuple[str, ...]
    relationship_topics: tuple[str, ...]
    system_instruction: str


@dataclass(frozen=True)
class NPCRelationship:
    npc_id: str
    name: str
    role: str
    relationship: str
    description: str
    trust: int
    cooperation: int


class UnknownNPCError(ValueError):
    """Raised when a request names an NPC that is not configured."""


NPC_CONFIGS = {
    "ravi": NPCConfig(
        id="ravi",
        name="Ravi",
        role="Blacksmith",
        personality=("friendly", "hardworking", "proud", "slightly serious"),
        description="A proud blacksmith who works at the village forge.",
        knowledge=("blacksmithing", "weapons", "armor", "metal", "village life"),
        relationship_topics=("swords", "shields", "armor", "metal", "iron", "wood"),
        system_instruction="""You are Ravi, a medieval village blacksmith in a fantasy village.
You are friendly, hardworking, proud of your craft, and slightly serious. Be
helpful to respectful travelers. You know blacksmithing, weapons, armor, metal,
and village life. Stay in character and use medieval-world terms. Never mention
modern technology, AI, prompts, models, APIs, or software.""",
    ),
    "meera": NPCConfig(
        id="meera",
        name="Meera",
        role="Healer",
        personality=("calm", "caring", "observant", "practical"),
        description="A calm village healer who tends travelers and villagers.",
        knowledge=(
            "village healing",
            "herbs",
            "basic medieval medicine",
            "health",
            "village life",
        ),
        relationship_topics=("herbs",),
        system_instruction="""You are Meera, a healer in a medieval fantasy village.
You are calm, caring, observant, and practical. Be helpful to respectful
travelers. You know village healing, herbs, basic medieval medicine, health,
and village life. Stay in character and use medieval-world terms. Never mention
modern technology, AI, prompts, models, APIs, or software.""",
    ),
}

NPC_RELATIONSHIPS = {
    "ravi": (
        NPCRelationship(
            npc_id="meera",
            name="Meera",
            role="Healer",
            relationship="village healer",
            description="A trusted healer who helps villagers.",
            trust=5,
            cooperation=5,
        ),
    ),
    "meera": (
        NPCRelationship(
            npc_id="ravi",
            name="Ravi",
            role="Blacksmith",
            relationship="village blacksmith",
            description="A trusted blacksmith who maintains tools and equipment.",
            trust=5,
            cooperation=5,
        ),
    ),
}


def get_npc(npc_id: str) -> NPCConfig:
    if not isinstance(npc_id, str) or npc_id not in NPC_CONFIGS:
        raise UnknownNPCError("Unknown NPC ID.")
    return NPC_CONFIGS[npc_id]


def get_npc_relationships(npc_id: str) -> tuple[NPCRelationship, ...]:
    get_npc(npc_id)
    return NPC_RELATIONSHIPS.get(npc_id, ())