from app.memory.database import (
    InventoryItem,
    PlayerState,
    _validate_session_id,
    get_inventory,
    get_player_state,
)
from app.npcs import UnknownNPCError, get_npc, get_npc_relationships
from app.services.language_support import normalize_language

MAX_NPC_MESSAGES_PER_REQUEST = 1
MAX_NPC_MESSAGE_LENGTH = 500


class NPCCommunicationError(ValueError):
    """Raised when an internal NPC communication request is invalid."""


def request_npc_message(
    from_npc_id: str,
    to_npc_id: str,
    message: str,
    session_id: str,
    language: str = "en",
) -> dict[str, str | bool]:
    language = normalize_language(language)
    try:
        sender = get_npc(from_npc_id)
        receiver = get_npc(to_npc_id)
    except UnknownNPCError:
        raise NPCCommunicationError("Unknown NPC ID.") from None

    if sender.id == receiver.id:
        raise NPCCommunicationError("NPCs cannot send messages to themselves.")
    if not any(
        relationship.npc_id == receiver.id
        for relationship in get_npc_relationships(sender.id)
    ):
        raise NPCCommunicationError("These NPCs are not configured to communicate.")

    try:
        _validate_session_id(session_id)
    except ValueError:
        raise NPCCommunicationError("Invalid session ID.") from None

    if not isinstance(message, str):
        raise NPCCommunicationError("Message must be text.")
    message = message.strip()
    if not message:
        raise NPCCommunicationError("Message must not be empty.")
    if len(message) > MAX_NPC_MESSAGE_LENGTH:
        raise NPCCommunicationError("Message is too long.")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in message):
        raise NPCCommunicationError("Message contains unsupported control characters.")

    game_state: PlayerState = get_player_state(session_id)
    inventory: list[InventoryItem] = get_inventory(session_id)

    from app.services.gemini_service import generate_peer_npc_reply

    reply = generate_peer_npc_reply(
        sender=sender,
        receiver=receiver,
        message=message,
        game_state=game_state,
        inventory=inventory,
        language=language,
    )
    if not isinstance(reply, str) or not reply.strip():
        raise NPCCommunicationError("The other NPC could not provide a reply.")

    return {
        "success": True,
        "from_npc": receiver.name,
        "message": reply.strip(),
    }
