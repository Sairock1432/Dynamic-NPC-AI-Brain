import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.npcs import UnknownNPCError, get_npc
from app.memory.database import (
    MAX_HISTORY_MESSAGES,
    ConversationSummary,
    InventoryItem,
    PlayerState,
    StoredConversationMessage,
    get_inventory,
    get_conversation_history,
    get_player_state,
    get_stored_conversation_messages,
    init_db,
    list_conversation_summaries,
    save_message,
)
from app.services.gemini_service import (
    GeminiConfigurationError,
    GeminiServiceError,
    generate_npc_reply,
    is_explicit_game_action_request,
)
from app.services.npc_decision import (
    NPCDecisionError,
    is_relationship_dialogue_candidate,
    select_npc_dialogue,
)
from app.services.npc_learning import learn_from_player_interaction
from app.services.npc_plan_executor import execute_npc_plan
from app.services.npc_planner import (
    NPCPlanError,
    classify_player_intent,
    create_player_plan,
)
from app.services.npc_relationship_progression import (
    NPCRelationshipError,
    apply_player_relationship_events,
    classify_player_relationship_events,
    evaluate_player_relationship,
)
from app.services.language_support import normalize_language
from app.services.semantic_memory import remember_user_facts, search_memories

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="Mystery Village Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    npc_id: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=4000)
    language: str = "en"

    @field_validator("message")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be empty.")
        return value

    @field_validator("language", mode="before")
    @classmethod
    def normalize_chat_language(cls, value: object) -> str:
        return normalize_language(value)


class ChatResponse(BaseModel):
    message: str


class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    npc_id: str
    npc_name: str
    title: str
    message_count: int
    started_at: str
    updated_at: str
    preview: str


class StoredConversationMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    npc_id: str
    npc_name: str
    messages: list[StoredConversationMessageResponse]


class InventoryItemResponse(BaseModel):
    item_name: str
    quantity: int


class InventoryResponse(BaseModel):
    session_id: str
    items: list[InventoryItemResponse]


class GameStateResponse(BaseModel):
    session_id: str
    location: str
    health: int
    gold: int
    game_time: str
    inventory: list[InventoryItemResponse]


@app.get("/history/{session_id}", response_model=list[ConversationSummaryResponse])
def read_conversation_summaries(
    session_id: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> list[ConversationSummaryResponse]:
    try:
        summaries: list[ConversationSummary] = list_conversation_summaries(session_id)
    except sqlite3.Error as error:
        logger.error("Unable to load conversation history (%s).", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="Conversation history could not be loaded. Please try again.",
        ) from None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID.") from None
    return [ConversationSummaryResponse(**summary) for summary in summaries]


@app.get(
    "/history/{session_id}/{npc_id}",
    response_model=ConversationHistoryResponse,
)
def read_conversation(
    session_id: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    npc_id: str = ApiPath(min_length=1, max_length=32),
) -> ConversationHistoryResponse:
    try:
        npc = get_npc(npc_id)
    except UnknownNPCError:
        raise HTTPException(status_code=400, detail="Unknown NPC ID.") from None
    try:
        messages: list[StoredConversationMessage] = get_stored_conversation_messages(
            session_id,
            npc.id,
        )
    except sqlite3.Error as error:
        logger.error("Unable to load conversation history (%s).", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="Conversation could not be loaded. Please try again.",
        ) from None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID.") from None
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationHistoryResponse(
        conversation_id=npc.id,
        npc_id=npc.id,
        npc_name=npc.name,
        messages=[StoredConversationMessageResponse(**message) for message in messages],
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Mystery Village backend is running",
    }


@app.get("/game-state/{session_id}", response_model=GameStateResponse)
def read_game_state(
    session_id: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> GameStateResponse:
    try:
        state: PlayerState = get_player_state(session_id)
        items: list[InventoryItem] = get_inventory(session_id)
    except sqlite3.Error as error:
        logger.error("Unable to load player state (%s).", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The game state could not be loaded. Please try again.",
        ) from None
    return GameStateResponse(**state, inventory=items)


@app.get("/inventory/{session_id}", response_model=InventoryResponse)
def read_inventory(
    session_id: str = ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> InventoryResponse:
    try:
        items = get_inventory(session_id)
    except sqlite3.Error as error:
        logger.error("Unable to load inventory (%s).", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The inventory could not be loaded. Please try again.",
        ) from None
    return InventoryResponse(session_id=session_id, items=items)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        npc = get_npc(request.npc_id)
    except UnknownNPCError:
        raise HTTPException(status_code=400, detail="Unknown NPC ID.") from None

    try:
        history = get_conversation_history(
            request.session_id,
            limit=MAX_HISTORY_MESSAGES - 1,
            npc_id=npc.id,
        )
        state = get_player_state(request.session_id)
        inventory = get_inventory(request.session_id)
        try:
            semantic_memories = search_memories(
                request.session_id,
                request.message,
                npc_id=npc.id,
            )
        except Exception as error:
            logger.warning(
                "Semantic memory retrieval failed (%s); continuing without it.",
                type(error).__name__,
            )
            semantic_memories = []

        plan_context = None
        supplemental_peer_response = None
        initial_npc_message_count = 0
        validated_plan_completed = False
        intent = classify_player_intent(npc.id, request.message)
        relationship_events = list(classify_player_relationship_events(request.message))
        relationship = None
        dialogue_decision = None
        relationship_dialogue_candidate = (
            not is_explicit_game_action_request(request.message)
            and is_relationship_dialogue_candidate(request.message, intent)
        )
        if relationship_dialogue_candidate or relationship_events or intent is not None:
            try:
                relationship = evaluate_player_relationship(request.session_id, npc.id)
            except (NPCDecisionError, NPCRelationshipError) as error:
                logger.warning(
                    "Relationship context unavailable (%s); continuing without relationship updates.",
                    type(error).__name__,
                )
            except Exception as error:
                logger.warning(
                    "Relationship context unavailable (%s); continuing without relationship updates.",
                    type(error).__name__,
                )
        if relationship_dialogue_candidate and relationship is not None:
            try:
                dialogue_decision = select_npc_dialogue(
                    npc,
                    relationship,
                    session_id=request.session_id,
                    message=request.message,
                    intent=intent,
                )
            except (NPCDecisionError, NPCRelationshipError) as error:
                logger.warning(
                    "Relationship-aware dialogue selection unavailable (%s).",
                    type(error).__name__,
                )
        if intent is not None:
            try:
                plan = create_player_plan(npc.id, intent)
                plan_result = execute_npc_plan(
                    request.session_id,
                    plan,
                    game_state=state,
                    inventory=inventory,
                    language=request.language,
                )
            except NPCPlanError:
                logger.warning("Rejected invalid NPC plan configuration.")
                raise HTTPException(
                    status_code=400,
                    detail="That request cannot be planned.",
                ) from None
            if plan_result["status"] != "completed":
                raise HTTPException(
                    status_code=503,
                    detail=f"{npc.name} cannot complete that request right now.",
                )
            validated_plan_completed = True
            plan_context = (
                f"VALIDATED PLAN INTENT: {intent}\n"
                f"VALIDATED TASK RESULT: status={plan_result['status']}, "
                f"completed_steps={plan_result['completed_steps']}, "
                f"total_steps={plan_result['total_steps']}.\n"
                "Use this only as a success signal. Respond naturally without "
                "revealing task IDs, plan steps, or execution details."
            )
            supplemental_peer_response = plan_result.get("peer_response")
            initial_npc_message_count = plan_result.get("communication_count", 0)
            if "offer_help" in plan.steps:
                relationship_events.append("successful_help")
            if "request_npc_help" in plan.steps and plan_result.get("peer_response"):
                relationship_events.append("successful_npc_communication")

        def collect_relationship_event(event: str) -> None:
            relationship_events.append(event)

        message = generate_npc_reply(
            request.message,
            [] if dialogue_decision is not None else history,
            npc=npc,
            session_id=request.session_id,
            game_state=state,
            inventory=inventory,
            semantic_memories=[] if dialogue_decision is not None else semantic_memories,
            plan_context=None if dialogue_decision is not None else plan_context,
            supplemental_peer_response=supplemental_peer_response,
            initial_npc_message_count=initial_npc_message_count,
            dialogue_decision=dialogue_decision,
            validated_plan_completed=(
                validated_plan_completed and dialogue_decision is not None
            ),
            relationship_event_callback=collect_relationship_event,
            language=request.language,
        )
    except GeminiConfigurationError:
        logger.warning("Gemini chat is unavailable because GEMINI_API_KEY is missing.")
        raise HTTPException(
            status_code=503,
            detail=f"{npc.name} cannot answer right now. Please try again later.",
        ) from None
    except GeminiServiceError as error:
        cause = error.__cause__
        error_type = type(cause).__name__ if cause is not None else type(error).__name__
        error_code = getattr(cause, "code", None)
        if isinstance(error_code, int):
            logger.error(
                "Gemini chat request failed (%s, status=%d).",
                error_type,
                error_code,
            )
        else:
            logger.error("Gemini chat request failed (%s).", error_type)
        raise HTTPException(
            status_code=502,
            detail=f"{npc.name} cannot answer right now. Please try again later.",
        ) from None
    except sqlite3.Error as error:
        logger.error("Unable to load conversation (%s).", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The conversation could not be loaded. Please try again.",
        ) from None

    try:
        user_message_id = save_message(request.session_id, "user", request.message, npc_id=npc.id)
        save_message(request.session_id, "assistant", message, npc_id=npc.id)
    except sqlite3.Error as error:
        logger.error("Unable to save conversation (%s).", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="The conversation could not be saved. Please try again.",
        ) from None

    if relationship_events:
        if relationship is None:
            try:
                relationship = evaluate_player_relationship(request.session_id, npc.id)
            except Exception as error:
                logger.warning(
                    "Relationship update skipped (%s).",
                    type(error).__name__,
                )
        if relationship is not None:
            try:
                apply_player_relationship_events(
                    request.session_id,
                    npc.id,
                    tuple(relationship_events),
                    source_message_id=user_message_id,
                    initial_score=relationship.trust_score,
                )
            except (NPCRelationshipError, sqlite3.Error) as error:
                logger.warning(
                    "Relationship update rejected (%s).",
                    type(error).__name__,
                )

    try:
        stored_count = remember_user_facts(
            request.session_id,
            request.message,
            npc_id=npc.id,
        )
        if stored_count:
            logger.info("Stored %d curated semantic memory item(s).", stored_count)
    except Exception as error:
        logger.warning(
            "Semantic memory write failed (%s); conversation remains saved.",
            type(error).__name__,
        )

    try:
        learned_count = learn_from_player_interaction(
            request.session_id,
            npc.id,
            request.message,
        )
        if learned_count:
            logger.info("Stored %d bounded NPC learning memory item(s).", learned_count)
    except Exception as error:
        logger.warning(
            "NPC learning write failed (%s); conversation remains saved.",
            type(error).__name__,
        )

    return ChatResponse(message=message)