from typing import Any

from app.memory.database import (
    InventoryItem,
    PlayerState,
    _validate_session_id,
)
from app.npcs import UnknownNPCError, get_npc, get_npc_relationships
from app.services.npc_communication import (
    NPCCommunicationError,
    request_npc_message,
)
from app.services.npc_movement import (
    MAX_MOVEMENTS_PER_NPC_PER_EVENT,
    NPCMovementError,
    move_npc_for_event,
)
from app.services.npc_planner import (
    ALLOWED_PLAN_STEPS,
    MAX_PLAN_STEPS,
    NPCPlan,
    NPCPlanError,
    validate_npc_plan,
)
from app.services.npc_task_progress import (
    NPCTaskProgressError,
    complete_step,
    complete_task,
    create_task_progress,
    fail_step,
    start_step,
    start_task,
    task_result,
)
from app.services.language_support import normalize_language


def _rejected_result(plan: NPCPlan | None, error: str) -> dict[str, Any]:
    return {
        "npc_id": plan.npc_id if isinstance(plan, NPCPlan) else None,
        "intent": plan.intent if isinstance(plan, NPCPlan) else None,
        "total_steps": len(plan.steps) if isinstance(plan, NPCPlan) else 0,
        "completed_steps": 0,
        "status": "rejected",
        "failed_step": None,
        "communication_count": 0,
        "reason": error,
    }


def _request_healer_help(
    *,
    npc_id: str,
    session_id: str,
    allow_communication: bool,
    language: str,
) -> dict[str, Any]:
    if not allow_communication:
        return {"success": False, "message": "The consultation limit has been reached."}
    relationships = get_npc_relationships(npc_id)
    healer = next(
        (
            relationship
            for relationship in relationships
            if relationship.role.casefold() == "healer"
            and relationship.trust >= 4
            and relationship.cooperation >= 4
        ),
        None,
    )
    if healer is None:
        return {"success": False, "message": "No trusted healer is configured."}
    try:
        if language == "en":
            return request_npc_message(
                npc_id,
                healer.npc_id,
                "A traveler needs care. What practical guidance may I share?",
                session_id,
            )
        return request_npc_message(
            npc_id,
            healer.npc_id,
            "A traveler needs care. What practical guidance may I share?",
            session_id,
            language=language,
        )
    except NPCCommunicationError:
        return {"success": False, "message": "The other NPC could not be consulted."}
    except Exception:
        return {"success": False, "message": "The other NPC could not be consulted."}


def _execute_step(
    step: str,
    *,
    npc_id: str,
    session_id: str,
    game_state: PlayerState,
    inventory: list[InventoryItem],
    schedule_location: str | None,
    event_name: str | None,
    allow_communication: bool,
    current_location: str | None,
    allow_movement: bool,
    language: str,
) -> dict[str, Any]:
    if step == "identify_player_need":
        return {"success": True}
    if step == "inspect_supplies":
        return {"success": True, "resource_count": len(inventory)}
    if step == "stay_at_location":
        return {"success": True, "location": schedule_location or game_state["location"]}
    if step == "request_npc_help":
        return _request_healer_help(
            npc_id=npc_id,
            session_id=session_id,
            allow_communication=allow_communication,
            language=language,
        )
    if step == "move_to_location":
        if not allow_movement or not event_name:
            return {"success": False, "message": "Movement is not allowed for this task."}
        try:
            movement = move_npc_for_event(
                session_id,
                npc_id,
                event_name,
                game_time=game_state["game_time"],
                current_location=current_location,
            )
        except NPCMovementError:
            return {"success": False, "message": "The configured movement was rejected."}
        return {"success": True, "movement": movement}
    if step in {
        "offer_help",
        "provide_sword_information",
        "provide_herb_information",
        "speak_to_player",
    }:
        return {"success": True}
    return {"success": False, "message": "Plan step is not allowed."}


def execute_npc_plan(
    session_id: str,
    plan: NPCPlan,
    *,
    game_state: PlayerState,
    inventory: list[InventoryItem],
    schedule_location: str | None = None,
    allow_communication: bool = True,
    current_location: str | None = None,
    allow_movement: bool = True,
    language: str = "en",
) -> dict[str, Any]:
    try:
        _validate_session_id(session_id)
        validate_npc_plan(plan)
        npc = get_npc(plan.npc_id)
        language = normalize_language(language)
    except (ValueError, UnknownNPCError, NPCPlanError) as error:
        return _rejected_result(plan, str(error))

    if game_state.get("session_id") != session_id:
        return _rejected_result(plan, "Game state does not match the plan session.")
    if len(plan.steps) > MAX_PLAN_STEPS or any(step not in ALLOWED_PLAN_STEPS for step in plan.steps):
        return _rejected_result(plan, "Plan steps are not allowed.")
    if plan.steps.count("move_to_location") > MAX_MOVEMENTS_PER_NPC_PER_EVENT:
        return _rejected_result(plan, "NPC movement limit was exceeded.")

    try:
        progress = create_task_progress(session_id, plan)
        start_task(progress)
    except NPCTaskProgressError as error:
        return _rejected_result(plan, str(error))

    peer_reply = None
    consulted_npc = None
    communication_count = 0
    movement_result = None
    for step in progress.steps:
        start_step(progress, step.name)
        if step.name == "request_npc_help":
            communication_count += 1
        try:
            operation_result = _execute_step(
                step.name,
                npc_id=npc.id,
                session_id=session_id,
                game_state=game_state,
                inventory=inventory,
                schedule_location=schedule_location,
                allow_communication=allow_communication,
                event_name=plan.event_name,
                current_location=current_location,
                allow_movement=allow_movement,
                language=language,
            )
        except Exception:
            operation_result = {"success": False}

        if not operation_result.get("success"):
            fail_step(progress, step.name)
            break
        complete_step(progress, step.name)
        if step.name == "request_npc_help":
            peer_reply = operation_result.get("message")
            consulted_npc = operation_result.get("from_npc")
        elif step.name == "move_to_location":
            movement_result = operation_result.get("movement")

    if progress.status == "running":
        try:
            complete_task(progress)
        except NPCTaskProgressError as error:
            return _rejected_result(plan, str(error))

    result = task_result(progress)
    response: dict[str, Any] = {
        "npc_id": npc.id,
        "intent": plan.intent,
        **result,
        "communication_count": communication_count,
    }
    if peer_reply:
        response["peer_reply"] = peer_reply
        response["consulted_npc"] = consulted_npc
        response["peer_response"] = {
            "npc_name": consulted_npc,
            "message": peer_reply,
        }
    if movement_result is not None:
        response["movement_result"] = movement_result
    return response
