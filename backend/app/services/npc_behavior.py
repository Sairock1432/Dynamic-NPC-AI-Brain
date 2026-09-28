import logging
from dataclasses import replace
from typing import Any

from app.memory.database import (
    InventoryItem,
    PlayerState,
    _validate_session_id,
    get_inventory,
    get_player_state,
)
from app.npcs import UnknownNPCError, get_npc
from app.services.gemini_service import (
    GeminiConfigurationError,
    GeminiServiceError,
    generate_npc_event_reaction,
)
from app.services.npc_communication import MAX_NPC_MESSAGES_PER_REQUEST
from app.services.npc_decision import (
    LOW_HEALTH_THRESHOLD,
    MAX_NPC_DECISIONS_PER_EVENT,
    NPCDecision,
    decide_npc_action,
)
from app.services.npc_plan_executor import execute_npc_plan
from app.services.npc_planner import NPCPlanError, create_event_plan
from app.services.npc_relationship_progression import (
    NPCRelationshipError,
    evaluate_player_relationship,
    select_relationship_decision,
)
from app.services.npc_schedules import (
    NPC_SCHEDULES,
    NPCScheduleError,
    get_npc_schedule_location,
)
from app.services.semantic_memory import RetrievedMemory, get_npc_learning_memories
from app.services.npc_movement import (
    MAX_MOVEMENTS_PER_NPC_PER_EVENT,
    MAX_NPC_MOVEMENTS_PER_EVENT,
)
from app.services.world_events import UnknownWorldEventError, WorldEvent, get_world_event

logger = logging.getLogger(__name__)

MAX_AUTONOMOUS_ACTIONS_PER_EVENT = 2
NPCBehaviorError = NPCScheduleError


def _selected_behaviors(event: WorldEvent, state: PlayerState) -> tuple[tuple[str, str], ...]:
    if event.event_id == "player_low_health" and state["health"] >= LOW_HEALTH_THRESHOLD:
        return ()
    limit = min(MAX_AUTONOMOUS_ACTIONS_PER_EVENT, MAX_NPC_DECISIONS_PER_EVENT)
    return event.npc_behaviors[:limit]


def dispatch_world_event(session_id: str, event_name: str) -> dict[str, Any]:
    try:
        _validate_session_id(session_id)
    except ValueError:
        raise NPCBehaviorError("Invalid session ID.") from None

    try:
        event = get_world_event(event_name)
    except UnknownWorldEventError:
        raise

    state = get_player_state(session_id)
    inventory = get_inventory(session_id)
    behaviors = _selected_behaviors(event, state)
    actions = []
    communication_count = 0
    movement_count = 0
    moved_npcs: set[str] = set()

    for npc_id, behavior in behaviors:
        npc = get_npc(npc_id)
        schedule_location = get_npc_schedule_location(npc_id, state["game_time"])
        try:
            learning_memories: list[RetrievedMemory] = get_npc_learning_memories(
                session_id,
                npc_id=npc.id,
            )
        except Exception as error:
            logger.warning("NPC learning retrieval failed (%s).", type(error).__name__)
            learning_memories = []
        try:
            relationship_evaluation = evaluate_player_relationship(
                session_id,
                npc.id,
                learning_memories=learning_memories,
            )
        except NPCRelationshipError as error:
            logger.warning("NPC relationship evaluation rejected (%s).", type(error).__name__)
            relationship_evaluation = None
        decision: NPCDecision | None = decide_npc_action(
            npc,
            state,
            inventory,
            event_name=event.event_id,
            schedule_location=schedule_location,
            learning_memories=learning_memories,
            relationship_evaluation=relationship_evaluation,
        )
        reaction = None
        consultation = None
        plan_result = None
        if decision is not None:
            try:
                plan = create_event_plan(npc.id, event.event_id, decision)
                requests_help = "request_npc_help" in plan.steps
                requests_movement = "move_to_location" in plan.steps
                allow_movement = (
                    requests_movement
                    and npc.id not in moved_npcs
                    and movement_count < MAX_NPC_MOVEMENTS_PER_EVENT
                )
                plan_result = execute_npc_plan(
                    session_id,
                    plan,
                    game_state=state,
                    inventory=inventory,
                    schedule_location=schedule_location,
                    allow_communication=(
                        not requests_help
                        or communication_count < MAX_NPC_MESSAGES_PER_REQUEST
                    ),
                    allow_movement=(not requests_movement or allow_movement),
                )
                if requests_help:
                    communication_count += 1
                if requests_movement:
                    moved_npcs.add(npc.id)
                    movement_count += 1
                if plan_result["status"] == "completed" and requests_help:
                    consultation = {
                        "success": True,
                        "from_npc": plan_result.get("consulted_npc"),
                        "message": plan_result.get("peer_reply"),
                    }
                elif plan_result["status"] != "completed" and requests_help:
                    consultation = {
                        "success": False,
                        "message": "The other NPC could not be consulted.",
                    }
                if plan_result["status"] == "completed":
                    try:
                        narration_decision = decision
                        progression_events = []
                        if "offer_help" in plan.steps:
                            progression_events.append("successful_help")
                        if requests_help:
                            progression_events.append("successful_npc_communication")
                        if progression_events:
                            try:
                                post_action_relationship = evaluate_player_relationship(
                                    session_id,
                                    npc.id,
                                    interaction_events=tuple(progression_events),
                                    learning_memories=learning_memories,
                                )
                                social_decision, relationship_topic = select_relationship_decision(
                                    post_action_relationship,
                                    event_name=event.event_id,
                                )
                                narration_decision = replace(
                                    decision,
                                    relationship_decision=social_decision,
                                    relationship_topic=relationship_topic,
                                )
                            except NPCRelationshipError as error:
                                logger.warning(
                                    "NPC relationship progression unavailable (%s).",
                                    type(error).__name__,
                                )
                        task_summary = {
                            key: plan_result[key]
                            for key in ("status", "completed_steps", "total_steps", "failed_step")
                        }
                        reaction = generate_npc_event_reaction(
                            npc=npc,
                            event_name=event.event_id,
                            behavior=behavior,
                            schedule_location=schedule_location,
                            game_state=state,
                            inventory=inventory,
                            decision=narration_decision,
                            task_result=task_summary,
                            movement_result=plan_result.get("movement_result"),
                        )
                    except (GeminiConfigurationError, GeminiServiceError) as error:
                        logger.warning("NPC event reaction unavailable (%s).", type(error).__name__)
            except NPCPlanError as error:
                logger.warning("NPC event plan rejected (%s).", type(error).__name__)
                plan_result = {
                    "status": "rejected",
                    "completed_steps": 0,
                    "step_results": [],
                }

        action_result: dict[str, Any] = {
                "npc_id": npc.id,
                "npc_name": npc.name,
                "behavior": behavior,
                "schedule_location": schedule_location,
                "decision": decision.decision if decision else None,
                "goal_id": decision.goal_id if decision else None,
                "goal_description": decision.goal_description if decision else None,
                "goal_priority": decision.goal_priority if decision else None,
                "reaction": reaction,
            }
        if plan_result is not None:
            action_result["plan_result"] = plan_result
            if plan_result.get("movement_result") is not None:
                action_result["movement"] = plan_result["movement_result"]
        if consultation is not None:
            action_result["consultation"] = consultation
        actions.append(action_result)

    result: dict[str, Any] = {
        "session_id": session_id,
        "event": event.event_id,
        "world_time": state["game_time"],
        "action_count": len(actions),
        "actions": actions,
    }
    plan_statuses = [
        action["plan_result"]["status"]
        for action in actions
        if "plan_result" in action
    ]
    if "rejected" in plan_statuses:
        result["status"] = "rejected"
    elif "failed" in plan_statuses:
        result["status"] = "failed"
    elif event.event_id == "player_low_health" and not actions:
        result["status"] = "not_triggered"
        result["reason"] = f"Health is not below {LOW_HEALTH_THRESHOLD}."
    else:
        result["status"] = "completed"
    return result
