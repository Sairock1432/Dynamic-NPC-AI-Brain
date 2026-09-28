from dataclasses import dataclass, field
from uuid import uuid4

from app.memory.database import _validate_session_id
from app.npcs import UnknownNPCError, get_npc
from app.services.npc_planner import MAX_PLAN_STEPS, NPCPlan, NPCPlanError, validate_npc_plan

MAX_EXECUTION_ATTEMPTS = 1
TASK_STATUSES = frozenset({"pending", "running", "completed", "failed", "rejected"})
STEP_STATUSES = frozenset({"pending", "running", "completed", "failed", "skipped"})
TASK_TRANSITIONS = {
    "pending": frozenset({"running", "rejected"}),
    "running": frozenset({"completed", "failed", "rejected"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "rejected": frozenset(),
}


class NPCTaskProgressError(ValueError):
    """Raised when task or step progress attempts an invalid transition."""


@dataclass
class TaskStepProgress:
    name: str
    status: str = "pending"
    execution_attempts: int = 0


@dataclass
class NPCTaskProgress:
    task_id: str
    npc_id: str
    session_id: str
    status: str
    current_step: str | None
    steps: list[TaskStepProgress]
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None

    @property
    def total_steps(self) -> int:
        return len(self.steps)


def validate_task_status(status: str) -> str:
    if not isinstance(status, str) or status not in TASK_STATUSES:
        raise NPCTaskProgressError("Task status is not allowed.")
    return status


def validate_step_status(status: str) -> str:
    if not isinstance(status, str) or status not in STEP_STATUSES:
        raise NPCTaskProgressError("Step status is not allowed.")
    return status


def create_task_progress(session_id: str, plan: NPCPlan) -> NPCTaskProgress:
    try:
        _validate_session_id(session_id)
        validate_npc_plan(plan)
        get_npc(plan.npc_id)
    except (ValueError, UnknownNPCError, NPCPlanError) as error:
        raise NPCTaskProgressError("Task plan, NPC, or session is invalid.") from error

    if not plan.steps or len(plan.steps) > MAX_PLAN_STEPS:
        raise NPCTaskProgressError("Task must contain between one and three steps.")
    return NPCTaskProgress(
        task_id=uuid4().hex,
        npc_id=plan.npc_id,
        session_id=session_id,
        status="pending",
        current_step=None,
        steps=[TaskStepProgress(name=step) for step in plan.steps],
    )


def transition_task(progress: NPCTaskProgress, next_status: str) -> None:
    validate_task_status(next_status)
    if next_status not in TASK_TRANSITIONS.get(progress.status, frozenset()):
        raise NPCTaskProgressError("Task status transition is not allowed.")
    progress.status = next_status


def start_task(progress: NPCTaskProgress) -> None:
    transition_task(progress, "running")


def start_step(progress: NPCTaskProgress, step_name: str) -> None:
    if progress.status != "running" or progress.current_step is not None:
        raise NPCTaskProgressError("Task is not ready to start another step.")
    next_step = next((step for step in progress.steps if step.status == "pending"), None)
    if next_step is None or next_step.name != step_name:
        raise NPCTaskProgressError("Step order is invalid.")
    validate_step_status("running")
    if next_step.execution_attempts >= MAX_EXECUTION_ATTEMPTS:
        raise NPCTaskProgressError("Step execution limit has been reached.")
    next_step.execution_attempts += 1
    next_step.status = "running"
    progress.current_step = step_name


def complete_step(progress: NPCTaskProgress, step_name: str) -> None:
    if progress.status != "running" or progress.current_step != step_name:
        raise NPCTaskProgressError("Only the currently running step can complete.")
    step = next(item for item in progress.steps if item.name == step_name)
    validate_step_status("completed")
    step.status = "completed"
    progress.completed_steps.append(step_name)
    progress.current_step = None


def fail_step(progress: NPCTaskProgress, step_name: str) -> None:
    if progress.status != "running" or progress.current_step != step_name:
        raise NPCTaskProgressError("Only the currently running step can fail.")
    step = next(item for item in progress.steps if item.name == step_name)
    validate_step_status("failed")
    step.status = "failed"
    progress.failed_step = step_name
    progress.current_step = None
    transition_task(progress, "failed")


def complete_task(progress: NPCTaskProgress) -> None:
    if any(step.status != "completed" for step in progress.steps):
        raise NPCTaskProgressError("All plan steps must complete before the task can finish.")
    transition_task(progress, "completed")


def reject_task(progress: NPCTaskProgress) -> None:
    transition_task(progress, "rejected")


def task_result(progress: NPCTaskProgress) -> dict[str, int | str | None]:
    return {
        "status": progress.status,
        "completed_steps": len(progress.completed_steps),
        "total_steps": progress.total_steps,
        "failed_step": progress.failed_step,
    }
