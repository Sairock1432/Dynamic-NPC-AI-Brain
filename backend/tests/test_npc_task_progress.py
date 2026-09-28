import unittest
from unittest.mock import patch

from app.memory import database
from app.services.npc_planner import NPCPlan, create_player_plan
from app.services.npc_task_progress import (
    MAX_EXECUTION_ATTEMPTS,
    MAX_PLAN_STEPS,
    NPCTaskProgressError,
    STEP_STATUSES,
    TASK_STATUSES,
    complete_step,
    complete_task,
    create_task_progress,
    fail_step,
    reject_task,
    start_step,
    start_task,
    task_result,
    transition_task,
    validate_step_status,
    validate_task_status,
)
from app.services import semantic_memory


class NPCTaskProgressTests(unittest.TestCase):
    def test_initial_task_and_step_state_is_pending_and_temporary(self) -> None:
        progress = create_task_progress(
            "session-a", create_player_plan("ravi", "ASK_SWORD")
        )
        self.assertEqual(progress.status, "pending")
        self.assertIsNone(progress.current_step)
        self.assertEqual(progress.completed_steps, [])
        self.assertIsNone(progress.failed_step)
        self.assertEqual(progress.total_steps, 3)
        self.assertEqual([step.status for step in progress.steps], ["pending"] * 3)
        self.assertTrue(progress.task_id)
        self.assertEqual(TASK_STATUSES, {"pending", "running", "completed", "failed", "rejected"})
        self.assertEqual(STEP_STATUSES, {"pending", "running", "completed", "failed", "skipped"})

    def test_only_allowlisted_task_and_step_statuses_are_accepted(self) -> None:
        for status in TASK_STATUSES:
            self.assertEqual(validate_task_status(status), status)
        for status in STEP_STATUSES:
            self.assertEqual(validate_step_status(status), status)
        with self.assertRaises(NPCTaskProgressError):
            validate_task_status("retrying")
        with self.assertRaises(NPCTaskProgressError):
            validate_step_status("waiting_for_gemini")

    def test_rejected_task_is_terminal(self) -> None:
        progress = create_task_progress("session-a", create_player_plan("ravi", "ASK_SWORD"))
        reject_task(progress)
        self.assertEqual(progress.status, "rejected")
        with self.assertRaises(NPCTaskProgressError):
            start_task(progress)

    def test_valid_transitions_and_successful_completion(self) -> None:
        progress = create_task_progress("session-a", create_player_plan("ravi", "ASK_SWORD"))
        start_task(progress)
        for step in progress.steps:
            start_step(progress, step.name)
            self.assertEqual(progress.status, "running")
            self.assertEqual(progress.current_step, step.name)
            self.assertEqual(step.status, "running")
            complete_step(progress, step.name)
        complete_task(progress)

        result = task_result(progress)
        self.assertEqual(result, {
            "status": "completed",
            "completed_steps": 3,
            "total_steps": 3,
            "failed_step": None,
        })
        self.assertEqual(progress.status, "completed")
        self.assertEqual(progress.completed_steps, list(create_player_plan("ravi", "ASK_SWORD").steps))
        with self.assertRaises(NPCTaskProgressError):
            start_task(progress)

    def test_failure_marks_only_current_step_and_leaves_later_steps_pending(self) -> None:
        progress = create_task_progress(
            "session-a", create_player_plan("ravi", "ASK_BLACKSMITH_HELP")
        )
        start_task(progress)
        start_step(progress, "identify_player_need")
        complete_step(progress, "identify_player_need")
        start_step(progress, "inspect_supplies")
        fail_step(progress, "inspect_supplies")

        self.assertEqual(progress.status, "failed")
        self.assertEqual(progress.failed_step, "inspect_supplies")
        self.assertEqual(progress.completed_steps, ["identify_player_need"])
        self.assertEqual([step.status for step in progress.steps], ["completed", "failed", "pending"])
        self.assertEqual(task_result(progress)["completed_steps"], 1)
        with self.assertRaises(NPCTaskProgressError):
            start_step(progress, "speak_to_player")

    def test_execution_attempts_are_limited_to_one_per_step(self) -> None:
        progress = create_task_progress("session-a", create_player_plan("ravi", "ASK_SWORD"))
        self.assertEqual(MAX_EXECUTION_ATTEMPTS, 1)
        start_task(progress)
        start_step(progress, "identify_player_need")
        with self.assertRaises(NPCTaskProgressError):
            start_step(progress, "identify_player_need")
        complete_step(progress, "identify_player_need")
        with self.assertRaises(NPCTaskProgressError):
            start_step(progress, "identify_player_need")
        self.assertEqual(progress.steps[0].execution_attempts, 1)

    def test_zero_long_invalid_npc_and_wrong_session_plans_are_rejected(self) -> None:
        valid = create_player_plan("ravi", "ASK_SWORD")
        invalid_plans = (
            NPCPlan("ravi", "player_request", "ASK_SWORD", (), "help_travelers"),
            NPCPlan(
                "ravi", "player_request", "ASK_SWORD",
                (*valid.steps, "speak_to_player"), "help_travelers"
            ),
            NPCPlan(
                "stranger", "player_request", "ASK_SWORD", valid.steps, "help_travelers"
            ),
            NPCPlan(
                "ravi", "player_request", "ASK_SWORD",
                ("identify_player_need", "arbitrary_action", "speak_to_player"),
                "help_travelers",
            ),
        )
        self.assertEqual(MAX_PLAN_STEPS, 3)
        for plan in invalid_plans:
            with self.subTest(plan=plan), self.assertRaises(NPCTaskProgressError):
                create_task_progress("session-a", plan)
        with self.assertRaises(NPCTaskProgressError):
            create_task_progress("invalid session", valid)

    def test_progress_is_not_persisted_to_sqlite_or_chroma_or_returned(self) -> None:
        with (
            patch.object(database, "_connect") as connect,
            patch.object(semantic_memory, "_get_collection") as collection,
        ):
            progress = create_task_progress(
                "session-a", create_player_plan("meera", "ASK_HERBS")
            )
            start_task(progress)
            for step in progress.steps:
                start_step(progress, step.name)
                complete_step(progress, step.name)
            complete_task(progress)
            result = task_result(progress)

        connect.assert_not_called()
        collection.assert_not_called()
        self.assertNotIn("task_id", result)
        self.assertNotIn("npc_id", result)
        self.assertNotIn("session_id", result)
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
