import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.db import (
    OperationalError,
    connections,
)
from django.test import TestCase, TransactionTestCase, override_settings
from openai_codex import CodexError
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse

from hitch.main.goals import autonomous_goal_prompts, autonomous_goal_proposal_stack
from hitch.main.models import (
    AutonomousGoal,
    AutonomousGoalMemory,
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SessionPullRequest,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import codex_events
from hitch.main.sessions import agent_tasks
from hitch.main.sessions.pr_prompts import PR_SLASH_PROMPT
from hitch.main.sessions.review_prompts import optional_review_prompt
from hitch.main.test.support import _make_project, _rollout_line
from hitch.main.workflows import (
    agent_io,
    autonomous_goals,
    engine,
    system_agents,
)


def _instance(
    *,
    thread_id: str = "thread-1",
    cwd: str = "/repo",
    prompt: str = "prompt",
    purpose: str = CodexInstance.PURPOSE_USER,
    workflow_id: int | None = None,
    events_path: str = "/dev/null",
    status: str = CodexInstance.STATUS_COMPLETED,
    agent_kind: str = "",
    display_author: str = "",
    auto_pr_enabled: bool = False,
    auto_qa_enabled: bool = False,
    plan_mode: bool = False,
    model: str = "",
    reasoning_effort: str = "",
    sandbox_policy: str = "",
    approval_mode: str = "",
    web_search_mode: str = "",
    developer_instructions: str = "",
    enable_memories: bool = False,
    user_message_index: int | None = None,
    error: str = "",
    codex_error_info: Any = None,
) -> CodexInstance:
    return CodexInstance.objects.create(
        pid=1,
        thread_id=thread_id,
        cwd=cwd,
        prompt=prompt,
        developer_instructions=developer_instructions,
        enable_memories=enable_memories,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox_policy=sandbox_policy,
        approval_mode=approval_mode,
        web_search_mode=web_search_mode,
        plan_mode=plan_mode,
        auto_pr_enabled=auto_pr_enabled,
        auto_qa_enabled=auto_qa_enabled,
        events_path=events_path,
        status=status,
        purpose=purpose,
        workflow_id=workflow_id,
        agent_kind=agent_kind,
        display_author=display_author,
        user_message_index=user_message_index,
        error=error,
        codex_error_info=codex_error_info,
    )


def _events_file(
    test: TestCase,
    payload: dict[str, object],
    *,
    thread_id: str = "thread-1",
    tokens_used: int | None = None,
) -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        if tokens_used is not None:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": thread_id,
                            "goal": {
                                "objective": "Autonomous goal",
                                "tokensUsed": tokens_used,
                            },
                        },
                    }
                )
                + "\n"
            )
        fh.write(
            json.dumps(
                {
                    "method": "item/completed",
                    "payload": {
                        "item": {
                            "id": "a1",
                            "type": "agentMessage",
                            "text": json.dumps(payload),
                        }
                    },
                }
            )
            + "\n"
        )
        events_path = fh.name
    test.addCleanup(Path(events_path).unlink, missing_ok=True)
    return events_path


def _agent_message_events_file(test: TestCase, text: str, *, phase: str | None = "final_answer") -> str:
    item = {
        "id": "msg-1",
        "type": "agentMessage",
        "text": text,
    }
    if phase is not None:
        item["phase"] = phase
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        fh.write(
            json.dumps(
                {
                    "method": "item/completed",
                    "payload": {"item": item},
                }
            )
            + "\n"
        )
        events_path = fh.name
    test.addCleanup(Path(events_path).unlink, missing_ok=True)
    return events_path


def _assert_response_schema_objects_are_strict(test: TestCase, schema: dict[str, Any], *, path: str = "$") -> None:
    schema_type = schema.get("type")
    is_object = schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type)
    if is_object:
        test.assertIs(schema.get("additionalProperties"), False, path)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            required = schema.get("required")
            if not isinstance(required, list):
                test.fail(path)
            test.assertEqual(set(required), set(properties), path)
            for name, child in properties.items():
                if isinstance(child, dict):
                    _assert_response_schema_objects_are_strict(test, child, path=f"{path}.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        _assert_response_schema_objects_are_strict(test, items, path=f"{path}[]")


def _raw_events_file(test: TestCase, events: list[dict[str, object]]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
        events_path = fh.name
    test.addCleanup(Path(events_path).unlink, missing_ok=True)
    return events_path


def _rollout_token_file(test: TestCase, total_tokens: int) -> str:
    """Write a rollout file whose token_count event reports ``total_tokens``."""
    line = {
        "timestamp": "2026-01-05T12:00:00Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": total_tokens,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": total_tokens,
                },
                "last_token_usage": {"total_tokens": total_tokens},
                "model_context_window": 200000,
            },
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        fh.write(json.dumps(line) + "\n")
        rollout_path = fh.name
    test.addCleanup(Path(rollout_path).unlink, missing_ok=True)
    return rollout_path


class AutoProposalQuotaPauseTests(TestCase):
    @patch("hitch.main.workflows.autonomous_goals._auto_proposal_quota_status")
    def test_unthrottled_quota_pause_maps_statuses(self, mock_quota_status: MagicMock) -> None:
        for status, expected_paused in (
            ("available", False),
            ("low", True),
            ("unavailable", True),
        ):
            with self.subTest(status=status):
                mock_quota_status.return_value = status

                self.assertIs(
                    autonomous_goals._auto_proposals_paused_by_usage_quota(),
                    expected_paused,
                )

    def test_rate_limit_window_pauses_below_half_linear_remaining_threshold(
        self,
    ) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        weekly_window_mins = 7 * 24 * 60
        half_week_from_now = int((now + timedelta(days=3, hours=12)).timestamp())
        just_below_threshold = SimpleNamespace(
            used_percent=76,
            resets_at=half_week_from_now,
            window_duration_mins=weekly_window_mins,
        )
        at_threshold = SimpleNamespace(
            used_percent=75,
            resets_at=half_week_from_now,
            window_duration_mins=weekly_window_mins,
        )

        self.assertTrue(autonomous_goals._rate_limit_window_below_auto_proposal_quota(just_below_threshold, now=now))
        self.assertFalse(autonomous_goals._rate_limit_window_below_auto_proposal_quota(at_threshold, now=now))

    @patch("hitch.main.workflows.system_agents.timezone.now")
    @patch("hitch.main.workflows.autonomous_goals.Codex")
    def test_auto_proposal_quota_pause_reads_account_rate_limits(
        self, mock_codex: MagicMock, mock_now: MagicMock
    ) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        mock_now.return_value = now
        ctx = mock_codex.return_value.__enter__.return_value
        ctx._client.request.return_value = SimpleNamespace(
            rate_limits=SimpleNamespace(
                primary=SimpleNamespace(
                    used_percent=0,
                    resets_at=int((now + timedelta(hours=5)).timestamp()),
                    window_duration_mins=5 * 60,
                ),
                secondary=SimpleNamespace(
                    used_percent=76,
                    resets_at=int((now + timedelta(days=3, hours=12)).timestamp()),
                    window_duration_mins=7 * 24 * 60,
                ),
            )
        )

        status = autonomous_goals._auto_proposal_quota_status()

        self.assertEqual(status, "low")
        ctx._client.request.assert_called_once_with(
            "account/rateLimits/read",
            None,
            response_model=GetAccountRateLimitsResponse,
        )

    @patch("hitch.main.workflows.autonomous_goals.Codex")
    def test_auto_proposal_quota_is_unavailable_without_usable_windows(self, mock_codex: MagicMock) -> None:
        ctx = mock_codex.return_value.__enter__.return_value
        ctx._client.request.return_value = SimpleNamespace(rate_limits=SimpleNamespace(primary=None, secondary=None))

        self.assertEqual(autonomous_goals._auto_proposal_quota_status(), "unavailable")

    def test_auto_proposal_quota_is_available_with_verified_windows(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        primary = SimpleNamespace(
            used_percent=0,
            resets_at=int((now + timedelta(hours=5)).timestamp()),
            window_duration_mins=5 * 60,
        )
        secondary = SimpleNamespace(
            used_percent=0,
            resets_at=int((now + timedelta(days=7)).timestamp()),
            window_duration_mins=7 * 24 * 60,
        )

        status = autonomous_goals._auto_proposal_quota_status_from_rate_limits(
            SimpleNamespace(primary=primary, secondary=secondary),
            now=now,
        )

        self.assertEqual(status, "available")

    def test_auto_proposal_quota_is_unavailable_with_malformed_weekly_window(
        self,
    ) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        primary = SimpleNamespace(
            used_percent=0,
            resets_at=int((now + timedelta(hours=5)).timestamp()),
            window_duration_mins=5 * 60,
        )
        weekly_reset = int((now + timedelta(days=7)).timestamp())
        malformed_windows = {
            "missing duration": SimpleNamespace(
                used_percent=0,
                resets_at=weekly_reset,
            ),
            "nonnumeric usage": SimpleNamespace(
                used_percent="unknown",
                resets_at=weekly_reset,
                window_duration_mins=7 * 24 * 60,
            ),
            "nonpositive duration": SimpleNamespace(
                used_percent=0,
                resets_at=weekly_reset,
                window_duration_mins=0,
            ),
        }

        for case, secondary in malformed_windows.items():
            with self.subTest(case=case):
                status = autonomous_goals._auto_proposal_quota_status_from_rate_limits(
                    SimpleNamespace(primary=primary, secondary=secondary),
                    now=now,
                )

                self.assertEqual(status, "unavailable")

    @patch("hitch.main.workflows.autonomous_goals.app_server_pool.borrow_codex")
    def test_auto_proposal_quota_pause_fails_closed_when_unavailable(self, mock_borrow_codex: MagicMock) -> None:
        mock_borrow_codex.return_value.__enter__.side_effect = CodexError("rate limits unavailable")

        self.assertEqual(autonomous_goals._auto_proposal_quota_status(), "unavailable")

    @patch("hitch.main.workflows.system_agents.logger")
    @patch("hitch.main.workflows.autonomous_goals.app_server_pool.borrow_codex")
    def test_auto_proposal_quota_pause_fails_closed_on_malformed_response(
        self, mock_borrow_codex: MagicMock, mock_logger: MagicMock
    ) -> None:
        ctx = mock_borrow_codex.return_value.__enter__.return_value
        ctx._client.request.return_value = SimpleNamespace(rate_limits=object())

        self.assertEqual(autonomous_goals._auto_proposal_quota_status(), "unavailable")
        mock_logger.exception.assert_called_once_with(
            "failed to verify account rate limits for auto-proposal quota pause"
        )

    @patch("hitch.main.workflows.system_agents.timezone.now")
    @patch("hitch.main.workflows.autonomous_goals._auto_proposal_quota_status")
    def test_quota_throttle_caches_verdict_within_ttl(self, mock_quota: MagicMock, mock_now: MagicMock) -> None:
        autonomous_goals._reset_auto_proposal_quota_cache()
        self.addCleanup(autonomous_goals._reset_auto_proposal_quota_cache)
        start = datetime(2026, 1, 1, tzinfo=UTC)
        mock_quota.return_value = "low"

        mock_now.return_value = start
        self.assertEqual(autonomous_goals._auto_proposal_quota_status_throttled(), "low")

        # A second call one minute later reuses the cached verdict without
        # re-querying, even though the underlying check would now say available.
        mock_quota.return_value = "available"
        mock_now.return_value = start + timedelta(minutes=1)
        self.assertEqual(autonomous_goals._auto_proposal_quota_status_throttled(), "low")
        mock_quota.assert_called_once()

        # Past the TTL the remote check runs again and the verdict refreshes.
        mock_now.return_value = start + timedelta(minutes=6)
        self.assertEqual(autonomous_goals._auto_proposal_quota_status_throttled(), "available")
        self.assertEqual(mock_quota.call_count, 2)


class AutonomousGoalAutoProposalConcurrencyTests(TransactionTestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        autonomous_goals._reset_auto_proposal_quota_cache()
        self.quota_patcher = patch(
            "hitch.main.workflows.autonomous_goals._auto_proposal_quota_status",
            return_value="available",
        )
        self.mock_auto_proposal_quota_status = self.quota_patcher.start()
        self.addCleanup(self.quota_patcher.stop)
        self.worktree_patcher = patch(
            "hitch.main.workflows.autonomous_goals.create_worktree_for_session",
            return_value=MagicMock(path=Path("/repo-worktree")),
        )
        self.mock_create_worktree = self.worktree_patcher.start()
        self.addCleanup(self.worktree_patcher.stop)

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_concurrent_auto_proposal_starts_share_global_queue_lock(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        first_project = _make_project()
        second_project = _make_project(name="Other", repo_path="/other")
        first_goal = AutonomousGoal.objects.create(
            project=first_project,
            title="Keep tests current",
            goal="Find small test improvements.",
            auto_proposal_enabled=True,
        )
        second_goal = AutonomousGoal.objects.create(
            project=second_project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            auto_proposal_enabled=True,
        )
        branch_lookup_barrier = threading.Barrier(2)
        spawn_lock = threading.Lock()
        spawned_threads: list[str] = []
        db_connection_lock = threading.Lock()
        worker_db_connections: list[Any] = []

        def branch_sha(_repo_path: str) -> str:
            branch_lookup_barrier.wait(timeout=10)
            return "a" * 40

        def spawn_instance(**_kwargs: object) -> CodexInstance:
            with spawn_lock:
                thread_id = f"candidate-thread-{len(spawned_threads) + 1}"
                spawned_threads.append(thread_id)
            return _instance(
                thread_id=thread_id,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            )

        def close_thread_db_connection(db_connection: Any) -> None:
            raw_connection = db_connection.connection
            db_connection.close()
            if raw_connection is not None:
                raw_connection.close()
                db_connection.connection = None

        def start(goal_id: int) -> bool:
            db_connection = connections["default"]
            db_connection.inc_thread_sharing()
            with db_connection_lock:
                worker_db_connections.append(db_connection)
            try:
                return autonomous_goals._maybe_start_auto_proposal_workflow(goal_id)
            finally:
                close_thread_db_connection(db_connection)

        mock_default_sha.side_effect = branch_sha
        mock_spawn.side_effect = spawn_instance

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(start, first_goal.pk),
                executor.submit(start, second_goal.pk),
            ]
            results = [future.result(timeout=10) for future in futures]

        for db_connection in worker_db_connections:
            close_thread_db_connection(db_connection)
            db_connection.dec_thread_sharing()

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 1)
        self.assertEqual(mock_spawn.call_count, 1)
        self.assertEqual(
            SystemWorkflow.objects.filter(
                kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                status=SystemWorkflow.STATUS_RUNNING,
                state__auto_proposal=True,
            ).count(),
            1,
        )


class AutonomousGoalWorkflowTests(TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        autonomous_goals._reset_auto_proposal_quota_cache()
        self.quota_patcher = patch(
            "hitch.main.workflows.autonomous_goals._auto_proposal_quota_status",
            return_value="available",
        )
        self.mock_auto_proposal_quota_status = self.quota_patcher.start()
        self.addCleanup(self.quota_patcher.stop)
        self.worktree_patcher = patch(
            "hitch.main.workflows.autonomous_goals.create_worktree_for_session",
            return_value=MagicMock(path=Path("/repo-worktree")),
        )
        self.mock_create_worktree = self.worktree_patcher.start()
        self.addCleanup(self.worktree_patcher.stop)

    def test_autonomous_goal_candidate_parser_rejects_invalid_wrapped_output(
        self,
    ) -> None:
        self.assertIsNone(
            agent_io._parse_autonomous_goal_candidate_output(json.dumps({"proposal": None, "message": "   "}))
        )
        self.assertIsNone(
            agent_io._parse_autonomous_goal_candidate_output(json.dumps({"proposal": "not an object", "message": ""}))
        )
        self.assertIsNone(
            agent_io._parse_autonomous_goal_candidate_output(json.dumps({"proposal": {"title": ""}, "message": ""}))
        )
        self.assertIsNone(
            agent_io._parse_autonomous_goal_candidate_output(json.dumps({"title": "", "summary": "", "impact": ""}))
        )

    def test_candidate_memory_summary_falls_back_to_proposal_details(self) -> None:
        parsed = agent_io._parse_autonomous_goal_candidate_output(
            json.dumps(
                {
                    "proposal": {
                        "title": "Add parser coverage",
                        "summary": "Cover parser edge cases.",
                        "impact": "Fewer regressions.",
                        "implemented_changes": "Added parser tests.",
                        "implementation_direction": "Add focused tests.",
                        "verification": "Not run.",
                        "rough_edges": "Needs cleanup.",
                        "suggested_continuation": "Polish and test parser work.",
                        "relevant_files": ["hitch/main/rollout.py"],
                    },
                    "message": "",
                    "next_steps_summary": "",
                    "memory_relevant_files": [],
                }
            )
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIn("Implemented: Added parser tests.", parsed["next_steps_summary"])
        self.assertIn(
            "Suggested continuation: Polish and test parser work.",
            parsed["next_steps_summary"],
        )
        message_fallback = agent_io._parse_autonomous_goal_candidate_output(
            json.dumps(
                {
                    "proposal": None,
                    "message": "Use the message as the durable summary.",
                    "next_steps_summary": "",
                    "memory_relevant_files": [],
                }
            )
        )

        self.assertIsNotNone(message_fallback)
        assert message_fallback is not None
        self.assertEqual(
            message_fallback["next_steps_summary"],
            "Use the message as the durable summary.",
        )

    def test_autonomous_goal_history_summary_parser(self) -> None:
        parsed = agent_io._parse_autonomous_goal_history_summary_output(
            json.dumps(
                {
                    "brief": "Use accepted parser helpers as precedent.",
                    "recent_stack": ["#2 superseded #1."],
                    "accepted_lessons": ["Parser helper extraction worked."],
                    "avoid_or_reconsider": ["Avoid broad rewrites."],
                    "promising_next_directions": ["Add focused parser tests."],
                    "important_files": ["hitch/main/rollout.py"],
                }
            )
        )

        self.assertEqual(
            parsed,
            {
                "brief": "Use accepted parser helpers as precedent.",
                "recent_stack": ["#2 superseded #1."],
                "accepted_lessons": ["Parser helper extraction worked."],
                "avoid_or_reconsider": ["Avoid broad rewrites."],
                "promising_next_directions": ["Add focused parser tests."],
                "important_files": ["hitch/main/rollout.py"],
            },
        )
        self.assertIsNone(agent_io._parse_autonomous_goal_history_summary_output(json.dumps({"brief": "   "})))
        self.assertIsNone(agent_io._parse_autonomous_goal_history_summary_output("not json"))

    def test_recent_proposal_references_cover_empty_and_missing_paths(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )

        self.assertEqual(
            autonomous_goal_prompts._autonomous_goal_recent_proposal_run_references(autonomous_goal),
            "(none)",
        )

        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        accepted = SessionMetadata.objects.create(
            thread_id="accepted-thread",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/rollout.py"],
            candidate_session=candidate,
            accepted_session=accepted,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_notes="Keep the focused parser helper extraction.",
            outcome_metadata={"stacked_diff_iteration": 2, "stacked_diff_depth": 5},
        )

        references = autonomous_goal_prompts._autonomous_goal_recent_proposal_run_references(autonomous_goal)

        self.assertIn("stack round 2 of 5", references)
        self.assertIn("Candidate: thread candidate-thread; session file (none)", references)
        self.assertIn("Accepted: thread accepted-thread; session file (none)", references)
        self.assertIn(
            "Outcome notes: Keep the focused parser helper extraction.",
            references,
        )

    def test_expected_agent_kind_includes_history_summary_step(self) -> None:
        workflow = SystemWorkflow(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            step=system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING,
        )

        self.assertEqual(
            system_agents._expected_system_agent_kinds_for_step(workflow),
            (system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,),
        )

    def _autonomous_goal(self) -> AutonomousGoal:
        project, _ = Project.objects.get_or_create(repo_path="/repo", defaults={"name": "Hitch"})
        return AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            ambition=AutonomousGoal.AMBITION_HIGH,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_LIVE,
            proposal_budget=25000,
        )

    def _stranded_autonomous_goal_workflow(self, step: str, goal: AutonomousGoal) -> SystemWorkflow:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
            cwd=goal.project.repo_path,
            status=SystemWorkflow.STATUS_RUNNING,
            step=step,
            state={"autonomous_goal_id": goal.pk},
        )
        # Age the row past the spawn-stale window to mimic a workflow whose
        # spawn handler was killed before the worker launched.
        SystemWorkflow.objects.filter(pk=workflow.pk).update(updated_at=datetime.now(UTC) - timedelta(minutes=20))
        return workflow

    def test_reconcile_blocks_stranded_candidate_spawn(self) -> None:
        # The candidate spawn creates a worktree and has step-specific dispatch,
        # so a stranded candidate is blocked rather than re-driven.
        goal = self._autonomous_goal()
        workflow = self._stranded_autonomous_goal_workflow(system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING, goal)
        self.assertTrue(autonomous_goals._autonomous_goal_running_workflow_exists(goal))

        system_agents.reconcile_terminal_workflow_instances(main_thread_id=workflow.main_thread_id)

        workflow.refresh_from_db()
        # No longer RUNNING, so the goal is unblocked for future proposals and
        # disk cleanup can reclaim any leaked worktree.
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertFalse(autonomous_goals._autonomous_goal_running_workflow_exists(goal))
        # A user-visible failure notice was recorded.
        self.assertTrue(
            ProposedSession.objects.filter(
                source_workflow=workflow,
                inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            ).exists()
        )

    @patch("hitch.main.workflows.autonomous_goals._block_autonomous_goal_spawn_failure_if_active")
    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_judge_or_block")
    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_history_summary_or_fallback")
    def test_recover_redrives_summary_and_judge_blocks_candidate(
        self,
        mock_history: MagicMock,
        mock_judge: MagicMock,
        mock_block: MagicMock,
    ) -> None:
        def reset() -> None:
            mock_history.reset_mock()
            mock_judge.reset_mock()
            mock_block.reset_mock()

        # HISTORY_SUMMARIZING: re-drive the summarizer (which falls back to the
        # candidate on its own failure), never block.
        workflow = self._stranded_autonomous_goal_workflow(
            system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING,
            self._autonomous_goal(),
        )
        autonomous_goals._recover_stranded_autonomous_goal_workflow(workflow)
        mock_history.assert_called_once()
        mock_judge.assert_not_called()
        mock_block.assert_not_called()

        # JUDGE_RUNNING with a persisted candidate: re-drive the read-only judge.
        reset()
        candidate = {"title": "t", "summary": "s", "impact": "i"}
        workflow = self._stranded_autonomous_goal_workflow(
            system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            self._autonomous_goal(),
        )
        workflow.state = {**workflow.state, "candidate": candidate}
        workflow.save(update_fields=["state"])
        autonomous_goals._recover_stranded_autonomous_goal_workflow(workflow)
        mock_judge.assert_called_once()
        self.assertEqual(mock_judge.call_args.args[2], candidate)
        mock_block.assert_not_called()

        # JUDGE_RUNNING without a persisted candidate cannot be re-driven: block.
        reset()
        workflow = self._stranded_autonomous_goal_workflow(
            system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            self._autonomous_goal(),
        )
        autonomous_goals._recover_stranded_autonomous_goal_workflow(workflow)
        mock_judge.assert_not_called()
        mock_block.assert_called_once()

        # CANDIDATE_RUNNING: always block (never re-driven).
        reset()
        workflow = self._stranded_autonomous_goal_workflow(
            system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            self._autonomous_goal(),
        )
        autonomous_goals._recover_stranded_autonomous_goal_workflow(workflow)
        mock_history.assert_not_called()
        mock_judge.assert_not_called()
        mock_block.assert_called_once()

    def test_reconcile_leaves_autonomous_goal_with_live_worker_alone(self) -> None:
        goal = self._autonomous_goal()
        workflow = self._stranded_autonomous_goal_workflow(system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING, goal)
        _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
        )

        system_agents.reconcile_terminal_workflow_instances(main_thread_id=workflow.main_thread_id)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)

    def test_reconcile_defers_autonomous_goal_with_routing_claim(self) -> None:
        # A finished worker mid-handoff has a fresh routing claim but no
        # recreated SystemAgentRun yet; recovery must not block it and discard a
        # valid completed result.
        goal = self._autonomous_goal()
        workflow = self._stranded_autonomous_goal_workflow(system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING, goal)
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_COMPLETED,
        )
        CodexInstance.objects.filter(pk=instance.pk).update(workflow_routing_started_at=datetime.now(UTC))

        self.assertFalse(autonomous_goals._autonomous_goal_spawn_needs_recovery(workflow))

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_candidate_prompt_includes_summary_and_prior_run_references(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
            codex_path="/root/.codex/sessions/prior-candidate.jsonl",
        )
        accepted = SessionMetadata.objects.create(
            thread_id="accepted-thread",
            cwd="/repo",
            project=project,
        )
        judge = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo",
            project=project,
            codex_path="/root/.codex/sessions/judge-thread.jsonl",
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary=("Summary: cleaned up parser setup.\n\nImplemented: moved parser setup into a shared helper."),
            prompt="Continue from the parser helper and add focused regression tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/rollout.py"],
            candidate_session=candidate,
            accepted_session=accepted,
            judge_session=judge,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        summary = _instance(
            thread_id="summary-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "brief": "Prefer parser helpers; do not repeat old setup.",
                    "recent_stack": ["#1 Prior parser cleanup was accepted."],
                    "accepted_lessons": ["Parser helper extraction worked."],
                    "avoid_or_reconsider": ["Avoid vague parser rewrites."],
                    "promising_next_directions": ["Add parser regression tests."],
                    "important_files": ["hitch/main/rollout.py"],
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        candidate_instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.side_effect = [summary, candidate_instance]

        workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        workflow.refresh_from_db()
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING)
        self.assertEqual(
            mock_spawn.call_args.kwargs["agent_kind"],
            system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(summary)
        workflow.refresh_from_db()

        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn(
            "Accepted/dismissed proposal history summary for candidate planning",
            prompt,
        )
        self.assertIn("Prefer parser helpers", prompt)
        self.assertIn("Recent proposal run references", prompt)
        self.assertIn("Proposal #", prompt)
        self.assertIn("Prior parser cleanup", prompt)
        self.assertIn("prior-candidate", prompt)
        self.assertIn("/root/.codex/sessions/prior-candidate.jsonl", prompt)
        self.assertIn("judge-thread", prompt)
        self.assertIn("/root/.codex/sessions/judge-thread.jsonl", prompt)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        run = SystemAgentRun.objects.get(thread_id="candidate-thread")
        self.assertEqual(run.input["proposal_history_count"], 1)
        self.assertFalse(run.input["proposal_history_compacted"])

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_history_summary_invalid_output_falls_back_to_candidate(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/rollout.py"],
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        summary = _instance(
            thread_id="summary-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(self, {"brief": ""}),
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        candidate_instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.side_effect = [summary, candidate_instance]

        workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        system_agents.on_codex_instance_finished(summary)
        workflow.refresh_from_db()

        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        self.assertNotIn(
            autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_HISTORY_SUMMARY_STATE_KEY,
            workflow.state,
        )
        self.assertIn("not valid JSON", workflow.state["proposal_history_summary_error"])
        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn("Prior parser cleanup", prompt)
        self.assertIn("Recent proposal run references", prompt)

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_history_summary_stops_when_it_exhausts_proposal_budget(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            proposal_budget=300,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        summary = _instance(
            thread_id="summary-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "brief": "Prefer parser helpers.",
                    "recent_stack": [],
                    "accepted_lessons": [],
                    "avoid_or_reconsider": [],
                    "promising_next_directions": [],
                    "important_files": [],
                },
                thread_id="summary-thread",
                tokens_used=350,
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        mock_spawn.return_value = summary

        workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        system_agents.on_codex_instance_finished(summary)
        workflow.refresh_from_db()

        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertIn("budget", workflow.state["error"])
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY],
            350,
        )
        run = SystemAgentRun.objects.get(thread_id="summary-thread")
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        notice = ProposedSession.objects.get(source_workflow=workflow)
        self.assertEqual(notice.inbox_kind, ProposedSession.INBOX_KIND_NOTICE)
        self.assertEqual(notice.outcome_metadata["proposal_budget_tokens_used"], 350)
        mock_spawn.assert_called_once()

    @override_settings(AUTONOMOUS_GOAL_HISTORY_SUMMARY_MODEL="gpt-small")
    @patch(
        "hitch.main.workflows.autonomous_goals._write_autonomous_goal_history_files",
        return_value=["/tmp/proposal_history.txt"],
    )
    @patch(
        "hitch.main.workflows.autonomous_goals._split_autonomous_goal_history",
        return_value=("inline proposal history", ["overflow proposal history"]),
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_history_summary_spawn_records_files_and_model(
        self,
        mock_spawn: MagicMock,
        mock_split: MagicMock,
        mock_write: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        summary = _instance(
            thread_id="summary-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        mock_spawn.return_value = summary

        workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        workflow.refresh_from_db()

        mock_split.assert_called_once()
        mock_write.assert_called_once_with(workflow, ["overflow proposal history"])
        self.assertEqual(workflow.state["proposal_history_files"], ["/tmp/proposal_history.txt"])
        self.assertTrue(workflow.state["proposal_history_summary_session_id"])
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(
            kwargs["agent_kind"],
            system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        self.assertEqual(kwargs["sandbox_policy"], "readOnly")
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["model"], "gpt-small")
        self.assertIn("/tmp/proposal_history.txt", kwargs["prompt"])
        run = SystemAgentRun.objects.get(thread_id="summary-thread")
        self.assertEqual(run.input["history_files"], ["/tmp/proposal_history.txt"])

    @patch("hitch.main.workflows.autonomous_goals.session_index.upsert_local_session")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_history_summary_spawn_failure_after_instance_cancels_summarizer(
        self,
        mock_spawn: MagicMock,
        mock_interrupt: MagicMock,
        mock_upsert: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        prior_candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            candidate_session=prior_candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        summary = _instance(
            thread_id="summary-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        candidate = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        mock_spawn.side_effect = [summary, candidate]
        mock_interrupt.return_value = summary
        mock_upsert.side_effect = [RuntimeError("index down"), candidate_metadata]

        workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        workflow.refresh_from_db()

        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        mock_interrupt.assert_called_once_with(summary.pk, expected_thread_id="summary-thread")
        summary_run = SystemAgentRun.objects.get(instance=summary)
        self.assertEqual(summary_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("index down", summary_run.error)
        self.assertIn("index down", workflow.state["proposal_history_summary_error"])
        self.assertEqual(
            mock_spawn.call_args.kwargs["agent_kind"],
            system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_history_summary_spawn_failure_without_run_blocks_not_fallback(
        self,
        mock_spawn: MagicMock,
        mock_interrupt: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        prior_candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            candidate_session=prior_candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        spawned: dict[str, CodexInstance] = {}

        def spawn_summary(**kwargs: Any) -> CodexInstance:
            summary = _instance(
                thread_id="summary-thread",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=int(kwargs["workflow_id"]),
                agent_kind=str(kwargs["agent_kind"]),
            )
            spawned["summary"] = summary
            return summary

        mock_spawn.side_effect = spawn_summary
        mock_interrupt.return_value = None

        with patch.object(
            SystemAgentRun.objects,
            "get_or_create",
            side_effect=RuntimeError("run table busy"),
        ):
            workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        workflow.refresh_from_db()
        summary = spawned["summary"]
        summary.refresh_from_db()

        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertIn("could not preserve a run", workflow.state["error"])
        self.assertEqual(summary.workflow_id, workflow.pk)
        self.assertEqual(
            summary.agent_kind,
            system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        self.assertFalse(SystemAgentRun.objects.filter(instance=summary).exists())
        mock_interrupt.assert_called_once_with(summary.pk, expected_thread_id="summary-thread")
        mock_spawn.assert_called_once()

        summary.status = CodexInstance.STATUS_COMPLETED
        summary.save(update_fields=["status"])
        self.assertTrue(system_agents.on_codex_instance_finished(summary))
        summary_run = SystemAgentRun.objects.get(instance=summary)
        self.assertEqual(summary_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(summary_run.error, workflow.state["error"])

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_history_summary_preserved_run_terminal_fails_after_inactive_interrupt_pending(
        self,
        mock_spawn: MagicMock,
        mock_interrupt: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        prior_candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            candidate_session=prior_candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        spawned: dict[str, CodexInstance] = {}

        def spawn_summary(**kwargs: Any) -> CodexInstance:
            summary = _instance(
                thread_id="summary-thread",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=int(kwargs["workflow_id"]),
                agent_kind=str(kwargs["agent_kind"]),
                status=CodexInstance.STATUS_RUNNING,
            )
            spawned["summary"] = summary
            return summary

        interrupt_calls = 0

        def interrupt_side_effect(instance_id: int, *, expected_thread_id: str) -> CodexInstance | None:
            nonlocal interrupt_calls
            interrupt_calls += 1
            CodexInstance.objects.get(pk=instance_id, thread_id=expected_thread_id)
            return None

        mock_spawn.side_effect = spawn_summary
        mock_interrupt.side_effect = interrupt_side_effect
        original_get_or_create = SystemAgentRun.objects.get_or_create
        get_or_create_calls = 0

        def flaky_get_or_create(*args: Any, **kwargs: Any) -> tuple[SystemAgentRun, bool]:
            nonlocal get_or_create_calls
            get_or_create_calls += 1
            if get_or_create_calls == 1:
                raise RuntimeError("run table busy")
            run, created = original_get_or_create(*args, **kwargs)
            system_agents._block_workflow(run.workflow, "stopped", surface_to_thread=False)
            return run, created

        with patch.object(
            SystemAgentRun.objects,
            "get_or_create",
            side_effect=flaky_get_or_create,
        ):
            workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        workflow.refresh_from_db()
        summary = spawned["summary"]
        summary_run = SystemAgentRun.objects.get(instance=summary)

        self.assertEqual(get_or_create_calls, 2)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(summary_run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(mock_interrupt.call_count, 2)
        mock_spawn.assert_called_once()

        summary.status = CodexInstance.STATUS_COMPLETED
        summary.save(update_fields=["status"])
        self.assertTrue(system_agents.on_codex_instance_finished(summary))
        summary_run.refresh_from_db()
        self.assertEqual(summary_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(summary_run.error, "stopped")

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    def test_history_summary_partial_spawn_cancel_without_run_detaches_instance(
        self, mock_interrupt: MagicMock
    ) -> None:
        summary = _instance(
            thread_id="summary-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=123,
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        mock_interrupt.return_value = summary

        cancelled = autonomous_goals._cancel_partially_spawned_autonomous_goal_history_summary(
            instance=summary,
            run=None,
            error="failed to start autonomous goal history summarizer",
        )
        summary.refresh_from_db()

        self.assertTrue(cancelled)
        mock_interrupt.assert_called_once_with(summary.pk, expected_thread_id="summary-thread")
        self.assertIsNone(summary.workflow_id)
        self.assertEqual(summary.agent_kind, "")

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_failed_history_summary_worker_falls_back_to_candidate(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="prior-candidate",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior parser cleanup",
            summary="Cleaned up parser setup.",
            prompt="Continue parser tests.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        summary = _instance(
            thread_id="summary-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            status=CodexInstance.STATUS_FAILED,
            error="worker died",
            agent_kind=system_agents.AUTONOMOUS_GOAL_HISTORY_SUMMARY_AGENT_KIND,
        )
        candidate_instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.side_effect = [summary, candidate_instance]

        workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)
        system_agents.on_codex_instance_finished(summary)
        workflow.refresh_from_db()

        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        self.assertIn("worker died", workflow.state["proposal_history_summary_error"])
        run = SystemAgentRun.objects.get(thread_id="summary-thread")
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(
            mock_spawn.call_args.kwargs["agent_kind"],
            system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_history_summary_spawn_noops_when_workflow_inactive(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING,
            state={"autonomous_goal_id": autonomous_goal.pk},
        )

        autonomous_goals._spawn_autonomous_goal_history_summary_or_fallback(workflow, autonomous_goal)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        mock_spawn.assert_not_called()

    def test_history_summary_fallback_blocks_when_goal_missing(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(12345),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING,
            state={"autonomous_goal_id": 12345},
        )

        autonomous_goals._record_autonomous_goal_history_summary_fallback_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=12345,
            error="summary failed",
        )

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertEqual(workflow.state["error"], "autonomous goal no longer exists")

    def test_history_summary_fallback_noops_when_inactive(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING,
            state={"autonomous_goal_id": autonomous_goal.pk},
        )

        autonomous_goals._record_autonomous_goal_history_summary_fallback_if_active(
            workflow_id=workflow.pk,
            autonomous_goal_id=autonomous_goal.pk,
            error="summary failed",
        )

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertNotIn("proposal_history_summary_error", workflow.state)

    def test_history_summary_worker_retry_kind(self) -> None:
        workflow = SystemWorkflow(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            step=system_agents.STEP_AUTONOMOUS_GOAL_HISTORY_SUMMARIZING,
        )

        self.assertEqual(
            autonomous_goals._autonomous_goal_worker_retry_kind(workflow),
            "autonomous_goal_history_summary",
        )

    @patch.object(autonomous_goal_prompts, "_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_MAX_ROWS", 1)
    def test_candidate_proposal_history_uses_metadata_and_outcome_notes(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Older parser proposal",
            summary="Older accepted context.",
            prompt="Continue older work.",
            confidence=AutonomousGoal.CONFIDENCE_MEDIUM,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Metadata-only proposal",
            prompt="Continue from the metadata-only result.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
            outcome_notes="Dismissed because a newer parser approach superseded it.",
            outcome_metadata={
                "implemented_changes": "Moved parser setup into a helper.",
                "verification": "Ran parser tests.",
                "rough_edges": "Could still trim duplicate fixtures.",
            },
        )

        history = autonomous_goal_prompts._autonomous_goal_candidate_proposal_history_context(autonomous_goal)

        self.assertTrue(history.compacted)
        self.assertIn("Metadata-only proposal", history.text)
        self.assertIn("Implemented: Moved parser setup into a helper.", history.text)
        self.assertIn("Verification: Ran parser tests.", history.text)
        self.assertIn(
            "Outcome notes: Dismissed because a newer parser approach superseded it.",
            history.text,
        )
        self.assertIn("1 older proposal history rows omitted.", history.text)
        bad_metadata_proposal = ProposedSession(summary="", outcome_metadata=["bad"])
        self.assertEqual(
            autonomous_goal_prompts._autonomous_goal_candidate_proposal_description(bad_metadata_proposal),
            "",
        )

    @patch.object(autonomous_goal_prompts, "_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS", 10)
    @patch.object(autonomous_goal_prompts, "_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_MAX_ROWS", 0)
    def test_candidate_proposal_history_truncates_marker_when_no_rows_fit(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Omitted proposal",
            summary="This proposal is outside the patched row cap.",
            prompt="Continue omitted work.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        history = autonomous_goal_prompts._autonomous_goal_candidate_proposal_history_context(autonomous_goal)

        self.assertTrue(history.compacted)
        self.assertEqual(history.count, 1)
        self.assertLessEqual(
            len(history.text),
            autonomous_goal_prompts._AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS,
        )

    @patch.object(autonomous_goal_prompts, "_AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS", 300)
    def test_candidate_proposal_history_keeps_row_with_long_files(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Older omitted proposal",
            summary="This row should be omitted when the newest row fills the budget.",
            prompt="Continue from older context.",
            confidence=AutonomousGoal.CONFIDENCE_MEDIUM,
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Prior proposal with long files",
            summary=("This accepted proposal summary should survive file compaction."),
            prompt="Continue from the accepted proposal.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/test/" + ("very_long_path_segment_" * 8) + f"{idx}.py" for idx in range(20)],
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )

        history = autonomous_goal_prompts._autonomous_goal_candidate_proposal_history_context(autonomous_goal)

        self.assertTrue(history.compacted)
        self.assertIn("Prior proposal with long files", history.text)
        self.assertIn("Outcome status: accepted", history.text)
        self.assertIn("summary should survive", history.text)
        self.assertNotIn("Older omitted proposal", history.text)
        self.assertNotEqual("1 older proposal history rows omitted.", history.text)
        self.assertLessEqual(
            len(history.text),
            autonomous_goal_prompts._AUTONOMOUS_GOAL_CANDIDATE_HISTORY_CONTEXT_CHARS,
        )

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_workflow_skips_candidate_spawn_when_goal_deleted_after_record_create(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )

        def fake_create(**kwargs: Any) -> tuple[SystemWorkflow, bool]:
            goal = kwargs["autonomous_goal"]
            workflow = SystemWorkflow.objects.create(
                kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
                cwd=goal.project.repo_path,
                status=SystemWorkflow.STATUS_RUNNING,
                step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
                state={"autonomous_goal_id": goal.pk},
            )
            AutonomousGoal.objects.filter(pk=goal.pk).update(deleted_at=datetime.now(UTC))
            return workflow, True

        with patch(
            "hitch.main.workflows.autonomous_goals._create_autonomous_goal_workflow_record",
            side_effect=fake_create,
        ):
            workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)

        mock_spawn.assert_not_called()
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertEqual(workflow.state["error"], "autonomous goal no longer exists")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    def test_accepted_stack_proposal_cancels_running_continuation_on_finish(self, mock_cleanup: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        accepted_candidate = SessionMetadata.objects.create(
            thread_id="candidate-2",
            cwd="/repo-worktree-2",
            project=project,
        )
        running_candidate = SessionMetadata.objects.create(
            thread_id="candidate-3",
            cwd="/repo-worktree-3",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Expand parser coverage",
            summary="Cover more parser edge cases.",
            candidate_session=accepted_candidate,
            accepted_session=accepted_candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 2,
                "stacked_diff_hidden_until_complete": False,
            },
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "proposal_id": proposal.pk,
                "candidate_session_id": running_candidate.pk,
                "session_cwd": "/repo-worktree-3",
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 3,
            },
        )
        instance = _instance(
            thread_id="candidate-3",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-3",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        proposal.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, system_agents.AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_stopped_reason"],
            "proposal_accepted",
        )
        self.assertEqual(ProposedSession.objects.count(), 1)
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, accepted_candidate)
        mock_cleanup.assert_called_once_with("/repo-worktree-3")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    def test_rejected_stack_proposal_cancels_running_continuation_on_finish(self, mock_cleanup: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        running_candidate = SessionMetadata.objects.create(
            thread_id="candidate-3",
            cwd="/repo-worktree-3",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Expand parser coverage",
            summary="Cover more parser edge cases.",
            candidate_session=SessionMetadata.objects.create(
                thread_id="candidate-2",
                cwd="/repo-worktree-2",
                project=project,
            ),
            outcome_status=ProposedSession.OUTCOME_REJECTED,
            outcome_notes="Not the right direction.",
            outcome_metadata={
                "resolved_by": "user",
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 2,
                "stacked_diff_hidden_until_complete": False,
            },
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "proposal_id": proposal.pk,
                "candidate_session_id": running_candidate.pk,
                "session_cwd": "/repo-worktree-3",
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 3,
            },
        )
        instance = _instance(
            thread_id="candidate-3",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-3",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        proposal.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, system_agents.AUTONOMOUS_GOAL_PROPOSAL_REJECTED_ERROR)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_stopped_reason"],
            "proposal_rejected",
        )
        self.assertEqual(ProposedSession.objects.count(), 1)
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertEqual(
            [call.args[0] for call in mock_cleanup.call_args_list],
            ["/repo-worktree-3", "/repo-worktree-2"],
        )

    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_accepted_stack_proposal_stop_ignores_different_proposal(self, mock_interrupt: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        accepted_session = SessionMetadata.objects.create(
            thread_id="accepted-thread",
            cwd="/repo-worktree-1",
            project=project,
        )
        accepted_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Older parser coverage",
            accepted_session=accepted_session,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={"accepted_by": "user"},
        )
        current_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Current parser coverage",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "proposal_id": current_proposal.pk,
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 3,
            },
        )
        instance = _instance(
            thread_id="candidate-3",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-3",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        stopped = autonomous_goals.stop_running_autonomous_goal_stack_after_proposal_resolution(
            autonomous_goal.pk,
            accepted_proposal.pk,
            ProposedSession.OUTCOME_ACCEPTED,
        )

        self.assertTrue(stopped)
        mock_interrupt.assert_not_called()
        workflow.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(run.status, SystemAgentRun.STATUS_RUNNING)

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_stack_proposal_stop_cleans_worktree_between_agent_turns(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        accepted_session = SessionMetadata.objects.create(
            thread_id="candidate-2",
            cwd="/repo-worktree-2",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Expand parser coverage",
            accepted_session=accepted_session,
            candidate_session=accepted_session,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 2,
                "stacked_diff_hidden_until_complete": False,
            },
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "proposal_id": proposal.pk,
                "session_cwd": "/repo-worktree-3",
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 3,
            },
        )
        finished_instance = _instance(
            thread_id="candidate-3",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_COMPLETED,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=finished_instance.thread_id,
            instance=finished_instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )

        stopped = autonomous_goals.stop_running_autonomous_goal_stack_after_proposal_resolution(
            autonomous_goal.pk,
            proposal.pk,
            ProposedSession.OUTCOME_ACCEPTED,
        )

        self.assertTrue(stopped)
        mock_interrupt.assert_not_called()
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_stopped_reason"],
            "proposal_accepted",
        )
        mock_cleanup.assert_called_once_with("/repo-worktree-3")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_stack_proposal_stop_keeps_accepted_worktree_before_next_candidate(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        accepted_session = SessionMetadata.objects.create(
            thread_id="candidate-2",
            cwd="/repo-worktree-2",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Expand parser coverage",
            accepted_session=accepted_session,
            candidate_session=accepted_session,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 2,
                "stacked_diff_hidden_until_complete": False,
            },
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "proposal_id": proposal.pk,
                "session_cwd": "/repo-worktree-2",
                autonomous_goals._AUTONOMOUS_GOAL_STACKED_FORK_CWD_STATE_KEY: ("/repo-worktree-2"),
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 3,
            },
        )

        stopped = autonomous_goals.stop_running_autonomous_goal_stack_after_proposal_resolution(
            autonomous_goal.pk,
            proposal.pk,
            ProposedSession.OUTCOME_ACCEPTED,
        )

        self.assertTrue(stopped)
        mock_interrupt.assert_not_called()
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(
            workflow.state["stacked_diff_stopped_reason"],
            "proposal_accepted",
        )
        mock_cleanup.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_accepted_stack_proposal_stop_leaves_live_uninterrupted_run(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Expand parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={"accepted_by": "user"},
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "proposal_id": proposal.pk,
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 3,
            },
        )
        interrupted_instance = _instance(
            thread_id="candidate-3a",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        live_instance = _instance(
            thread_id="candidate-3b",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        interrupted_run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=interrupted_instance.thread_id,
            instance=interrupted_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        live_run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=live_instance.thread_id,
            instance=live_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        def interrupt_side_effect(_instance_id: int, *, expected_thread_id: str) -> CodexInstance | None:
            if expected_thread_id == interrupted_instance.thread_id:
                return interrupted_instance
            return None

        mock_interrupt.side_effect = interrupt_side_effect

        stopped = autonomous_goals.stop_running_autonomous_goal_stack_after_proposal_resolution(
            autonomous_goal.pk,
            proposal.pk,
            ProposedSession.OUTCOME_ACCEPTED,
        )

        self.assertFalse(stopped)
        workflow.refresh_from_db()
        interrupted_run.refresh_from_db()
        live_run.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(interrupted_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(
            interrupted_run.error,
            system_agents.AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR,
        )
        self.assertEqual(live_run.status, SystemAgentRun.STATUS_RUNNING)
        mock_cleanup.assert_not_called()

        interrupted_instance.status = CodexInstance.STATUS_COMPLETED
        interrupted_instance.save(update_fields=["status"])

        handled = system_agents.on_codex_instance_finished(interrupted_instance)

        self.assertTrue(handled)
        mock_cleanup.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        return_value="c" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_stacked_diff_rejection_stops_with_existing_proposal(
        self,
        mock_spawn: MagicMock,
        _mock_default_sha: MagicMock,
        mock_snapshot: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        self.mock_create_worktree.side_effect = [
            MagicMock(path=Path("/repo-worktree-1")),
            MagicMock(path=Path("/repo-worktree-2")),
        ]
        candidate_1 = _instance(
            thread_id="candidate-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_1 = _instance(
            thread_id="judge-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "The first candidate is useful.",
                    "rationale": "It is focused.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        candidate_2 = _instance(
            thread_id="candidate-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_2 = _instance(
            thread_id="judge-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "confidence": "medium",
                    "summary": "The second candidate is not ready.",
                    "rationale": "It is not confident enough.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        mock_spawn.side_effect = [candidate_1, judge_1, candidate_2, judge_2]

        workflow = autonomous_goals.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )
        candidate_1.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Add parser coverage",
                    "summary": "Cover parser edge cases.",
                    "impact": "Fewer regressions.",
                    "implementation_direction": "Finish parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Selected parser coverage.",
                "memory_relevant_files": [],
            },
        )
        candidate_1.save(update_fields=["events_path"])
        system_agents.on_codex_instance_finished(candidate_1)
        system_agents.on_codex_instance_finished(judge_1)
        first_proposal = ProposedSession.objects.get()
        self.assertEqual(first_proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertFalse(first_proposal.outcome_metadata["stacked_diff_hidden_until_complete"])

        candidate_2.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Expand parser coverage",
                    "summary": "Cover more parser edge cases.",
                    "impact": "Even fewer regressions.",
                    "implementation_direction": "Finish broader parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Expanded parser coverage.",
                "memory_relevant_files": [],
            },
        )
        candidate_2.save(update_fields=["events_path"])
        system_agents.on_codex_instance_finished(candidate_2)
        system_agents.on_codex_instance_finished(judge_2)

        workflow.refresh_from_db()
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertFalse(proposal.outcome_metadata["stacked_diff_hidden_until_complete"])
        self.assertIsNotNone(proposal.candidate_session)
        assert proposal.candidate_session is not None
        self.assertEqual(proposal.candidate_session.thread_id, "candidate-1")
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_stopped_reason"],
            "judge_confidence_below_threshold",
        )
        stop_reason_key = autonomous_goal_proposal_stack._AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY
        self.assertEqual(
            proposal.outcome_metadata[stop_reason_key],
            "judge_confidence_below_threshold",
        )
        mock_snapshot.assert_called_once_with("/repo-worktree-1", message="Add parser coverage")
        mock_cleanup.assert_called_once_with("/repo-worktree-2")

    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        side_effect=RuntimeError("snapshot failed"),
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_stacked_diff_continuation_failure_publishes_existing_proposal(
        self,
        mock_spawn: MagicMock,
        _mock_default_sha: MagicMock,
        mock_snapshot: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=2,
        )
        self.mock_create_worktree.return_value = MagicMock(path=Path("/repo-worktree-1"))
        candidate_1 = _instance(
            thread_id="candidate-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_1 = _instance(
            thread_id="judge-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "The first candidate is useful.",
                    "rationale": "It is focused.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        mock_spawn.side_effect = [candidate_1, judge_1]

        workflow = autonomous_goals.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )
        candidate_1.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Add parser coverage",
                    "summary": "Cover parser edge cases.",
                    "impact": "Fewer regressions.",
                    "implementation_direction": "Finish parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Selected parser coverage.",
                "memory_relevant_files": [],
            },
        )
        candidate_1.save(update_fields=["events_path"])

        system_agents.on_codex_instance_finished(candidate_1)
        system_agents.on_codex_instance_finished(judge_1)

        workflow.refresh_from_db()
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.inbox_kind, ProposedSession.INBOX_KIND_PROPOSAL)
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertFalse(proposal.outcome_metadata["stacked_diff_hidden_until_complete"])
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_stopped_reason"],
            "stacked_diff_continuation_failed",
        )
        mock_snapshot.assert_called_once_with("/repo-worktree-1", message="Add parser coverage")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        return_value="c" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_stacked_diff_candidate_parse_failure_publishes_existing_proposal(
        self,
        mock_spawn: MagicMock,
        _mock_default_sha: MagicMock,
        _mock_snapshot: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=2,
        )
        self.mock_create_worktree.side_effect = [
            MagicMock(path=Path("/repo-worktree-1")),
            MagicMock(path=Path("/repo-worktree-2")),
        ]
        candidate_1 = _instance(
            thread_id="candidate-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_1 = _instance(
            thread_id="judge-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "The first candidate is useful.",
                    "rationale": "It is focused.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        candidate_2 = _instance(
            thread_id="candidate-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(self, {"unexpected": "shape"}),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.side_effect = [candidate_1, judge_1, candidate_2]

        workflow = autonomous_goals.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )
        candidate_1.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Add parser coverage",
                    "summary": "Cover parser edge cases.",
                    "impact": "Fewer regressions.",
                    "implementation_direction": "Finish parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Selected parser coverage.",
                "memory_relevant_files": [],
            },
        )
        candidate_1.save(update_fields=["events_path"])
        system_agents.on_codex_instance_finished(candidate_1)
        system_agents.on_codex_instance_finished(judge_1)

        system_agents.on_codex_instance_finished(candidate_2)

        workflow.refresh_from_db()
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertFalse(proposal.outcome_metadata["stacked_diff_hidden_until_complete"])
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_continuation_error"],
            "autonomous goal candidate output was not valid JSON",
        )
        mock_cleanup.assert_called_once_with("/repo-worktree-2")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        return_value="c" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_stacked_diff_no_proposal_stall_publishes_existing_proposal(
        self,
        mock_spawn: MagicMock,
        _mock_default_sha: MagicMock,
        _mock_snapshot: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=2,
            proposal_budget=100_000_000,
        )
        self.mock_create_worktree.side_effect = [
            MagicMock(path=Path("/repo-worktree-1")),
            MagicMock(path=Path("/repo-worktree-2")),
        ]
        candidate_1 = _instance(
            thread_id="candidate-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_1 = _instance(
            thread_id="judge-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "The first candidate is useful.",
                    "rationale": "It is focused.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        candidate_2 = _instance(
            thread_id="candidate-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "proposal": None,
                    "message": "The continuation did not improve the proposal.",
                    "next_steps_summary": "No stronger proposal found.",
                    "memory_relevant_files": [],
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.side_effect = [candidate_1, judge_1, candidate_2]

        workflow = autonomous_goals.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )
        candidate_1.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Add parser coverage",
                    "summary": "Cover parser edge cases.",
                    "impact": "Fewer regressions.",
                    "implementation_direction": "Finish parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Selected parser coverage.",
                "memory_relevant_files": [],
            },
        )
        candidate_1.save(update_fields=["events_path"])
        system_agents.on_codex_instance_finished(candidate_1)
        system_agents.on_codex_instance_finished(judge_1)

        workflow.refresh_from_db()
        workflow.state = {
            **workflow.state,
            autonomous_goals._AUTONOMOUS_GOAL_NO_PROPOSAL_STREAK_STATE_KEY: (
                autonomous_goals._AUTONOMOUS_GOAL_NO_PROPOSAL_RETRY_LIMIT
            ),
        }
        workflow.save(update_fields=["state", "updated_at"])

        system_agents.on_codex_instance_finished(candidate_2)

        workflow.refresh_from_db()
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertFalse(proposal.outcome_metadata["stacked_diff_hidden_until_complete"])
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_stopped_reason"],
            "candidate_no_proposal_stall_limit",
        )
        stop_reason_key = autonomous_goal_proposal_stack._AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY
        self.assertEqual(
            proposal.outcome_metadata[stop_reason_key],
            "candidate_no_proposal_stall_limit",
        )
        self.assertEqual(proposal.outcome_metadata["no_proposal_retries"], 3)
        self.assertEqual(proposal.outcome_metadata["no_proposal_retry_limit"], 3)
        mock_cleanup.assert_called_once_with("/repo-worktree-2")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        return_value="c" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_stacked_diff_judge_parse_failure_publishes_existing_proposal(
        self,
        mock_spawn: MagicMock,
        _mock_default_sha: MagicMock,
        _mock_snapshot: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=2,
        )
        self.mock_create_worktree.side_effect = [
            MagicMock(path=Path("/repo-worktree-1")),
            MagicMock(path=Path("/repo-worktree-2")),
        ]
        candidate_1 = _instance(
            thread_id="candidate-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_1 = _instance(
            thread_id="judge-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "The first candidate is useful.",
                    "rationale": "It is focused.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        candidate_2 = _instance(
            thread_id="candidate-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_2 = _instance(
            thread_id="judge-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(self, {"unexpected": "shape"}),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        mock_spawn.side_effect = [candidate_1, judge_1, candidate_2, judge_2]

        workflow = autonomous_goals.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )
        candidate_1.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Add parser coverage",
                    "summary": "Cover parser edge cases.",
                    "impact": "Fewer regressions.",
                    "implementation_direction": "Finish parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Selected parser coverage.",
                "memory_relevant_files": [],
            },
        )
        candidate_1.save(update_fields=["events_path"])
        system_agents.on_codex_instance_finished(candidate_1)
        system_agents.on_codex_instance_finished(judge_1)
        candidate_2.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Expand parser coverage",
                    "summary": "Cover more parser edge cases.",
                    "impact": "Even fewer regressions.",
                    "implementation_direction": "Finish broader parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Expanded parser coverage.",
                "memory_relevant_files": [],
            },
        )
        candidate_2.save(update_fields=["events_path"])
        system_agents.on_codex_instance_finished(candidate_2)

        system_agents.on_codex_instance_finished(judge_2)

        workflow.refresh_from_db()
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertFalse(proposal.outcome_metadata["stacked_diff_hidden_until_complete"])
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertEqual(
            workflow.state["stacked_diff_continuation_error"],
            "autonomous goal judge output was not valid JSON",
        )
        mock_cleanup.assert_called_once_with("/repo-worktree-2")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_worktree")
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.autonomous_goals.create_worktree_for_session")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_workflow_cleans_up_candidate_worktree_when_spawn_fails(
        self,
        mock_spawn: MagicMock,
        mock_worktree: MagicMock,
        _mock_default_sha: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
        )
        managed_worktree = MagicMock(path=Path("/repo-worktree"))
        mock_worktree.return_value = managed_worktree
        mock_spawn.side_effect = RuntimeError("boom")

        workflow = autonomous_goals.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )

        mock_cleanup.assert_called_once_with(managed_worktree)
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_waits_when_manual_goal_is_running(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        running_goal = AutonomousGoal.objects.create(
            project=project,
            title="Running goal",
            goal="Manual work owns the queue.",
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Auto goal",
            goal="This should wait.",
            auto_proposal_enabled=True,
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(running_goal.pk),
            cwd=project.repo_path,
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": running_goal.pk, "auto_proposal": False},
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        self.assertEqual(SystemWorkflow.objects.count(), 1)
        mock_default_sha.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        return_value="c" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_continues_from_pending_stack_proposal(
        self,
        mock_spawn: MagicMock,
        _mock_default_sha: MagicMock,
        mock_snapshot: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            auto_proposal_last_no_proposal_sha="a" * 40,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=2,
            proposal_budget=2000,
        )
        candidate_session = SessionMetadata.objects.create(
            thread_id="candidate-1",
            cwd="/repo-worktree-1",
            project=project,
        )
        previous_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Add parser coverage",
            summary="Cover parser edge cases.",
            candidate_session=candidate_session,
            outcome_metadata={
                "stacked_diff_depth": 2,
                "stacked_diff_iteration": 1,
                "proposal_budget": 2000,
                "proposal_budget_tokens_used": 350,
                "proposal_budget_failed_attempts": 1,
            },
        )
        self.mock_create_worktree.return_value = MagicMock(path=Path("/repo-worktree-2"))
        candidate_2 = _instance(
            thread_id="candidate-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        judge_2 = _instance(
            thread_id="judge-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "The second candidate is better.",
                    "rationale": "It builds on the first candidate.",
                },
                thread_id="judge-2",
                tokens_used=275,
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        mock_spawn.side_effect = [candidate_2, judge_2]

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        workflow = SystemWorkflow.objects.get()
        self.assertEqual(workflow.state["proposal_id"], previous_proposal.pk)
        self.assertEqual(workflow.state["proposal_budget_tokens_used"], 350)
        self.assertEqual(workflow.state["proposal_budget_failed_attempts"], 1)
        self.assertEqual(workflow.state["stacked_diff_iteration"], 2)
        self.assertEqual(workflow.state["session_cwd"], "/repo-worktree-2")
        self.assertEqual(workflow.state["default_branch_sha"], "a" * 40)
        mock_snapshot.assert_called_once_with("/repo-worktree-1", message="Add parser coverage")
        self.mock_create_worktree.assert_called_with("/repo", base_ref="c" * 40)
        self.assertIn("candidate round 2 of 2", mock_spawn.call_args.kwargs["prompt"])
        previous_proposal.refresh_from_db()
        self.assertEqual(previous_proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertEqual(previous_proposal.outcome_notes, "")
        self.assertFalse(previous_proposal.outcome_metadata["stacked_diff_hidden_until_complete"])

        candidate_2.events_path = _events_file(
            self,
            {
                "proposal": {
                    "title": "Expand parser coverage",
                    "summary": "Cover more parser edge cases.",
                    "impact": "Even fewer regressions.",
                    "implementation_direction": "Finish broader parser tests.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                "message": "",
                "next_steps_summary": "Expanded parser coverage.",
                "memory_relevant_files": [],
            },
            thread_id="candidate-2",
            tokens_used=125,
        )
        candidate_2.save(update_fields=["events_path"])
        system_agents.on_codex_instance_finished(candidate_2)
        workflow.refresh_from_db()
        workflow.state = {
            **workflow.state,
            autonomous_goal_prompts._AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY: (
                autonomous_goals._AUTONOMOUS_GOAL_NO_PROGRESS_RETRY_LIMIT
            ),
            autonomous_goal_prompts._AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY: {
                "reason": "judge_confidence_below_threshold"
            },
        }
        workflow.save(update_fields=["state", "updated_at"])
        system_agents.on_codex_instance_finished(judge_2)

        workflow.refresh_from_db()
        proposals = list(ProposedSession.objects.order_by("pk"))
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0].pk, previous_proposal.pk)
        self.assertEqual(proposals[0].outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertEqual(
            proposals[0].outcome_notes,
            f"Replaced by stacked diff proposal #{proposals[1].pk}.",
        )
        self.assertEqual(proposals[1].outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertEqual(proposals[1].outcome_metadata["stacked_diff_iteration"], 2)
        self.assertEqual(proposals[1].outcome_metadata["proposal_budget_tokens_used"], 750)
        self.assertEqual(proposals[1].outcome_metadata["proposal_budget_failed_attempts"], 1)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertNotIn(
            autonomous_goal_prompts._AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY,
            workflow.state,
        )
        self.assertNotIn(
            autonomous_goal_prompts._AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY,
            workflow.state,
        )
        mock_cleanup.assert_called_once_with("/repo-worktree-1")

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    @patch("hitch.main.workflows.autonomous_goals._claim_autonomous_goal_stack_continuation_proposal")
    def test_auto_proposal_does_not_start_when_stack_claim_loses_race(
        self,
        mock_claim: MagicMock,
        mock_spawn: MagicMock,
        mock_default_sha: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        candidate_session = SessionMetadata.objects.create(
            thread_id="candidate-1",
            cwd="/repo-worktree-1",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Stack proposal",
            candidate_session=candidate_session,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )
        mock_default_sha.return_value = "a" * 40
        mock_claim.return_value = None

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        self.assertFalse(SystemWorkflow.objects.exists())
        mock_spawn.assert_not_called()

    def test_pending_proposal_state_empty_input_has_no_blockers(self) -> None:
        state = autonomous_goal_proposal_stack._autonomous_goal_pending_proposal_state([])

        self.assertEqual(state.blocking_goal_ids, set())
        self.assertEqual(state.continuable_stack_goal_ids, set())

    def test_pending_proposal_state_blocks_exhausted_stack_budget(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
            proposal_budget=1000,
        )
        candidate_session = SessionMetadata.objects.create(
            thread_id="candidate-1",
            cwd="/repo-worktree-1",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Exhausted stack proposal",
            candidate_session=candidate_session,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
                "proposal_budget_tokens_used": 1000,
            },
        )

        state = autonomous_goal_proposal_stack._autonomous_goal_pending_proposal_state([autonomous_goal])

        self.assertFalse(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_allows_stack_continuation(
                proposal, autonomous_goal
            )
        )
        self.assertFalse(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_budget_allows_stack_continuation(
                proposal, autonomous_goal
            )
        )
        self.assertEqual(state.blocking_goal_ids, {autonomous_goal.pk})
        self.assertEqual(state.continuable_stack_goal_ids, set())

    def test_stack_continuation_helpers_reject_invalid_proposal_states(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        candidate_session = SessionMetadata.objects.create(
            thread_id="candidate-1",
            cwd="/repo-worktree-1",
            project=project,
        )
        dismissed_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Dismissed stack proposal",
            candidate_session=candidate_session,
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )
        self.assertFalse(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_allows_stack_continuation(
                dismissed_proposal, autonomous_goal
            )
        )
        self.assertIsNone(
            autonomous_goal_proposal_stack._claim_autonomous_goal_stack_continuation_proposal(dismissed_proposal)
        )

        notice_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Stack notice",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            candidate_session=candidate_session,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )
        self.assertFalse(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_allows_stack_continuation(
                notice_proposal, autonomous_goal
            )
        )

        repo_candidate = SessionMetadata.objects.create(
            thread_id="candidate-repo",
            cwd="/repo",
            project=project,
        )
        repo_cwd_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Repo cwd proposal",
            candidate_session=repo_candidate,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )
        self.assertFalse(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_allows_stack_continuation(
                repo_cwd_proposal, autonomous_goal
            )
        )

        propose_only_goal = AutonomousGoal.objects.create(
            project=project,
            title="Propose only goal",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            stacked_diff_depth=3,
        )
        too_shallow_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=propose_only_goal,
            title="Too shallow stack proposal",
            candidate_session=candidate_session,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )
        self.assertIsNone(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_stack_continuation_metadata(
                too_shallow_proposal, propose_only_goal
            )
        )

        completed_stack_proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Completed stack proposal",
            candidate_session=candidate_session,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 3,
            },
        )
        self.assertIsNone(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_stack_continuation_metadata(
                completed_stack_proposal, autonomous_goal
            )
        )
        self.assertEqual(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_stack_iteration(completed_stack_proposal),
            3,
        )
        plain_proposal = ProposedSession(outcome_metadata={})
        self.assertEqual(
            autonomous_goal_proposal_stack._autonomous_goal_proposal_stack_iteration(plain_proposal),
            1,
        )

    def test_create_workflow_record_rejects_invalid_stack_continuation_metadata(
        self,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        candidate_session = SessionMetadata.objects.create(
            thread_id="candidate-1",
            cwd="/repo-worktree-1",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Stack proposal without metadata",
            candidate_session=candidate_session,
        )

        with self.assertRaisesRegex(ValueError, "stack continuation proposal missing stack metadata"):
            autonomous_goals._create_autonomous_goal_workflow_record(
                autonomous_goal=autonomous_goal,
                auto_proposal=True,
                default_branch_sha="a" * 40,
                use_worktrees=True,
                stack_continuation_proposal=proposal,
            )

        self.assertFalse(SystemWorkflow.objects.exists())

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        return_value="c" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_does_not_retry_stopped_stack_continuation(
        self,
        mock_spawn: MagicMock,
        _mock_default_sha: MagicMock,
        _mock_snapshot: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=2,
        )
        candidate_session = SessionMetadata.objects.create(
            thread_id="candidate-1",
            cwd="/repo-worktree-1",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Add parser coverage",
            summary="Cover parser edge cases.",
            candidate_session=candidate_session,
            outcome_metadata={
                "stacked_diff_depth": 2,
                "stacked_diff_iteration": 1,
            },
        )
        self.mock_create_worktree.return_value = MagicMock(path=Path("/repo-worktree-2"))
        candidate_2 = _instance(
            thread_id="candidate-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            events_path=_events_file(
                self,
                {
                    "proposal": None,
                    "message": "No useful continuation was found.",
                    "next_steps_summary": "Stop after checking parser coverage.",
                    "memory_relevant_files": [],
                },
            ),
        )
        mock_spawn.return_value = candidate_2

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)
        system_agents.on_codex_instance_finished(candidate_2)
        started_again = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        self.assertEqual(started_again, 0)
        self.assertEqual(SystemWorkflow.objects.count(), 1)
        self.assertEqual(mock_spawn.call_count, 1)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        stop_reason_key = autonomous_goal_proposal_stack._AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY
        self.assertEqual(
            proposal.outcome_metadata[stop_reason_key],
            "candidate_no_proposal",
        )
        mock_cleanup.assert_called_once_with("/repo-worktree-2")

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_rechecks_enablement_after_lock(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            auto_proposal_enabled=False,
        )

        started = autonomous_goals._maybe_start_auto_proposal_workflow(autonomous_goal.pk)

        self.assertFalse(started)
        self.assertFalse(SystemWorkflow.objects.exists())
        mock_default_sha.assert_not_called()
        mock_spawn.assert_not_called()

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals._lock_auto_proposal_queue",
        side_effect=OperationalError("schema changed"),
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_reraises_non_lock_operational_error(
        self,
        mock_spawn: MagicMock,
        mock_lock_queue: MagicMock,
        mock_default_sha: MagicMock,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            auto_proposal_enabled=True,
        )

        with self.assertRaisesRegex(OperationalError, "schema changed"):
            autonomous_goals._maybe_start_auto_proposal_workflow(autonomous_goal.pk)

        self.assertFalse(SystemWorkflow.objects.exists())
        mock_default_sha.assert_called_once_with("/repo")
        mock_lock_queue.assert_called_once_with()
        mock_spawn.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_rechecks_enablement_after_sha_lookup(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            auto_proposal_enabled=True,
        )

        def disable_goal(_repo_path: str) -> str:
            AutonomousGoal.objects.filter(pk=autonomous_goal.pk).update(auto_proposal_enabled=False)
            return "a" * 40

        mock_default_sha.side_effect = disable_goal

        started = autonomous_goals._maybe_start_auto_proposal_workflow(autonomous_goal.pk)

        self.assertFalse(started)
        self.assertFalse(SystemWorkflow.objects.exists())
        mock_default_sha.assert_called_once_with("/repo")
        mock_spawn.assert_not_called()

    def test_auto_proposal_batch_survives_a_goal_raising_mid_iteration(self) -> None:
        # The goal ids are a snapshot, so a goal (or its project) deleted between
        # the snapshot and the select_for_update().get() makes the per-goal call
        # raise. One bad row must not abort the rest of the batch.
        project = _make_project()
        first = AutonomousGoal.objects.create(
            project=project,
            title="First",
            goal="First goal.",
            auto_proposal_enabled=True,
        )
        second = AutonomousGoal.objects.create(
            project=project,
            title="Second",
            goal="Second goal.",
            auto_proposal_enabled=True,
        )

        def fake_start(goal_id: int) -> bool:
            if goal_id == first.pk:
                raise AutonomousGoal.DoesNotExist
            return True

        with patch.object(
            autonomous_goals,
            "_maybe_start_auto_proposal_workflow",
            side_effect=fake_start,
        ) as mock_start:
            started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        self.assertEqual(
            [invocation.args[0] for invocation in mock_start.call_args_list],
            [first.pk, second.pk],
        )

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_pauses_when_usage_quota_is_low(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        self.mock_auto_proposal_quota_status.return_value = "low"
        project = _make_project()
        AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            auto_proposal_enabled=True,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        self.assertFalse(SystemWorkflow.objects.exists())
        mock_default_sha.assert_not_called()
        mock_spawn.assert_not_called()

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_skips_pending_proposal_but_not_notice(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        pending_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        notice_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve docs",
            goal="Find useful documentation increments.",
            auto_proposal_enabled=True,
        )
        ProposedSession.objects.create(
            autonomous_goal=pending_goal,
            title="Add parser coverage",
        )
        ProposedSession.objects.create(
            autonomous_goal=notice_goal,
            title="No proposal from Improve docs",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        workflow = SystemWorkflow.objects.get()
        self.assertEqual(
            workflow.main_thread_id,
            autonomous_goals._autonomous_goal_main_thread_id(notice_goal.pk),
        )

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_blocks_transient_proposal_start_claim(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Accepted proposal start",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "auto_qa_enabled": True,
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (datetime.now(UTC).isoformat()),
            },
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        mock_spawn.assert_not_called()

    def test_proposal_start_claim_activity_parses_only_fresh_timestamps(self) -> None:
        now = datetime.now(UTC)
        claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY

        self.assertFalse(ProposedSession.accepted_session_start_claim_is_active(None, now=now))
        self.assertFalse(ProposedSession.accepted_session_start_claim_is_active({claim_key: 123}, now=now))
        self.assertFalse(ProposedSession.accepted_session_start_claim_is_active({claim_key: "not-a-date"}, now=now))
        self.assertFalse(
            ProposedSession.accepted_session_start_claim_is_active(
                {
                    claim_key: (
                        now - ProposedSession.ACCEPTED_SESSION_START_CLAIM_TTL - timedelta(seconds=1)
                    ).isoformat()
                },
                now=now,
            )
        )
        self.assertTrue(
            ProposedSession.accepted_session_start_claim_is_active(
                {claim_key: now.replace(tzinfo=None).isoformat()}, now=now
            )
        )
        self.assertFalse(
            ProposedSession.accepted_session_start_claim_is_active(
                {
                    claim_key: (
                        now + ProposedSession.ACCEPTED_SESSION_START_CLAIM_CLOCK_SKEW + timedelta(seconds=1)
                    ).isoformat()
                },
                now=now,
            )
        )

    @patch(
        "hitch.main.goals.autonomous_goal_proposal_stack.rollout.session_stage_data",
        side_effect=ValueError("broken rollout"),
    )
    def test_accepted_session_rollout_error_is_logged(self, mock_stage_data: MagicMock) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        rollout_path = Path(temp_dir.name) / "rollout.jsonl"
        rollout_path.write_text("{}\n", encoding="utf-8")

        with self.assertLogs(
            autonomous_goal_proposal_stack.logger,
            level="ERROR",
        ) as captured:
            evidence = autonomous_goal_proposal_stack._accepted_session_rollout_evidence(str(rollout_path))

        assert evidence is not None
        self.assertFalse(evidence.done)
        self.assertFalse(evidence.superseded_by_lifecycle)
        self.assertIn(
            f"failed to read accepted-session rollout stage from {rollout_path}",
            "\n".join(captured.output),
        )
        mock_stage_data.assert_called_once_with(rollout_path)

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_blocks_stale_done_after_resumed_accepted_session(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        rollout_path = Path(temp_dir.name) / "rollout.jsonl"
        pr_url = "https://github.com/cberner/hitch/pull/94"
        rollout_path.write_text(
            "\n".join(
                [
                    _rollout_line(
                        "event_msg",
                        {
                            "type": "user_message",
                            "message": PR_SLASH_PROMPT,
                        },
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "function_call",
                            "name": "github_fetch_pr",
                            "arguments": json.dumps(
                                {
                                    "repo_full_name": "cberner/hitch",
                                    "pr_number": 94,
                                }
                            ),
                            "call_id": "call-fetch",
                        },
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "function_call_output",
                            "call_id": "call-fetch",
                            "output": json.dumps(
                                {
                                    "url": pr_url,
                                    "state": "closed",
                                    "merged": True,
                                }
                            ),
                        },
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Merged."}],
                            "phase": "final_answer",
                        },
                    ),
                    _rollout_line(
                        "event_msg",
                        {"type": "user_message", "message": "Follow-up work"},
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Updated."}],
                            "phase": "final_answer",
                        },
                    ),
                ]
            ),
            encoding="utf-8",
        )
        implementation = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
            codex_path=str(rollout_path),
            derived_stage="done_merged",
            derived_stage_source_mtime_ns=1,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Automated proposal",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=implementation,
            outcome_metadata={"accepted_by": "autonomous_goal_autonomy"},
        )
        SessionPullRequest.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            state={
                "pr_handoff": {
                    "url": pr_url,
                    "state": "closed",
                    "merged": True,
                }
            },
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        mock_spawn.assert_not_called()

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_allows_watch_owned_pr_before_final_session_activity(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        now = datetime.now(UTC)
        implementation = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
            codex_updated_at=now,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Automated proposal",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=implementation,
            outcome_metadata={"accepted_by": "autonomous_goal_autonomy"},
        )
        registered_pr = SessionPullRequest.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            state={
                "pr_handoff": {
                    "url": "https://github.com/cberner/hitch/pull/94",
                    "state": "closed",
                    "merged": True,
                },
                SessionPullRequest.WATCH_OWNER_INSTANCE_STATE_KEY: 7,
            },
        )
        SessionPullRequest.objects.filter(pk=registered_pr.pk).update(
            updated_at=now - timedelta(minutes=1)
        )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        workflow = SystemWorkflow.objects.get(kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND)
        self.assertEqual(
            workflow.main_thread_id,
            autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
        )

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_blocks_registered_pr_older_than_session_activity(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        now = datetime.now(UTC)
        implementation = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
            codex_updated_at=now,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Automated proposal",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=implementation,
            outcome_metadata={"accepted_by": "autonomous_goal_autonomy"},
        )
        registered_pr = SessionPullRequest.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            state={
                "pr_handoff": {
                    "url": "https://github.com/cberner/hitch/pull/94",
                    "state": "closed",
                    "merged": True,
                }
            },
        )
        SessionPullRequest.objects.filter(pk=registered_pr.pk).update(
            updated_at=now - timedelta(minutes=1)
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(
            project=project
        )

        self.assertEqual(started, 0)
        mock_spawn.assert_not_called()

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_allows_uncached_done_accepted_session_from_rollout(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        rollout_path = Path(temp_dir.name) / "rollout.jsonl"
        pr_url = "https://github.com/cberner/hitch/pull/94"
        rollout_path.write_text(
            "\n".join(
                [
                    _rollout_line(
                        "event_msg",
                        {
                            "type": "user_message",
                            "message": PR_SLASH_PROMPT,
                        },
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "function_call",
                            "name": "github_fetch_pr",
                            "arguments": json.dumps(
                                {
                                    "repo_full_name": "cberner/hitch",
                                    "pr_number": 94,
                                }
                            ),
                            "call_id": "call-fetch",
                        },
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "function_call_output",
                            "call_id": "call-fetch",
                            "output": json.dumps(
                                {
                                    "url": pr_url,
                                    "state": "closed",
                                    "merged": True,
                                }
                            ),
                        },
                    ),
                    _rollout_line(
                        "response_item",
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Merged."}],
                            "phase": "final_answer",
                        },
                    ),
                ]
            ),
            encoding="utf-8",
        )
        implementation = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
            codex_path=str(rollout_path),
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Automated proposal",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=implementation,
            outcome_metadata={"accepted_by": "autonomous_goal_autonomy"},
        )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        workflow = SystemWorkflow.objects.get()
        self.assertEqual(
            workflow.main_thread_id,
            autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
        )

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_blocks_in_flight_pr_task_for_automation(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        blocker_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve docs",
            goal="Find useful documentation increments.",
            auto_proposal_enabled=True,
        )
        implementation = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=blocker_goal,
            title="Automated proposal",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=implementation,
            outcome_metadata={"accepted_by": "autonomous_goal_autonomy"},
        )
        _instance(
            thread_id="implementation-thread",
            cwd="/repo",
            status=CodexInstance.STATUS_RUNNING,
            agent_kind="pr_publish",
        )
        for index in range(25):
            session = SessionMetadata.objects.create(
                thread_id=f"completed-implementation-{index}",
                cwd="/repo",
                project=project,
                derived_stage="done_merged",
            )
            ProposedSession.objects.create(
                project=project,
                autonomous_goal=autonomous_goal,
                title=f"Completed automated proposal {index}",
                outcome_status=ProposedSession.OUTCOME_ACCEPTED,
                accepted_session=session,
                outcome_metadata={"accepted_by": "autonomous_goal_autonomy"},
            )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        workflow = SystemWorkflow.objects.get(kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND)
        self.assertEqual(
            workflow.main_thread_id,
            autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
        )

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_blocks_unresolved_failure_notice(
        self, mock_spawn: MagicMock, _mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Autonomous goal failed: Improve tests",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            outcome_metadata={"automation_status": "failed"},
        )
        for index in range(25):
            ProposedSession.objects.create(
                project=project,
                autonomous_goal=autonomous_goal,
                title=f"No proposal from Improve tests {index}",
                inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        mock_spawn.assert_not_called()

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value=None,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_proposal_waits_when_base_branch_is_unavailable(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        mock_spawn.assert_not_called()
        mock_default_sha.assert_called_once_with("/repo")

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch(
        "hitch.main.management.commands.run_auto_proposals.reconciliation.reconcile_dead",
        return_value=0,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_run_auto_proposals_command_starts_eligible_goals(
        self,
        mock_spawn: MagicMock,
        mock_reconcile_dead: MagicMock,
        _mock_default_sha: MagicMock,
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        eligible_goal = AutonomousGoal.objects.create(
            project=project,
            title="Keep docs current",
            goal="Find small documentation improvements.",
            auto_proposal_enabled=True,
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Disabled goal",
            goal="This goal should require manual runs.",
            auto_proposal_enabled=False,
        )
        AutonomousGoal.objects.create(
            project=other_project,
            title="Other project goal",
            goal="This belongs to a different project.",
            auto_proposal_enabled=True,
        )
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        output = call_command("run_auto_proposals", project_id=project.pk)

        self.assertEqual(output, "Started 1 auto-proposal workflow(s).")
        workflow = SystemWorkflow.objects.get()
        self.assertEqual(
            workflow.main_thread_id,
            autonomous_goals._autonomous_goal_main_thread_id(eligible_goal.pk),
        )
        mock_reconcile_dead.assert_called_once_with()
        mock_spawn.assert_called_once()

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_no_proposal_records_and_suppresses_until_branch_changes(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        mock_default_sha.return_value = "a" * 40
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        instance = CodexInstance.objects.get(thread_id="candidate-thread")
        instance.events_path = _events_file(
            self,
            {
                "proposal": None,
                "message": "No concrete test increment was worth proposing.",
                "next_steps_summary": "Try a different area next.",
                "memory_relevant_files": [],
            },
        )
        instance.save(update_fields=["events_path"])

        system_agents.on_codex_instance_finished(instance)

        autonomous_goal.refresh_from_db()
        self.assertEqual(autonomous_goal.auto_proposal_last_no_proposal_sha, "a" * 40)

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        self.assertEqual(mock_spawn.call_count, 1)

        mock_default_sha.return_value = "b" * 40
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )

        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 1)
        self.assertEqual(mock_spawn.call_count, 2)

    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_history_summary_or_candidate")
    def test_manual_start_if_queue_idle_starts_candidate(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Waiting goal",
            goal="Should get queue admission.",
        )
        workflow = autonomous_goals.start_autonomous_goal_workflow_if_queue_idle(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )

        self.assertIsNotNone(workflow)
        assert workflow is not None
        self.assertEqual(
            workflow.main_thread_id,
            f"autonomous-goal:{autonomous_goal.pk}",
        )
        self.assertFalse(workflow.state["auto_proposal"])
        self.assertTrue(workflow.state["use_worktrees"])
        mock_spawn.assert_called_once()

    def test_compacted_memory_context_includes_older_summary_section(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        memories = [
            AutonomousGoalMemory.objects.create(
                autonomous_goal=autonomous_goal,
                title=f"Processed file {idx}",
                summary=f"Run {idx} left a concise continuation.",
                relevant_files=[f"hitch/main/test/test_{idx}.py"],
            )
            for idx in range(agent_io._AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT + 1)
        ]

        compacted = agent_io._compact_autonomous_goal_memories(memories)

        self.assertIn("Older compacted summaries:", compacted)
        self.assertIn(
            f"Processed file {agent_io._AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT}",
            compacted,
        )

    @patch.object(agent_io, "_AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS", 190)
    def test_fit_memory_context_uses_line_when_full_section_does_not_fit(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        memory = AutonomousGoalMemory.objects.create(
            autonomous_goal=autonomous_goal,
            title="Short",
            summary="Target parser assertions next.",
        )

        compacted = agent_io._fit_autonomous_goal_memory_context([memory], "")

        self.assertIn("Target parser assertions next.", compacted)
        self.assertNotIn("Memory ID:", compacted)
        self.assertLessEqual(len(compacted), agent_io._AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS)

    @patch.object(agent_io, "_AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS", 450)
    @patch.object(agent_io, "_AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT", 1)
    def test_fit_memory_context_includes_older_compacted_summaries(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        memories = [
            AutonomousGoalMemory.objects.create(
                autonomous_goal=autonomous_goal,
                title=f"Processed file {idx}",
                summary=f"Run {idx} left a concise continuation.",
            )
            for idx in range(2)
        ]

        compacted = agent_io._fit_autonomous_goal_memory_context(memories, "")

        self.assertIn("Older compacted summaries:", compacted)
        self.assertIn("Processed file 1", compacted)
        self.assertLessEqual(len(compacted), agent_io._AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS)

    @patch.object(agent_io, "_AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS", 260)
    @patch.object(agent_io, "_AUTONOMOUS_GOAL_MEMORY_COMPACT_RECENT_COUNT", 1)
    def test_fit_memory_context_stops_before_older_summary_that_would_overflow(
        self,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        memories = [
            AutonomousGoalMemory.objects.create(
                autonomous_goal=autonomous_goal,
                title=f"File {idx}",
                summary=f"Run {idx} next.",
            )
            for idx in range(2)
        ]

        compacted = agent_io._fit_autonomous_goal_memory_context(memories, "")

        self.assertIn("File 0", compacted)
        self.assertNotIn("Older compacted summaries:", compacted)
        self.assertNotIn("File 1", compacted)
        self.assertLessEqual(len(compacted), agent_io._AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS)

    @patch.object(agent_io, "_AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS", 240)
    def test_compacted_memory_context_enforces_budget_with_long_files(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        for idx in range(5):
            AutonomousGoalMemory.objects.create(
                autonomous_goal=autonomous_goal,
                title=f"Processed file {idx}",
                summary="Chose one file and left a long next-step summary. " * 12,
                relevant_files=["hitch/main/test/" + ("very_long_path_segment_" * 8) + f"{idx}.py"],
            )

        memory_context = agent_io._autonomous_goal_memory_context(autonomous_goal)

        self.assertTrue(memory_context.compacted)
        self.assertIn("Compacted from 5 prior candidate summaries", memory_context.text)
        self.assertLessEqual(len(memory_context.text), agent_io._AUTONOMOUS_GOAL_MEMORY_CONTEXT_CHARS)

    @patch.object(agent_io, "_AUTONOMOUS_GOAL_MEMORY_MAX_ROWS", 2)
    def test_memory_context_caps_recent_rows_before_compaction(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Process one test file",
            goal="Pick one test file and improve it.",
        )
        for idx in range(4):
            AutonomousGoalMemory.objects.create(
                autonomous_goal=autonomous_goal,
                title=f"Processed file {idx}",
                summary=f"Summary for file {idx}.",
                relevant_files=[f"hitch/main/test/test_{idx}.py"],
            )

        memory_context = agent_io._autonomous_goal_memory_context(autonomous_goal)

        self.assertTrue(memory_context.compacted)
        self.assertEqual(memory_context.count, 4)
        self.assertIn("Compacted from 4 prior candidate summaries", memory_context.text)
        self.assertIn("2 older memory rows are outside this prompt cap", memory_context.text)
        self.assertIn("Processed file 3", memory_context.text)
        self.assertNotIn("Processed file 0", memory_context.text)

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_no_proposal_records_workflow_start_sha_snapshot(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        mock_default_sha.return_value = "a" * 40
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        workflow = autonomous_goals.start_autonomous_goal_workflow(
            autonomous_goal=autonomous_goal,
            auto_proposal=True,
        )
        instance = CodexInstance.objects.get(thread_id="candidate-thread")
        instance.events_path = _events_file(
            self,
            {
                "proposal": None,
                "message": "No concrete test increment was worth proposing.",
                "next_steps_summary": "Try a different area next.",
                "memory_relevant_files": [],
            },
        )
        instance.save(update_fields=["events_path"])
        mock_default_sha.return_value = "b" * 40

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        autonomous_goal.refresh_from_db()
        self.assertEqual(workflow.state["default_branch_sha"], "a" * 40)
        self.assertEqual(autonomous_goal.auto_proposal_last_no_proposal_sha, "a" * 40)
        mock_default_sha.assert_called_once_with("/repo")

    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_candidate_spawn_interrupts_worker_when_goal_deleted_mid_spawn(
        self, mock_spawn: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )

        def spawn_candidate(**_kwargs: Any) -> CodexInstance:
            AutonomousGoal.objects.filter(pk=autonomous_goal.pk).update(deleted_at=datetime.now(UTC))
            instance = _instance(
                thread_id="candidate-thread",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                status=CodexInstance.STATUS_RUNNING,
                agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            )
            mock_interrupt.return_value = instance
            return instance

        mock_spawn.side_effect = spawn_candidate

        workflow = autonomous_goals.start_autonomous_goal_workflow(autonomous_goal=autonomous_goal)

        workflow.refresh_from_db()
        run = SystemAgentRun.objects.get()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.state["error"], "autonomous goal no longer exists")
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, "autonomous goal no longer exists")
        mock_interrupt.assert_called_once_with(run.instance_id, expected_thread_id=run.thread_id)

    def test_dead_autonomous_goal_candidate_worker_blocks_after_retry_budget(
        self,
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Production issues",
            goal="Inspect production logs and the main database.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread-1",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                system_agents._WORKFLOW_TURN_DEATH_RETRY_STATE_KEY: {
                    autonomous_goals._AUTONOMOUS_GOAL_CANDIDATE_RETRY_KIND: 1
                },
            },
        )
        instance = _instance(
            thread_id="candidate-thread-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_FAILED,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            error=("worker process exited before reporting completion; last event: command failed"),
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertIn("worker process exited", workflow.state["error"])
        notice = ProposedSession.objects.get()
        self.assertEqual(notice.inbox_kind, ProposedSession.INBOX_KIND_NOTICE)
        self.assertEqual(notice.candidate_session, candidate_metadata)

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_dead_candidate_worker_retries_within_proposal_budget_after_death_retry(
        self, mock_spawn: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Production issues",
            goal="Inspect production logs and the main database.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread-1",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                system_agents._WORKFLOW_TURN_DEATH_RETRY_STATE_KEY: {
                    autonomous_goals._AUTONOMOUS_GOAL_CANDIDATE_RETRY_KIND: 1
                },
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: 400,
                autonomous_goals._AUTONOMOUS_GOAL_NO_PROPOSAL_STREAK_STATE_KEY: 2,
                autonomous_goals._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY: {
                    "candidate-thread-1": 400,
                },
            },
        )
        instance = _instance(
            thread_id="candidate-thread-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_FAILED,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            error=("worker process exited before reporting completion; last event: command failed"),
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        retry_instance = _instance(
            thread_id="candidate-thread-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.return_value = retry_instance

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY],
            400,
        )
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY],
            1,
        )
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY],
            1,
        )
        failure = workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY]
        self.assertEqual(failure["reason"], "candidate_failed")
        self.assertIn("worker process exited", failure["error"])
        self.assertEqual(
            workflow.state[autonomous_goals._AUTONOMOUS_GOAL_NO_PROPOSAL_STREAK_STATE_KEY],
            2,
        )
        retry_run = SystemAgentRun.objects.get(instance=retry_instance)
        self.assertEqual(retry_run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(retry_run.input["proposal_budget_tokens_used"], 400)
        self.assertFalse(ProposedSession.objects.exists())
        mock_spawn.assert_called_once()

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_dead_autonomous_goal_judge_worker_is_retried_once(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        candidate = {
            "title": "Add parser coverage",
            "implementation_direction": "Add focused rollout parser tests.",
            "relevant_files": ["hitch/main/rollout.py"],
        }
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                "candidate": candidate,
            },
        )
        instance = _instance(
            thread_id="judge-thread-1",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_FAILED,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            error=(
                "worker process exited before reporting completion; "
                'last event: command failed: `/bin/bash -lc "which sqlite3"`'
            ),
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id="judge-thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        mock_spawn.return_value = _instance(
            thread_id="judge-thread-2",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("which sqlite3", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING)
        self.assertEqual(workflow.state["candidate"], candidate)
        self.assertEqual(
            workflow.state[system_agents._WORKFLOW_TURN_DEATH_RETRY_STATE_KEY],
            {autonomous_goals._AUTONOMOUS_GOAL_JUDGE_RETRY_KIND: 1},
        )
        replacement_run = SystemAgentRun.objects.get(thread_id="judge-thread-2")
        self.assertEqual(
            replacement_run.agent_kind,
            system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        self.assertEqual(replacement_run.status, SystemAgentRun.STATUS_RUNNING)
        judge_metadata = SessionMetadata.objects.get(thread_id="judge-thread-2")
        self.assertEqual(workflow.state["judge_session_id"], judge_metadata.pk)
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["agent_kind"], system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND)
        self.assertIn("Add parser coverage", kwargs["prompt"])

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_yolo_candidate_completion_starts_judge_thread_with_yolo_guidance(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_YOLO,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": autonomous_goal.pk},
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow.state = {
            **workflow.state,
            "candidate_session_id": candidate_metadata.pk,
        }
        workflow.save(update_fields=["state", "updated_at"])
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "title": "Consolidate command tests",
                    "summary": "Merge duplicated command-routing tests.",
                    "impact": "Less duplicated test maintenance.",
                    "implementation_direction": "Refactor adjacent tests.",
                    "relevant_files": ["hitch/main/test/test_views.py"],
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )
        mock_spawn.return_value = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )

        system_agents.on_codex_instance_finished(instance)

        prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn("bold, high-leverage progress", prompt)
        self.assertIn("substantial and high-upside", prompt)
        self.assertNotIn("incremental", prompt.lower())

    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_judge_spawn_interrupts_worker_when_goal_deleted_mid_spawn(
        self, mock_spawn: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": autonomous_goal.pk},
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow.state = {
            **workflow.state,
            "candidate_session_id": candidate_metadata.pk,
        }
        workflow.save(update_fields=["state", "updated_at"])
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "proposal": {
                        "title": "Add parser coverage",
                        "summary": "Cover parser edge cases.",
                        "impact": "Fewer regressions.",
                        "implementation_direction": "Finish the candidate changes.",
                        "relevant_files": ["hitch/main/rollout.py"],
                    },
                    "message": "",
                    "next_steps_summary": "Selected parser coverage.",
                    "memory_relevant_files": [],
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        candidate_run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )

        def spawn_judge(**_kwargs: Any) -> CodexInstance:
            AutonomousGoal.objects.filter(pk=autonomous_goal.pk).update(deleted_at=datetime.now(UTC))
            judge_instance = _instance(
                thread_id="judge-thread",
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                status=CodexInstance.STATUS_RUNNING,
                agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            )
            mock_interrupt.return_value = judge_instance
            return judge_instance

        mock_spawn.side_effect = spawn_judge

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        candidate_run.refresh_from_db()
        judge_run = SystemAgentRun.objects.get(agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND)
        self.assertEqual(candidate_run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(judge_run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(judge_run.error, "autonomous goal no longer exists")
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        mock_interrupt.assert_called_once_with(judge_run.instance_id, expected_thread_id=judge_run.thread_id)

    def test_judge_creates_proposal_when_confidence_meets_threshold(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            auto_proposal_last_no_proposal_sha="a" * 40,
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        judge_metadata = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                "judge_session_id": judge_metadata.pk,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": (
                        "Add focused rollout parser regression tests before touching parser behavior."
                    ),
                    "relevant_files": ["hitch/main/rollout.py"],
                },
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: 500,
                system_agents._WORKFLOW_TURN_DEATH_RETRY_STATE_KEY: {
                    autonomous_goals._AUTONOMOUS_GOAL_JUDGE_RETRY_KIND: 1
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
                thread_id="judge-thread",
                tokens_used=200,
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        self.assertNotIn(
            system_agents._WORKFLOW_TURN_DEATH_RETRY_STATE_KEY,
            workflow.state,
        )
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.title, "Add parser coverage")
        self.assertEqual(proposal.confidence, AutonomousGoal.CONFIDENCE_HIGH)
        self.assertEqual(proposal.outcome_metadata["proposal_budget"], 1000)
        self.assertEqual(proposal.outcome_metadata["proposal_budget_tokens_used"], 700)
        self.assertEqual(proposal.outcome_metadata["proposal_budget_failed_attempts"], 0)
        self.assertIn("Implementation guidance:", proposal.prompt)
        self.assertIn(
            "Add focused rollout parser regression tests before touching parser behavior.",
            proposal.prompt,
        )
        self.assertEqual(proposal.candidate_session, candidate_metadata)
        self.assertEqual(proposal.judge_session, judge_metadata)
        autonomous_goal.refresh_from_db()
        self.assertEqual(autonomous_goal.auto_proposal_last_no_proposal_sha, "")

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value=None,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_auto_draft_patch_does_not_revalidate_until_user_continuation(
        self, mock_spawn: MagicMock, mock_default_sha: MagicMock
    ) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "auto_proposal": True,
                "default_branch_sha": "a" * 40,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": "Add focused tests.",
                    "relevant_files": [],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED)
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertEqual(
            proposal.outcome_metadata["automation_status"],
            "proposed",
        )
        mock_default_sha.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_draft_patch_auto_qa_setting_is_recorded_for_pending_proposal(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": "Add focused tests.",
                    "relevant_files": [],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )
        system_agents.on_codex_instance_finished(instance)

        proposal = ProposedSession.objects.get()
        mock_spawn.assert_not_called()
        self.assertFalse(proposal.outcome_metadata["auto_pr_enabled"])
        self.assertTrue(proposal.outcome_metadata["auto_qa_enabled"])
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_new_session")
    def test_draft_pr_autonomy_records_auto_pr_for_pending_proposal(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PR,
            web_search_mode=AutonomousGoal.WEB_SEARCH_DISABLED,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "web_search_mode": AutonomousGoal.WEB_SEARCH_DISABLED,
                "candidate": {
                    "title": "Add parser coverage",
                    "implementation_direction": "Add focused tests.",
                    "relevant_files": [],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "This adds focused parser coverage.",
                    "rationale": "The files are well-scoped.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )
        system_agents.on_codex_instance_finished(instance)

        proposal = ProposedSession.objects.get()
        mock_spawn.assert_not_called()
        self.assertTrue(proposal.outcome_metadata["auto_pr_enabled"])
        self.assertFalse(proposal.outcome_metadata["auto_qa_enabled"])
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)

    def test_completed_autonomous_goal_run_blocks_when_goal_was_deleted(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            deleted_at=datetime.now(UTC),
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": autonomous_goal.pk},
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "proposal": {
                        "title": "Add parser coverage",
                        "summary": "Cover parser edge cases.",
                        "impact": "Fewer regressions.",
                        "implementation_direction": "Finish the candidate changes.",
                        "relevant_files": ["hitch/main/rollout.py"],
                    },
                    "message": "",
                    "next_steps_summary": "Selected parser coverage.",
                    "memory_relevant_files": [],
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, "autonomous goal no longer exists")
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertFalse(ProposedSession.objects.exists())

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    def test_deleted_autonomous_goal_terminal_callback_cleans_workflow_worktree(self, mock_cleanup: MagicMock) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id="autonomous-goal:1",
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step=system_agents.STEP_BLOCKED,
            state={
                "autonomous_goal_id": 1,
                "session_cwd": "/repo-worktree",
                "error": system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_FAILED,
            error=system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_FAILED,
            error=system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
        )

        routed = system_agents.on_codex_instance_finished(instance)

        self.assertTrue(routed)
        mock_cleanup.assert_called_once_with("/repo-worktree")

    def test_judge_skips_proposal_below_threshold(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_VERY_HIGH,
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        judge_metadata = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo-worktree",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "auto_proposal": True,
                "default_branch_sha": "a" * 40,
                "candidate_session_id": candidate_metadata.pk,
                "judge_session_id": judge_metadata.pk,
                "candidate": {"title": "Maybe add tests", "relevant_files": []},
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "Useful but not certain.",
                    "rationale": "There is some ambiguity.",
                },
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED)
        notice = ProposedSession.objects.get()
        self.assertEqual(notice.inbox_kind, ProposedSession.INBOX_KIND_NOTICE)
        self.assertEqual(notice.title, "Skipped proposal: Maybe add tests")
        self.assertEqual(notice.candidate_session, candidate_metadata)
        self.assertEqual(notice.judge_session, judge_metadata)
        self.assertEqual(
            notice.summary,
            'Found candidate "Maybe add tests", but judge confidence was high '
            "and this goal requires very high. Judge summary: Useful but not "
            "certain.",
        )
        self.assertEqual(notice.outcome_metadata["automation_status"], "skipped")
        self.assertEqual(
            notice.outcome_metadata["skip_reason"],
            "judge_confidence_below_threshold",
        )
        self.assertEqual(notice.outcome_metadata["judge_confidence"], "high")
        self.assertEqual(
            notice.outcome_metadata["confidence_threshold"],
            AutonomousGoal.CONFIDENCE_VERY_HIGH,
        )
        self.assertEqual(notice.outcome_metadata["candidate_title"], "Maybe add tests")
        autonomous_goal.refresh_from_db()
        self.assertEqual(autonomous_goal.auto_proposal_last_no_proposal_sha, "a" * 40)

    def test_instance_tokens_used_reads_rollout_totals(self) -> None:
        # Codex only emits thread/goal/updated when the model sets a thread
        # goal, which hidden candidate/judge sessions normally never do -- the
        # rollout file's TokenCount totals are the reliable source and goal
        # events are only a fallback (the larger of the two wins when both
        # exist, since each is a cumulative thread total).
        project = _make_project()
        SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
            codex_path=_rollout_token_file(self, 700),
        )
        no_goal_events = _instance(
            thread_id="candidate-thread",
            events_path=_raw_events_file(self, []),
        )
        self.assertEqual(
            autonomous_goals._autonomous_goal_instance_tokens_used(no_goal_events),
            700,
        )
        with_goal_events = _instance(
            thread_id="candidate-thread",
            events_path=_events_file(self, {}, thread_id="candidate-thread", tokens_used=900),
        )
        self.assertEqual(
            autonomous_goals._autonomous_goal_instance_tokens_used(with_goal_events),
            900,
        )
        goal_events_only = _instance(
            thread_id="other-thread",
            events_path=_events_file(self, {}, thread_id="other-thread", tokens_used=120),
        )
        self.assertEqual(
            autonomous_goals._autonomous_goal_instance_tokens_used(goal_events_only),
            120,
        )
        no_sources = _instance(
            thread_id="other-thread",
            events_path=_raw_events_file(self, []),
        )
        self.assertIsNone(autonomous_goals._autonomous_goal_instance_tokens_used(no_sources))

    def test_proposal_budget_helper_edges(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id="autonomous-goal:1",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
        )

        autonomous_goals._record_autonomous_goal_proposal_budget_tokens(workflow, instance, None)

        self.assertNotIn(
            autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY,
            workflow.state,
        )
        self.assertTrue(
            autonomous_goals._autonomous_goal_proposal_budget_allows_retry(workflow, tokens_used=None, token_delta=0)
        )
        workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY] = (
            autonomous_goals._AUTONOMOUS_GOAL_NO_PROGRESS_RETRY_LIMIT
        )
        self.assertFalse(
            autonomous_goals._autonomous_goal_proposal_budget_allows_retry(workflow, tokens_used=None, token_delta=0)
        )
        self.assertTrue(
            autonomous_goals._autonomous_goal_proposal_budget_allows_retry(workflow, tokens_used=101, token_delta=101)
        )
        self.assertIsNone(
            autonomous_goals._retry_budgeted_failed_autonomous_goal_candidate(
                run,
                workflow,
                error="candidate failed",
                raw_output="raw",
                tokens_used=100,
                token_delta=100,
            )
        )
        workflow.step = system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING
        self.assertIsNone(
            autonomous_goals._retry_budgeted_unaccepted_autonomous_goal_candidate(
                workflow,
                reason="candidate_no_proposal",
                tokens_used=100,
                token_delta=100,
            )
        )
        self.assertEqual(
            autonomous_goal_prompts._format_autonomous_goal_last_failure_context(workflow),
            "(none)",
        )

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_exhausted_candidate_budget_persists_tokens_before_blocking(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 300,
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_raw_events_file(
                self,
                [
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "candidate-thread",
                            "goal": {
                                "objective": "Autonomous goal",
                                "tokensUsed": 350,
                            },
                        },
                    },
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "a1",
                                "type": "agentMessage",
                                "text": "not json",
                            }
                        },
                    },
                ],
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY],
            350,
        )
        self.assertEqual(
            workflow.state[autonomous_goals._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY],
            {"candidate-thread": 350},
        )
        notice = ProposedSession.objects.get()
        self.assertEqual(notice.outcome_metadata["proposal_budget_tokens_used"], 350)
        mock_spawn.assert_not_called()

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_candidate_budget_retries_without_new_token_progress(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: 350,
                autonomous_goals._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY: {
                    "candidate-thread": 350,
                },
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_raw_events_file(
                self,
                [
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "candidate-thread",
                            "goal": {
                                "objective": "Autonomous goal",
                                "tokensUsed": 350,
                            },
                        },
                    },
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "id": "a1",
                                "type": "agentMessage",
                                "text": "not json",
                            }
                        },
                    },
                ],
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        retry_instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.return_value = retry_instance

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY],
            350,
        )
        self.assertEqual(
            workflow.state[autonomous_goals._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY],
            {"candidate-thread": 350},
        )
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY],
            1,
        )
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_NO_PROGRESS_RETRIES_STATE_KEY],
            1,
        )
        retry_run = SystemAgentRun.objects.get(instance=retry_instance)
        self.assertEqual(retry_run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(retry_run.input["proposal_budget_tokens_used"], 350)
        self.assertEqual(retry_run.input["retry_attempt"], 1)
        self.assertFalse(ProposedSession.objects.exists())
        mock_spawn.assert_called_once()

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_no_proposal_retries_candidate_within_proposal_budget(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "proposal": None,
                    "message": "No safe target found this time.",
                    "next_steps_summary": "Looked for parser work but found none.",
                    "memory_relevant_files": [],
                },
                thread_id="candidate-thread",
                tokens_used=250,
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        retry_instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.return_value = retry_instance

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        self.assertNotIn("candidate", workflow.state)
        failure = workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY]
        self.assertEqual(failure["reason"], "candidate_no_proposal")
        self.assertEqual(failure["tokens_used"], 250)
        self.assertEqual(failure["message"], "No safe target found this time.")
        retry_run = SystemAgentRun.objects.get(instance=retry_instance)
        self.assertEqual(retry_run.input["proposal_budget_tokens_used"], 250)
        self.assertIn("No safe target found", mock_spawn.call_args.kwargs["prompt"])
        self.assertFalse(ProposedSession.objects.exists())

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_consecutive_no_proposals_stop_before_large_budget_is_exhausted(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Pursue hard theorem",
            goal="Only propose mathematically honest progress.",
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 100_000_000,
                autonomous_goals._AUTONOMOUS_GOAL_NO_PROPOSAL_STREAK_STATE_KEY: (
                    autonomous_goals._AUTONOMOUS_GOAL_NO_PROPOSAL_RETRY_LIMIT
                ),
            },
        )
        instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "proposal": None,
                    "message": "No honest implementation clears the threshold.",
                    "next_steps_summary": "A theorem is still missing.",
                    "memory_relevant_files": [],
                },
                thread_id="candidate-thread",
                tokens_used=250,
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id="candidate-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_COMPLETED)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED)
        self.assertEqual(workflow.state["proposal_budget_tokens_used"], 250)
        notice = ProposedSession.objects.get()
        self.assertIn("No honest implementation", notice.summary)
        self.assertEqual(notice.outcome_metadata["automation_status"], "skipped")
        self.assertEqual(
            notice.outcome_metadata["skip_reason"],
            "candidate_no_proposal_stall_limit",
        )
        self.assertEqual(notice.outcome_metadata["no_proposal_retries"], 3)
        self.assertEqual(notice.outcome_metadata["no_proposal_retry_limit"], 3)
        mock_spawn.assert_not_called()

    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_below_threshold_retries_candidate_within_proposal_budget(self, mock_spawn: MagicMock) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_VERY_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_LIVE,
        )
        candidate_metadata = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        judge_metadata = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo-worktree",
            project=project,
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate_session_id": candidate_metadata.pk,
                "judge_session_id": judge_metadata.pk,
                "session_cwd": "/repo-worktree",
                "web_search_mode": AutonomousGoal.WEB_SEARCH_LIVE,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
                "candidate": {
                    "title": "Maybe add tests",
                    "summary": "Add a broad test sweep.",
                    "implementation_direction": "Try a broader pass.",
                    "relevant_files": ["hitch/main/rollout.py"],
                },
            },
        )
        instance = _instance(
            thread_id="judge-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            events_path=_events_file(
                self,
                {
                    "confidence": "high",
                    "summary": "Useful but not certain.",
                    "rationale": "The candidate is too narrow for the threshold.",
                },
                thread_id="judge-thread",
                tokens_used=400,
            ),
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
        )
        judge_run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_JUDGE_AGENT_KIND,
            thread_id="judge-thread",
            instance=instance,
        )
        retry_instance = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            status=CodexInstance.STATUS_RUNNING,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        mock_spawn.return_value = retry_instance

        system_agents.on_codex_instance_finished(instance)

        workflow.refresh_from_db()
        judge_run.refresh_from_db()
        self.assertEqual(judge_run.status, SystemAgentRun.STATUS_COMPLETED)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        self.assertEqual(workflow.step, system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING)
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY],
            400,
        )
        self.assertEqual(
            workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY],
            1,
        )
        self.assertNotIn("candidate", workflow.state)
        self.assertNotIn("judge_session_id", workflow.state)
        failure = workflow.state[autonomous_goal_prompts._AUTONOMOUS_GOAL_LAST_FAILURE_STATE_KEY]
        self.assertEqual(failure["reason"], "judge_confidence_below_threshold")
        self.assertEqual(failure["tokens_used"], 400)
        self.assertIn("too narrow", failure["judgment"]["rationale"])
        self.assertEqual(failure["candidate"]["title"], "Maybe add tests")
        retry_run = SystemAgentRun.objects.get(instance=retry_instance)
        self.assertEqual(retry_run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(retry_run.input["proposal_budget"], 1000)
        self.assertEqual(retry_run.input["proposal_budget_tokens_used"], 400)
        self.assertEqual(retry_run.input["retry_attempt"], 1)
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "candidate-thread")
        self.assertEqual(kwargs["cwd"], "/repo-worktree")
        self.assertEqual(
            kwargs["sandbox_policy"],
            system_agents.AUTONOMOUS_GOAL_IMPLEMENTATION_SANDBOX_POLICY,
        )
        self.assertEqual(kwargs["web_search_mode"], AutonomousGoal.WEB_SEARCH_LIVE)
        self.assertEqual(kwargs["output_schema"]["properties"]["message"]["type"], "string")
        self.assertIn("Last failed attempt context", kwargs["prompt"])
        self.assertIn("judge_confidence_below_threshold", kwargs["prompt"])
        self.assertIn("too narrow", kwargs["prompt"])
        self.assertIn("Proposal budget tokens used so far: 400", kwargs["prompt"])
        self.assertFalse(ProposedSession.objects.exists())

    def test_candidate_retry_spawn_blocks_when_candidate_session_missing(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
            },
        )

        autonomous_goals._spawn_autonomous_goal_candidate_retry_or_block(workflow, autonomous_goal)

        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertIn("candidate session is unavailable", workflow.state["error"])
        notice = ProposedSession.objects.get()
        self.assertEqual(notice.inbox_kind, ProposedSession.INBOX_KIND_NOTICE)
        self.assertIn("candidate session is unavailable", notice.summary)
        self.assertEqual(notice.outcome_metadata["proposal_budget"], 1000)

    def test_candidate_retry_spawn_noops_for_inactive_workflow(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(autonomous_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": autonomous_goal.pk},
        )

        autonomous_goals._spawn_autonomous_goal_candidate_retry_or_block(workflow, autonomous_goal)

        self.assertFalse(SystemAgentRun.objects.exists())

    def test_publish_unset_stack_proposal_records_budget_metadata(self) -> None:
        project = _make_project()
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id="autonomous-goal:1",
            cwd="/repo",
            state={
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_STATE_KEY: 1000,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: 450,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_FAILED_ATTEMPTS_STATE_KEY: 2,
            },
        )
        existing = ProposedSession.objects.create(
            project=project,
            title="Existing",
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        budgeted = ProposedSession.objects.create(
            project=project,
            title="Budgeted",
            outcome_status=ProposedSession.OUTCOME_UNSET,
            outcome_metadata={"existing": True},
        )

        self.assertTrue(autonomous_goals._publish_current_stack_proposal(existing))
        self.assertTrue(autonomous_goals._publish_current_stack_proposal(budgeted, workflow=workflow))

        budgeted.refresh_from_db()
        self.assertTrue(budgeted.outcome_metadata["existing"])
        self.assertEqual(budgeted.outcome_metadata["proposal_budget"], 1000)
        self.assertEqual(budgeted.outcome_metadata["proposal_budget_tokens_used"], 450)
        self.assertEqual(budgeted.outcome_metadata["proposal_budget_failed_attempts"], 2)

    def test_current_stack_proposal_falls_back_to_source_workflow_for_legacy_state(
        self,
    ) -> None:
        project = _make_project()
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id="autonomous-goal:1",
            cwd="/repo",
            state={},
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Legacy stack proposal",
            source_workflow=workflow,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        workflow.state = {"proposal_id": proposal.pk}
        workflow.save(update_fields=["state"])

        self.assertEqual(autonomous_goals._autonomous_goal_current_stack_proposal(workflow), proposal)

    def test_below_threshold_notice_copy_handles_missing_candidate_title(self) -> None:
        project = _make_project()
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            confidence_threshold=AutonomousGoal.CONFIDENCE_VERY_HIGH,
        )
        judgment = {"confidence": "high", "summary": "", "rationale": ""}

        self.assertEqual(
            autonomous_goals._below_threshold_notice_title({}, autonomous_goal),
            "Skipped proposal from Improve tests",
        )
        self.assertEqual(
            autonomous_goals._below_threshold_notice_summary({}, judgment, autonomous_goal.confidence_threshold),
            "Found a candidate, but judge confidence was high and this goal requires very high.",
        )


class AutoReviewTaskTests(TestCase):
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_auto_qa_starts_one_ordinary_review_turn(
        self, mock_spawn: MagicMock
    ) -> None:
        mock_spawn.return_value = SimpleNamespace(pk=101)
        instance = _instance(
            thread_id="auto-qa-thread",
            cwd="/repo",
            auto_qa_enabled=True,
            model="gpt-test",
            reasoning_effort="high",
            sandbox_policy="workspaceWrite",
            approval_mode="prompt_user",
            web_search_mode="live",
            developer_instructions="Keep it small.",
            enable_memories=True,
            user_message_index=7,
        )

        self.assertFalse(system_agents.on_codex_instance_finished(instance))
        self.assertFalse(system_agents.on_codex_instance_finished(instance))

        instance.refresh_from_db()
        self.assertIsNotNone(instance.auto_qa_triggered_at)
        mock_spawn.assert_called_once_with(
            thread_id="auto-qa-thread",
            cwd="/repo",
            prompt=optional_review_prompt(prepare_pull_request=False),
            sandbox_policy="workspaceWrite",
            approval_mode="prompt_user",
            model="gpt-test",
            stored_model="gpt-test",
            reasoning_effort="high",
            stored_reasoning_effort="high",
            developer_instructions="Keep it small.",
            enable_memories=True,
            web_search_mode="live",
            user_message_index=8,
            agent_kind=agent_tasks.REVIEW_AGENT_KIND,
        )
        self.assertFalse(SystemWorkflow.objects.exists())

    @patch(
        "hitch.main.sessions.session_resume.thread_has_dynamic_tool",
        return_value=True,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_auto_pr_uses_review_publish_prompt_and_records_proposal_turn(
        self,
        mock_spawn: MagicMock,
        _mock_has_tool: MagicMock,
    ) -> None:
        mock_spawn.return_value = SimpleNamespace(pk=202)
        project = _make_project()
        metadata = SessionMetadata.objects.create(
            thread_id="auto-pr-thread",
            cwd="/repo",
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="  Improve   parser coverage  ",
            accepted_session=metadata,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={"auto_pr_enabled": True, "keep": "value"},
        )
        instance = _instance(
            thread_id="auto-pr-thread",
            cwd="/repo",
            auto_pr_enabled=True,
            approval_mode="deny_all",
            user_message_index=2,
        )

        system_agents.on_codex_instance_finished(instance)

        kwargs = mock_spawn.call_args.kwargs
        self.assertIn(PR_SLASH_PROMPT, kwargs["prompt"])
        self.assertIn(
            "Use this pull request title: Improve parser coverage",
            kwargs["prompt"],
        )
        self.assertEqual(kwargs["agent_kind"], agent_tasks.PR_PUBLISH_AGENT_KIND)
        self.assertEqual(kwargs["approval_mode"], "deny_all")
        self.assertEqual(kwargs["user_message_index"], 3)
        instance.refresh_from_db()
        self.assertIsNotNone(instance.auto_pr_triggered_at)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_metadata["auto_pr_status"], "started")
        self.assertEqual(proposal.outcome_metadata["auto_pr_instance_id"], 202)
        self.assertEqual(proposal.outcome_metadata["keep"], "value")
        self.assertFalse(SystemWorkflow.objects.exists())

    @patch(
        "hitch.main.sessions.session_resume.thread_has_dynamic_tool",
        return_value=False,
    )
    @patch("hitch.main.workflows.system_agents.codex_pool.spawn_turn")
    def test_auto_pr_without_watch_tool_remains_retryable(
        self,
        mock_spawn: MagicMock,
        _mock_has_tool: MagicMock,
    ) -> None:
        instance = _instance(auto_pr_enabled=True)

        system_agents.on_codex_instance_finished(instance)

        instance.refresh_from_db()
        self.assertIsNone(instance.auto_pr_triggered_at)
        mock_spawn.assert_not_called()

    @patch(
        "hitch.main.workflows.system_agents.codex_pool.spawn_turn",
        side_effect=RuntimeError("spawn failed"),
    )
    def test_spawn_failure_releases_auto_qa_trigger(
        self, _mock_spawn: MagicMock
    ) -> None:
        instance = _instance(auto_qa_enabled=True)

        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            system_agents.on_codex_instance_finished(instance)

        instance.refresh_from_db()
        self.assertIsNone(instance.auto_qa_triggered_at)


class AutoReviewIntentionallySkippedTests(TestCase):
    @patch(
        "hitch.main.sessions.session_resume.thread_has_dynamic_tool",
        return_value=True,
    )
    def test_user_approval_mode_is_not_skipped(
        self, _mock_has_tool: MagicMock
    ) -> None:
        instance = _instance(approval_mode="prompt_user", auto_pr_enabled=True)
        self.assertFalse(system_agents.auto_review_intentionally_skipped(instance))


class ClaimWorkflowTransitionTests(TestCase):
    def _workflow(self, **overrides: Any) -> SystemWorkflow:
        defaults: dict[str, Any] = {
            "kind": SystemWorkflow.KIND_AUTONOMOUS_GOAL_RUN,
            "main_thread_id": "main-thread",
            "cwd": "/repo",
            "status": SystemWorkflow.STATUS_RUNNING,
            "step": "current_step",
            "state": {"revision": 1},
        }
        defaults.update(overrides)
        return SystemWorkflow.objects.create(**defaults)

    def test_returns_none_without_applying_on_step_mismatch(self) -> None:
        workflow = self._workflow(step="other_step")
        apply = MagicMock()

        self.assertIsNone(
            engine.claim_workflow_transition(
                workflow,
                apply,
                expect_step="current_step",
            )
        )
        apply.assert_not_called()

    def test_returns_none_for_inactive_unless_opted_out(self) -> None:
        workflow = self._workflow(status=SystemWorkflow.STATUS_BLOCKED)
        apply = MagicMock(return_value=True)

        self.assertIsNone(engine.claim_workflow_transition(workflow, apply))
        apply.assert_not_called()
        self.assertTrue(engine.claim_workflow_transition(workflow, apply, require_active=False))
