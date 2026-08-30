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
from django.test import TestCase, TransactionTestCase
from openai_codex import CodexError
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse

from hitch.main.goals import autonomous_goal_prompts, autonomous_goal_proposal_stack
from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    Project,
    ProposedSession,
    SessionMetadata,
    SessionPullRequest,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import codex_events
from hitch.main.sessions.pr_prompts import PR_SLASH_PROMPT
from hitch.main.test.support import _make_project, _rollout_line
from hitch.main.workflows import (
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


def _reset_auto_proposal_quota_cache() -> None:
    with autonomous_goals._quota_cache_lock:
        autonomous_goals._quota_cache_status = "available"
        autonomous_goals._quota_cache_checked_at = None


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
        _reset_auto_proposal_quota_cache()
        self.addCleanup(_reset_auto_proposal_quota_cache)
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
        _reset_auto_proposal_quota_cache()
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
        self.created_thread_count = 0

        def create_thread(**_kwargs: Any) -> tuple[str, str]:
            self.created_thread_count += 1
            thread_id = f"ag-thread-{self.created_thread_count}"
            return thread_id, f"/rollouts/{thread_id}.jsonl"

        self.thread_patcher = patch(
            "hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path",
            side_effect=create_thread,
        )
        self.mock_create_thread = self.thread_patcher.start()
        self.addCleanup(self.thread_patcher.stop)

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_concurrent_auto_proposal_starts_share_global_queue_lock(
        self,
        mock_spawn: MagicMock,
        mock_create_thread: MagicMock,
        mock_default_sha: MagicMock,
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

        def create_thread(**_kwargs: object) -> tuple[str, str]:
            with spawn_lock:
                thread_id = f"candidate-thread-{len(spawned_threads) + 1}"
                spawned_threads.append(thread_id)
            return thread_id, f"/rollouts/{thread_id}.jsonl"

        def spawn_instance(**kwargs: object) -> CodexInstance:
            thread_id = str(kwargs["thread_id"])
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
        mock_create_thread.side_effect = create_thread
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
        self.assertEqual(mock_create_thread.call_count, 1)
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
        _reset_auto_proposal_quota_cache()
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
        self.thread_patcher = patch(
            "hitch.main.workflows.autonomous_goals.codex_pool.create_session_thread_with_path",
            return_value=("candidate-thread", "/rollouts/candidate-thread.jsonl"),
        )
        self.thread_patcher.start()
        self.addCleanup(self.thread_patcher.stop)




    def test_expected_agent_kind_routes_legacy_history_step(self) -> None:
        workflow = SystemWorkflow(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            step=system_agents.LEGACY_STEP_AUTONOMOUS_GOAL_HISTORY,
        )

        self.assertEqual(
            system_agents._expected_system_agent_kinds_for_step(workflow),
            (system_agents.LEGACY_AUTONOMOUS_GOAL_HISTORY_AGENT_KIND,),
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
    def test_recover_retires_legacy_workflows_and_redrives_tool_judge(
        self, mock_judge: MagicMock, mock_block: MagicMock
    ) -> None:
        workflow = self._stranded_autonomous_goal_workflow(
            system_agents.LEGACY_STEP_AUTONOMOUS_GOAL_HISTORY,
            self._autonomous_goal(),
        )
        autonomous_goals._recover_stranded_autonomous_goal_workflow(workflow)
        mock_judge.assert_not_called()
        mock_block.assert_called_once()

        mock_block.reset_mock()
        candidate = {"title": "t", "summary": "s", "impact": "i"}
        workflow = self._stranded_autonomous_goal_workflow(
            system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            self._autonomous_goal(),
        )
        workflow.state = {
            **workflow.state,
            "candidate": candidate,
            "tool_protocol": True,
        }
        workflow.save(update_fields=["state"])
        autonomous_goals._recover_stranded_autonomous_goal_workflow(workflow)
        mock_judge.assert_called_once()
        self.assertEqual(mock_judge.call_args.args[2], candidate)
        mock_block.assert_not_called()

        mock_judge.reset_mock()
        workflow = self._stranded_autonomous_goal_workflow(
            system_agents.STEP_AUTONOMOUS_GOAL_JUDGE_RUNNING,
            self._autonomous_goal(),
        )
        workflow.state = {**workflow.state, "candidate": candidate}
        workflow.save(update_fields=["state"])
        autonomous_goals._recover_stranded_autonomous_goal_workflow(workflow)
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




    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
            workflow = autonomous_goals.start_autonomous_goal_workflow_if_queue_idle(
                autonomous_goal=autonomous_goal
            )
        assert workflow is not None

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

    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    def test_accepted_stack_proposal_stop_waits_when_one_force_stop_fails(
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

        def interrupt_side_effect(
            _instance_id: int,
            *,
            expected_thread_id: str,
            force: bool,
            error: str,
        ) -> CodexInstance | None:
            self.assertTrue(force)
            self.assertEqual(
                error, system_agents.AUTONOMOUS_GOAL_PROPOSAL_ACCEPTED_ERROR
            )
            if expected_thread_id == interrupted_instance.thread_id:
                interrupted_instance.status = CodexInstance.STATUS_FAILED
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
        self.assertEqual(interrupted_run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(live_run.status, SystemAgentRun.STATUS_RUNNING)
        mock_cleanup.assert_not_called()






    @patch("hitch.main.workflows.autonomous_goals.cleanup_worktree")
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.autonomous_goals.create_worktree_for_session")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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

        workflow = autonomous_goals.start_autonomous_goal_workflow_if_queue_idle(
            autonomous_goal=autonomous_goal,
            use_worktrees=True,
        )
        assert workflow is not None

        mock_cleanup.assert_called_once_with(managed_worktree)
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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

    @patch(
        "hitch.main.workflows.autonomous_goals.snapshot_worktree_to_commit",
        return_value="c" * 40,
    )
    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_auto_proposal_continues_from_pending_stack_proposal(
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
        mock_spawn.return_value = candidate_2

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
        self.assertIn("Stack round: 2 of 2", mock_spawn.call_args.kwargs["prompt"])
        previous_proposal.refresh_from_db()
        self.assertEqual(previous_proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertEqual(previous_proposal.outcome_notes, "")
        self.assertFalse(previous_proposal.outcome_metadata["stacked_diff_hidden_until_complete"])

    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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


    @patch("hitch.main.workflows.autonomous_goals.default_branch_commit_hash")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_auto_proposal_blocks_stale_ownerless_migrated_pr(
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
        mock_spawn.return_value = _instance(
            thread_id="candidate-thread",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
    def test_auto_proposal_blocks_rollout_only_done_accepted_session(
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
        started = autonomous_goals.maybe_start_auto_proposal_workflows(project=project)

        self.assertEqual(started, 0)
        mock_spawn.assert_not_called()

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
            SessionPullRequest.objects.create(
                thread_id=session.thread_id,
                cwd=session.cwd,
                state={
                    "pr_handoff": {
                        "url": f"https://github.com/cberner/hitch/pull/{100 + index}",
                        "state": "closed",
                        "merged": True,
                    }
                },
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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


    @patch("hitch.main.workflows.autonomous_goals._spawn_autonomous_goal_candidate_or_block")
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








    @patch("hitch.main.workflows.autonomous_goals.codex_pool.interrupt_instance")
    @patch("hitch.main.workflows.autonomous_goals.codex_pool.spawn_turn")
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

        workflow = autonomous_goals.start_autonomous_goal_workflow_if_queue_idle(
            autonomous_goal=autonomous_goal
        )
        assert workflow is not None

        workflow.refresh_from_db()
        run = SystemAgentRun.objects.get()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.state["error"], "autonomous goal no longer exists")
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, "autonomous goal no longer exists")
        mock_interrupt.assert_called_once_with(
            run.instance_id,
            expected_thread_id=run.thread_id,
            force=True,
            error="autonomous goal no longer exists",
        )










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
