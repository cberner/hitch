"""Coverage for the detached Codex worker plumbing: spawning, launching the
subprocess, the worker management command, and the bookkeeping that keeps
the CodexInstance row in sync with the OS process.
"""

import dataclasses
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone
from openai_codex import ApprovalMode
from openai_codex._message_router import MessageRouter
from openai_codex.generated.v2_all import (
    ApprovalsReviewer,
    AskForApprovalValue,
    DangerFullAccessSandboxPolicy,
    ReasoningEffort,
    SandboxPolicy,
    ThreadGoal,
    ThreadGoalClearedNotification,
    ThreadGoalStatus,
    ThreadGoalUpdatedNotification,
    ThreadSource,
    Turn,
    TurnCompletedNotification,
    TurnError,
    TurnStartParams,
    TurnStatus,
    WorkspaceWriteSandboxPolicy,
)
from openai_codex.models import Notification
from pydantic import BaseModel

from hitch.main import codex_events, codex_pool, coding_agents, demo, streaming
from hitch.main.management.commands import codex_worker as codex_worker_module
from hitch.main.management.commands.codex_worker import (
    _DEFAULT_COLLABORATION_INSTRUCTIONS,
    _forward_goal_notifications,
    _install_notification_sequencer,
    _make_approval_handler,
    _serialize_event,
    _start_goal_event_forwarder,
)
from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    SessionDemo,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
)


def _events_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory()


def _stub_codex_thread_start(
    mock_codex: MagicMock, thread_id: str = "t", thread_path: str = ""
) -> MagicMock:
    codex: MagicMock = mock_codex.return_value.__enter__.return_value
    codex.thread_start.return_value = SimpleNamespace(id=thread_id)
    codex._client.thread_start.return_value = SimpleNamespace(
        thread=SimpleNamespace(id=thread_id, path=thread_path)
    )
    return codex


def _thread_start_payload(codex: MagicMock) -> dict[str, Any]:
    codex._client.thread_start.assert_called_once()
    payload = codex._client.thread_start.call_args.args[0]
    assert isinstance(payload, dict)
    return payload


def _completed_event(turn_id: str, status: TurnStatus, error_message: str | None = None) -> SimpleNamespace:
    """Build a turn/completed event whose payload is a real
    TurnCompletedNotification — the worker's status logic narrows on the
    SDK type, not on duck-typed shapes, so the test must use the real model.
    """
    return SimpleNamespace(
        method="turn/completed",
        payload=TurnCompletedNotification(
            thread_id="thread-1",
            turn=Turn(
                id=turn_id,
                items=[],
                status=status,
                error=TurnError(message=error_message) if error_message else None,
            ),
        ),
    )


def _stub_thread_resume(events: list[SimpleNamespace], turn_id: str = "turn-1") -> object:
    """Return an object shaped like ``thread_resume(...).turn(...).stream()``."""
    return SimpleNamespace(
        turn=lambda _input: SimpleNamespace(id=turn_id, stream=lambda: iter(events)),
    )


def _wait_for_linux_proc_state(pid: int, state: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if codex_pool._linux_proc_state(pid) == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"pid {pid} did not reach Linux process state {state!r}")


def _wait_for_process_exit(
    proc: subprocess.Popen[bytes], timeout: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            return
        time.sleep(0.01)
    raise AssertionError(f"pid {proc.pid} was not reaped")


def _forget_worker_pid(pid: int) -> None:
    with codex_pool._TRACKED_WORKER_PROCS_LOCK:
        codex_pool._TRACKED_WORKER_PROCS.pop(pid, None)
        codex_pool._REAPED_WORKERS.difference_update(
            {worker for worker in codex_pool._REAPED_WORKERS if worker[0] == pid}
        )


class SpawnNewSessionTests(TestCase):
    def test_hitch_instructions_expand_proposal_env_values_at_call_site(self) -> None:
        self.assertIn(
            '$HITCH_PROPOSE_SESSION_COMMAND run --project "$HITCH_PROJECT_DIR" '
            '"$HITCH_MANAGE_PY" propose_session --cwd "$HITCH_CWD" '
            '--source-thread-id "$HITCH_THREAD_ID"',
            coding_agents.HITCH_BASE_INSTRUCTIONS,
        )
        self.assertNotIn(
            'HITCH_PROPOSE_SESSION_COMMAND` with `--title',
            coding_agents.HITCH_BASE_INSTRUCTIONS,
        )

    def test_memories_are_explicitly_disabled_by_default(self) -> None:
        config = codex_pool.app_server_config()

        self.assertEqual(config.config_overrides, ("features.memories=false",))

    def test_stamps_deployment_marker_for_nuke_scoping(self) -> None:
        config = codex_pool.app_server_config()

        self.assertEqual(
            config.env,
            {
                codex_pool._APP_SERVER_DEPLOYMENT_ENV: (
                    codex_pool._app_server_deployment_id()
                )
            },
        )

    def test_app_server_config_rejects_invalid_web_search_mode(self) -> None:
        with self.assertRaises(ValueError):
            codex_pool.app_server_config(
                web_search_mode='live"\napproval_policy="never'
            )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_creates_thread_then_spawns_worker(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        thread_path = "/root/.codex/sessions/rollout-thread-abc.jsonl"
        codex = _stub_codex_thread_start(
            mock_codex, "thread-abc", thread_path=thread_path
        )
        mock_launch.return_value = SimpleNamespace(pid=4242)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(cwd="/repo", prompt="hi")

        self.assertEqual(instance.thread_id, "thread-abc")
        self.assertEqual(instance.cwd, "/repo")
        self.assertEqual(instance.prompt, "hi")
        self.assertEqual(instance.developer_instructions, "")
        self.assertEqual(instance.base_instructions, "")
        self.assertEqual(instance.pid, 4242)
        self.assertEqual(instance.status, CodexInstance.STATUS_STARTING)
        self.assertTrue(instance.events_path.endswith(f"{instance.pk}.jsonl"))
        self.assertEqual(codex_pool.thread_path_for_instance(instance), thread_path)
        # ``model=None`` means "fall back to whatever Codex's config picks".
        payload = _thread_start_payload(codex)
        self.assertEqual(payload["cwd"], "/repo")
        self.assertIsNone(payload["developerInstructions"])
        self.assertIsNone(payload["model"])
        self.assertEqual(payload["dynamicTools"][0]["namespace"], "hitch")
        self.assertEqual(payload["dynamicTools"][0]["name"], "propose_session")
        # ``thread/start`` defers writing the rollout file to disk, so the
        # cross-process ``thread/resume`` the worker and the session view both
        # rely on would fail with "no rollout found" without an explicit
        # metadata write to materialise the rollout. ``thread/set-name`` is the
        # cheapest such write; it must happen inside the same Codex context as
        # ``thread/start`` so the in-memory thread is still loaded.
        codex._client.thread_set_name.assert_called_once_with("thread-abc", "hi")
        # Worker subprocess only receives the row id and (when set) the
        # reasoning effort, sandbox policy, and approval mode; the prompt is
        # read from the row to avoid argparse misinterpreting prompts that
        # begin with '-'.
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_systemd_launch_leaves_pid_for_worker_and_records_unit(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        _stub_codex_thread_start(mock_codex, "thread-scoped")
        mock_launch.return_value = codex_pool.WorkerLaunch(
            pid=0,
            scope_unit="hitch-codex-worker-7.service",
        )

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(cwd="/repo", prompt="hi")

        self.assertEqual(instance.pid, 0)
        self.assertEqual(
            instance.systemd_scope_unit,
            "hitch-codex-worker-7.service",
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_systemd_post_launch_pid_write_does_not_clobber_worker_pid(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        # Under systemd isolation the launcher can leave pid unset; the worker
        # overwrites pid with its own real pid as its first action. If a future
        # launch path reports a non-worker pid and the parent's post-launch
        # write lands after the worker's, it must not clobber the real pid.
        _stub_codex_thread_start(mock_codex, "thread-scoped")
        real_worker_pid = 4242
        wrapper_pid = 999

        def _launch(
            *, instance_id: int, **_kwargs: object
        ) -> codex_pool.WorkerLaunch:
            # Simulate the worker winning the pid/status handshake race.
            CodexInstance.objects.filter(pk=instance_id).update(
                pid=real_worker_pid, status=CodexInstance.STATUS_RUNNING
            )
            return codex_pool.WorkerLaunch(
                pid=wrapper_pid, scope_unit="hitch-codex-worker-7.service"
            )

        mock_launch.side_effect = _launch

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(cwd="/repo", prompt="hi")

        instance.refresh_from_db()
        self.assertEqual(instance.pid, real_worker_pid)
        self.assertEqual(
            instance.systemd_scope_unit,
            "hitch-codex-worker-7.service",
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_spawn_new_session_persists_auto_qa_and_auto_merge_fields(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        _stub_codex_thread_start(mock_codex, "thread-auto-merge")
        mock_launch.return_value = SimpleNamespace(pid=4242)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                auto_qa_enabled=True,
                auto_merge_to_local_branch=True,
                auto_merge_branch="release",
            )

        self.assertTrue(instance.auto_qa_enabled)
        self.assertFalse(instance.auto_pr_enabled)
        self.assertTrue(instance.auto_merge_to_local_branch)
        self.assertEqual(instance.auto_merge_branch, "release")
        instance.refresh_from_db()
        self.assertTrue(instance.auto_qa_enabled)
        self.assertFalse(instance.auto_pr_enabled)
        self.assertTrue(instance.auto_merge_to_local_branch)
        self.assertEqual(instance.auto_merge_branch, "release")

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_initial_thread_name_derivation(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        """The initial name is the first line of the prompt, trimmed of
        whitespace, clipped to 200 chars; whitespace-only prompts fall back
        to a static placeholder rather than being passed through verbatim
        (Codex rejects whitespace-only names)."""
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        cases = [
            ("  Refactor the parser \nthen write tests\n", "Refactor the parser"),
            ("a" * 500, "a" * 200),
            ("   \n\t  ", "New session"),
        ]
        for prompt, expected in cases:
            with self.subTest(prompt=prompt[:30]):
                codex._client.thread_set_name.reset_mock()
                with (
                    _events_dir() as events_dir,
                    override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
                ):
                    codex_pool.spawn_new_session(cwd="/repo", prompt=prompt)
                codex._client.thread_set_name.assert_called_once_with("t", expected)

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_initial_thread_name_can_use_explicit_task_title(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        codex = _stub_codex_thread_start(mock_codex, "thread-abc")
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="Go ahead and implement this proposed session.",
                thread_name="Add parser coverage",
            )

        self.assertEqual(instance.prompt, "Go ahead and implement this proposed session.")
        codex._client.thread_set_name.assert_called_once_with(
            "thread-abc", "Add parser coverage"
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_forwards_base_instructions_to_thread_start(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                base_instructions="Base override.",
            )

        payload = _thread_start_payload(codex)
        self.assertEqual(payload["cwd"], "/repo")
        self.assertIsNone(payload["developerInstructions"])
        self.assertIsNone(payload["model"])
        self.assertEqual(payload["baseInstructions"], "Base override.")
        self.assertEqual(instance.base_instructions, "Base override.")

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_system_agent_thread_start_forwards_source_without_hitch_tools(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                thread_source=ThreadSource.subagent,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            )

        payload = _thread_start_payload(codex)
        self.assertEqual(payload["threadSource"], ThreadSource.subagent.value)
        self.assertNotIn("dynamicTools", payload)

    @patch("hitch.main.codex_pool.Codex")
    def test_create_session_thread_forwards_base_instructions(
        self, mock_codex: MagicMock
    ) -> None:
        codex = _stub_codex_thread_start(mock_codex, "thread-abc")

        thread_id = codex_pool.create_session_thread(
            cwd="/repo",
            name="QA",
            base_instructions="Base override.",
        )

        self.assertEqual(thread_id, "thread-abc")
        payload = _thread_start_payload(codex)
        self.assertEqual(payload["cwd"], "/repo")
        self.assertIsNone(payload["developerInstructions"])
        self.assertIsNone(payload["model"])
        self.assertEqual(payload["baseInstructions"], "Base override.")
        # Visible sessions are real user threads, so they register the Hitch
        # dynamic tools (e.g. propose_session) like spawn_new_session does.
        self.assertEqual(payload["dynamicTools"][0]["namespace"], "hitch")
        self.assertEqual(payload["dynamicTools"][0]["name"], "propose_session")
        codex._client.thread_set_name.assert_called_once_with("thread-abc", "QA")

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_forwards_model_effort_sandbox_and_approval(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        """The settings dialog's model selector flows into
        ``thread_start(model=...)``; effort, sandbox policy and approval
        mode flow into the worker as CLI args. Pin every wiring so a
        refactor can't quietly drop one of them.
        """
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                developer_instructions="Prefer small, typed changes.",
                model="gpt-5",
                reasoning_effort="high",
                sandbox_policy="workspaceWrite",
                approval_mode="deny_all",
            )

        payload = _thread_start_payload(codex)
        self.assertEqual(payload["cwd"], "/repo")
        self.assertEqual(
            payload["developerInstructions"], "Prefer small, typed changes."
        )
        self.assertEqual(payload["model"], "gpt-5")
        self.assertEqual(instance.developer_instructions, "Prefer small, typed changes.")
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort="high",
            sandbox_policy="workspaceWrite",
            approval_mode="deny_all",
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_enable_memories_sets_config_override_and_worker_flag(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                enable_memories=True,
            )

        self.assertTrue(instance.enable_memories)
        mock_codex.assert_called_once()
        config = mock_codex.call_args.kwargs["config"]
        self.assertEqual(config.config_overrides, ("features.memories=true",))
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
            enable_memories=True,
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_web_search_mode_sets_config_override_and_worker_flag(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                web_search_mode="live",
            )

        self.assertEqual(instance.web_search_mode, "live")
        config = mock_codex.call_args.kwargs["config"]
        self.assertEqual(
            config.config_overrides,
            ('features.memories=false', 'web_search="live"'),
        )
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
            web_search_mode="live",
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_plan_mode_forwards_model_to_worker(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        codex = _stub_codex_thread_start(mock_codex)
        mock_launch.return_value = SimpleNamespace(pid=1)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                model="gpt-5.4",
                plan_mode=True,
            )

        payload = _thread_start_payload(codex)
        self.assertEqual(payload["cwd"], "/repo")
        self.assertIsNone(payload["developerInstructions"])
        self.assertEqual(payload["model"], "gpt-5.4")
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            model="gpt-5.4",
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
            plan_mode=True,
        )


class SqliteHomeTests(SimpleTestCase):
    """Per-role ``sqlite_home`` isolation (openai/codex#20213 mitigation)."""

    def test_concurrent_leases_fan_out_across_pool(self) -> None:
        with (
            tempfile.TemporaryDirectory() as base,
            override_settings(
                CODEX_SQLITE_HOME_BASE=Path(base),
                CODEX_WORKER_SQLITE_POOL_SIZE=2,
            ),
        ):
            # Hold the first lease while taking the second: the flock on
            # worker-0 forces the second worker onto worker-1.
            lease0 = codex_pool.acquire_worker_sqlite_home(10)
            lease1 = codex_pool.acquire_worker_sqlite_home(11)
            try:
                self.assertEqual(lease0.home.name, "worker-0")
                self.assertEqual(lease1.home.name, "worker-1")
                self.assertFalse(lease0.overflow)
                self.assertFalse(lease1.overflow)
                self.assertTrue(lease0.home.is_dir())
                self.assertTrue(lease1.home.is_dir())
            finally:
                lease0.release()
                lease1.release()

    def test_released_slot_is_reused(self) -> None:
        with (
            tempfile.TemporaryDirectory() as base,
            override_settings(
                CODEX_SQLITE_HOME_BASE=Path(base),
                CODEX_WORKER_SQLITE_POOL_SIZE=2,
            ),
        ):
            first = codex_pool.acquire_worker_sqlite_home(1)
            first.release()
            # With the slot free again, the next lease takes worker-0, not a
            # fresh slot -- keeping the backfilled home reused.
            second = codex_pool.acquire_worker_sqlite_home(2)
            try:
                self.assertEqual(first.home.name, "worker-0")
                self.assertEqual(second.home.name, "worker-0")
            finally:
                second.release()

    def test_overflow_home_when_pool_exhausted(self) -> None:
        with (
            tempfile.TemporaryDirectory() as base,
            override_settings(
                CODEX_SQLITE_HOME_BASE=Path(base),
                CODEX_WORKER_SQLITE_POOL_SIZE=1,
            ),
        ):
            slot = codex_pool.acquire_worker_sqlite_home(5)
            overflow = codex_pool.acquire_worker_sqlite_home(6)
            try:
                self.assertEqual(slot.home.name, "worker-0")
                self.assertFalse(slot.overflow)
                self.assertTrue(overflow.overflow)
                self.assertEqual(overflow.home.name, "worker-overflow-6")
                self.assertTrue(overflow.home.is_dir())
            finally:
                slot.release()
                overflow.release()
            # The private overflow home is removed on release; the reused
            # slot home is left in place.
            self.assertFalse(overflow.home.exists())
            self.assertTrue(slot.home.is_dir())

    def test_web_home_is_distinct_and_created(self) -> None:
        with (
            tempfile.TemporaryDirectory() as base,
            override_settings(CODEX_SQLITE_HOME_BASE=Path(base)),
        ):
            web = codex_pool.web_sqlite_home()
            lease = codex_pool.acquire_worker_sqlite_home(0)
            try:
                self.assertEqual(web.name, "web")
                self.assertTrue(web.is_dir())
                self.assertNotEqual(web, lease.home)
            finally:
                lease.release()

    def test_app_server_config_injects_explicit_sqlite_home(self) -> None:
        config = codex_pool.app_server_config(sqlite_home="/srv/codex-state")
        assert config.env is not None
        self.assertEqual(
            config.env.get(codex_pool._CODEX_SQLITE_HOME_ENV), "/srv/codex-state"
        )

    def test_app_server_config_defaults_to_web_home_outside_tests(self) -> None:
        with tempfile.TemporaryDirectory() as base:
            with override_settings(
                CODEX_SQLITE_HOME_BASE=Path(base), TESTING=False
            ):
                config = codex_pool.app_server_config()
            assert config.env is not None
            self.assertEqual(
                config.env.get(codex_pool._CODEX_SQLITE_HOME_ENV),
                str(Path(base) / "web"),
            )

    def test_app_server_config_omits_sqlite_home_under_tests(self) -> None:
        # The default request path must not create state dirs (or perturb the
        # exact-env assertion) while running the suite.
        config = codex_pool.app_server_config()
        assert config.env is not None
        self.assertNotIn(codex_pool._CODEX_SQLITE_HOME_ENV, config.env)

    def test_codex_home_dir_fallback_overrides_inherited_env(self) -> None:
        # The lease-failure degrade points at $CODEX_HOME explicitly so it
        # overrides any CODEX_SQLITE_HOME the deployment exported (a bare omission
        # would leave the inherited value in effect via the SDK's env merge).
        with patch.dict(os.environ, {"CODEX_HOME": "/srv/codex"}):
            self.assertEqual(codex_pool.codex_home_dir(), Path("/srv/codex"))
            config = codex_pool.app_server_config(
                sqlite_home=str(codex_pool.codex_home_dir())
            )
        assert config.env is not None
        self.assertEqual(
            config.env.get(codex_pool._CODEX_SQLITE_HOME_ENV), "/srv/codex"
        )


class PruneWorkerLogsDbTests(SimpleTestCase):
    def _write(self, path: Path, size: int) -> None:
        path.write_bytes(b"\0" * size)

    def test_deletes_oversized_log_db_and_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            home_path = Path(home)
            logs = home_path / "logs_2.sqlite"
            wal = home_path / "logs_2.sqlite-wal"
            shm = home_path / "logs_2.sqlite-shm"
            state = home_path / "state_5.sqlite"
            self._write(logs, 80)
            self._write(wal, 60)  # main+wal exceed the 100-byte cap
            self._write(shm, 5)
            self._write(state, 1000)

            freed = codex_pool.prune_worker_logs_db(home_path, max_bytes=100)

            self.assertEqual(freed, 145)
            self.assertFalse(logs.exists())
            self.assertFalse(wal.exists())
            self.assertFalse(shm.exists())
            # The state DB is never touched, even when much larger than the cap.
            self.assertTrue(state.exists())

    def test_keeps_log_db_under_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            logs = Path(home) / "logs_2.sqlite"
            self._write(logs, 50)

            freed = codex_pool.prune_worker_logs_db(home, max_bytes=100)

            self.assertEqual(freed, 0)
            self.assertTrue(logs.exists())

    def test_uses_settings_default_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            logs = Path(home) / "logs_2.sqlite"
            self._write(logs, 200)
            with override_settings(CODEX_WORKER_LOGS_DB_MAX_BYTES=100):
                freed = codex_pool.prune_worker_logs_db(home)
            self.assertEqual(freed, 200)
            self.assertFalse(logs.exists())

    def test_missing_home_is_noop(self) -> None:
        freed = codex_pool.prune_worker_logs_db(
            Path("/nonexistent/codex-home"), max_bytes=1
        )
        self.assertEqual(freed, 0)


class SpawnFailureTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_marks_instance_failed_when_launch_raises(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.side_effect = OSError("boom")

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            image_path = codex_pool.input_attachments_dir() / "req" / "1.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")
            with self.assertRaises(OSError):
                codex_pool.spawn_turn(
                    thread_id="t",
                    cwd="/repo",
                    prompt="hi",
                    input_image_paths=[str(image_path)],
                )
            self.assertFalse(image_path.exists())

            # The exception propagates to the caller, but the row is left in a
            # terminal state so it isn't treated as forever-pending.
        instance = CodexInstance.objects.latest("started_at")
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.pid, 0)
        self.assertIn("boom", instance.error)
        self.assertIsNotNone(instance.ended_at)
        self.assertEqual(instance.input_image_paths, [])
        self.assertEqual(instance.input_attachment_paths, [])


class SpawnTurnTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_resumes_existing_thread_without_calling_codex(
        self, mock_launch: MagicMock
    ) -> None:
        """``spawn_turn`` operates entirely on a known thread id — it must
        never reach out to the Codex app-server, which keeps it cheap on the
        request-serving path."""
        mock_launch.return_value = SimpleNamespace(pid=1234)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz",
                cwd="/repo",
                prompt="follow-up",
                input_image_paths=["/tmp/screen.png"],
            )

        self.assertEqual(instance.thread_id, "thread-xyz")
        self.assertEqual(instance.prompt, "follow-up")
        self.assertEqual(instance.input_image_paths, ["/tmp/screen.png"])
        self.assertEqual(instance.input_attachment_paths, ["/tmp/screen.png"])
        self.assertEqual(instance.developer_instructions, "")
        self.assertEqual(instance.pid, 1234)
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_spawn_turn_persists_auto_qa_enabled(self, mock_launch: MagicMock) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz",
                cwd="/repo",
                prompt="follow-up",
                auto_qa_enabled=True,
            )

        instance.refresh_from_db()
        self.assertTrue(instance.auto_qa_enabled)
        self.assertFalse(instance.auto_pr_enabled)

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_spawn_turn_marks_only_user_reviewer_approval_modes_live_editable(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)

        for approval_mode, expected in (
            ("prompt_user", True),
            ("approve_all", True),
            ("auto_review", False),
            ("deny_all", False),
            (None, False),
        ):
            with self.subTest(approval_mode=approval_mode):
                with (
                    _events_dir() as events_dir,
                    override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
                ):
                    instance = codex_pool.spawn_turn(
                        thread_id=f"thread-{approval_mode or 'default'}",
                        cwd="/repo",
                        prompt="follow-up",
                        approval_mode=approval_mode,
                    )

                self.assertEqual(instance.approval_mode_live_editable, expected)

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_plan_mode_turn_forwards_model_and_plan_flag(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz",
                cwd="/repo",
                prompt="make a plan",
                model="gpt-5.4",
                plan_mode=True,
            )

        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            model="gpt-5.4",
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
            plan_mode=True,
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_copies_developer_instructions_from_previous_turn(
        self, mock_launch: MagicMock
    ) -> None:
        """Follow-up workers have to re-supply the thread's developer
        instructions on resume; copy them from the latest known row so
        they are not lost when each turn runs in a fresh process."""
        mock_launch.return_value = SimpleNamespace(pid=1234)
        CodexInstance.objects.create(
            pid=999,
            thread_id="thread-xyz",
            cwd="/repo",
            prompt="first",
            developer_instructions="Prefer small, typed changes.",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz", cwd="/repo", prompt="follow-up"
            )

        self.assertEqual(instance.developer_instructions, "Prefer small, typed changes.")

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_copies_base_instructions_from_previous_turn(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)
        CodexInstance.objects.create(
            pid=999,
            thread_id="thread-xyz",
            cwd="/repo",
            prompt="first",
            base_instructions="Base override.",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz", cwd="/repo", prompt="follow-up"
            )

        self.assertEqual(instance.base_instructions, "Base override.")

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_omitted_web_search_mode_does_not_inherit_previous_turn(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)
        CodexInstance.objects.create(
            pid=999,
            thread_id="thread-xyz",
            cwd="/repo",
            prompt="first",
            web_search_mode="live",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz", cwd="/repo", prompt="follow-up"
            )

        self.assertEqual(instance.web_search_mode, "")
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_explicit_blank_web_search_mode_clears_previous_turn(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)
        CodexInstance.objects.create(
            pid=999,
            thread_id="thread-xyz",
            cwd="/repo",
            prompt="first",
            web_search_mode="live",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz",
                cwd="/repo",
                prompt="follow-up",
                web_search_mode="",
            )

        self.assertEqual(instance.web_search_mode, "")
        mock_launch.assert_called_once_with(
            instance_id=instance.pk,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
        )

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_explicit_base_instructions_override_previous_turn(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)
        CodexInstance.objects.create(
            pid=999,
            thread_id="thread-xyz",
            cwd="/repo",
            prompt="first",
            base_instructions="Old base.",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz",
                cwd="/repo",
                prompt="follow-up",
                base_instructions="Fresh base.",
            )

        self.assertEqual(instance.base_instructions, "Fresh base.")

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_explicit_developer_instructions_override_previous_turn(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = SimpleNamespace(pid=1234)
        CodexInstance.objects.create(
            pid=999,
            thread_id="thread-xyz",
            cwd="/repo",
            prompt="first",
            developer_instructions="Old instructions.",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
        )

        with (
            _events_dir() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_turn(
                thread_id="thread-xyz",
                cwd="/repo",
                prompt="follow-up",
                developer_instructions="Fresh instructions.",
                user_message_index=7,
            )

        self.assertEqual(instance.developer_instructions, "Fresh instructions.")
        self.assertEqual(instance.user_message_index, 7)

    @unittest.skipUnless(Path("/proc").exists(), "requires Linux /proc")
    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_tracks_real_popen_handles_for_reaping(
        self, mock_launch: MagicMock
    ) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mock_launch.return_value = proc
        try:
            with (
                _events_dir() as events_dir,
                override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
            ):
                instance = codex_pool.spawn_turn(
                    thread_id="thread-xyz", cwd="/repo", prompt="follow-up"
                )

            self.assertEqual(instance.pid, proc.pid)
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                self.assertEqual(
                    codex_pool._TRACKED_WORKER_PROCS[proc.pid],
                    (instance.pk, proc),
                )
        finally:
            if proc.returncode is None:
                proc.kill()
                proc.wait(timeout=5)
            _forget_worker_pid(proc.pid)


class LaunchWorkerProcessTests(TestCase):
    @override_settings(CODEX_WORKER_ISOLATION="direct")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_direct_launches_manage_command_in_new_session(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=Path(raw)),
        ):
            launch = codex_pool._launch_worker_process(instance_id=7)

            args, kwargs = mock_popen.call_args
            argv = args[0]
            self.assertEqual(launch.pid, 999)
            self.assertEqual(launch.scope_unit, "")
            self.assertEqual(argv[1].endswith("manage.py"), True)
            self.assertEqual(argv[2], "codex_worker")
            self.assertIn("--instance-id", argv)
            self.assertEqual(argv[argv.index("--instance-id") + 1], "7")
            # The prompt is *not* passed on the command line — it lives on the
            # CodexInstance row so a leading '-' can't be reinterpreted as an
            # argparse option.
            self.assertNotIn("--prompt", argv)
            # The worker must outlive the Django process, so it gets its own
            # session; stderr is kept in a per-worker log for crash forensics.
            self.assertTrue(kwargs["start_new_session"])
            from subprocess import DEVNULL

            self.assertEqual(kwargs["stdin"], DEVNULL)
            self.assertEqual(kwargs["stdout"], DEVNULL)
            self.assertEqual(Path(kwargs["stderr"].name), Path(raw) / "7.log")
            self.assertTrue(kwargs["stderr"].closed)
            self.assertTrue(kwargs["close_fds"])
            self.assertEqual(
                kwargs["env"]["DJANGO_SETTINGS_MODULE"],
                "hitch.settings.dev",
            )
            # No effort passed → no --reasoning-effort on the CLI; Codex's own
            # default takes over inside the worker.
            self.assertNotIn("--reasoning-effort", argv)


class SystemdInstallRecipeTests(SimpleTestCase):
    def test_systemd_deployment_forces_systemd_worker_isolation(self) -> None:
        justfile = (Path(settings.BASE_DIR) / "justfile").read_text()

        self.assertIn("Environment=HITCH_CODEX_WORKER_ISOLATION=systemd", justfile)

    def test_systemd_deployment_runs_runserver_without_autoreloader(self) -> None:
        justfile = (Path(settings.BASE_DIR) / "justfile").read_text()

        self.assertIn(
            'ExecStart="${UV_BIN}" run ./manage.py runserver --noreload '
            "--settings hitch.settings.dev",
            justfile,
        )

    def test_global_worker_isolation_default_stays_auto(self) -> None:
        common_settings = (
            Path(settings.BASE_DIR) / "hitch" / "settings" / "common.py"
        ).read_text()
        default_line = (
            'CODEX_WORKER_ISOLATION = os.environ.get('
            '"HITCH_CODEX_WORKER_ISOLATION", "auto")'
        )

        self.assertIn(default_line, common_settings)
        self.assertNotIn("_CODEX_WORKER_ISOLATION_DEFAULT", common_settings)


class LaunchWorkerProcessSystemdTests(TestCase):
    @override_settings(
        CODEX_WORKER_ISOLATION="systemd",
        CODEX_WORKER_MEMORY_HIGH="4G",
        CODEX_WORKER_MEMORY_MAX="12G",
        CODEX_WORKER_MEMORY_SWAP_MAX="0",
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
    )
    @patch("hitch.main.codex_pool._ensure_systemd_worker_slice")
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemd-run")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_systemd_launches_worker_in_memory_capped_service(
        self,
        mock_popen: MagicMock,
        mock_which: MagicMock,
        mock_ensure_slice: MagicMock,
    ) -> None:
        proc = MagicMock()
        proc.pid = 999
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=Path(raw)),
        ):
            launch = codex_pool._launch_worker_process(instance_id=7)

        args, kwargs = mock_popen.call_args
        argv = args[0]
        self.assertEqual(launch.pid, 0)
        self.assertIsNone(launch.proc)
        self.assertEqual(launch.scope_unit, "hitch-codex-worker-7.service")
        self.assertEqual(
            argv[:5],
            [
                "/usr/bin/systemd-run",
                "--user",
                "--quiet",
                "--collect",
                "--service-type=exec",
            ],
        )
        self.assertNotIn("--pipe", argv)
        self.assertIn("--unit=hitch-codex-worker-7", argv)
        self.assertIn("--slice=hitch-codex-workers.slice", argv)
        self.assertIn("--setenv=DJANGO_SETTINGS_MODULE", argv)
        self.assertIn("--property=StandardOutput=null", argv)
        self.assertIn(
            f"--property=StandardError=append:{Path(raw) / '7.log'}", argv
        )
        # MemoryHigh/MemoryMax are silently ignored on hosts that do not
        # default to memory accounting, so the service must opt in explicitly —
        # otherwise the per-worker cap would not actually bound the worker.
        self.assertIn("--property=MemoryAccounting=yes", argv)
        self.assertIn("--property=MemoryHigh=4G", argv)
        self.assertIn("--property=MemoryMax=12G", argv)
        # Cap swap so MemoryMax is a true ceiling that OOM-kills a runaway
        # worker instead of letting it thrash swap forever.
        self.assertIn("--property=MemorySwapMax=0", argv)
        separator = argv.index("--")
        worker_argv = argv[separator + 1 :]
        self.assertEqual(worker_argv[1].endswith("manage.py"), True)
        self.assertEqual(worker_argv[2], "codex_worker")
        self.assertEqual(worker_argv[worker_argv.index("--instance-id") + 1], "7")
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(
            kwargs["env"]["DJANGO_SETTINGS_MODULE"],
            "hitch.settings.dev",
        )
        proc.wait.assert_called_once_with(timeout=0.25)
        mock_which.assert_called_once_with("systemd-run")
        mock_ensure_slice.assert_called_once_with()

    @override_settings(CODEX_WORKER_ISOLATION="systemd")
    @patch("hitch.main.codex_pool._ensure_systemd_worker_slice")
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemd-run")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_systemd_launch_keeps_slow_client_pending(
        self,
        mock_popen: MagicMock,
        _mock_which: MagicMock,
        _mock_ensure_slice: MagicMock,
    ) -> None:
        proc = MagicMock()
        proc.pid = 999
        proc.wait.side_effect = subprocess.TimeoutExpired("systemd-run", 0.25)
        mock_popen.return_value = proc

        launch = codex_pool._launch_worker_process(instance_id=7)

        self.assertEqual(launch.pid, 0)
        self.assertIs(launch.proc, proc)
        self.assertEqual(launch.scope_unit, "hitch-codex-worker-7.service")
        proc.wait.assert_called_once_with(timeout=0.25)
        proc.kill.assert_not_called()

    @override_settings(
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
        CODEX_WORKER_SLICE_MEMORY_HIGH="8G",
        CODEX_WORKER_SLICE_MEMORY_MAX="10G",
        CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0",
        CODEX_PARENT_SLICE="",
    )
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_configures_worker_slice_aggregate_memory_cap(
        self, mock_run: MagicMock, mock_which: MagicMock
    ) -> None:
        mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

        codex_pool._ensure_systemd_worker_slice()

        argv = mock_run.call_args.args[0]
        self.assertEqual(
            argv[:5],
            [
                "/usr/bin/systemctl",
                "--user",
                "set-property",
                "--runtime",
                "hitch-codex-workers.slice",
            ],
        )
        self.assertIn("MemoryAccounting=yes", argv)
        self.assertIn("MemoryHigh=8G", argv)
        self.assertIn("MemoryMax=10G", argv)
        # The aggregate slice caps swap too, so workers collectively can't
        # exceed their RAM budget by spilling to swap.
        self.assertIn("MemorySwapMax=0", argv)
        mock_which.assert_called_once_with("systemctl")

    @override_settings(
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
        CODEX_WORKER_SLICE_MEMORY_HIGH="8G",
        CODEX_WORKER_SLICE_MEMORY_MAX="",
        CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="",
        CODEX_PARENT_SLICE="",
    )
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_slice_resets_cleared_caps_to_infinity(
        self, mock_run: MagicMock, _mock_which: MagicMock
    ) -> None:
        # `set-property --runtime` only changes the properties it is handed, so
        # a cleared cap must be reset to `infinity` rather than omitted —
        # otherwise a previously-applied MemoryMax/MemorySwapMax would linger on
        # the runtime slice and keep clipping workers after the operator cleared
        # it (and the hierarchy warning would wrongly treat it as unlimited).
        mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

        codex_pool._ensure_systemd_worker_slice()

        argv = mock_run.call_args.args[0]
        self.assertIn("MemoryHigh=8G", argv)
        self.assertIn("MemoryMax=infinity", argv)
        self.assertIn("MemorySwapMax=infinity", argv)

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="2G",
        CODEX_WORKER_MEMORY_MAX="4G",
        CODEX_WORKER_MEMORY_SWAP_MAX="0",
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
    )
    def test_per_worker_service_enables_memory_accounting(self) -> None:
        # Regression: a worker unit launched with MemoryHigh/MemoryMax but no
        # MemoryAccounting=yes has its limits silently dropped on any host that
        # does not default to memory accounting (DefaultMemoryAccounting=no or
        # legacy cgroup v1). The aggregate slice opts in, so without this the
        # per-worker cap is the only one that fails to bind and a single
        # runaway worker can consume the whole slice budget and OOM-kill its
        # sibling QA-panel lanes.
        argv = codex_pool._systemd_scope_argv(
            systemd_run="/usr/bin/systemd-run",
            scope_unit="hitch-codex-worker-7.service",
            worker_argv=["python", "manage.py", "codex_worker"],
        )

        self.assertIn("--property=MemoryAccounting=yes", argv)
        self.assertIn("--property=MemoryHigh=2G", argv)
        self.assertIn("--property=MemoryMax=4G", argv)

    def test_systemd_service_env_args_copy_valid_names_without_values(self) -> None:
        argv = codex_pool._systemd_scope_argv(
            systemd_run="/usr/bin/systemd-run",
            scope_unit="hitch-codex-worker-7.service",
            worker_argv=["python", "manage.py", "codex_worker"],
            env={
                "DJANGO_SETTINGS_MODULE": "hitch.settings.dev",
                "CODEX_HOME": "/home/user/.codex",
                "INVOCATION_ID": "parent-unit",
                "bad-name": "ignored",
            },
        )

        self.assertIn("--setenv=CODEX_HOME", argv)
        self.assertIn("--setenv=DJANGO_SETTINGS_MODULE", argv)
        self.assertNotIn("--setenv=INVOCATION_ID", argv)
        self.assertNotIn("--setenv=bad-name", argv)
        self.assertFalse(
            [arg for arg in argv if arg.startswith("--setenv=CODEX_HOME=")]
        )

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="2G",
        CODEX_WORKER_MEMORY_MAX="4G",
        CODEX_WORKER_MEMORY_SWAP_MAX="0",
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
    )
    def test_per_worker_service_caps_swap(self) -> None:
        # Regression: cgroup v2 counts only RAM toward MemoryMax, so without a
        # swap cap a runaway worker is reclaimed to swap rather than OOM-killed.
        # The hard cap then never fires and the turn thrashes the host
        # indefinitely instead of failing, so MemoryMax must ride with a swap
        # cap to be a true ceiling.
        argv = codex_pool._systemd_scope_argv(
            systemd_run="/usr/bin/systemd-run",
            scope_unit="hitch-codex-worker-7.service",
            worker_argv=["python", "manage.py", "codex_worker"],
        )

        self.assertIn("--property=MemorySwapMax=0", argv)

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="2G",
        CODEX_WORKER_MEMORY_MAX="4G",
        CODEX_WORKER_MEMORY_SWAP_MAX="1G",
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
    )
    def test_per_worker_service_honors_swap_override(self) -> None:
        # A non-zero cap grants a bounded swap cushion rather than forbidding
        # swap outright; the configured value must be passed through verbatim.
        argv = codex_pool._systemd_scope_argv(
            systemd_run="/usr/bin/systemd-run",
            scope_unit="hitch-codex-worker-7.service",
            worker_argv=["python", "manage.py", "codex_worker"],
        )

        self.assertIn("--property=MemorySwapMax=1G", argv)

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="",
        CODEX_WORKER_MEMORY_MAX="",
        CODEX_WORKER_MEMORY_SWAP_MAX="0",
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
    )
    def test_per_worker_service_skips_accounting_without_limits(self) -> None:
        # Accounting is only worth enabling when a limit rides along with it;
        # an unconfigured cap must not emit a bare MemoryAccounting property,
        # and a stray swap cap must not disable swap on an otherwise-unbounded
        # unit (it only completes a real MemoryMax ceiling).
        argv = codex_pool._systemd_scope_argv(
            systemd_run="/usr/bin/systemd-run",
            scope_unit="hitch-codex-worker-7.service",
            worker_argv=["python", "manage.py", "codex_worker"],
        )

        self.assertNotIn("--property=MemoryAccounting=yes", argv)
        self.assertFalse([arg for arg in argv if arg.startswith("--property=Memory")])

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="2G",
        CODEX_WORKER_MEMORY_MAX="",
        CODEX_WORKER_MEMORY_SWAP_MAX="0",
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
    )
    def test_per_worker_service_keeps_swap_for_soft_only_throttle(self) -> None:
        # MemoryHigh is a soft throttle that usage may exceed (graceful
        # degradation, no OOM); with no hard MemoryMax there is no ceiling for a
        # swap cap to make "true", so swap must NOT be disabled out from under a
        # config that deliberately avoided fail-fast OOMs. Accounting and the
        # soft throttle itself still apply.
        argv = codex_pool._systemd_scope_argv(
            systemd_run="/usr/bin/systemd-run",
            scope_unit="hitch-codex-worker-7.service",
            worker_argv=["python", "manage.py", "codex_worker"],
        )

        self.assertIn("--property=MemoryAccounting=yes", argv)
        self.assertIn("--property=MemoryHigh=2G", argv)
        self.assertNotIn("--property=MemoryMax=", argv)
        self.assertFalse(
            [arg for arg in argv if arg.startswith("--property=MemorySwapMax")]
        )

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="2G",
        CODEX_WORKER_MEMORY_MAX="infinity",
        CODEX_WORKER_MEMORY_SWAP_MAX="0",
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
    )
    def test_per_worker_service_keeps_swap_when_hard_cap_is_infinity(self) -> None:
        # systemd treats MemoryMax=infinity as no limit, so it is not a real
        # ceiling for the swap cap to make "true"; capping swap here would
        # reintroduce the soft-only/unbounded regression. The explicit
        # MemoryMax=infinity is still passed through.
        argv = codex_pool._systemd_scope_argv(
            systemd_run="/usr/bin/systemd-run",
            scope_unit="hitch-codex-worker-7.service",
            worker_argv=["python", "manage.py", "codex_worker"],
        )

        self.assertIn("--property=MemoryMax=infinity", argv)
        self.assertFalse(
            [arg for arg in argv if arg.startswith("--property=MemorySwapMax")]
        )

    @override_settings(CODEX_WORKER_SLICE="", CODEX_PARENT_SLICE="hitch.slice")
    @patch("hitch.main.codex_pool.shutil.which")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_skips_all_slice_configuration_when_worker_slice_disabled(
        self, mock_run: MagicMock, mock_which: MagicMock
    ) -> None:
        # Without a worker slice, workers are not placed under the parent slice
        # (`_systemd_scope_argv` omits `--slice`), so the parent-slice bias is
        # skipped too even though CODEX_PARENT_SLICE is set — it could not reach
        # those workers, and a failure on that unrelated slice would needlessly
        # abort every launch.
        codex_pool._ensure_systemd_worker_slice()

        mock_run.assert_not_called()
        mock_which.assert_not_called()

    @override_settings(
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
        CODEX_WORKER_SLICE_MEMORY_HIGH="8G",
        CODEX_WORKER_SLICE_MEMORY_MAX="10G",
    )
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_worker_slice_configuration_failure_is_fatal(
        self, mock_run: MagicMock, _mock_which: MagicMock
    ) -> None:
        mock_run.return_value = SimpleNamespace(
            returncode=1,
            stderr=b"Failed to connect to bus\n",
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to connect to bus"):
            codex_pool._ensure_systemd_worker_slice()

    @override_settings(CODEX_PARENT_SLICE_CPU_WEIGHT="20")
    def test_parent_slice_properties_default(self) -> None:
        self.assertEqual(
            codex_pool._systemd_parent_slice_properties(), ["CPUWeight=20"]
        )

    @override_settings(CODEX_PARENT_SLICE_CPU_WEIGHT="")
    def test_parent_slice_properties_cleared_resets_to_default(self) -> None:
        # A cleared weight must render declaratively as the cgroup-v2 default so
        # a previously-applied --runtime weight doesn't linger on the slice.
        self.assertEqual(
            codex_pool._systemd_parent_slice_properties(), ["CPUWeight=100"]
        )

    @override_settings(CODEX_PARENT_SLICE_CPU_WEIGHT="500")
    def test_parent_slice_properties_passes_arbitrary_valid_weight(self) -> None:
        self.assertEqual(
            codex_pool._systemd_parent_slice_properties(), ["CPUWeight=500"]
        )

    @override_settings(CODEX_PARENT_SLICE_CPU_WEIGHT="20000")
    def test_parent_slice_properties_drops_out_of_range_weight(self) -> None:
        # Out of the cgroup-v2 1..10000 range: dropped (slice keeps its current
        # weight) with a loud warning rather than failing every worker spawn.
        with self.assertLogs("hitch.main.codex_pool", level="WARNING"):
            self.assertEqual(codex_pool._systemd_parent_slice_properties(), [])

    @override_settings(CODEX_PARENT_SLICE_CPU_WEIGHT="heavy")
    def test_parent_slice_properties_drops_non_integer_weight(self) -> None:
        with self.assertLogs("hitch.main.codex_pool", level="WARNING"):
            self.assertEqual(codex_pool._systemd_parent_slice_properties(), [])

    @override_settings(
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
        CODEX_WORKER_SLICE_MEMORY_HIGH="8G",
        CODEX_WORKER_SLICE_MEMORY_MAX="10G",
        CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0",
        CODEX_PARENT_SLICE="hitch.slice",
        CODEX_PARENT_SLICE_CPU_WEIGHT="20",
    )
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_biases_parent_slice_after_workers_slice(
        self, mock_run: MagicMock, _mock_which: MagicMock
    ) -> None:
        # The CPU weight belongs on the parent slice (sibling to the runserver's
        # slice), applied after the workers-slice memory config. Setting it on
        # the workers slice would be inert — workers have no siblings there.
        mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

        codex_pool._ensure_systemd_worker_slice()

        self.assertEqual(mock_run.call_count, 2)
        workers_argv = mock_run.call_args_list[0].args[0]
        parent_argv = mock_run.call_args_list[1].args[0]
        self.assertEqual(workers_argv[4], "hitch-codex-workers.slice")
        self.assertEqual(
            parent_argv,
            [
                "/usr/bin/systemctl",
                "--user",
                "set-property",
                "--runtime",
                "hitch.slice",
                "CPUWeight=20",
            ],
        )

    @override_settings(
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
        CODEX_WORKER_SLICE_MEMORY_HIGH="8G",
        CODEX_WORKER_SLICE_MEMORY_MAX="10G",
        CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0",
        CODEX_PARENT_SLICE="",
    )
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_parent_slice_configuration_skipped_when_disabled(
        self, mock_run: MagicMock, _mock_which: MagicMock
    ) -> None:
        # An empty CODEX_PARENT_SLICE disables the CPU-weight bias entirely
        # (deployments not under systemd-user) while the workers slice is still
        # configured — only the one set-property call fires.
        mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

        codex_pool._ensure_systemd_worker_slice()

        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(
            mock_run.call_args.args[0][4], "hitch-codex-workers.slice"
        )

    @patch("hitch.main.codex_pool.shutil.which", return_value=None)
    def test_apply_slice_properties_requires_systemctl(
        self, _mock_which: MagicMock
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "systemctl is required"):
            codex_pool._apply_slice_properties("hitch.slice", ["CPUWeight=20"])

    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run", side_effect=OSError("boom"))
    def test_apply_slice_properties_wraps_exec_failure(
        self, _mock_run: MagicMock, _mock_which: MagicMock
    ) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "failed to configure Codex slice hitch.slice"
        ):
            codex_pool._apply_slice_properties("hitch.slice", ["CPUWeight=20"])

    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_apply_slice_properties_reports_status_without_detail(
        self, mock_run: MagicMock, _mock_which: MagicMock
    ) -> None:
        # A non-zero exit with no stderr still surfaces a fatal error naming the
        # exit status rather than failing silently.
        mock_run.return_value = SimpleNamespace(returncode=2, stderr=b"")

        with self.assertRaisesRegex(RuntimeError, "exited with status 2"):
            codex_pool._apply_slice_properties("hitch.slice", ["CPUWeight=20"])

    @override_settings(CODEX_WORKER_ISOLATION="systemd")
    @patch("hitch.main.codex_pool.shutil.which", return_value=None)
    def test_systemd_launch_fails_closed_when_systemd_run_missing(
        self, mock_which: MagicMock
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "systemd-run is required"):
            codex_pool._launch_worker_process(instance_id=7)

        mock_which.assert_called_once_with("systemd-run")

    @override_settings(CODEX_WORKER_ISOLATION="auto")
    @patch("hitch.main.codex_pool._systemd_user_manager_available", return_value=False)
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemd-run")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_auto_launch_falls_back_to_direct_when_user_manager_unavailable(
        self,
        mock_popen: MagicMock,
        mock_which: MagicMock,
        mock_user_manager: MagicMock,
    ) -> None:
        mock_popen.return_value = SimpleNamespace(pid=999)

        launch = codex_pool._launch_worker_process(instance_id=7)

        argv = mock_popen.call_args.args[0]
        self.assertEqual(launch.pid, 999)
        self.assertEqual(launch.scope_unit, "")
        self.assertEqual(argv[2], "codex_worker")
        mock_which.assert_called_once_with("systemd-run")
        mock_user_manager.assert_called_once_with()

    @override_settings(CODEX_WORKER_ISOLATION="auto")
    @patch("hitch.main.codex_pool._systemd_user_manager_available", return_value=True)
    @patch("hitch.main.codex_pool._ensure_systemd_worker_slice")
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemd-run")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_auto_launch_uses_systemd_when_user_manager_available(
        self,
        mock_popen: MagicMock,
        _mock_which: MagicMock,
        mock_ensure_slice: MagicMock,
        mock_user_manager: MagicMock,
    ) -> None:
        proc = MagicMock()
        proc.pid = 999
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        launch = codex_pool._launch_worker_process(instance_id=7)

        argv = mock_popen.call_args.args[0]
        self.assertEqual(launch.pid, 0)
        self.assertEqual(launch.scope_unit, "hitch-codex-worker-7.service")
        self.assertEqual(argv[0], "/usr/bin/systemd-run")
        mock_user_manager.assert_called_once_with()
        mock_ensure_slice.assert_called_once_with()

    @override_settings(CODEX_WORKER_ISOLATION="auto")
    @patch("hitch.main.codex_pool._systemd_user_manager_available", return_value=True)
    @patch("hitch.main.codex_pool._ensure_systemd_worker_slice")
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemd-run")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_auto_launch_fails_closed_when_systemd_run_fails(
        self,
        mock_popen: MagicMock,
        _mock_which: MagicMock,
        _mock_ensure_slice: MagicMock,
        _mock_user_manager: MagicMock,
    ) -> None:
        failed_proc = MagicMock()
        failed_proc.wait.return_value = 1

        def popen_side_effect(*_args: object, **kwargs: object) -> object:
            stderr = cast(Any, kwargs["stderr"])
            stderr.write(b"Failed to connect to bus\n")
            stderr.flush()
            return failed_proc

        mock_popen.side_effect = popen_side_effect

        with self.assertRaisesRegex(RuntimeError, "Failed to connect to bus"):
            codex_pool._launch_worker_process(instance_id=7)

        self.assertEqual(mock_popen.call_count, 1)

    @override_settings(CODEX_WORKER_ISOLATION="systemd")
    @patch("hitch.main.codex_pool._ensure_systemd_worker_slice")
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemd-run")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_systemd_launch_fails_promptly_when_systemd_run_exits_nonzero(
        self,
        mock_popen: MagicMock,
        _mock_which: MagicMock,
        _mock_ensure_slice: MagicMock,
    ) -> None:
        proc = MagicMock()
        proc.wait.return_value = 1

        def popen_side_effect(*_args: object, **kwargs: object) -> MagicMock:
            stderr = cast(Any, kwargs["stderr"])
            stderr.write(b"Failed to connect to bus: Operation not permitted\n")
            stderr.flush()
            return proc

        mock_popen.side_effect = popen_side_effect

        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to connect to bus: Operation not permitted",
        ):
            codex_pool._launch_worker_process(instance_id=7)

        proc.wait.assert_called_once_with(timeout=0.25)

    @override_settings(CODEX_WORKER_ISOLATION="bogus")
    def test_rejects_unknown_worker_isolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid CODEX_WORKER_ISOLATION"):
            codex_pool._launch_worker_process(instance_id=7)

    @override_settings(CODEX_WORKER_ISOLATION="direct")
    @patch("hitch.main.codex_pool.subprocess.Popen")
    def test_forwards_optional_cli_args(
        self, mock_popen: MagicMock
    ) -> None:
        cases: list[tuple[dict[str, Any], list[tuple[str, str | None]]]] = [
            ({"reasoning_effort": "high"}, [("--reasoning-effort", "high")]),
            (
                {"sandbox_policy": "workspaceWrite"},
                [("--sandbox-policy", "workspaceWrite")],
            ),
            ({"approval_mode": "deny_all"}, [("--approval-mode", "deny_all")]),
            ({"web_search_mode": "live"}, [("--web-search-mode", "live")]),
            ({"enable_memories": True}, [("--enable-memories", None)]),
            (
                {"collaboration_mode": "default"},
                [("--collaboration-mode", "default")],
            ),
            (
                {"model": "gpt-5.4", "plan_mode": True},
                [("--model", "gpt-5.4"), ("--plan-mode", None)],
            ),
        ]
        for kwargs, expected_args in cases:
            with self.subTest(kwargs=kwargs):
                mock_popen.reset_mock()
                mock_popen.return_value = SimpleNamespace(pid=999)
                codex_pool._launch_worker_process(instance_id=7, **kwargs)

                argv = mock_popen.call_args.args[0]
                for flag, expected_value in expected_args:
                    self.assertIn(flag, argv)
                    if expected_value is not None:
                        self.assertEqual(argv[argv.index(flag) + 1], expected_value)


class SwapCapHierarchyWarningTests(TestCase):
    """The per-worker swap cushion is meaningless if the parent slice forbids
    swap: cgroup v2 swap limits are hierarchical, so a worker can never exceed
    its enclosing slice's MemorySwapMax. Raising only the worker setting is a
    silent no-op, so _warn_on_swap_cap_hierarchy surfaces the mismatch.
    """

    BASE = {
        "CODEX_WORKER_MEMORY_MAX": "4G",
        "CODEX_WORKER_SLICE_MEMORY_MAX": "10G",
    }

    @override_settings(
        CODEX_WORKER_MEMORY_SWAP_MAX="1G", CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0"
    )
    def test_warns_when_zero_slice_cap_nullifies_worker_cushion(self) -> None:
        with (
            override_settings(**self.BASE),
            self.assertLogs("hitch.main.codex_pool", level="WARNING") as logs,
        ):
            codex_pool._warn_on_swap_cap_hierarchy()
        self.assertTrue(
            any("hierarchical" in line and "1G" in line for line in logs.output),
            logs.output,
        )

    @override_settings(
        CODEX_WORKER_MEMORY_SWAP_MAX="", CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0"
    )
    def test_warns_when_worker_opts_out_but_slice_still_caps(self) -> None:
        # Clearing the worker swap cap (with a hard MemoryMax still set) reads
        # as opting the worker out of a per-scope cap, but the slice still
        # enforces MemorySwapMax=0, so the worker gets no swap regardless. Warn
        # rather than letting the cleared setting silently do nothing.
        with (
            override_settings(**self.BASE),
            self.assertLogs("hitch.main.codex_pool", level="WARNING") as logs,
        ):
            codex_pool._warn_on_swap_cap_hierarchy()
        self.assertTrue(
            any("uncapped" in line and "hierarchical" in line for line in logs.output),
            logs.output,
        )

    @override_settings(
        CODEX_WORKER_MEMORY_SWAP_MAX="1G", CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="512M"
    )
    def test_warns_when_slice_cap_smaller_than_worker_cushion(self) -> None:
        # 512M < 1G, so the parent still clips the per-worker cushion even
        # though it is not a hard zero.
        with (
            override_settings(**self.BASE),
            self.assertLogs("hitch.main.codex_pool", level="WARNING") as logs,
        ):
            codex_pool._warn_on_swap_cap_hierarchy()
        self.assertTrue(any("512M" in line for line in logs.output), logs.output)

    @override_settings(
        CODEX_WORKER_MEMORY_SWAP_MAX="1G", CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="2G"
    )
    def test_silent_when_slice_cap_accommodates_worker_cushion(self) -> None:
        with (
            override_settings(**self.BASE),
            self.assertNoLogs("hitch.main.codex_pool", level="WARNING"),
        ):
            codex_pool._warn_on_swap_cap_hierarchy()

    @override_settings(
        CODEX_WORKER_MEMORY_SWAP_MAX="1G", CODEX_WORKER_SLICE_MEMORY_SWAP_MAX=""
    )
    def test_silent_when_slice_swap_is_unlimited(self) -> None:
        # An empty slice cap leaves the parent's swap unlimited, so the worker
        # cushion applies unhindered and there is nothing to warn about.
        with (
            override_settings(**self.BASE),
            self.assertNoLogs("hitch.main.codex_pool", level="WARNING"),
        ):
            codex_pool._warn_on_swap_cap_hierarchy()

    @override_settings(
        CODEX_WORKER_MEMORY_SWAP_MAX="0", CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0"
    )
    def test_silent_for_fail_fast_defaults(self) -> None:
        # The shipped 0/0 default forbids swap everywhere by design; there is no
        # cushion to defend, so it must not warn.
        with (
            override_settings(**self.BASE),
            self.assertNoLogs("hitch.main.codex_pool", level="WARNING"),
        ):
            codex_pool._warn_on_swap_cap_hierarchy()

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="2G",
        CODEX_WORKER_MEMORY_MAX="",
        CODEX_WORKER_MEMORY_SWAP_MAX="1G",
        CODEX_WORKER_SLICE_MEMORY_MAX="10G",
        CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0",
    )
    def test_silent_when_worker_is_soft_only_throttle(self) -> None:
        # A high-only worker has no hard MemoryMax, so its swap cap is never
        # emitted (mirrors _memory_cgroup_properties) and there is no effective
        # cushion for the slice to nullify — nothing to warn about.
        with self.assertNoLogs("hitch.main.codex_pool", level="WARNING"):
            codex_pool._warn_on_swap_cap_hierarchy()

    @override_settings(
        CODEX_WORKER_MEMORY_HIGH="",
        CODEX_WORKER_MEMORY_MAX="",
        CODEX_WORKER_MEMORY_SWAP_MAX="1G",
        CODEX_WORKER_SLICE_MEMORY_MAX="10G",
        CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0",
    )
    def test_silent_when_worker_has_no_memory_limit(self) -> None:
        # Without a worker memory limit the swap cap is never emitted onto the
        # scope (mirrors _memory_cgroup_properties), so there is no cushion that
        # the slice could nullify.
        with self.assertNoLogs("hitch.main.codex_pool", level="WARNING"):
            codex_pool._warn_on_swap_cap_hierarchy()

    def test_parse_memory_bytes_handles_units_and_rejects_ambiguous(self) -> None:
        self.assertEqual(codex_pool._parse_memory_bytes("0"), 0)
        self.assertEqual(codex_pool._parse_memory_bytes("1024"), 1024)
        self.assertEqual(codex_pool._parse_memory_bytes("2G"), 2 * 1024**3)
        self.assertEqual(codex_pool._parse_memory_bytes("512m"), 512 * 1024**2)
        # Decimals, percentages, infinity, and empty are not comparable.
        for ambiguous in ("", "1.5G", "20%", "infinity"):
            self.assertIsNone(codex_pool._parse_memory_bytes(ambiguous), ambiguous)

    @override_settings(
        CODEX_WORKER_SLICE="hitch-codex-workers.slice",
        CODEX_WORKER_MEMORY_MAX="4G",
        CODEX_WORKER_SLICE_MEMORY_MAX="10G",
        CODEX_WORKER_MEMORY_SWAP_MAX="1G",
        CODEX_WORKER_SLICE_MEMORY_SWAP_MAX="0",
    )
    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    def test_ensure_slice_warns_once_per_process(
        self, mock_run: MagicMock, _mock_which: MagicMock
    ) -> None:
        mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")
        original = codex_pool._swap_hierarchy_warned
        codex_pool._swap_hierarchy_warned = False
        try:
            with self.assertLogs("hitch.main.codex_pool", level="WARNING") as logs:
                codex_pool._ensure_systemd_worker_slice()
                # A second launch must not re-emit the same warning.
                codex_pool._ensure_systemd_worker_slice()
                # assertLogs fails if nothing is logged, so emit a sentinel to
                # assert against rather than relying on the warning count alone.
                codex_pool.logger.warning("sentinel")
        finally:
            codex_pool._swap_hierarchy_warned = original
        hierarchy_warnings = [
            line for line in logs.output if "hierarchical" in line
        ]
        self.assertEqual(len(hierarchy_warnings), 1, logs.output)


class IsAliveTests(TestCase):
    def test_known_pid_states(self) -> None:
        self.assertTrue(codex_pool.is_alive(os.getpid()))
        self.assertFalse(codex_pool.is_alive(0))
        self.assertFalse(codex_pool.is_alive(-1))
        # 2**22 is well above the default pid_max on Linux/macOS.
        self.assertFalse(codex_pool.is_alive(2**22))

    @unittest.skipUnless(Path("/proc").exists(), "requires Linux /proc")
    def test_zombie_pid_is_not_alive(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import os; os._exit(0)"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_linux_proc_state(proc.pid, "Z")

            self.assertFalse(codex_pool.is_alive(proc.pid))
        finally:
            if proc.returncode is None:
                proc.wait(timeout=5)

    def test_linux_proc_state_tolerates_non_utf8_comm(self) -> None:
        # A recycled pid can point at a foreign process whose comm (executable
        # basename) holds arbitrary non-UTF-8 bytes. Reading its state must not
        # raise UnicodeDecodeError out of the liveness check.
        raw = b"1234 (we\xffird) R 1 0 0 0 -1"
        with patch("hitch.main.codex_pool.Path") as mock_path:
            proc_root = mock_path.return_value
            proc_root.exists.return_value = True
            stat_path = proc_root.__truediv__.return_value.__truediv__.return_value
            stat_path.read_bytes.return_value = raw
            self.assertEqual(codex_pool._linux_proc_state(1234), "R")

    def test_worker_is_alive_uses_reaped_instance_key(self) -> None:
        pid = 98765
        instance = CodexInstance.objects.create(
            pid=pid,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        try:
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                codex_pool._REAPED_WORKERS.add((pid, instance.pk))

            self.assertFalse(codex_pool.worker_is_alive(instance))
        finally:
            _forget_worker_pid(pid)

    def test_tracked_finished_worker_is_reaped_by_liveness_check(self) -> None:
        pid = 98766
        proc = MagicMock()
        proc.wait.return_value = 0
        try:
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                codex_pool._TRACKED_WORKER_PROCS[pid] = (
                    42,
                    cast(subprocess.Popen[bytes], proc),
                )

            self.assertFalse(codex_pool.is_alive(pid))

            proc.wait.assert_called_once_with(timeout=0)
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                self.assertNotIn(pid, codex_pool._TRACKED_WORKER_PROCS)
                self.assertIn((pid, 42), codex_pool._REAPED_WORKERS)
        finally:
            _forget_worker_pid(pid)

    def test_worker_is_alive_handles_unset_pid_and_tracked_running_process(self) -> None:
        self.assertFalse(codex_pool.worker_is_alive(CodexInstance(pid=0)))

        pid = 98767
        instance = CodexInstance.objects.create(
            pid=pid,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        proc = MagicMock()
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="worker", timeout=0)
        try:
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                codex_pool._TRACKED_WORKER_PROCS[pid] = (
                    instance.pk,
                    cast(subprocess.Popen[bytes], proc),
                )

            self.assertTrue(codex_pool.worker_is_alive(instance))

            proc.wait.assert_called_once_with(timeout=0)
        finally:
            _forget_worker_pid(pid)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=False)
    @patch("hitch.main.codex_pool.is_alive", return_value=True)
    def test_worker_is_alive_rejects_recycled_untracked_pid(
        self, mock_alive: MagicMock, mock_identity: MagicMock
    ) -> None:
        instance = CodexInstance.objects.create(
            pid=4321,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

        self.assertFalse(codex_pool.worker_is_alive(instance))
        mock_identity.assert_called_once_with(4321, instance.pk)
        mock_alive.assert_not_called()

    def test_linux_proc_state_defensive_branches(self) -> None:
        with patch("hitch.main.codex_pool.Path") as mock_path:
            proc_root = mock_path.return_value
            proc_root.exists.return_value = False
            self.assertIsNone(codex_pool._linux_proc_state(1))

        with patch("hitch.main.codex_pool.Path") as mock_path:
            proc_root = mock_path.return_value
            proc_root.exists.return_value = True
            stat_path = proc_root.__truediv__.return_value.__truediv__.return_value
            stat_path.read_bytes.side_effect = FileNotFoundError
            self.assertEqual(codex_pool._linux_proc_state(1), "")

        with patch("hitch.main.codex_pool.Path") as mock_path:
            proc_root = mock_path.return_value
            proc_root.exists.return_value = True
            stat_path = proc_root.__truediv__.return_value.__truediv__.return_value
            stat_path.read_bytes.return_value = b"malformed"
            self.assertIsNone(codex_pool._linux_proc_state(1))

    @patch("hitch.main.codex_pool.os.kill")
    def test_permission_error_means_alive(self, mock_kill: MagicMock) -> None:
        # ``os.kill(pid, 0)`` raises PermissionError when the pid exists but
        # is owned by another user; the process is still alive in that case.
        mock_kill.side_effect = PermissionError
        self.assertTrue(codex_pool.is_alive(1234))

    @patch("hitch.main.codex_pool.os.kill")
    def test_other_os_error_is_treated_as_dead(self, mock_kill: MagicMock) -> None:
        mock_kill.side_effect = OSError
        self.assertFalse(codex_pool.is_alive(1234))


class CodexInstanceModelTests(TestCase):
    def test_str_includes_pid_thread_and_status(self) -> None:
        instance = CodexInstance(
            pid=42,
            thread_id="abc",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        rendered = str(instance)
        self.assertIn("pid=42", rendered)
        self.assertIn("thread_id=abc", rendered)
        self.assertIn("status=running", rendered)


class ApprovalRequestModelTests(TestCase):
    def test_str_surfaces_method_and_pending_state(self) -> None:
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )
        approval = ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "ls"}},
        )
        rendered = str(approval)
        self.assertIn(f"pk={approval.pk}", rendered)
        self.assertIn("method=item/commandExecution/requestApproval", rendered)
        # Empty decision renders as "pending" so a glance at logs/admin
        # makes the row's open state obvious.
        self.assertIn("decision=pending", rendered)

        approval.decision = "accept"
        approval.save(update_fields=["decision"])
        self.assertIn("decision=accept", str(approval))


class ReconcileAndLookupTests(TestCase):
    def _make(
        self,
        *,
        pid: int = 1,
        thread_id: str = "t",
        status: str | None = None,
        purpose: str = CodexInstance.PURPOSE_USER,
        events_path: str = "/dev/null",
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id=thread_id,
            cwd="/r",
            events_path=events_path,
            status=status or CodexInstance.STATUS_COMPLETED,
            purpose=purpose,
        )

    @patch("hitch.main.codex_pool.worker_is_alive")
    def test_reconcile_marks_only_dead_pending_rows_failed(
        self, mock_worker_alive: MagicMock
    ) -> None:
        dead_running = self._make(pid=10, status=CodexInstance.STATUS_RUNNING)
        live_running = self._make(pid=11, status=CodexInstance.STATUS_RUNNING)
        completed = self._make(pid=12, status=CodexInstance.STATUS_COMPLETED)
        mock_worker_alive.side_effect = lambda instance: instance.pk == live_running.pk

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        dead_running.refresh_from_db()
        live_running.refresh_from_db()
        completed.refresh_from_db()
        self.assertEqual(dead_running.status, CodexInstance.STATUS_FAILED)
        self.assertIsNotNone(dead_running.ended_at)
        self.assertEqual(live_running.status, CodexInstance.STATUS_RUNNING)
        self.assertIsNone(live_running.ended_at)
        self.assertEqual(completed.status, CodexInstance.STATUS_COMPLETED)
        self.assertIn("exited", dead_running.error)

    @patch("hitch.main.disk_cleanup.run_finished_session_disk_cleanup")
    @patch("hitch.main.codex_pool.cleanup_requested_input_images_for")
    @patch("hitch.main.codex_pool._notify_system_agents_if_needed")
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_mark_dead_instances_failed_does_not_fan_out_disk_cleanup(
        self,
        _mock_alive: MagicMock,
        _mock_notify: MagicMock,
        _mock_cleanup_images: MagicMock,
        mock_disk_cleanup: MagicMock,
    ) -> None:
        # Both the per-instance kill branch and the already-terminal branch of
        # _mark_dead_instances_failed used to fan out a full ~/.hitch walk per
        # row. A single reconcile sweep must trigger zero disk cleanups.
        dead = self._make(pid=30, status=CodexInstance.STATUS_RUNNING)
        late = self._make(pid=31, status=CodexInstance.STATUS_RUNNING)
        pending = list(CodexInstance.objects.filter(pk__in=[dead.pk, late.pk]))
        CodexInstance.objects.filter(pk=late.pk).update(
            status=CodexInstance.STATUS_COMPLETED
        )

        n = codex_pool._mark_dead_instances_failed(pending)

        self.assertEqual(n, 1)
        mock_disk_cleanup.assert_not_called()

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_dead_records_last_auto_approval_timeout(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = Path(raw) / "events.jsonl"
            events_path.write_text(
                json.dumps(
                    {
                        "method": "item/autoApprovalReview/completed",
                        "payload": {
                            "action": {"command": "pkill -f target/release/bench"},
                            "review": {
                                "status": "timedOut",
                                "rationale": (
                                    "Automatic approval review timed out while "
                                    "evaluating the requested approval."
                                ),
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instance = self._make(
                pid=13,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(events_path),
            )

            n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn(
            "worker process exited before reporting completion", instance.error
        )
        self.assertIn("auto-approval review timed out", instance.error)
        self.assertIn("pkill -f target/release/bench", instance.error)

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_dead_records_pending_auto_approval(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = Path(raw) / "events.jsonl"
            events_path.write_text(
                json.dumps(
                    {
                        "method": "item/started",
                        "payload": {
                            "item": {
                                "type": "commandExecution",
                                "status": "inProgress",
                                "command": "/bin/bash -lc 'just test'",
                            },
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "method": "item/autoApprovalReview/started",
                        "payload": {
                            "action": {"command": "/bin/bash -lc 'just test'"},
                            "review": {"status": "inProgress"},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instance = self._make(
                pid=14,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(events_path),
            )

            n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn(
            "worker process exited before reporting completion", instance.error
        )
        self.assertIn("auto-approval review in progress", instance.error)
        self.assertIn("/bin/bash -lc 'just test'", instance.error)

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_dead_ignores_completed_auto_approval_start(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = Path(raw) / "events.jsonl"
            action = {"command": "/bin/bash -lc 'just test'"}
            events_path.write_text(
                json.dumps(
                    {
                        "method": "item/autoApprovalReview/started",
                        "payload": {
                            "action": action,
                            "review": {"status": "inProgress"},
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "method": "item/autoApprovalReview/completed",
                        "payload": {
                            "action": action,
                            "review": {"status": "approved"},
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "type": "agentMessage",
                                "text": "Continuing after approved command.",
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instance = self._make(
                pid=15,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(events_path),
            )

            n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("agent last said: Continuing after approved command.", instance.error)
        self.assertNotIn("auto-approval review in progress", instance.error)

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_dead_records_last_agent_message(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = Path(raw) / "events.jsonl"
            events_path.write_text(
                json.dumps(
                    {
                        "method": "item/completed",
                        "payload": {
                            "item": {
                                "type": "agentMessage",
                                "text": "Running the base side of the paired measurement now.",
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instance = self._make(
                pid=15,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(events_path),
            )

            n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn(
            "worker process exited before reporting completion", instance.error
        )
        self.assertIn(
            "agent last said: Running the base side of the paired measurement now.",
            instance.error,
        )

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_dead_records_worker_log_tail(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=Path(raw) / "logs"),
        ):
            instance = self._make(
                pid=16,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "missing.jsonl"),
            )
            log_path = codex_pool.worker_log_path(instance.pk)
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                "worker starting\n"
                "Traceback (most recent call last):\n"
                "SystemExit: transport closed\n",
                encoding="utf-8",
            )

            n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn(
            "worker process exited before reporting completion", instance.error
        )
        self.assertIn("worker log:", instance.error)
        self.assertIn("SystemExit: transport closed", instance.error)

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_dead_ignores_worker_log_tail_when_test_logging_disabled(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(HITCH_HOME_DIR=Path(raw), CODEX_WORKER_LOG_DIR=None),
        ):
            instance = self._make(
                pid=17,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "missing.jsonl"),
            )
            log_path = codex_pool.worker_log_path(instance.pk)
            log_path.parent.mkdir(parents=True)
            log_path.write_text("stale host log\n", encoding="utf-8")

            n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(
            instance.error,
            "worker process exited before reporting completion",
        )

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_cancels_pending_requests_of_dead_worker(
        self, _mock_alive: MagicMock
    ) -> None:
        # A worker that dies mid-approval leaves the ApprovalRequest/UserInputRequest
        # rows pending. Nothing alive will ever consume them, so the reconcile sweep
        # must cancel them — otherwise the browser keeps an actionable card for a
        # dead turn and a click returns 200 while the decision is silently dropped.
        instance = self._make(pid=30, status=CodexInstance.STATUS_RUNNING)
        approval = ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={},
        )
        input_request = UserInputRequest.objects.create(
            instance=instance,
            method="item/userInput/request",
            params={},
        )

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        approval.refresh_from_db()
        input_request.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(approval.decision, ApprovalRequest.DECISION_CANCEL)
        self.assertIsNotNone(approval.decided_at)
        self.assertEqual(input_request.response, {"answers": {}})
        self.assertIsNotNone(input_request.responded_at)

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_preserves_resolved_request_of_dead_worker(
        self, _mock_alive: MagicMock
    ) -> None:
        # A decision the user already made (and the worker may have consumed) must
        # not be retroactively overwritten with "cancel" by the reconcile sweep.
        instance = self._make(pid=31, status=CodexInstance.STATUS_RUNNING)
        approval = ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={},
            decision=ApprovalRequest.DECISION_ACCEPT,
        )

        codex_pool.reconcile_dead()

        approval.refresh_from_db()
        self.assertEqual(approval.decision, ApprovalRequest.DECISION_ACCEPT)

    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_preserves_row_that_completed_during_sweep(
        self, _mock_alive: MagicMock
    ) -> None:
        # The worker can write its terminal status in the gap between the
        # pending-row read and the failed-marking write; the conditional UPDATE
        # must not retroactively rewrite that completed turn as failed.
        instance = self._make(pid=20, status=CodexInstance.STATUS_RUNNING)
        pending = list(CodexInstance.objects.filter(pk=instance.pk))
        CodexInstance.objects.filter(pk=instance.pk).update(
            status=CodexInstance.STATUS_COMPLETED
        )

        n = codex_pool._mark_dead_instances_failed(pending)

        instance.refresh_from_db()
        self.assertEqual(n, 0)
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(instance.error, "")
        self.assertIsNone(instance.ended_at)

    @patch("hitch.main.codex_pool.cleanup_requested_input_images_for")
    @patch("hitch.main.codex_pool._notify_system_agents_if_needed")
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_still_routes_instance_that_completed_during_sweep(
        self,
        _mock_alive: MagicMock,
        mock_notify: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        # A worker that saved its terminal status but died before its own notify
        # must still be routed by the reclaim sweep — demo system-agent rows are
        # excluded from reconcile_terminal_workflow_instances() and have no other
        # backstop, so skipping routing here would strand them.
        instance = self._make(pid=21, status=CodexInstance.STATUS_RUNNING)
        pending = list(CodexInstance.objects.filter(pk=instance.pk))
        CodexInstance.objects.filter(pk=instance.pk).update(
            status=CodexInstance.STATUS_COMPLETED
        )

        n = codex_pool._mark_dead_instances_failed(pending)

        self.assertEqual(n, 0)
        mock_notify.assert_called_once()
        routed = mock_notify.call_args.args[0]
        self.assertEqual(routed.pk, instance.pk)
        self.assertEqual(routed.status, CodexInstance.STATUS_COMPLETED)

    def test_reconcile_spares_freshly_spawned_worker_with_unassigned_pid(self) -> None:
        # ``_spawn_worker`` commits the CodexInstance row with ``pid=0`` before
        # ``subprocess.Popen`` returns. Between those two writes, reconcile_dead
        # — invoked on essentially every page render — must not interpret the
        # row's transient unassigned pid as "worker process exited"; doing so
        # poisons a completing turn with a leftover error message and prematurely
        # routes system-agent workers through their workflow's failure handler.
        starting = self._make(pid=0, status=CodexInstance.STATUS_STARTING)

        n = codex_pool.reconcile_dead()

        starting.refresh_from_db()
        self.assertEqual(starting.status, CodexInstance.STATUS_STARTING)
        self.assertEqual(starting.error, "")
        self.assertIsNone(starting.ended_at)
        self.assertEqual(n, 0)

    def test_reconcile_fails_stale_unassigned_pid_after_grace_window(self) -> None:
        # The pid=0 reprieve must be bounded: if the Django parent crashed
        # between row commit and pid assignment, the orphaned row would
        # otherwise pin the session in ``starting`` forever. After the launch
        # grace window expires, reconcile_dead reclaims it.
        instance = self._make(pid=0, status=CodexInstance.STATUS_STARTING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            started_at=timezone.now() - timedelta(minutes=10)
        )

        n = codex_pool.reconcile_dead()

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(n, 1)

    @patch("hitch.main.system_agents.reconcile_terminal_workflow_instances")
    @patch("hitch.main.system_agents.on_codex_instance_finished", return_value=True)
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_dead_for_workflow_scopes_pending_rows(
        self,
        _mock_worker_alive: MagicMock,
        _mock_notify: MagicMock,
        mock_reconcile: MagicMock,
    ) -> None:
        target_workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )
        other_workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="other-thread",
            cwd="/repo",
        )
        target = self._make(
            pid=10,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        )
        other = self._make(
            pid=11,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
        )
        CodexInstance.objects.filter(pk=target.pk).update(
            workflow_id=target_workflow.pk
        )
        CodexInstance.objects.filter(pk=other.pk).update(workflow_id=other_workflow.pk)

        n = codex_pool.reconcile_dead_for_workflow(
            target_workflow.pk,
            main_thread_id=target_workflow.main_thread_id,
        )

        self.assertEqual(n, 1)
        target.refresh_from_db()
        other.refresh_from_db()
        self.assertEqual(target.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(other.status, CodexInstance.STATUS_RUNNING)
        mock_reconcile.assert_called_once_with(
            main_thread_id=target_workflow.main_thread_id,
            workflow_id=target_workflow.pk,
        )

    @patch("hitch.main.codex_pool.reconcile_orphaned_workers", return_value=0)
    @patch("hitch.main.system_agents.reconcile_terminal_workflow_instances")
    def test_reconcile_dead_for_workflow_reaps_orphans(
        self, _mock_reconcile: MagicMock, mock_reap: MagicMock
    ) -> None:
        # A hidden workflow stream may be the only thing reconciling, so it must
        # still run the orphan reap to free a wedged worker's DB lock.
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="main-thread",
            cwd="/repo",
        )

        codex_pool.reconcile_dead_for_workflow(
            workflow.pk, main_thread_id=workflow.main_thread_id
        )

        mock_reap.assert_called_once()

    @patch("hitch.main.system_agents.on_codex_instance_finished")
    def test_reconcile_does_not_notify_system_agents_for_unassigned_pid(
        self, mock_notify: MagicMock
    ) -> None:
        # A system-agent worker in its pid=0 launch window must not be routed
        # through ``on_codex_instance_finished``. The autonomous-goal and
        # PR-QA workflows treat a system-agent ``finished`` event as terminal
        # and block on the recorded error — so the workflow would be marked
        # failed before its worker subprocess has even started running.
        self._make(
            pid=0,
            status=CodexInstance.STATUS_STARTING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )

        codex_pool.reconcile_dead()

        mock_notify.assert_not_called()

    @patch("hitch.main.codex_pool.worker_is_alive")
    def test_reconcile_dead_retains_pending_attachments(
        self, mock_worker_alive: MagicMock
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            dead_path = codex_pool.input_attachments_dir() / "dead" / "1.png"
            live_path = codex_pool.input_attachments_dir() / "live" / "1.png"
            completed_path = codex_pool.input_attachments_dir() / "done" / "1.png"
            for path in (dead_path, live_path, completed_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            dead_running = self._make(
                pid=10,
                status=CodexInstance.STATUS_RUNNING,
            )
            live_running = self._make(
                pid=11,
                status=CodexInstance.STATUS_RUNNING,
            )
            completed = self._make(
                pid=12,
                status=CodexInstance.STATUS_COMPLETED,
            )
            CodexInstance.objects.filter(pk=dead_running.pk).update(
                input_image_paths=[str(dead_path)],
                input_attachment_paths=[str(dead_path)],
            )
            CodexInstance.objects.filter(pk=live_running.pk).update(
                input_attachment_paths=[str(live_path)]
            )
            CodexInstance.objects.filter(pk=completed.pk).update(
                input_attachment_paths=[str(completed_path)]
            )
            mock_worker_alive.side_effect = (
                lambda instance: instance.pk == live_running.pk
            )

            n = codex_pool.reconcile_dead()

            self.assertEqual(n, 1)
            self.assertTrue(dead_path.exists())
            self.assertTrue(live_path.exists())
            self.assertTrue(completed_path.exists())
            dead_running.refresh_from_db()
            live_running.refresh_from_db()
            completed.refresh_from_db()
            self.assertEqual(dead_running.input_image_paths, [str(dead_path)])
            self.assertEqual(dead_running.input_attachment_paths, [str(dead_path)])
            self.assertEqual(live_running.input_attachment_paths, [str(live_path)])
            self.assertEqual(completed.input_attachment_paths, [str(completed_path)])

    def test_cleanup_keeps_attachment_ledger_for_unlink_failures(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            image_path = codex_pool.input_attachments_dir() / "busy" / "1.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"image")
            instance = self._make(status=CodexInstance.STATUS_FAILED)
            CodexInstance.objects.filter(pk=instance.pk).update(
                input_image_paths=[str(image_path)],
                input_attachment_paths=[str(image_path)],
            )

            with patch.object(Path, "unlink", side_effect=OSError("busy")):
                codex_pool.cleanup_input_images_for(instance)

            instance.refresh_from_db()
            self.assertEqual(instance.input_image_paths, [])
            self.assertEqual(instance.input_attachment_paths, [str(image_path)])
            self.assertTrue(instance.input_attachment_cleanup_requested)
            self.assertTrue(image_path.exists())

    def test_cleanup_preserves_concurrently_added_attachment(self) -> None:
        # A steer image can land between the candidate read and the ledger
        # write; the locked read-modify-write must not clobber it.
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            attachments = codex_pool.input_attachments_dir()
            path_a = attachments / "a" / "1.png"
            path_b = attachments / "b" / "1.png"
            for path in (path_a, path_b):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            instance = self._make(status=CodexInstance.STATUS_FAILED)
            CodexInstance.objects.filter(pk=instance.pk).update(
                input_attachment_paths=[str(path_a)],
            )

            injected = {"done": False}

            def _inject_concurrent_add(*args: Any, **kwargs: Any) -> None:
                if not injected["done"]:
                    injected["done"] = True
                    CodexInstance.objects.filter(pk=instance.pk).update(
                        input_attachment_paths=[str(path_a), str(path_b)],
                    )
                return None

            with patch.object(Path, "unlink", side_effect=_inject_concurrent_add):
                codex_pool.cleanup_input_images_for(instance)

            instance.refresh_from_db()
            self.assertEqual(instance.input_attachment_paths, [str(path_b)])
            self.assertTrue(instance.input_attachment_cleanup_requested)

    def test_retry_failed_input_image_cleanups_retries_requested_rows(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            retry_path = codex_pool.input_attachments_dir() / "retry" / "1.png"
            retained_path = codex_pool.input_attachments_dir() / "retained" / "1.png"
            for path in (retry_path, retained_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            retry_instance = self._make(status=CodexInstance.STATUS_FAILED)
            retained_instance = self._make(status=CodexInstance.STATUS_COMPLETED)
            CodexInstance.objects.filter(pk=retry_instance.pk).update(
                input_attachment_paths=[str(retry_path)],
                input_attachment_cleanup_requested=True,
            )
            CodexInstance.objects.filter(pk=retained_instance.pk).update(
                input_image_paths=[str(retained_path)],
                input_attachment_paths=[str(retained_path)],
            )

            retried = codex_pool.retry_failed_input_image_cleanups()

            self.assertEqual(retried, 1)
            self.assertFalse(retry_path.exists())
            self.assertTrue(retained_path.exists())
            retry_instance.refresh_from_db()
            retained_instance.refresh_from_db()
            self.assertEqual(retry_instance.input_attachment_paths, [])
            self.assertFalse(retry_instance.input_attachment_cleanup_requested)
            self.assertEqual(
                retained_instance.input_attachment_paths, [str(retained_path)]
            )

    def test_cleanup_input_images_for_thread_deletes_retained_thread_images(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            target_path = codex_pool.input_attachments_dir() / "target" / "1.png"
            active_path = codex_pool.input_attachments_dir() / "active" / "1.png"
            other_path = codex_pool.input_attachments_dir() / "other" / "1.png"
            for path in (target_path, active_path, other_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            target = self._make(thread_id="target", status=CodexInstance.STATUS_COMPLETED)
            active = self._make(thread_id="target", status=CodexInstance.STATUS_RUNNING)
            other = self._make(thread_id="other", status=CodexInstance.STATUS_COMPLETED)
            CodexInstance.objects.filter(pk=target.pk).update(
                input_image_paths=[str(target_path)],
                input_attachment_paths=[str(target_path)],
            )
            CodexInstance.objects.filter(pk=active.pk).update(
                input_image_paths=[str(active_path)],
                input_attachment_paths=[str(active_path)],
            )
            CodexInstance.objects.filter(pk=other.pk).update(
                input_image_paths=[str(other_path)],
                input_attachment_paths=[str(other_path)],
            )

            codex_pool.cleanup_input_images_for_thread("target")

            self.assertFalse(target_path.exists())
            self.assertTrue(active_path.exists())
            self.assertTrue(other_path.exists())
            target.refresh_from_db()
            active.refresh_from_db()
            other.refresh_from_db()
            self.assertEqual(target.input_image_paths, [])
            self.assertEqual(target.input_attachment_paths, [])
            self.assertFalse(target.input_attachment_cleanup_requested)
            self.assertEqual(active.input_attachment_paths, [str(active_path)])
            self.assertTrue(active.input_attachment_cleanup_requested)
            self.assertEqual(other.input_attachment_paths, [str(other_path)])

    def test_cleanup_keeps_attachment_ledger_for_paths_outside_root(self) -> None:
        outside_path = "/tmp/not-hitch-input.png"
        instance = self._make(
            status=CodexInstance.STATUS_FAILED,
        )
        CodexInstance.objects.filter(pk=instance.pk).update(
            input_image_paths=[outside_path],
            input_attachment_paths=[outside_path],
        )

        with tempfile.TemporaryDirectory() as raw, override_settings(
            CODEX_EVENTS_DIR=Path(raw)
        ):
            codex_pool.cleanup_input_images_for(instance)

        instance.refresh_from_db()
        self.assertEqual(instance.input_image_paths, [])
        self.assertEqual(instance.input_attachment_paths, [outside_path])
        self.assertTrue(instance.input_attachment_cleanup_requested)

    @patch("hitch.main.system_agents.on_codex_instance_finished")
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_notifies_system_agents_for_dead_system_rows(
        self, _mock_worker_alive: MagicMock, mock_notify: MagicMock
    ) -> None:
        system_agent = self._make(
            pid=10,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        mock_notify.assert_called_once()
        notified = mock_notify.call_args.args[0]
        self.assertEqual(notified.pk, system_agent.pk)
        self.assertEqual(notified.status, CodexInstance.STATUS_FAILED)

    @patch("hitch.main.demo.on_codex_instance_finished")
    @patch("hitch.main.system_agents.on_codex_instance_finished")
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_does_not_double_route_demo_system_agent(
        self,
        _mock_worker_alive: MagicMock,
        mock_system_notify: MagicMock,
        mock_demo_notify: MagicMock,
    ) -> None:
        mock_system_notify.return_value = True
        system_agent = self._make(
            pid=10,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        system_agent.agent_kind = demo.DEMO_AGENT_KIND
        system_agent.save(update_fields=["agent_kind"])

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        mock_system_notify.assert_called_once()
        mock_demo_notify.assert_not_called()

    @patch("hitch.main.demo.on_codex_instance_finished")
    @patch("hitch.main.system_agents.on_codex_instance_finished", return_value=False)
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_keeps_demo_fallback_when_system_agents_noop(
        self,
        _mock_worker_alive: MagicMock,
        mock_system_notify: MagicMock,
        mock_demo_notify: MagicMock,
    ) -> None:
        system_agent = self._make(
            pid=10,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        system_agent.agent_kind = demo.DEMO_AGENT_KIND
        system_agent.save(update_fields=["agent_kind"])

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        mock_system_notify.assert_called_once()
        mock_demo_notify.assert_called_once_with(system_agent)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=False)
    @patch("hitch.main.codex_pool.is_alive", return_value=True)
    def test_reconcile_marks_recycled_pid_failed(
        self, mock_alive: MagicMock, mock_identity: MagicMock
    ) -> None:
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        n = codex_pool.reconcile_dead()

        self.assertEqual(n, 1)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("exited", instance.error)
        mock_identity.assert_called_once_with(4321, instance.pk)
        mock_alive.assert_not_called()

    @unittest.skipUnless(Path("/proc").exists(), "requires Linux /proc")
    def test_tracked_exited_workers_are_reaped_and_reconciled(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import os; os._exit(0)"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        instance: CodexInstance | None = None
        try:
            instance = self._make(pid=proc.pid, status=CodexInstance.STATUS_RUNNING)
            codex_pool._track_worker_process(instance.pk, proc)
            _wait_for_process_exit(proc)

            n = codex_pool.reconcile_dead()

            self.assertEqual(n, 1)
            self.assertEqual(proc.returncode, 0)
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                self.assertNotIn(proc.pid, codex_pool._TRACKED_WORKER_PROCS)
                self.assertNotIn((proc.pid, instance.pk), codex_pool._REAPED_WORKERS)
            instance.refresh_from_db()
            self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
            self.assertIn("exited", instance.error)
        finally:
            if proc.returncode is None:
                proc.wait(timeout=5)
            _forget_worker_pid(proc.pid)

    def test_reconcile_sweeps_finished_tracked_workers(self) -> None:
        pid = 98768
        instance = self._make(pid=pid, status=CodexInstance.STATUS_RUNNING)
        proc = MagicMock()
        proc.wait.return_value = 0
        try:
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                codex_pool._TRACKED_WORKER_PROCS[pid] = (
                    instance.pk,
                    cast(subprocess.Popen[bytes], proc),
                )

            n = codex_pool.reconcile_dead()

            self.assertEqual(n, 1)
            proc.wait.assert_called_once_with(timeout=0)
            instance.refresh_from_db()
            self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
            with codex_pool._TRACKED_WORKER_PROCS_LOCK:
                self.assertNotIn(pid, codex_pool._TRACKED_WORKER_PROCS)
                self.assertNotIn((pid, instance.pk), codex_pool._REAPED_WORKERS)
        finally:
            _forget_worker_pid(pid)

    def test_list_and_latest_for_thread(self) -> None:
        first = self._make(thread_id="t1")
        second = self._make(thread_id="t1")
        self._make(thread_id="t2")

        self.assertEqual(
            [r.pk for r in codex_pool.list_for_thread("t1")],
            [second.pk, first.pk],
        )
        self.assertEqual(codex_pool.latest_for_thread("t1"), second)
        self.assertIsNone(codex_pool.latest_for_thread("nothing"))

    def test_latest_active_for_thread(self) -> None:
        # ``send_message`` can stack workers on a thread, so a newer
        # terminal row must not mask an older still-running one — the
        # streaming UI needs to stay up as long as *any* worker for the
        # thread is in progress.
        older = self._make(thread_id="t-active", status=CodexInstance.STATUS_RUNNING)
        newer_terminal = self._make(
            thread_id="t-active", status=CodexInstance.STATUS_FAILED
        )
        self.assertEqual(codex_pool.latest_active_for_thread("t-active"), older)

        # When multiple actives exist, return the newest by started_at.
        newer_active = self._make(
            thread_id="t-active", status=CodexInstance.STATUS_STARTING
        )
        self.assertEqual(codex_pool.latest_active_for_thread("t-active"), newer_active)

        # All terminal → None; never-existed thread → None.
        older.status = CodexInstance.STATUS_COMPLETED
        older.save(update_fields=["status"])
        newer_active.status = CodexInstance.STATUS_COMPLETED
        newer_active.save(update_fields=["status"])
        self.assertIsNone(codex_pool.latest_active_for_thread("t-active"))
        self.assertIsNone(codex_pool.latest_active_for_thread("never-existed"))
        # newer_terminal was already terminal; referenced here so it isn't
        # flagged as unused by future readers.
        self.assertEqual(newer_terminal.status, CodexInstance.STATUS_FAILED)


    def test_latest_id_for_thread(self) -> None:
        # ``latest_id_for_thread`` is the baseline the idle SSE stream
        # uses to spot any out-of-band turn — including fast turns that
        # complete between two polls. It must return the highest pk for
        # the thread regardless of status and ``None`` when the thread
        # has never had a worker.
        self.assertIsNone(codex_pool.latest_id_for_thread("never-existed"))
        first = self._make(thread_id="t-id", status=CodexInstance.STATUS_RUNNING)
        self.assertEqual(codex_pool.latest_id_for_thread("t-id"), first.pk)
        second = self._make(thread_id="t-id", status=CodexInstance.STATUS_COMPLETED)
        self.assertEqual(codex_pool.latest_id_for_thread("t-id"), second.pk)
        # Workers on other threads must not bleed into this thread's pk.
        self._make(thread_id="other", status=CodexInstance.STATUS_RUNNING)
        self.assertEqual(codex_pool.latest_id_for_thread("t-id"), second.pk)


class ReapScopeCgroupTests(TestCase):
    def _make(self, *, systemd_scope_unit: str = "", pid: int = 7) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_FAILED,
            systemd_scope_unit=systemd_scope_unit,
        )

    @patch("hitch.main.codex_pool._scope_has_live_worker", return_value=False)
    @patch("hitch.main.codex_pool._force_kill_instance")
    def test_kills_cgroup_of_scoped_worker(
        self, mock_force_kill: MagicMock, _mock_live: MagicMock
    ) -> None:
        instance = self._make(systemd_scope_unit="hitch-codex-worker-7.service")

        codex_pool._reap_scope_cgroup(instance)

        mock_force_kill.assert_called_once_with(instance)

    @patch("hitch.main.codex_pool._force_kill_instance")
    def test_skips_non_scoped_worker(self, mock_force_kill: MagicMock) -> None:
        # A direct launch has no cgroup to sweep, and its already-dead pid must
        # never be re-signaled (it may have been recycled).
        codex_pool._reap_scope_cgroup(self._make(systemd_scope_unit=""))

        mock_force_kill.assert_not_called()

    @patch("hitch.main.codex_pool._scope_has_live_worker", return_value=True)
    @patch("hitch.main.codex_pool._force_kill_instance")
    def test_skips_scope_reused_by_a_live_worker(
        self, mock_force_kill: MagicMock, _mock_live: MagicMock
    ) -> None:
        # Scope names are not deployment-unique. Our worker is already dead, so a
        # live worker now in the scope means another checkout reused the name --
        # signaling it would kill that launch, so skip.
        instance = self._make(systemd_scope_unit="hitch-codex-worker-7.service")

        codex_pool._reap_scope_cgroup(instance)

        mock_force_kill.assert_not_called()

    @patch("hitch.main.codex_pool._scope_has_live_worker", return_value=False)
    @patch("hitch.main.codex_pool._force_kill_instance")
    def test_swallows_missing_scope(
        self, mock_force_kill: MagicMock, _mock_live: MagicMock
    ) -> None:
        # An already empty/collected scope reports ProcessLookupError; that is the
        # success case (nothing left to reap), not an error to propagate.
        mock_force_kill.side_effect = ProcessLookupError
        instance = self._make(systemd_scope_unit="hitch-codex-worker-7.service")

        codex_pool._reap_scope_cgroup(instance)  # must not raise

    @patch("hitch.main.codex_pool._scope_has_live_worker", return_value=False)
    @patch("hitch.main.codex_pool._force_kill_instance")
    def test_swallows_kill_failure(
        self, mock_force_kill: MagicMock, _mock_live: MagicMock
    ) -> None:
        mock_force_kill.side_effect = OSError("boom")
        instance = self._make(systemd_scope_unit="hitch-codex-worker-7.service")

        codex_pool._reap_scope_cgroup(instance)  # best-effort: must not raise

    @patch("hitch.main.codex_pool._scope_has_live_worker", return_value=False)
    @patch("hitch.main.codex_pool._force_kill_instance")
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=False)
    def test_reconcile_reaps_dead_systemd_worker_cgroup(
        self,
        _mock_alive: MagicMock,
        mock_force_kill: MagicMock,
        _mock_live: MagicMock,
    ) -> None:
        # A wedged/OOM-killed systemd worker reaped by the reconcile sweep has its
        # cgroup swept so a reparented grandchild can't keep holding memory.
        scoped = CodexInstance.objects.create(
            pid=10,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            systemd_scope_unit="hitch-codex-worker-10.service",
        )
        direct = CodexInstance.objects.create(
            pid=11,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

        codex_pool._mark_dead_instances_failed(
            list(CodexInstance.objects.filter(pk__in=[scoped.pk, direct.pk]))
        )

        scoped.refresh_from_db()
        self.assertEqual(scoped.status, CodexInstance.STATUS_FAILED)
        # Only the systemd worker's cgroup is swept; the direct worker is skipped.
        mock_force_kill.assert_called_once()
        self.assertEqual(mock_force_kill.call_args.args[0].pk, scoped.pk)


class ScopeHasLiveWorkerTests(SimpleTestCase):
    def _proc(self, entries: dict[str, dict[str, bytes]]) -> MagicMock:
        # Build a fake /proc whose iterdir yields entries; each entry exposes
        # ``cmdline``/``cgroup`` files via the ``/`` operator.
        def make_entry(name: str, files: dict[str, bytes]) -> MagicMock:
            entry = MagicMock()
            entry.name = name

            def child(fname: str) -> MagicMock:
                f = MagicMock()
                if fname in files:
                    f.read_bytes.return_value = files[fname]
                else:
                    f.read_bytes.side_effect = FileNotFoundError
                return f

            entry.__truediv__.side_effect = child
            return entry

        proc_root = MagicMock()
        proc_root.exists.return_value = True
        proc_root.iterdir.return_value = [
            make_entry(name, files) for name, files in entries.items()
        ]
        return proc_root

    def test_true_when_worker_runs_in_systemd_service(self) -> None:
        proc_root = self._proc(
            {
                "100": {
                    "cmdline": b"python\x00manage.py\x00codex_worker\x00",
                    "cgroup": b"0::/u.slice/hitch-codex-worker-7.service\n",
                },
            }
        )
        self.assertTrue(
            codex_pool._scope_has_live_worker(
                "hitch-codex-worker-7.service", proc_root=proc_root
            )
        )

    def test_true_for_legacy_scope_worker(self) -> None:
        proc_root = self._proc(
            {
                "101": {
                    "cmdline": b"python\x00manage.py\x00codex_worker\x00",
                    "cgroup": b"0::/u.slice/hitch-codex-worker-7.scope\n",
                },
            }
        )
        self.assertTrue(
            codex_pool._scope_has_live_worker(
                "hitch-codex-worker-7.scope", proc_root=proc_root
            )
        )

    def test_false_when_only_a_grandchild_runs_in_scope(self) -> None:
        # A leaked ``cargo bench`` is not a codex_worker, so the unit is still
        # ours to reap.
        proc_root = self._proc(
            {
                "200": {
                    "cmdline": b"cargo\x00bench\x00",
                    "cgroup": b"0::/u.slice/hitch-codex-worker-7.service\n",
                },
            }
        )
        self.assertFalse(
            codex_pool._scope_has_live_worker(
                "hitch-codex-worker-7.service", proc_root=proc_root
            )
        )

    def test_false_when_worker_is_in_a_different_scope(self) -> None:
        proc_root = self._proc(
            {
                "300": {
                    "cmdline": b"python\x00manage.py\x00codex_worker\x00",
                    "cgroup": b"0::/u.slice/hitch-codex-worker-99.service\n",
                },
            }
        )
        self.assertFalse(
            codex_pool._scope_has_live_worker(
                "hitch-codex-worker-7.service", proc_root=proc_root
            )
        )

    def test_false_without_proc(self) -> None:
        proc_root = MagicMock()
        proc_root.exists.return_value = False
        self.assertFalse(
            codex_pool._scope_has_live_worker(
                "hitch-codex-worker-7.service", proc_root=proc_root
            )
        )

    def test_skips_non_numeric_and_unreadable_entries(self) -> None:
        # Non-pid entries are ignored, and a pid whose cmdline or cgroup file
        # vanishes mid-scan is skipped rather than raising.
        proc_root = self._proc(
            {
                "cpuinfo": {},  # not a pid
                "400": {},  # cmdline read fails
                "500": {  # worker, but cgroup read fails
                    "cmdline": b"python\x00manage.py\x00codex_worker\x00",
                },
            }
        )
        self.assertFalse(
            codex_pool._scope_has_live_worker(
                "hitch-codex-worker-7.service", proc_root=proc_root
            )
        )

    def test_worker_unit_from_pid_cgroup_returns_service(self) -> None:
        with tempfile.TemporaryDirectory() as proc_root:
            pid_dir = Path(proc_root) / "700"
            pid_dir.mkdir()
            (pid_dir / "cgroup").write_bytes(
                b"0::/user.slice/hitch-codex-worker-7.service\n"
            )

            unit = codex_pool._worker_unit_from_pid_cgroup(
                700, 7, proc_root=Path(proc_root)
            )

        self.assertEqual(unit, "hitch-codex-worker-7.service")

    def test_worker_unit_from_pid_cgroup_returns_legacy_scope(self) -> None:
        with tempfile.TemporaryDirectory() as proc_root:
            pid_dir = Path(proc_root) / "701"
            pid_dir.mkdir()
            (pid_dir / "cgroup").write_bytes(
                b"0::/user.slice/hitch-codex-worker-7.scope\n"
            )

            unit = codex_pool._worker_unit_from_pid_cgroup(
                701, 7, proc_root=Path(proc_root)
            )

        self.assertEqual(unit, "hitch-codex-worker-7.scope")

    def test_worker_unit_from_pid_cgroup_ignores_wrong_instance(self) -> None:
        with tempfile.TemporaryDirectory() as proc_root:
            pid_dir = Path(proc_root) / "702"
            pid_dir.mkdir()
            (pid_dir / "cgroup").write_bytes(
                b"0::/user.slice/hitch-codex-worker-99.scope\n"
            )

            unit = codex_pool._worker_unit_from_pid_cgroup(
                702, 7, proc_root=Path(proc_root)
            )

        self.assertIsNone(unit)


class ReconcileOrphanedWorkersTests(TestCase):
    def _make(
        self,
        *,
        pid: int = 1,
        status: str = CodexInstance.STATUS_RUNNING,
        purpose: str = CodexInstance.PURPOSE_USER,
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=status,
            purpose=purpose,
        )

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_worker_whose_instance_is_terminal(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        done = self._make(pid=5001, status=CodexInstance.STATUS_COMPLETED)
        mock_iter.return_value = [(5001, done.pk)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_called_once_with(5001, signal.SIGKILL)

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_worker_with_no_instance_row(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        mock_iter.return_value = [(5002, 999999)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_called_once_with(5002, signal.SIGKILL)

    @patch("hitch.main.codex_pool._force_kill_instance")
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._worker_unit_from_pid_cgroup", return_value=None)
    @patch("hitch.main.codex_pool._pid_is_our_worker")
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_systemd_worker_with_no_instance_row(
        self,
        mock_iter: MagicMock,
        mock_identity: MagicMock,
        _mock_cgroup_unit: MagicMock,
        mock_killpg: MagicMock,
        mock_force_kill: MagicMock,
    ) -> None:
        # Row gone (reset DB) but a systemd worker is still alive: it is not a
        # session leader, so the killpg path would skip it. Reap it through its
        # derived unit instead so its app-server / DB lock is released.
        mock_identity.side_effect = (
            lambda pid, iid, require_session_leader=True: not require_session_leader
        )
        mock_iter.return_value = [(5002, 999999)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_not_called()
        mock_force_kill.assert_called_once()
        target = mock_force_kill.call_args.args[0]
        self.assertEqual(
            target.systemd_scope_unit, codex_pool._scope_unit_for_instance(999999)
        )

    @patch("hitch.main.codex_pool._force_kill_instance")
    @patch("hitch.main.codex_pool.os.killpg")
    @patch(
        "hitch.main.codex_pool._worker_unit_from_pid_cgroup",
        return_value="hitch-codex-worker-999999.scope",
    )
    @patch("hitch.main.codex_pool._pid_is_our_worker")
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_legacy_scope_worker_with_no_instance_row(
        self,
        mock_iter: MagicMock,
        mock_identity: MagicMock,
        _mock_cgroup_unit: MagicMock,
        mock_killpg: MagicMock,
        mock_force_kill: MagicMock,
    ) -> None:
        mock_identity.side_effect = (
            lambda pid, iid, require_session_leader=True: not require_session_leader
        )
        mock_iter.return_value = [(5002, 999999)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_not_called()
        mock_force_kill.assert_called_once()
        target = mock_force_kill.call_args.args[0]
        self.assertEqual(target.systemd_scope_unit, "hitch-codex-worker-999999.scope")

    @patch("hitch.main.codex_pool._force_kill_instance")
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._worker_unit_from_pid_cgroup", return_value=None)
    @patch("hitch.main.codex_pool._pid_is_our_worker")
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_systemd_worker_when_row_lacks_unit(
        self,
        mock_iter: MagicMock,
        mock_identity: MagicMock,
        _mock_cgroup_unit: MagicMock,
        mock_killpg: MagicMock,
        mock_force_kill: MagicMock,
    ) -> None:
        # The row exists but never saved its systemd unit (parent died after
        # systemd-run returned): the worker is still systemd-managed (not a
        # session leader), so derive the unit rather than falling through to
        # killpg.
        done = self._make(pid=5006, status=CodexInstance.STATUS_COMPLETED)
        mock_identity.side_effect = (
            lambda pid, iid, require_session_leader=True: not require_session_leader
        )
        mock_iter.return_value = [(5006, done.pk)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_not_called()
        mock_force_kill.assert_called_once()
        target = mock_force_kill.call_args.args[0]
        self.assertEqual(
            target.systemd_scope_unit, codex_pool._scope_unit_for_instance(done.pk)
        )

    @patch("hitch.main.codex_pool._force_kill_instance")
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker")
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_legacy_scope_worker_when_row_lacks_unit(
        self,
        mock_iter: MagicMock,
        mock_identity: MagicMock,
        mock_killpg: MagicMock,
        mock_force_kill: MagicMock,
    ) -> None:
        done = self._make(pid=5006, status=CodexInstance.STATUS_COMPLETED)
        mock_identity.side_effect = (
            lambda pid, iid, require_session_leader=True: not require_session_leader
        )
        mock_iter.return_value = [(5006, done.pk)]
        with patch(
            "hitch.main.codex_pool._worker_unit_from_pid_cgroup",
            return_value=f"hitch-codex-worker-{done.pk}.scope",
        ):
            killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_not_called()
        mock_force_kill.assert_called_once()
        target = mock_force_kill.call_args.args[0]
        self.assertEqual(
            target.systemd_scope_unit, f"hitch-codex-worker-{done.pk}.scope"
        )

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_spares_running_user_and_system_workers(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        user = self._make(pid=5003, status=CodexInstance.STATUS_RUNNING)
        starting = self._make(pid=5004, status=CodexInstance.STATUS_STARTING)
        # A running system-agent worker is expected too and must never be reaped.
        system = self._make(
            pid=5005,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        mock_iter.return_value = [
            (5003, user.pk),
            (5004, starting.pk),
            (5005, system.pk),
        ]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 0)
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_reaps_only_the_orphan_among_a_mix(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        live = self._make(pid=5006, status=CodexInstance.STATUS_RUNNING)
        leaked = self._make(pid=5007, status=CodexInstance.STATUS_FAILED)
        mock_iter.return_value = [(5006, live.pk), (5007, leaked.pk)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_called_once_with(5007, signal.SIGKILL)

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=False)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_does_not_kill_when_identity_recheck_fails(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # The pid was recycled between the /proc scan and the kill: never signal.
        done = self._make(pid=5008, status=CodexInstance.STATUS_COMPLETED)
        mock_iter.return_value = [(5008, done.pk)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 0)
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool._force_kill_instance")
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_scoped_worker_killed_via_force_kill_instance(
        self,
        mock_iter: MagicMock,
        mock_identity: MagicMock,
        mock_killpg: MagicMock,
        mock_force_kill: MagicMock,
    ) -> None:
        scoped = self._make(pid=5009, status=CodexInstance.STATUS_COMPLETED)
        scoped.systemd_scope_unit = "codex-worker-5009.scope"
        scoped.save(update_fields=["systemd_scope_unit"])
        mock_iter.return_value = [(5009, scoped.pk)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        # Scoped workers are killed through systemctl, not a raw killpg -- but
        # only after reverifying the scanned pid belongs to this deployment.
        mock_identity.assert_called_once_with(5009, scoped.pk, require_session_leader=False)
        mock_force_kill.assert_called_once()
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_spares_terminal_worker_within_grace(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # Just reached a terminal status: the worker is still running its
        # post-terminal hooks (notify system agents, image cleanup), so it must
        # not be killed yet.
        done = self._make(pid=5011, status=CodexInstance.STATUS_COMPLETED)
        done.ended_at = timezone.now()
        done.save(update_fields=["ended_at"])
        mock_iter.return_value = [(5011, done.pk)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 0)
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_terminal_worker_past_grace(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        done = self._make(pid=5012, status=CodexInstance.STATUS_COMPLETED)
        done.ended_at = timezone.now() - codex_pool._ORPHAN_REAP_GRACE - timedelta(
            seconds=5
        )
        done.save(update_fields=["ended_at"])
        mock_iter.return_value = [(5012, done.pk)]

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_called_once_with(5012, signal.SIGKILL)

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_spares_supervised_worker_mid_terminal_commit(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # Terminal status but no ended_at recorded yet and still supervised by
        # this process: it is mid terminal-commit, not leaked.
        done = self._make(pid=5013, status=CodexInstance.STATUS_COMPLETED)
        self.assertIsNone(done.ended_at)
        mock_iter.return_value = [(5013, done.pk)]
        with codex_pool._TRACKED_WORKER_PROCS_LOCK:
            codex_pool._TRACKED_WORKER_PROCS[5013] = (done.pk, cast(Any, MagicMock()))
        self.addCleanup(_forget_worker_pid, 5013)

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 0)
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_reaps_supervised_worker_wedged_past_grace(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # A worker this process spawned that is terminal and hung past the grace
        # is the exact same-process process holding the lock: it MUST be killed,
        # not exempted forever by the tracked check.
        done = self._make(pid=5015, status=CodexInstance.STATUS_COMPLETED)
        done.ended_at = timezone.now() - codex_pool._ORPHAN_REAP_GRACE - timedelta(
            seconds=5
        )
        done.save(update_fields=["ended_at"])
        mock_iter.return_value = [(5015, done.pk)]
        with codex_pool._TRACKED_WORKER_PROCS_LOCK:
            codex_pool._TRACKED_WORKER_PROCS[5015] = (done.pk, cast(Any, MagicMock()))
        self.addCleanup(_forget_worker_pid, 5015)

        killed = codex_pool.reconcile_orphaned_workers()

        self.assertEqual(killed, 1)
        mock_killpg.assert_called_once_with(5015, signal.SIGKILL)

    @patch("hitch.main.codex_pool.os.killpg", side_effect=ProcessLookupError)
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kill_already_gone_counts_as_not_killed(
        self, mock_iter: MagicMock, _mock_identity: MagicMock, _mock_killpg: MagicMock
    ) -> None:
        # The leaked process exited between scan and signal: not counted, no raise.
        done = self._make(pid=5014, status=CodexInstance.STATUS_COMPLETED)
        mock_iter.return_value = [(5014, done.pk)]

        self.assertEqual(codex_pool.reconcile_orphaned_workers(), 0)

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_kills_nothing_when_expected_set_unreadable(
        self, mock_iter: MagicMock, mock_killpg: MagicMock
    ) -> None:
        done = self._make(pid=5010, status=CodexInstance.STATUS_COMPLETED)
        mock_iter.return_value = [(5010, done.pk)]

        with patch.object(
            CodexInstance.objects,
            "filter",
            side_effect=Exception("db down"),
        ):
            killed = codex_pool.reconcile_orphaned_workers()

        # Never act on incomplete information.
        self.assertEqual(killed, 0)
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool._finalize_reaped_instance")
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool._iter_running_worker_pids")
    def test_finalizes_each_reaped_instance(
        self,
        mock_iter: MagicMock,
        _mock_identity: MagicMock,
        _mock_killpg: MagicMock,
        mock_finalize: MagicMock,
    ) -> None:
        done = self._make(pid=5016, status=CodexInstance.STATUS_COMPLETED)
        mock_iter.return_value = [(5016, done.pk)]

        codex_pool.reconcile_orphaned_workers()

        mock_finalize.assert_called_once_with(done.pk)


class FinalizeReapedInstanceTests(TestCase):
    def _make(
        self,
        *,
        status: str,
        auto_pr_enabled: bool = False,
        auto_qa_enabled: bool = False,
        auto_pr_triggered_at: datetime | None = None,
        plan_mode: bool = False,
        workflow_id: int | None = None,
        purpose: str = CodexInstance.PURPOSE_USER,
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=1,
            thread_id="t",
            cwd="/r",
            events_path="/dev/null",
            status=status,
            purpose=purpose,
            auto_pr_enabled=auto_pr_enabled,
            auto_qa_enabled=auto_qa_enabled,
            auto_pr_triggered_at=auto_pr_triggered_at,
            plan_mode=plan_mode,
            workflow_id=workflow_id,
        )

    @patch("hitch.main.codex_pool._resolve_dangling_requests")
    def test_failed_turn_resolves_dangling_requests(
        self, mock_resolve: MagicMock
    ) -> None:
        failed = self._make(status=CodexInstance.STATUS_FAILED)

        codex_pool._finalize_reaped_instance(failed.pk)

        mock_resolve.assert_called_once_with(failed.pk)

    @patch("hitch.main.codex_pool.cleanup_requested_input_images_for")
    @patch("hitch.main.codex_pool._notify_system_agents_if_needed")
    def test_routes_finish_hooks_for_reaped_terminal_turn(
        self, mock_notify: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        # A reaped demo/system-agent (or workflow) turn relies on the same
        # idempotent finish routing as a row that died after saving terminal
        # status; without it the SessionDemo/SystemAgentRun follow-up strands.
        agent = self._make(
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )

        codex_pool._finalize_reaped_instance(agent.pk)

        self.assertEqual(mock_notify.call_args.args[0].pk, agent.pk)
        self.assertEqual(mock_cleanup.call_args.args[0].pk, agent.pk)

    def test_non_terminal_instance_is_a_noop(self) -> None:
        running = self._make(status=CodexInstance.STATUS_RUNNING, auto_pr_enabled=True)

        codex_pool._finalize_reaped_instance(running.pk)

        running.refresh_from_db()
        self.assertEqual(running.status, CodexInstance.STATUS_RUNNING)

    @patch("hitch.main.disk_cleanup.run_finished_session_disk_cleanup")
    def test_does_not_fan_out_disk_cleanup(
        self, mock_disk_cleanup: MagicMock
    ) -> None:
        # Disk cleanup walks all of ~/.hitch; finalizing a reaped worker must
        # not trigger it — the 10-minute scheduler owns that cadence.
        done = self._make(status=CodexInstance.STATUS_COMPLETED)

        codex_pool._finalize_reaped_instance(done.pk)

        mock_disk_cleanup.assert_not_called()

    @patch(
        "hitch.main.system_agents.auto_review_intentionally_skipped",
        return_value=False,
    )
    def test_completed_auto_pr_not_fired_is_marked_failed(
        self, _mock_skipped: MagicMock
    ) -> None:
        done = self._make(
            status=CodexInstance.STATUS_COMPLETED, auto_pr_enabled=True
        )

        codex_pool._finalize_reaped_instance(done.pk)

        done.refresh_from_db()
        self.assertEqual(done.status, CodexInstance.STATUS_FAILED)
        self.assertIn("auto-PR", done.error)
        self.assertIn("retry", done.error)

    @patch(
        "hitch.main.system_agents.auto_review_intentionally_skipped",
        return_value=True,
    )
    def test_completed_auto_pr_intentionally_skipped_is_preserved(
        self, _mock_skipped: MagicMock
    ) -> None:
        # Visible-approval mode / pending proposed plan: the null trigger is by
        # design, so a successful turn must not be rewritten as failed.
        done = self._make(
            status=CodexInstance.STATUS_COMPLETED, auto_pr_enabled=True
        )

        codex_pool._finalize_reaped_instance(done.pk)

        done.refresh_from_db()
        self.assertEqual(done.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(done.error, "")

    @patch(
        "hitch.main.system_agents.auto_review_intentionally_skipped",
        side_effect=RuntimeError("boom"),
    )
    def test_completed_auto_pr_preserved_when_intent_check_errors(
        self, _mock_skipped: MagicMock
    ) -> None:
        # If we cannot determine auto-review intent, bias against a false
        # failure and leave the completed turn intact.
        done = self._make(
            status=CodexInstance.STATUS_COMPLETED, auto_pr_enabled=True
        )

        codex_pool._finalize_reaped_instance(done.pk)

        done.refresh_from_db()
        self.assertEqual(done.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(done.error, "")

    def test_completed_without_auto_review_is_untouched(self) -> None:
        done = self._make(status=CodexInstance.STATUS_COMPLETED)

        codex_pool._finalize_reaped_instance(done.pk)

        done.refresh_from_db()
        self.assertEqual(done.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(done.error, "")

    def test_completed_auto_pr_already_fired_is_untouched(self) -> None:
        done = self._make(
            status=CodexInstance.STATUS_COMPLETED,
            auto_pr_enabled=True,
            auto_pr_triggered_at=timezone.now(),
        )

        codex_pool._finalize_reaped_instance(done.pk)

        done.refresh_from_db()
        self.assertEqual(done.status, CodexInstance.STATUS_COMPLETED)

    def test_missing_instance_is_a_noop(self) -> None:
        codex_pool._finalize_reaped_instance(999999)  # must not raise


class IterRunningWorkerPidsTests(SimpleTestCase):
    def _write_cmdline(self, proc_root: Path, pid: int, argv: list[bytes]) -> None:
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir()
        (pid_dir / "cmdline").write_bytes(b"\0".join(argv) + b"\0")

    def test_matches_our_worker_and_scopes_out_other_deployments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            ours = "/srv/hitch/manage.py"
            # Our worker.
            self._write_cmdline(
                proc_root,
                100,
                [b"python", ours.encode(), b"codex_worker", b"--instance-id", b"7"],
            )
            # Another Hitch checkout under the same user: same codex_worker marker
            # but a different manage.py path -- must be ignored.
            self._write_cmdline(
                proc_root,
                101,
                [
                    b"python",
                    b"/other/hitch/manage.py",
                    b"codex_worker",
                    b"--instance-id",
                    b"7",
                ],
            )
            # Unrelated process.
            self._write_cmdline(proc_root, 102, [b"bash", b"-l"])
            # A non-pid entry must be skipped without error.
            (proc_root / "not-a-pid").mkdir()
            # A pid dir whose cmdline cannot be read is skipped, not fatal.
            (proc_root / "104").mkdir()

            found = sorted(
                codex_pool._iter_running_worker_pids(
                    proc_root=proc_root, manage_py=ours
                )
            )

        self.assertEqual(found, [(100, 7)])

    def test_skips_malformed_instance_id_and_missing_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            ours = "/srv/hitch/manage.py"
            self._write_cmdline(
                proc_root,
                200,
                [b"python", ours.encode(), b"codex_worker", b"--instance-id", b"x"],
            )
            self._write_cmdline(
                proc_root,
                201,
                [b"python", ours.encode(), b"codex_worker"],
            )

            found = list(
                codex_pool._iter_running_worker_pids(
                    proc_root=proc_root, manage_py=ours
                )
            )

        self.assertEqual(found, [])

    def test_no_proc_root_yields_nothing(self) -> None:
        missing = Path(tempfile.gettempdir()) / "definitely-not-here-12345"
        self.assertEqual(
            list(
                codex_pool._iter_running_worker_pids(
                    proc_root=missing, manage_py="/srv/hitch/manage.py"
                )
            ),
            [],
        )


class IterCodexAppServerPidsTests(SimpleTestCase):
    _DEPLOYMENT = "/srv/hitch"
    _MARKER = f"{codex_pool._APP_SERVER_DEPLOYMENT_ENV}={_DEPLOYMENT}".encode()
    _APP_SERVER_ARGV = [
        b"/usr/local/bin/codex",
        b"--config",
        b"features.memories=false",
        b"app-server",
        b"--listen",
        b"stdio://",
    ]

    def _write_proc(
        self,
        proc_root: Path,
        pid: int,
        argv: list[bytes],
        environ: list[bytes] | None = None,
        ppid: int | None = None,
    ) -> None:
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir()
        (pid_dir / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
        if environ is not None:
            (pid_dir / "environ").write_bytes(b"\0".join(environ) + b"\0")
        if ppid is not None:
            # Minimal /proc/<pid>/stat: "pid (comm) state ppid ...". The comm is
            # deliberately given spaces and a ")" to exercise the parser.
            (pid_dir / "stat").write_text(f"{pid} (codex app) S {ppid} 0 0\n")

    def test_matches_our_app_server_and_scopes_out_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            # Our app-server: SDK invocation + our deployment marker.
            self._write_proc(
                proc_root, 100, self._APP_SERVER_ARGV, [b"PATH=/usr/bin", self._MARKER]
            )
            # Another checkout's app-server: identical command line, but its
            # environ carries a different deployment id -- must be ignored.
            self._write_proc(
                proc_root,
                101,
                self._APP_SERVER_ARGV,
                [
                    f"{codex_pool._APP_SERVER_DEPLOYMENT_ENV}=/other/hitch".encode(),
                ],
            )
            # An app-server with no marker at all (e.g. pre-upgrade) -- ignored.
            self._write_proc(proc_root, 102, self._APP_SERVER_ARGV, [b"PATH=/usr/bin"])
            # An interactive codex TUI -- no app-server stdio markers -- ignored
            # even though it carries our marker.
            self._write_proc(
                proc_root, 103, [b"/usr/local/bin/codex", b"chat"], [self._MARKER]
            )
            # Unrelated process.
            self._write_proc(proc_root, 104, [b"bash", b"-l"], [self._MARKER])
            # Non-pid entry and an unreadable pid dir are skipped, not fatal.
            (proc_root / "not-a-pid").mkdir()
            (proc_root / "105").mkdir()

            found = sorted(
                codex_pool._iter_codex_app_server_pids(
                    proc_root=proc_root, deployment_id=self._DEPLOYMENT
                )
            )

        self.assertEqual(found, [100])

    def test_yields_both_node_wrapper_and_native_child(self) -> None:
        # The codex CLI is a node wrapper that re-execs a native child; both
        # carry the SDK app-server argv and our marker. The iterator drives the
        # nuke sweep, so it must yield *both* halves -- SIGKILL is not delivered
        # to children, and the native child is the lock-holder. Deduping to one
        # logical app-server is the job of count_running_codex_app_servers.
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            # Worker (parent of the wrapper) -- not itself an app-server.
            self._write_proc(proc_root, 50, [b"python", b"manage.py"], ppid=1)
            # node wrapper (parent is the worker) + native re-exec child.
            self._write_proc(
                proc_root, 100, self._APP_SERVER_ARGV, [self._MARKER], ppid=50
            )
            self._write_proc(
                proc_root, 200, self._APP_SERVER_ARGV, [self._MARKER], ppid=100
            )

            found = sorted(
                codex_pool._iter_codex_app_server_pids(
                    proc_root=proc_root, deployment_id=self._DEPLOYMENT
                )
            )

        self.assertEqual(found, [100, 200])

    def test_unreadable_environ_is_not_matched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            # cmdline matches but environ is missing -- cannot confirm ownership.
            self._write_proc(proc_root, 200, self._APP_SERVER_ARGV)

            found = list(
                codex_pool._iter_codex_app_server_pids(
                    proc_root=proc_root, deployment_id=self._DEPLOYMENT
                )
            )

        self.assertEqual(found, [])

    def test_defaults_deployment_id_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            marker = (
                f"{codex_pool._APP_SERVER_DEPLOYMENT_ENV}="
                f"{codex_pool._app_server_deployment_id()}"
            ).encode()
            self._write_proc(proc_root, 300, self._APP_SERVER_ARGV, [marker])

            found = list(
                codex_pool._iter_codex_app_server_pids(proc_root=proc_root)
            )

        self.assertEqual(found, [300])

    def test_no_proc_root_yields_nothing(self) -> None:
        missing = Path(tempfile.gettempdir()) / "definitely-not-here-67890"
        self.assertEqual(
            list(
                codex_pool._iter_codex_app_server_pids(
                    proc_root=missing, deployment_id=self._DEPLOYMENT
                )
            ),
            [],
        )


class NukeCodexAppServersTests(SimpleTestCase):
    _DEPLOYMENT = "/srv/hitch"

    def _proc_root(self, tmp: str) -> Path:
        return Path(tmp)

    def _write_app_server(
        self, proc_root: Path, pid: int, ppid: int | None = None
    ) -> None:
        marker = (
            f"{codex_pool._APP_SERVER_DEPLOYMENT_ENV}="
            f"{codex_pool._app_server_deployment_id()}"
        ).encode()
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir()
        (pid_dir / "cmdline").write_bytes(
            b"\0".join(
                [b"/usr/local/bin/codex", b"app-server", b"--listen", b"stdio://"]
            )
            + b"\0"
        )
        (pid_dir / "environ").write_bytes(marker + b"\0")
        if ppid is not None:
            (pid_dir / "stat").write_text(f"{pid} (codex) S {ppid} 0 0\n")

    @patch("hitch.main.codex_pool.os.kill")
    def test_sigkills_each_app_server(self, mock_kill: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 111)
            self._write_app_server(proc_root, 222)

            killed = codex_pool.nuke_codex_app_servers(proc_root=proc_root)

        self.assertEqual(killed, 2)
        mock_kill.assert_any_call(111, signal.SIGKILL)
        mock_kill.assert_any_call(222, signal.SIGKILL)

    @patch("hitch.main.codex_pool.os.kill")
    def test_signals_both_wrapper_and_native_child(
        self, mock_kill: MagicMock
    ) -> None:
        # A logical app-server is a node wrapper plus a native re-exec child.
        # SIGKILL is not delivered to children, and the native child holds the
        # CODEX_HOME state-DB lock, so the sweep must signal both directly --
        # killing only the wrapper would orphan the lock-holder.
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 111, ppid=50)
            self._write_app_server(proc_root, 222, ppid=111)

            killed = codex_pool.nuke_codex_app_servers(proc_root=proc_root)

        self.assertEqual(killed, 2)
        mock_kill.assert_any_call(111, signal.SIGKILL)
        mock_kill.assert_any_call(222, signal.SIGKILL)

    @patch("hitch.main.codex_pool.os.kill")
    def test_recycled_pid_failing_recheck_is_not_signaled(
        self, mock_kill: MagicMock
    ) -> None:
        # The pid matched during the scan but its /proc entry no longer looks
        # like our app-server at signal time (recycled): it must not be killed.
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 111)

            with patch(
                "hitch.main.codex_pool._iter_codex_app_server_pids",
                return_value=iter([111, 999]),
            ):
                killed = codex_pool.nuke_codex_app_servers(proc_root=proc_root)

        self.assertEqual(killed, 1)
        mock_kill.assert_called_once_with(111, signal.SIGKILL)

    @patch("hitch.main.codex_pool.os.kill", side_effect=ProcessLookupError)
    def test_already_gone_pid_is_not_counted(self, _mock_kill: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 111)

            self.assertEqual(
                codex_pool.nuke_codex_app_servers(proc_root=proc_root), 0
            )

    @patch("hitch.main.codex_pool.os.kill", side_effect=PermissionError)
    def test_unsignalable_pid_is_skipped(self, _mock_kill: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 333)

            self.assertEqual(
                codex_pool.nuke_codex_app_servers(proc_root=proc_root), 0
            )


class CountRunningCodexAppServersTests(SimpleTestCase):
    def _write_app_server(
        self, proc_root: Path, pid: int, ppid: int | None = None
    ) -> None:
        marker = (
            f"{codex_pool._APP_SERVER_DEPLOYMENT_ENV}="
            f"{codex_pool._app_server_deployment_id()}"
        ).encode()
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir()
        (pid_dir / "cmdline").write_bytes(
            b"\0".join(
                [b"/usr/local/bin/codex", b"app-server", b"--listen", b"stdio://"]
            )
            + b"\0"
        )
        (pid_dir / "environ").write_bytes(marker + b"\0")
        if ppid is not None:
            (pid_dir / "stat").write_text(f"{pid} (codex app) S {ppid} 0 0\n")

    def test_counts_one_per_wrapper_child_pair(self) -> None:
        # Two logical app-servers, each a node wrapper plus its native child, so
        # four matched pids must collapse to a count of two.
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 100, ppid=50)  # wrapper A
            self._write_app_server(proc_root, 200, ppid=100)  # native child A
            self._write_app_server(proc_root, 300, ppid=51)  # wrapper B
            self._write_app_server(proc_root, 400, ppid=300)  # native child B

            count = codex_pool.count_running_codex_app_servers(proc_root=proc_root)

        self.assertEqual(count, 2)

    def test_missing_stat_counts_pid(self) -> None:
        # ppid unknown (no stat) -> count it rather than silently drop, so a
        # leaked app-server is never undercounted.
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 100)

            self.assertEqual(
                codex_pool.count_running_codex_app_servers(proc_root=proc_root), 1
            )

    def test_orphaned_native_child_is_counted(self) -> None:
        # A native child whose wrapper already died (parent reparented to init,
        # not in the matched set) is itself a leaked logical app-server.
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_app_server(proc_root, 200, ppid=1)

            self.assertEqual(
                codex_pool.count_running_codex_app_servers(proc_root=proc_root), 1
            )

    def test_no_proc_root_counts_zero(self) -> None:
        missing = Path(tempfile.gettempdir()) / "definitely-not-here-13579"
        self.assertEqual(
            codex_pool.count_running_codex_app_servers(proc_root=missing), 0
        )


class InterruptActiveTests(TestCase):
    def _make(
        self,
        *,
        pid: int = 1,
        thread_id: str = "t",
        status: str = CodexInstance.STATUS_RUNNING,
        systemd_scope_unit: str = "",
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            systemd_scope_unit=systemd_scope_unit,
            thread_id=thread_id,
            cwd="/r",
            events_path="/dev/null",
            status=status,
        )

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_first_stop_sends_sigterm_and_records_timestamp(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # Polite interrupt: SIGTERM is sent to the worker pid alone (not
        # the group) so the worker's handler can call the SDK's
        # ``turn.interrupt()``. The row's status is left for the worker
        # to update — flipping it here would be overwritten when the
        # worker's stream loop finishes and saves its own terminal state.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_called_once_with(4321, signal.SIGTERM)
        mock_killpg.assert_not_called()
        mock_identity.assert_called_once_with(4321, instance.pk)
        instance.refresh_from_db()
        # Status untouched — worker writes it when the SDK interrupt
        # surfaces as a turn/completed event with status=interrupted.
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)
        self.assertEqual(instance.error, "")
        # Timestamp recorded so the next click can detect "polite
        # already issued" and escalate to SIGKILL.
        self.assertIsNotNone(instance.interrupt_requested_at)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_second_stop_escalates_to_sigkill(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # Worker didn't honour the polite stop; user clicks again.
        # Escalate to SIGKILL on the whole process group (so the codex
        # app-server child dies with the worker) and write status
        # ourselves since the worker no longer has the chance to.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_called_once_with(4321, signal.SIGKILL)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "forcibly stopped by user")
        self.assertIsNotNone(instance.ended_at)

    @patch("hitch.main.disk_cleanup.run_finished_session_disk_cleanup")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_force_stop_does_not_fan_out_disk_cleanup(
        self,
        _mock_kill: MagicMock,
        _mock_killpg: MagicMock,
        _mock_identity: MagicMock,
        mock_disk_cleanup: MagicMock,
    ) -> None:
        # Flipping a row to FAILED via _mark_failed must not trigger the
        # recursive ~/.hitch disk walk on every worker death.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        codex_pool.interrupt_active("t")

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        mock_disk_cleanup.assert_not_called()

    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_second_stop_escalates_systemd_worker_to_systemctl_kill(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        instance = self._make(
            pid=4321,
            status=CodexInstance.STATUS_RUNNING,
            systemd_scope_unit="hitch-codex-worker-7.service",
        )
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=["systemctl", "kill"], returncode=0
        )

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        mock_identity.assert_called_once_with(
            4321, instance.pk, require_session_leader=False
        )
        mock_which.assert_called_once_with("systemctl")
        mock_run.assert_called_once_with(
            [
                "/usr/bin/systemctl",
                "--user",
                "kill",
                "--kill-whom=all",
                "--signal=SIGKILL",
                "hitch-codex-worker-7.service",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "forcibly stopped by user")

    @patch("hitch.main.codex_pool.shutil.which", return_value="/usr/bin/systemctl")
    @patch("hitch.main.codex_pool.subprocess.run")
    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_second_stop_treats_vanished_systemd_unit_as_stopped(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
        mock_run: MagicMock,
        _mock_which: MagicMock,
    ) -> None:
        instance = self._make(
            pid=4321,
            status=CodexInstance.STATUS_RUNNING,
            systemd_scope_unit="hitch-codex-worker-7.service",
        )
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["systemctl", "kill"], returncode=1, stderr=b"Unit not loaded\n"
            ),
            subprocess.CompletedProcess(
                args=["systemctl", "show"], returncode=0, stdout=b"not-found\n"
            ),
        ]

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        mock_identity.assert_called_once_with(
            4321, instance.pk, require_session_leader=False
        )
        self.assertEqual(mock_run.call_count, 2)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "forcibly stopped by user")

    @patch("hitch.main.codex_pool.os.kill")
    @patch("hitch.main.codex_pool.os.killpg")
    def test_no_active_worker_is_a_noop(
        self, mock_killpg: MagicMock, mock_kill: MagicMock
    ) -> None:
        # An already-completed turn must not 500 the stop endpoint; the user
        # may have raced the agent finishing.
        self._make(pid=99, status=CodexInstance.STATUS_COMPLETED)

        self.assertIsNone(codex_pool.interrupt_active("t"))
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=False)
    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_dead_or_recycled_pid_skips_signal_but_marks_row_failed(
        self,
        mock_kill: MagicMock,
        mock_killpg: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # Identity check rejects the pid: no safe target for SIGTERM or
        # SIGKILL, but the leftover RUNNING row is still flipped to
        # failed so the UI exits streaming mode.
        instance = self._make(pid=12345, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "interrupted by user")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_worker_exit_between_identity_and_sigterm_marks_failed(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        # TOCTOU: worker exited between the cmdline check and the signal.
        # ESRCH is tolerated and the leftover row is flipped to failed.
        mock_kill.side_effect = ProcessLookupError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_eperm_from_sigterm_leaves_row_active(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Real signal failure: the worker is still running. Don't claim
        # the turn was stopped — leave the row active so the user can
        # retry rather than seeing a phantom "failed" state.
        mock_kill.side_effect = PermissionError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)
        self.assertEqual(instance.error, "")
        self.assertIsNone(instance.interrupt_requested_at)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    def test_worker_exit_between_identity_and_sigkill_marks_failed(
        self, mock_killpg: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Same TOCTOU as the SIGTERM case, but on the escalation path:
        # ESRCH from SIGKILL is tolerated and we still flip the row.
        mock_killpg.side_effect = ProcessLookupError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        result = codex_pool.interrupt_active("t")

        self.assertIsNotNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "forcibly stopped by user")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    def test_completed_status_under_race_is_preserved_on_sigkill(
        self, mock_killpg: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Escalation path: worker raced to completion between the row
        # read and the killpg. ``_mark_failed`` uses a conditional
        # UPDATE so the genuine completed status is preserved and the
        # helper returns None rather than rewriting it.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        def flip_to_completed(*_args: object, **_kwargs: object) -> None:
            CodexInstance.objects.filter(pk=instance.pk).update(
                status=CodexInstance.STATUS_COMPLETED
            )

        mock_killpg.side_effect = flip_to_completed

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.killpg")
    def test_eperm_from_sigkill_leaves_row_active(
        self, mock_killpg: MagicMock, mock_identity: MagicMock
    ) -> None:
        # Same as above, but on the escalation path (second click): a
        # SIGKILL that we can't deliver must not flip the row to
        # failed, because the worker is still running.
        mock_killpg.side_effect = PermissionError
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        CodexInstance.objects.filter(pk=instance.pk).update(
            interrupt_requested_at=timezone.now()
        )

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_completed_status_under_race_is_preserved_on_first_stop(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        # The worker can legitimately finish in the window between the
        # row read and the signal. The first-click path no longer
        # writes a terminal status, but it does record a timestamp;
        # the timestamp write must not resurrect an already-terminal
        # row by re-enabling the Stop button.
        instance = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)

        def flip_to_completed(*_args: object, **_kwargs: object) -> None:
            CodexInstance.objects.filter(pk=instance.pk).update(
                status=CodexInstance.STATUS_COMPLETED
            )

        mock_kill.side_effect = flip_to_completed

        result = codex_pool.interrupt_active("t")

        # Helper returns the refreshed instance (with whatever state
        # the DB now reflects); the test cares that ``status`` is
        # preserved.
        self.assertIsNotNone(result)
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.codex_pool._pid_is_our_worker")
    @patch("hitch.main.codex_pool.os.kill")
    @patch("hitch.main.codex_pool.os.killpg")
    def test_unset_pid_is_noop(
        self,
        mock_killpg: MagicMock,
        mock_kill: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # The codex_worker subprocess may already be alive and will set
        # status=RUNNING any moment now. Flipping the row to failed
        # here would be silently undone, so the helper refuses.
        instance = self._make(pid=0, status=CodexInstance.STATUS_STARTING)

        result = codex_pool.interrupt_active("t")

        self.assertIsNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        mock_identity.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_STARTING)
        self.assertEqual(instance.error, "")


class InterruptInstanceTests(TestCase):
    """Targeted-interrupt entry point used by the Stop button.

    ``interrupt_instance`` differs from ``interrupt_active`` by stopping
    the *specific* worker the page is showing, not "whichever worker is
    latest at click time" — protecting against a stale tab aborting an
    overlapping newer turn.
    """

    def _make(
        self,
        *,
        pid: int = 1,
        thread_id: str = "t",
        status: str = CodexInstance.STATUS_RUNNING,
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id=thread_id,
            cwd="/r",
            events_path="/dev/null",
            status=status,
        )

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_stops_specific_instance(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        target = self._make(pid=4321, status=CodexInstance.STATUS_RUNNING)
        # A second, newer active worker that would be picked by
        # ``latest_active_for_thread`` is left untouched: the targeted
        # entry point must hit the exact instance the page asked for.
        bystander = self._make(pid=9999, status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_instance(target.pk, expected_thread_id="t")

        self.assertIsNotNone(result)
        # First-click path: polite SIGTERM to the worker pid only.
        mock_kill.assert_called_once_with(4321, signal.SIGTERM)
        target.refresh_from_db()
        bystander.refresh_from_db()
        # Status is left for the worker to update; the timestamp marks
        # that polite interrupt was issued so a re-click can escalate.
        self.assertIsNotNone(target.interrupt_requested_at)
        self.assertIsNone(bystander.interrupt_requested_at)
        self.assertEqual(bystander.status, CodexInstance.STATUS_RUNNING)

    def test_unknown_instance_returns_none(self) -> None:
        # A stale form value for a row that's been deleted (or never
        # existed) must not 500 the stop endpoint.
        self.assertIsNone(
            codex_pool.interrupt_instance(99999, expected_thread_id="t")
        )

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_thread_id_mismatch_refuses(
        self, mock_kill: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # A tampered/stale form post that targets a worker belonging to
        # a different thread must not stop it.
        instance = self._make(thread_id="other", status=CodexInstance.STATUS_RUNNING)

        result = codex_pool.interrupt_instance(instance.pk, expected_thread_id="t")

        self.assertIsNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_RUNNING)

    @patch("hitch.main.codex_pool.os.killpg")
    @patch("hitch.main.codex_pool.os.kill")
    def test_already_terminal_returns_none(
        self, mock_kill: MagicMock, mock_killpg: MagicMock
    ) -> None:
        # The page may be stale; clicking Stop on a worker that already
        # finished must be a clean no-op, not an overwrite.
        instance = self._make(
            pid=4321, thread_id="t", status=CodexInstance.STATUS_COMPLETED
        )

        result = codex_pool.interrupt_instance(instance.pk, expected_thread_id="t")

        self.assertIsNone(result)
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)


class SteerInstanceTests(TestCase):
    """Targeted-steer entry point used by active composer submissions."""

    def _make(
        self,
        *,
        pid: int = 1,
        thread_id: str = "t",
        status: str = CodexInstance.STATUS_RUNNING,
        events_path: str,
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=pid,
            thread_id=thread_id,
            cwd="/r",
            events_path=events_path,
            status=status,
        )

    @patch("hitch.main.codex_pool._steer_instance")
    def test_steer_active_targets_latest_active_instance(
        self, mock_steer: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            active = self._make(
                thread_id="t",
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "active.jsonl"),
            )
            self._make(
                thread_id="other",
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "other.jsonl"),
            )
            mock_steer.return_value = active

            result = codex_pool.steer_active("t", prompt="also do this")

        self.assertEqual(result, active)
        mock_steer.assert_called_once_with(active, prompt="also do this")

    @patch("hitch.main.codex_pool._steer_instance")
    def test_steer_active_returns_none_without_active_instance(
        self, mock_steer: MagicMock
    ) -> None:
        result = codex_pool.steer_active("missing", prompt="also do this")

        self.assertIsNone(result)
        mock_steer.assert_not_called()

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_queues_payload_and_signals_running_instance(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "target.jsonl"),
            )
            bystander = self._make(
                pid=9999,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "bystander.jsonl"),
            )

            result = codex_pool.steer_instance(
                target.pk,
                expected_thread_id="t",
                prompt="also update the tests",
            )

            self.assertIsNotNone(result)
            mock_identity.assert_called_once_with(4321, target.pk)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)
            line = codex_pool.control_path_for(target).read_text(encoding="utf-8")
            self.assertEqual(
                json.loads(line),
                {"op": "steer", "input": "also update the tests"},
            )
            self.assertFalse(codex_pool.control_path_for(bystander).exists())

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_queues_image_paths_for_steer(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "target.jsonl"),
            )

            result = codex_pool.steer_instance(
                target.pk,
                expected_thread_id="t",
                prompt="use this",
                input_image_paths=["/tmp/screen.png"],
            )

            self.assertIsNotNone(result)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)
            line = codex_pool.control_path_for(target).read_text(encoding="utf-8")
            self.assertEqual(
                json.loads(line),
                {
                    "op": "steer",
                    "input": "use this",
                    "inputImagePaths": ["/tmp/screen.png"],
                },
            )
            target.refresh_from_db()
            self.assertEqual(target.input_image_paths, [])
            self.assertEqual(target.input_attachment_paths, ["/tmp/screen.png"])

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_queues_image_only_steer(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "target.jsonl"),
            )

            result = codex_pool.steer_instance(
                target.pk,
                expected_thread_id="t",
                prompt="",
                input_image_paths=["/tmp/screen.png"],
            )

            self.assertIsNotNone(result)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)
            line = codex_pool.control_path_for(target).read_text(encoding="utf-8")
            self.assertEqual(
                json.loads(line),
                {
                    "op": "steer",
                    "input": "",
                    "inputImagePaths": ["/tmp/screen.png"],
                },
            )

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_image_steer_records_ledger_before_control_request(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        original_append = codex_pool._append_control_request

        def append_and_assert_tracked(
            instance: CodexInstance,
            payload: dict[str, Any],
        ) -> None:
            instance.refresh_from_db()
            self.assertEqual(instance.input_attachment_paths, ["/tmp/screen.png"])
            original_append(instance, payload)

        with tempfile.TemporaryDirectory() as raw:
            target = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "target.jsonl"),
            )

            with patch(
                "hitch.main.codex_pool._append_control_request",
                side_effect=append_and_assert_tracked,
            ):
                result = codex_pool.steer_instance(
                    target.pk,
                    expected_thread_id="t",
                    prompt="use this",
                    input_image_paths=["/tmp/screen.png"],
                )

            self.assertIsNotNone(result)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_starting_instance_queues_without_signal(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_STARTING,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also inspect migrations",
            )

            self.assertIsNotNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_not_called()
            self.assertTrue(codex_pool.control_path_for(instance).exists())

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_starting_image_steer_does_not_change_initial_inputs(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_STARTING,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="use this later",
                input_image_paths=["/tmp/steer.png"],
            )

            self.assertIsNotNone(result)
            mock_kill.assert_not_called()
            instance.refresh_from_db()
            self.assertEqual(instance.input_image_paths, [])
            self.assertEqual(instance.input_attachment_paths, ["/tmp/steer.png"])

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_terminal_after_image_tracking_rolls_back_steer_ledger(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        original_add = codex_pool._add_input_attachment_paths

        def add_and_finish(instance: CodexInstance, paths: list[str]) -> None:
            original_add(instance, paths)
            CodexInstance.objects.filter(pk=instance.pk).update(
                status=CodexInstance.STATUS_FAILED
            )

        with tempfile.TemporaryDirectory() as raw:
            image_path = Path(raw) / "screen.png"
            image_path.write_bytes(b"image")
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            with patch(
                "hitch.main.codex_pool._add_input_attachment_paths",
                side_effect=add_and_finish,
            ):
                result = codex_pool.steer_instance(
                    instance.pk,
                    expected_thread_id="t",
                    prompt="use this",
                    input_image_paths=[str(image_path)],
                )

            self.assertIsNone(result)
            mock_kill.assert_not_called()
            instance.refresh_from_db()
            self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
            self.assertEqual(instance.input_attachment_paths, [])
            self.assertTrue(image_path.exists())

    def test_attachment_tracking_merges_from_current_row(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            stale = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            CodexInstance.objects.filter(pk=stale.pk).update(
                input_attachment_paths=["/tmp/first.png"]
            )

            codex_pool._add_input_attachment_paths(stale, ["/tmp/second.png"])

            stale.refresh_from_db()
            self.assertEqual(
                stale.input_attachment_paths,
                ["/tmp/first.png", "/tmp/second.png"],
            )

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_image_steer_rejects_aggregate_attachment_over_cap(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            existing_paths = [
                f"/tmp/screen-{index}.png"
                for index in range(codex_pool._MAX_INPUT_ATTACHMENT_PATHS_PER_INSTANCE)
            ]
            CodexInstance.objects.filter(pk=instance.pk).update(
                input_attachment_paths=existing_paths
            )

            with self.assertRaises(codex_pool.InputAttachmentLimitExceededError):
                codex_pool.steer_instance(
                    instance.pk,
                    expected_thread_id="t",
                    prompt="use one more",
                    input_image_paths=["/tmp/too-many.png"],
                )

            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_not_called()
            self.assertFalse(codex_pool.control_path_for(instance).exists())
            instance.refresh_from_db()
            self.assertEqual(instance.input_attachment_paths, existing_paths)

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_image_steer_rejects_thread_attachment_over_cap(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            active = self._make(
                pid=4321,
                thread_id="t",
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "active.jsonl"),
            )
            existing = self._make(
                pid=0,
                thread_id="t",
                status=CodexInstance.STATUS_COMPLETED,
                events_path=str(Path(raw) / "done.jsonl"),
            )
            existing_paths = [
                f"/tmp/thread-{index}.png"
                for index in range(codex_pool._MAX_INPUT_ATTACHMENT_PATHS_PER_THREAD)
            ]
            CodexInstance.objects.filter(pk=existing.pk).update(
                input_attachment_paths=existing_paths
            )

            with self.assertRaises(codex_pool.InputAttachmentLimitExceededError):
                codex_pool.steer_instance(
                    active.pk,
                    expected_thread_id="t",
                    prompt="use one more",
                    input_image_paths=["/tmp/too-many.png"],
                )

            mock_identity.assert_called_once_with(4321, active.pk)
            mock_kill.assert_not_called()
            active.refresh_from_db()
            self.assertEqual(active.input_attachment_paths, [])

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_starting_instance_reports_not_steered_if_terminal_after_append(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        original_append = codex_pool._append_control_request

        def append_and_finish(
            instance: CodexInstance,
            payload: dict[str, Any],
        ) -> None:
            original_append(instance, payload)
            CodexInstance.objects.filter(pk=instance.pk).update(
                status=CodexInstance.STATUS_FAILED
            )

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_STARTING,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            with patch(
                "hitch.main.codex_pool._append_control_request",
                side_effect=append_and_finish,
            ):
                result = codex_pool.steer_instance(
                    instance.pk,
                    expected_thread_id="t",
                    prompt="also inspect migrations",
                )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_not_called()
            self.assertTrue(codex_pool.control_path_for(instance).exists())

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_starting_image_steer_rolls_back_if_terminal_after_append(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        original_append = codex_pool._append_control_request

        def append_and_finish(
            instance: CodexInstance,
            payload: dict[str, Any],
        ) -> None:
            original_append(instance, payload)
            CodexInstance.objects.filter(pk=instance.pk).update(
                status=CodexInstance.STATUS_FAILED
            )

        with tempfile.TemporaryDirectory() as raw:
            image_path = Path(raw) / "screen.png"
            image_path.write_bytes(b"image")
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_STARTING,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            with patch(
                "hitch.main.codex_pool._append_control_request",
                side_effect=append_and_finish,
            ):
                result = codex_pool.steer_instance(
                    instance.pk,
                    expected_thread_id="t",
                    prompt="use this",
                    input_image_paths=[str(image_path)],
                )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_not_called()
            instance.refresh_from_db()
            self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
            self.assertEqual(instance.input_attachment_paths, [])
            self.assertTrue(image_path.exists())

    @patch("hitch.main.codex_pool.os.kill")
    def test_thread_id_mismatch_refuses(self, mock_kill: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                thread_id="other",
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
            )

            self.assertIsNone(result)
            mock_kill.assert_not_called()
            self.assertFalse(codex_pool.control_path_for(instance).exists())

    @patch("hitch.main.codex_pool.os.kill")
    def test_terminal_instance_refuses(self, mock_kill: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                status=CodexInstance.STATUS_COMPLETED,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
            )

            self.assertIsNone(result)
            mock_kill.assert_not_called()
            self.assertFalse(codex_pool.control_path_for(instance).exists())

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=False)
    @patch("hitch.main.codex_pool.os.kill")
    def test_dead_pid_marks_failed_but_reports_not_steered(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
            )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_not_called()
            instance.refresh_from_db()
            self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
            self.assertEqual(instance.error, "worker process unavailable for steer")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_process_lookup_error_marks_failed_but_reports_not_steered(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        mock_kill.side_effect = ProcessLookupError
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
            )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)
            instance.refresh_from_db()
            self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
            self.assertEqual(instance.error, "worker process exited before steer")

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_image_steer_retains_ledger_when_signal_fails(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        mock_kill.side_effect = OSError("interrupted")
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
                input_image_paths=["/tmp/screen.png"],
            )

            self.assertIsNotNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)
            self.assertTrue(codex_pool.control_path_for(instance).exists())
            instance.refresh_from_db()
            self.assertEqual(instance.input_attachment_paths, ["/tmp/screen.png"])

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_running_instance_reports_not_steered_if_terminal_after_signal(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            def finish_after_signal(_pid: int, _signal: int) -> None:
                CodexInstance.objects.filter(pk=instance.pk).update(
                    status=CodexInstance.STATUS_COMPLETED
                )

            mock_kill.side_effect = finish_after_signal

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
            )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)
            self.assertTrue(codex_pool.control_path_for(instance).exists())

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    def test_image_steer_rolls_back_ledger_when_terminal_after_signal(
        self, mock_kill: MagicMock, mock_identity: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                status=CodexInstance.STATUS_RUNNING,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            def finish_after_signal(_pid: int, _signal: int) -> None:
                CodexInstance.objects.filter(pk=instance.pk).update(
                    status=CodexInstance.STATUS_COMPLETED
                )

            mock_kill.side_effect = finish_after_signal

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
                input_image_paths=["/tmp/screen.png"],
            )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_called_once_with(4321, signal.SIGUSR1)
            instance.refresh_from_db()
            self.assertEqual(instance.input_attachment_paths, [])

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    @patch("hitch.main.codex_pool._append_control_request")
    def test_control_file_write_error_reports_not_steered(
        self,
        mock_append: MagicMock,
        mock_kill: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        mock_append.side_effect = OSError("disk full")
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
            )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_append.assert_called_once()
            mock_kill.assert_not_called()

    @patch("hitch.main.codex_pool._pid_is_our_worker", return_value=True)
    @patch("hitch.main.codex_pool.os.kill")
    @patch("hitch.main.codex_pool._append_control_request")
    def test_control_file_write_error_rolls_back_image_ledger(
        self,
        mock_append: MagicMock,
        mock_kill: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        mock_append.side_effect = OSError("disk full")
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=4321,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
                input_image_paths=["/tmp/screen.png"],
            )

            self.assertIsNone(result)
            mock_identity.assert_called_once_with(4321, instance.pk)
            mock_kill.assert_not_called()
            instance.refresh_from_db()
            self.assertEqual(instance.input_attachment_paths, [])

    @patch("hitch.main.codex_pool._pid_is_our_worker")
    @patch("hitch.main.codex_pool.os.kill")
    def test_unset_pid_refuses(self, mock_kill: MagicMock, mock_identity: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make(
                pid=0,
                status=CodexInstance.STATUS_STARTING,
                events_path=str(Path(raw) / "events.jsonl"),
            )

            result = codex_pool.steer_instance(
                instance.pk,
                expected_thread_id="t",
                prompt="also do this",
            )

            self.assertIsNone(result)
            mock_identity.assert_not_called()
            mock_kill.assert_not_called()
            self.assertFalse(codex_pool.control_path_for(instance).exists())


class PidIsOurWorkerTests(TestCase):
    """The cmdline-based identity guard that protects against PID reuse."""

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_matches_when_cmdline_carries_instance_id(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        mock_getsid.return_value = 4321
        marker = codex_pool._our_manage_py().encode()
        cmdline = (
            b"/usr/bin/python\x00" + marker + b"\x00codex_worker\x00"
            b"--instance-id\x0042\x00"
        )
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = cmdline

        self.assertTrue(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_cmdline_lacks_our_manage_py(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # Another Hitch checkout under the same user: same codex_worker marker
        # and even the same instance id, but a different manage.py path. Killing
        # it would terminate the other deployment's process group.
        mock_getsid.return_value = 4321
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = (
            b"/usr/bin/python\x00/other/hitch/manage.py\x00codex_worker\x00"
            b"--instance-id\x0042\x00"
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_cmdline_lacks_codex_worker(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # An unrelated session leader has inherited the recycled pid:
        # session-leader check passes, cmdline check rules it out.
        mock_getsid.return_value = 4321
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = (
            b"/usr/bin/bash\x00-l\x00"
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_instance_id_flag_missing(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # Defensive: ``codex_worker`` is always invoked with
        # ``--instance-id`` today, but a malformed cmdline (truncated,
        # different worker variant) must not pass identity.
        mock_getsid.return_value = 4321
        marker = codex_pool._our_manage_py().encode()
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = (
            b"python\x00" + marker + b"\x00codex_worker\x00"
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_wrong_instance_id(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # cmdline names a codex_worker but for a different instance —
        # another worker, not ours.
        mock_getsid.return_value = 4321
        marker = codex_pool._our_manage_py().encode()
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = (
            b"python\x00" + marker + b"\x00codex_worker\x00--instance-id\x0099\x00"
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_not_session_leader(
        self, mock_getsid: MagicMock
    ) -> None:
        mock_getsid.return_value = 999  # different from pid

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_scoped_worker_identity_does_not_require_session_leader(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        mock_getsid.return_value = 999
        marker = codex_pool._our_manage_py().encode()
        cmdline = (
            b"/usr/bin/python\x00" + marker + b"\x00codex_worker\x00"
            b"--instance-id\x0042\x00"
        )
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.return_value = cmdline

        self.assertTrue(
            codex_pool._pid_is_our_worker(
                4321, 42, require_session_leader=False
            )
        )
        mock_getsid.assert_not_called()

    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_pid_gone(self, mock_getsid: MagicMock) -> None:
        mock_getsid.side_effect = ProcessLookupError

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_falls_back_to_getsid_when_proc_missing(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # macOS dev: /proc doesn't exist. Cmdline layer is unavailable;
        # trust the session-leader check rather than refuse to stop.
        mock_getsid.return_value = 4321
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.side_effect = (
            FileNotFoundError
        )

        self.assertTrue(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_scoped_worker_rejects_when_proc_missing(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.side_effect = (
            FileNotFoundError
        )

        self.assertFalse(
            codex_pool._pid_is_our_worker(
                4321, 42, require_session_leader=False
            )
        )
        mock_getsid.assert_not_called()

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_when_pid_vanishes_between_getsid_and_cmdline(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # Linux TOCTOU: getsid says the pid is a session leader, but
        # ``/proc/<pid>/cmdline`` is gone moments later because the
        # worker exited. The pid is now at risk of being recycled to an
        # unrelated process; falling back to the session-leader check
        # could let us signal a stranger's group. ``/proc`` itself
        # still exists, so we must refuse rather than trust the cheap
        # check that's no longer authoritative.
        mock_getsid.return_value = 4321
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.side_effect = (
            FileNotFoundError
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))

    @patch("hitch.main.codex_pool.Path")
    @patch("hitch.main.codex_pool.os.getsid")
    def test_rejects_on_other_cmdline_read_error(
        self, mock_getsid: MagicMock, mock_path: MagicMock
    ) -> None:
        # Permission error or any other non-ENOENT failure: be
        # conservative — refuse rather than signaling something we
        # could not identify.
        mock_getsid.return_value = 4321
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.read_bytes.side_effect = (
            PermissionError
        )

        self.assertFalse(codex_pool._pid_is_our_worker(4321, 42))


class EventsDirTests(TestCase):
    def test_uses_setting_when_configured(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            self.assertEqual(codex_pool.events_dir(), Path(raw))

    def test_falls_back_to_home_dir(self) -> None:
        with override_settings(CODEX_EVENTS_DIR=None):
            self.assertEqual(
                codex_pool.events_dir(),
                Path.home() / ".hitch" / "codex_events",
            )

    def test_worker_logs_dir_uses_setting_when_configured(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=Path(raw)),
        ):
            self.assertEqual(codex_pool.worker_logs_dir(), Path(raw))
            self.assertEqual(codex_pool.worker_log_path(7), Path(raw) / "7.log")

    def test_worker_logs_dir_falls_back_under_hitch_home(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=None, HITCH_HOME_DIR=Path(raw)),
        ):
            self.assertEqual(codex_pool.worker_logs_dir(), Path(raw) / "worker_logs")
            self.assertEqual(
                codex_pool.worker_log_path(7), Path(raw) / "worker_logs" / "7.log"
            )


class _FakePayload(BaseModel):
    method_kind: str = "demo"
    detail: str


class SerializeEventTests(TestCase):
    def test_serializes_payload_shapes(self) -> None:
        @dataclasses.dataclass
        class Params:
            params: dict[str, str]

        # pydantic, dataclass, plain dict all flatten to a JSON payload field.
        cases = [
            (_FakePayload(detail="hello"), {"method_kind": "demo", "detail": "hello"}),
            (Params(params={"k": "v"}), {"params": {"k": "v"}}),
            ({"k": 1}, {"k": 1}),
        ]
        for payload, expected in cases:
            with self.subTest(payload=type(payload).__name__):
                parsed = json.loads(
                    _serialize_event("m", payload, recorded_at=11, event_seq=7)
                )
                self.assertEqual(parsed["recordedAt"], 11)
                self.assertEqual(parsed["eventSeq"], 7)
                self.assertEqual(parsed["method"], "m")
                self.assertEqual(parsed["payload"], expected)

    def test_omits_order_metadata_when_absent(self) -> None:
        parsed = json.loads(_serialize_event("m", {"k": 1}))

        self.assertNotIn("recordedAt", parsed)
        self.assertNotIn("eventSeq", parsed)

    def test_redacts_local_image_paths(self) -> None:
        parsed = json.loads(
            _serialize_event(
                "turn/item",
                {
                    "local_images": ["/tmp/private/from-array.png"],
                    "content": [
                        {"type": "text", "text": "see attached"},
                        {"type": "localImage", "path": "/tmp/private/screen.png"},
                    ]
                },
            )
        )

        self.assertEqual(parsed["payload"]["content"][1]["path"], "[redacted]")
        self.assertEqual(parsed["payload"]["local_images"], ["[redacted]"])
        self.assertNotIn("/tmp/private", json.dumps(parsed))


class _FakeNotificationSource:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    def next_notification(self) -> Any:
        if not self.events:
            raise RuntimeError("transport closed")
        return self.events.pop(0)


class GoalNotificationForwarderTests(TestCase):
    def test_forwards_only_goal_notifications_for_current_thread(self) -> None:
        written: list[tuple[str, object]] = []
        discarded: list[Notification] = []
        matching_update = Notification(
            method=codex_events.GOAL_UPDATED_METHOD,
            payload=cast(
                Any,
                ThreadGoalUpdatedNotification(
                    thread_id="thread-1",
                    turn_id=None,
                    goal=ThreadGoal(
                        thread_id="thread-1",
                        objective="Implement goal status",
                        status=ThreadGoalStatus.active,
                        token_budget=None,
                        tokens_used=10,
                        time_used_seconds=2,
                        created_at=1,
                        updated_at=2,
                    ),
                ),
            ),
        )
        account_updated = Notification(
            method="account/updated",
            payload=cast(Any, _FakePayload(detail="x")),
        )
        other_thread_clear = Notification(
            method=codex_events.GOAL_CLEARED_METHOD,
            payload=cast(
                Any,
                ThreadGoalClearedNotification(thread_id="other-thread"),
            ),
        )
        matching_clear = Notification(
            method=codex_events.GOAL_CLEARED_METHOD,
            payload=cast(Any, ThreadGoalClearedNotification(thread_id="thread-1")),
        )

        _forward_goal_notifications(
            source=_FakeNotificationSource(
                [
                    matching_update,
                    account_updated,
                    other_thread_clear,
                    matching_clear,
                ]
            ),
            thread_id="thread-1",
            write_notification=lambda event: written.append((event.method, event.payload)),
            discard_notification=discarded.append,
        )

        self.assertEqual(
            [method for method, _payload in written],
            [codex_events.GOAL_UPDATED_METHOD, codex_events.GOAL_CLEARED_METHOD],
        )
        self.assertEqual(discarded, [account_updated, other_thread_clear])

    def test_exits_on_unexpected_notification_shape(self) -> None:
        written: list[tuple[str, object]] = []
        discarded: list[Notification] = []

        _forward_goal_notifications(
            source=_FakeNotificationSource([object()]),
            thread_id="thread-1",
            write_notification=lambda event: written.append((event.method, event.payload)),
            discard_notification=discarded.append,
        )

        self.assertEqual(written, [])
        self.assertEqual(discarded, [])

    def test_start_goal_event_forwarder_runs_in_background(self) -> None:
        written: list[tuple[str, object]] = []
        discarded: list[Notification] = []

        thread = _start_goal_event_forwarder(
            _FakeNotificationSource(
                [
                    Notification(
                        method=codex_events.GOAL_CLEARED_METHOD,
                        payload=cast(
                            Any,
                            ThreadGoalClearedNotification(thread_id="thread-1"),
                        ),
                    )
                ]
            ),
            thread_id="thread-1",
            write_notification=lambda event: written.append((event.method, event.payload)),
            discard_notification=discarded.append,
        )
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(discarded, [])
        self.assertEqual(
            [method for method, _payload in written],
            [codex_events.GOAL_CLEARED_METHOD],
        )

    def test_notification_sequencer_preserves_router_arrival_order(self) -> None:
        class Router:
            def __init__(self) -> None:
                self.routed: list[Notification] = []

            def route_notification(self, notification: Notification) -> None:
                self.routed.append(notification)

        router = Router()
        codex = SimpleNamespace(_client=SimpleNamespace(_router=router))
        with patch(
            "hitch.main.management.commands.codex_worker.time.time_ns",
            side_effect=[2_000_000, 1_000_000, 3_000_000],
        ):
            order_for = _install_notification_sequencer(cast(Any, codex))
            first = Notification(
                method=codex_events.GOAL_CLEARED_METHOD,
                payload=cast(Any, ThreadGoalClearedNotification(thread_id="thread-1")),
            )
            second = Notification(
                method=codex_events.GOAL_CLEARED_METHOD,
                payload=cast(Any, ThreadGoalClearedNotification(thread_id="thread-1")),
            )
            unrouted = Notification(
                method=codex_events.GOAL_CLEARED_METHOD,
                payload=cast(Any, ThreadGoalClearedNotification(thread_id="thread-1")),
            )

            router.route_notification(first)
            router.route_notification(second)

            self.assertEqual(router.routed, [first, second])
            self.assertEqual(order_for(second), (2_000, 2))
            self.assertEqual(order_for(first), (2_000, 1))
            self.assertEqual(order_for(unrouted), (3_000, 3))

    def test_forwarder_discards_order_for_skipped_notifications(self) -> None:
        class Router:
            def __init__(self) -> None:
                self.routed: list[Notification] = []

            def route_notification(self, notification: Notification) -> None:
                self.routed.append(notification)

        router = Router()
        codex = SimpleNamespace(_client=SimpleNamespace(_router=router))
        skipped = Notification(
            method="account/updated",
            payload=cast(Any, _FakePayload(detail="x")),
        )
        written: list[Notification] = []
        discarded_orders: list[tuple[int, int]] = []

        with patch(
            "hitch.main.management.commands.codex_worker.time.time_ns",
            side_effect=[2_000_000, 3_000_000],
        ):
            order_for = _install_notification_sequencer(cast(Any, codex))
            router.route_notification(skipped)

            _forward_goal_notifications(
                source=_FakeNotificationSource([skipped]),
                thread_id="thread-1",
                write_notification=written.append,
                discard_notification=lambda event: discarded_orders.append(order_for(event)),
            )

            self.assertEqual(written, [])
            self.assertEqual(discarded_orders, [(2_000, 1)])
            self.assertEqual(order_for(skipped), (3_000, 2))

    def test_preserves_early_turn_completed_until_stream_registration(self) -> None:
        """Fast turns may complete before ``TurnHandle.stream`` registers.

        The pinned SDK drops that early completion by default; Hitch's worker
        router shim must keep it pending so the stream loop can observe the
        terminal event and mark the row completed.
        """
        router = MessageRouter()
        codex = SimpleNamespace(_client=SimpleNamespace(_router=router))
        _install_notification_sequencer(cast(Any, codex))
        completed = Notification(
            method="turn/completed",
            payload=cast(
                Any,
                TurnCompletedNotification(
                    thread_id="thread-1",
                    turn=Turn(id="turn-1", items=[], status=TurnStatus.completed),
                ),
            ),
        )

        router.route_notification(completed)
        router.register_turn("turn-1")

        queue = router._turn_notifications["turn-1"]
        self.assertEqual(queue.qsize(), 1)
        self.assertIs(queue.get_nowait(), completed)


class _BrokenStderr:
    def write(self, _value: str) -> int:
        raise OSError("stderr unavailable")

    def flush(self) -> None:
        raise OSError("stderr unavailable")


class CodexWorkerCommandTests(TestCase):
    def _make_instance(
        self,
        events_dir: Path,
        *,
        prompt: str = "hi",
        base_instructions: str = "",
        developer_instructions: str = "",
        enable_memories: bool = False,
        web_search_mode: str = "",
    ) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=12345,
            thread_id="thread-1",
            cwd="/repo",
            prompt=prompt,
            base_instructions=base_instructions,
            developer_instructions=developer_instructions,
            enable_memories=enable_memories,
            web_search_mode=web_search_mode,
            events_path=str(events_dir / "events.jsonl"),
            status=CodexInstance.STATUS_STARTING,
        )

    def _attach_input_image(self, instance: CodexInstance, name: str = "1.png") -> Path:
        image_path = codex_pool.input_attachments_dir() / f"req-{instance.pk}" / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
        instance.input_image_paths = [str(image_path)]
        instance.input_attachment_paths = [str(image_path)]
        instance.save(update_fields=["input_image_paths", "input_attachment_paths"])
        return image_path

    def _make_pending_requests(
        self, instance: CodexInstance
    ) -> tuple[ApprovalRequest, UserInputRequest]:
        approval = ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={},
        )
        input_request = UserInputRequest.objects.create(
            instance=instance,
            method="request_user_input",
            params={},
        )
        return approval, input_request

    def _assert_requests_cancelled(
        self, approval: ApprovalRequest, input_request: UserInputRequest
    ) -> None:
        approval.refresh_from_db()
        input_request.refresh_from_db()
        self.assertEqual(approval.decision, ApprovalRequest.DECISION_CANCEL)
        self.assertIsNotNone(approval.decided_at)
        self.assertEqual(input_request.response, {"answers": {}})
        self.assertIsNotNone(input_request.responded_at)

    @patch("hitch.main.demo.on_codex_instance_finished")
    @patch("hitch.main.system_agents.on_codex_instance_finished")
    def test_notify_system_agents_does_not_double_route_demo_system_agent(
        self, mock_system_notify: MagicMock, mock_demo_notify: MagicMock
    ) -> None:
        mock_system_notify.return_value = True
        instance = CodexInstance.objects.create(
            pid=12345,
            thread_id="thread-1",
            cwd="/repo",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        codex_worker_module._notify_system_agents(instance)

        mock_system_notify.assert_called_once_with(instance)
        mock_demo_notify.assert_not_called()

    @patch("hitch.main.demo.on_codex_instance_finished")
    @patch("hitch.main.system_agents.on_codex_instance_finished", return_value=False)
    def test_notify_system_agents_keeps_demo_fallback_when_system_agents_noop(
        self, mock_system_notify: MagicMock, mock_demo_notify: MagicMock
    ) -> None:
        instance = CodexInstance.objects.create(
            pid=12345,
            thread_id="thread-1",
            cwd="/repo",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
        )

        codex_worker_module._notify_system_agents(instance)

        mock_system_notify.assert_called_once_with(instance)
        mock_demo_notify.assert_called_once_with(instance)

    @override_settings(CODEX_WORKER_OOM_SCORE_ADJ=1000)
    def test_apply_worker_oom_score_adjust_writes_configured_score(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "oom_score_adj"

            codex_worker_module._apply_worker_oom_score_adjust(path)

            self.assertEqual(path.read_text(encoding="utf-8"), "1000\n")

    @override_settings(CODEX_WORKER_OOM_SCORE_ADJ=1500)
    def test_apply_worker_oom_score_adjust_clamps_score(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "oom_score_adj"

            codex_worker_module._apply_worker_oom_score_adjust(path)

            self.assertEqual(path.read_text(encoding="utf-8"), "1000\n")

    @override_settings(CODEX_WORKER_OOM_SCORE_ADJ=0)
    def test_apply_worker_oom_score_adjust_skips_zero_score(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "oom_score_adj"

            codex_worker_module._apply_worker_oom_score_adjust(path)

            self.assertFalse(path.exists())

    @override_settings(CODEX_WORKER_OOM_SCORE_ADJ="not-an-int")
    @patch("hitch.main.management.commands.codex_worker.logger.warning")
    def test_apply_worker_oom_score_adjust_ignores_invalid_score(
        self, mock_warning: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "oom_score_adj"

            codex_worker_module._apply_worker_oom_score_adjust(path)

            self.assertFalse(path.exists())
        mock_warning.assert_called_once_with(
            "invalid CODEX_WORKER_OOM_SCORE_ADJ: %r",
            "not-an-int",
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_streams_notifications_and_marks_completed(self, mock_codex: MagicMock) -> None:
        events = [
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=_FakePayload(detail="chunk-1"),
            ),
            _completed_event("turn-1", TurnStatus.completed),
        ]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

            with open(instance.events_path, encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh]

        codex_ctx.thread_resume.assert_called_once_with("thread-1")
        self.assertEqual(os.environ["HITCH_THREAD_ID"], "thread-1")
        self.assertEqual(os.environ["HITCH_CWD"], "/repo")
        self.assertEqual(os.environ["HITCH_PROJECT_DIR"], str(Path(settings.BASE_DIR)))
        self.assertEqual(
            os.environ["HITCH_MANAGE_PY"],
            str(Path(settings.BASE_DIR) / "manage.py"),
        )
        self.assertEqual(os.environ["HITCH_MANAGE_COMMAND"], "uv")
        self.assertEqual(os.environ["HITCH_PROPOSE_SESSION_COMMAND"], "uv")
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["method"], "item/agentMessage/delta")
        self.assertEqual(lines[0]["payload"]["detail"], "chunk-1")
        self.assertEqual(lines[1]["method"], "turn/completed")

        instance.refresh_from_db()
        self.assertEqual(instance.pid, os.getpid())
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        self.assertIsNotNone(instance.ended_at)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.management.commands.codex_worker._notify_system_agents")
    @patch("hitch.main.management.commands.codex_worker._run_turn")
    def test_worker_completed_save_does_not_resurrect_parent_forced_stop(
        self, mock_run_turn: MagicMock, _mock_notify: MagicMock
    ) -> None:
        # A forced Stop (``_force_kill_instance``/``_mark_failed``) or a
        # ``reconcile_dead`` sweep flips the row to FAILED with a conditional
        # UPDATE while the turn is mid-flight. The worker's own end-of-turn save
        # must honour that terminal state rather than resurrecting the row as
        # COMPLETED -- which would silently drop the user's Stop.
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))

            def _forced_stop_then_complete(*args: object, **kwargs: object) -> SimpleNamespace:
                CodexInstance.objects.filter(pk=instance.pk).update(
                    status=CodexInstance.STATUS_FAILED,
                    ended_at=timezone.now(),
                    error="forcibly stopped by user",
                )
                return SimpleNamespace(status=TurnStatus.completed)

            mock_run_turn.side_effect = _forced_stop_then_complete
            call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "forcibly stopped by user")

    @patch("hitch.main.management.commands.codex_worker._notify_system_agents")
    @patch("hitch.main.management.commands.codex_worker._run_turn")
    def test_worker_bumps_session_index_on_completion(
        self, mock_run_turn: MagicMock, _mock_notify: MagicMock
    ) -> None:
        # Worker turns write to an isolated sqlite_home the web index never reads,
        # so the worker must bump the session's recency directly on completion.
        mock_run_turn.return_value = SimpleNamespace(status=TurnStatus.completed)
        stale = timezone.now() - timedelta(days=3)
        SessionMetadata.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            codex_created_at=stale,
            codex_updated_at=stale,
            codex_last_synced_at=stale,
        )
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        assert metadata.codex_updated_at is not None
        self.assertGreater(metadata.codex_updated_at, stale)

    @patch(
        "hitch.main.management.commands.codex_worker.disk_cleanup"
        ".run_finished_session_disk_cleanup"
    )
    @patch("hitch.main.management.commands.codex_worker._notify_system_agents")
    @patch("hitch.main.management.commands.codex_worker._run_turn")
    @patch("hitch.main.management.commands.codex_worker.acquire_worker_sqlite_home")
    def test_worker_leases_and_releases_sqlite_home(
        self,
        mock_acquire: MagicMock,
        mock_run_turn: MagicMock,
        _mock_notify: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        mock_run_turn.return_value = SimpleNamespace(status=TurnStatus.completed)
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw) / "worker-0"
            home.mkdir()
            lease = MagicMock()
            lease.home = home
            mock_acquire.return_value = lease
            instance = self._make_instance(Path(raw))
            # TESTING=False takes the real lease path (the app-server is still
            # mocked, and disk cleanup is patched out).
            with override_settings(TESTING=False):
                call_command("codex_worker", "--instance-id", str(instance.pk))

        mock_acquire.assert_called_once_with(instance.pk)
        lease.release.assert_called_once_with()
        self.assertEqual(mock_run_turn.call_args.kwargs.get("sqlite_home"), str(home))

    @patch(
        "hitch.main.management.commands.codex_worker.disk_cleanup"
        ".run_finished_session_disk_cleanup"
    )
    @patch("hitch.main.management.commands.codex_worker._notify_system_agents")
    @patch("hitch.main.management.commands.codex_worker._run_turn")
    @patch("hitch.main.management.commands.codex_worker.acquire_worker_sqlite_home")
    def test_worker_inherits_codex_home_when_lease_fails(
        self,
        mock_acquire: MagicMock,
        mock_run_turn: MagicMock,
        _mock_notify: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        # A broken state-dir filesystem degrades to $CODEX_HOME, not the web home
        # under the same (just-failed) base.
        mock_acquire.side_effect = OSError("read-only file system")
        mock_run_turn.return_value = SimpleNamespace(status=TurnStatus.completed)
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            with override_settings(TESTING=False):
                call_command("codex_worker", "--instance-id", str(instance.pk))

        self.assertEqual(
            mock_run_turn.call_args.kwargs.get("sqlite_home"),
            str(codex_pool.codex_home_dir()),
        )

    @patch("hitch.main.management.commands.codex_worker.sys.stderr", new_callable=StringIO)
    @patch("hitch.main.management.commands.codex_worker._notify_system_agents")
    @patch("hitch.main.management.commands.codex_worker._run_turn")
    def test_worker_records_base_exception_before_reraising(
        self,
        mock_run_turn: MagicMock,
        _mock_notify: MagicMock,
        stderr: StringIO,
    ) -> None:
        mock_run_turn.side_effect = SystemExit("transport closed")
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=Path(raw) / "worker-logs"),
        ):
            instance = self._make_instance(Path(raw))

            with self.assertRaises(SystemExit):
                call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "SystemExit('transport closed')")
        self.assertIn("failed with SystemExit", stderr.getvalue())

    @patch("hitch.main.management.commands.codex_worker.sys.stderr", new_callable=_BrokenStderr)
    @patch("hitch.main.management.commands.codex_worker._notify_system_agents")
    @patch("hitch.main.management.commands.codex_worker._run_turn")
    def test_worker_stderr_errors_do_not_block_completed_status(
        self,
        mock_run_turn: MagicMock,
        _mock_notify: MagicMock,
        _stderr: _BrokenStderr,
    ) -> None:
        mock_run_turn.return_value = SimpleNamespace(status=TurnStatus.completed)
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=Path(raw) / "worker-logs"),
        ):
            instance = self._make_instance(Path(raw))

            call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(instance.error, "")

    @patch("hitch.main.management.commands.codex_worker.sys.stderr", new_callable=_BrokenStderr)
    @patch("hitch.main.management.commands.codex_worker._notify_system_agents")
    @patch("hitch.main.management.commands.codex_worker._run_turn")
    def test_worker_stderr_errors_do_not_skip_failed_status(
        self,
        mock_run_turn: MagicMock,
        _mock_notify: MagicMock,
        _stderr: _BrokenStderr,
    ) -> None:
        mock_run_turn.side_effect = RuntimeError("boom")
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_WORKER_LOG_DIR=Path(raw) / "worker-logs"),
        ):
            instance = self._make_instance(Path(raw))

            with self.assertRaises(RuntimeError):
                call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(instance.error, "RuntimeError('boom')")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_enable_memories_row_sets_app_server_override(
        self, mock_codex: MagicMock
    ) -> None:
        events = [_completed_event("turn-1", TurnStatus.completed)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw), enable_memories=True)
            call_command("codex_worker", "--instance-id", str(instance.pk))

        config = mock_codex.call_args.kwargs["config"]
        self.assertEqual(config.config_overrides, ("features.memories=true",))

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_web_search_row_sets_app_server_override(
        self, mock_codex: MagicMock
    ) -> None:
        events = [_completed_event("turn-1", TurnStatus.completed)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw), web_search_mode="live")
            call_command("codex_worker", "--instance-id", str(instance.pk))

        config = mock_codex.call_args.kwargs["config"]
        self.assertEqual(
            config.config_overrides,
            ('features.memories=false', 'web_search="live"'),
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_web_search_cli_flag_sets_app_server_override(
        self, mock_codex: MagicMock
    ) -> None:
        events = [_completed_event("turn-1", TurnStatus.completed)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw), web_search_mode="live")
            call_command(
                "codex_worker",
                "--instance-id",
                str(instance.pk),
                "--web-search-mode",
                "cached",
            )

        config = mock_codex.call_args.kwargs["config"]
        self.assertEqual(
            config.config_overrides,
            ('features.memories=false', 'web_search="cached"'),
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_invalid_web_search_row_is_rejected_before_codex_starts(
        self, mock_codex: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(
                Path(raw),
                web_search_mode='live"\napproval_policy="never',
            )

            with self.assertRaises(ValueError):
                call_command("codex_worker", "--instance-id", str(instance.pk))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        mock_codex.assert_not_called()

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_forwards_developer_instructions_on_resume(
        self, mock_codex: MagicMock
    ) -> None:
        """Developer instructions originate at thread creation, but each
        worker starts from a fresh app-server and must re-supply them when
        resuming the thread before the turn starts."""
        events = [_completed_event("turn-1", TurnStatus.completed)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(
                Path(raw),
                developer_instructions="Prefer small, typed changes.",
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        codex_ctx.thread_resume.assert_called_once_with(
            "thread-1",
            developer_instructions="Prefer small, typed changes.",
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_forwards_base_instructions_on_resume(
        self, mock_codex: MagicMock
    ) -> None:
        events = [_completed_event("turn-1", TurnStatus.completed)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(
                Path(raw),
                base_instructions="Base override.",
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        codex_ctx.thread_resume.assert_called_once_with(
            "thread-1",
            base_instructions="Base override.",
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_reads_prompt_from_instance_row(self, mock_codex: MagicMock) -> None:
        """The prompt isn't a CLI arg — verify it round-trips via the row,
        including a leading-dash value that argparse would reject."""
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object) -> object:
            captured["input"] = input_obj
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw), prompt="- a markdown bullet")
            call_command("codex_worker", "--instance-id", str(instance.pk))

        self.assertEqual(getattr(captured["input"], "text", None), "- a markdown bullet")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_reads_input_images_from_instance_row(self, mock_codex: MagicMock) -> None:
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object) -> object:
            captured["input"] = input_obj
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            image_paths = [
                codex_pool.input_attachments_dir() / "req" / "1.png",
                codex_pool.input_attachments_dir() / "req" / "2.jpg",
            ]
            image_paths[0].parent.mkdir(parents=True)
            for image_path in image_paths:
                image_path.write_bytes(b"image")
            instance = self._make_instance(Path(raw), prompt="use this")
            instance.input_image_paths = [str(path) for path in image_paths]
            instance.input_attachment_paths = [str(path) for path in image_paths]
            instance.save(update_fields=["input_image_paths", "input_attachment_paths"])
            call_command("codex_worker", "--instance-id", str(instance.pk))

        input_items = captured["input"]
        assert isinstance(input_items, list)
        self.assertEqual(getattr(input_items[0], "text", None), "use this")
        self.assertEqual(
            [getattr(item, "path", None) for item in input_items[1:]],
            [str(path) for path in image_paths],
        )

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_attachment_ledger_is_not_initial_turn_input(
        self, mock_codex: MagicMock
    ) -> None:
        captured: dict[str, object] = {}

        def _capture_turn(input_obj: object) -> object:
            captured["input"] = input_obj
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            image_path = codex_pool.input_attachments_dir() / "steer" / "1.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"steer")
            instance = self._make_instance(Path(raw), prompt="initial prompt")
            instance.input_attachment_paths = [str(image_path)]
            instance.save(update_fields=["input_attachment_paths"])

            call_command("codex_worker", "--instance-id", str(instance.pk))

            self.assertEqual(getattr(captured["input"], "text", None), "initial prompt")
            self.assertTrue(image_path.exists())

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_terminal_worker_retains_input_images(self, mock_codex: MagicMock) -> None:
        events = [_completed_event("turn-1", TurnStatus.completed)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            image_path = codex_pool.input_attachments_dir() / "req" / "1.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")
            instance = self._make_instance(Path(raw), prompt="use this")
            instance.input_image_paths = [str(image_path)]
            instance.input_attachment_paths = [str(image_path)]
            instance.save(update_fields=["input_image_paths", "input_attachment_paths"])

            call_command("codex_worker", "--instance-id", str(instance.pk))

            self.assertTrue(image_path.exists())
            instance.refresh_from_db()
            self.assertEqual(instance.input_image_paths, [str(image_path)])
            self.assertEqual(instance.input_attachment_paths, [str(image_path)])

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_terminal_worker_cleans_images_when_archive_requested(
        self, mock_codex: MagicMock
    ) -> None:
        events = [_completed_event("turn-1", TurnStatus.completed)]
        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = _stub_thread_resume(events)

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            image_path = codex_pool.input_attachments_dir() / "req" / "1.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")
            instance = self._make_instance(Path(raw), prompt="use this")
            instance.input_image_paths = [str(image_path)]
            instance.input_attachment_paths = [str(image_path)]
            instance.input_attachment_cleanup_requested = True
            instance.save(
                update_fields=[
                    "input_image_paths",
                    "input_attachment_paths",
                    "input_attachment_cleanup_requested",
                ]
            )

            call_command("codex_worker", "--instance-id", str(instance.pk))

            self.assertFalse(image_path.exists())
            instance.refresh_from_db()
            self.assertEqual(instance.input_image_paths, [])
            self.assertEqual(instance.input_attachment_paths, [])
            self.assertFalse(instance.input_attachment_cleanup_requested)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_terminal_worker_retains_image_steers_added_after_load(
        self, mock_codex: MagicMock
    ) -> None:
        captured: dict[str, Path] = {}

        def _capture_turn(input_obj: object) -> object:
            assert not isinstance(input_obj, list)
            steer_path = codex_pool.input_attachments_dir() / "steer" / "1.png"
            steer_path.parent.mkdir(parents=True)
            steer_path.write_bytes(b"steer")
            captured["steer_path"] = steer_path
            CodexInstance.objects.filter(pk=instance.pk).update(
                input_attachment_paths=[str(steer_path)]
            )
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            instance = self._make_instance(Path(raw), prompt="keep going")

            call_command("codex_worker", "--instance-id", str(instance.pk))

            self.assertTrue(captured["steer_path"].exists())
            instance.refresh_from_db()
            self.assertEqual(instance.input_image_paths, [])
            self.assertEqual(instance.input_attachment_paths, [str(captured["steer_path"])])

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_failed_for_non_completed_outcomes(self, mock_codex: MagicMock) -> None:
        """Failed / interrupted turn statuses and a stream that ends without
        a turn/completed event all leave the row in STATUS_FAILED with a
        human-readable error blurb."""
        codex_ctx = mock_codex.return_value.__enter__.return_value

        cases = [
            (
                [_completed_event("turn-1", TurnStatus.failed, error_message="model said no")],
                "model said no",
            ),
            ([_completed_event("turn-1", TurnStatus.interrupted)], "interrupted"),
            ([], "turn/completed"),
        ]
        for events, expected_in_error in cases:
            with self.subTest(case=expected_in_error):
                codex_ctx.thread_resume.reset_mock()
                codex_ctx.thread_resume.return_value = _stub_thread_resume(events)
                with (
                    tempfile.TemporaryDirectory() as raw,
                    override_settings(CODEX_EVENTS_DIR=Path(raw)),
                ):
                    instance = self._make_instance(Path(raw))
                    approval, input_request = self._make_pending_requests(instance)
                    image_path = self._attach_input_image(instance)
                    call_command("codex_worker", "--instance-id", str(instance.pk))
                    self.assertTrue(image_path.exists())
                    instance.refresh_from_db()
                self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
                self.assertIn(expected_in_error, instance.error)
                self.assertIsNotNone(instance.ended_at)
                self.assertEqual(instance.input_image_paths, [str(image_path)])
                self.assertEqual(instance.input_attachment_paths, [str(image_path)])
                self._assert_requests_cancelled(approval, input_request)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_records_failure_when_codex_raises(self, mock_codex: MagicMock) -> None:
        mock_codex.return_value.__enter__.side_effect = RuntimeError("boom")

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            instance = self._make_instance(Path(raw))
            approval, input_request = self._make_pending_requests(instance)
            image_path = self._attach_input_image(instance)
            with self.assertRaises(RuntimeError):
                call_command(
                    "codex_worker",
                    "--instance-id",
                    str(instance.pk),
                    stderr=StringIO(),
                )
            self.assertTrue(image_path.exists())

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertIn("boom", instance.error)
        self.assertIsNotNone(instance.ended_at)
        self.assertEqual(instance.input_image_paths, [str(image_path)])
        self.assertEqual(instance.input_attachment_paths, [str(image_path)])
        self._assert_requests_cancelled(approval, input_request)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_typed_cli_args_round_trip_to_turn_kwargs(self, mock_codex: MagicMock) -> None:
        """Known CLI values reach ``Thread.turn`` as SDK types; stale values
        are dropped so Codex's own defaults take over."""
        captured: dict[str, object] = {}

        def _assert_value(key: str, expected: object) -> Callable[[dict[str, object]], None]:
            def _assert(capture: dict[str, object]) -> None:
                self.assertEqual(capture.get(key), expected)

            return _assert

        def _assert_absent(key: str) -> Callable[[dict[str, object]], None]:
            def _assert(capture: dict[str, object]) -> None:
                self.assertNotIn(key, capture)

            return _assert

        def _assert_sandbox_variant(
            expected_variant: type[object],
        ) -> Callable[[dict[str, object]], None]:
            def _assert(capture: dict[str, object]) -> None:
                policy = capture.get("sandbox_policy")
                assert isinstance(policy, SandboxPolicy)
                self.assertIsInstance(policy.root, expected_variant)

            return _assert

        def _capture_turn(input_obj: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                id="turn-1",
                stream=lambda: iter([_completed_event("turn-1", TurnStatus.completed)]),
            )

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.return_value = SimpleNamespace(turn=_capture_turn)

        cases: list[tuple[str, str, Callable[[dict[str, object]], None]]] = [
            ("--reasoning-effort", "high", _assert_value("effort", ReasoningEffort.high)),
            ("--reasoning-effort", "ludicrous", _assert_absent("effort")),
            (
                "--sandbox-policy",
                "workspaceWrite",
                _assert_sandbox_variant(WorkspaceWriteSandboxPolicy),
            ),
            (
                "--sandbox-policy",
                "dangerFullAccess",
                _assert_sandbox_variant(DangerFullAccessSandboxPolicy),
            ),
            ("--sandbox-policy", "phantomPolicy", _assert_absent("sandbox_policy")),
            ("--approval-mode", "deny_all", _assert_value("approval_mode", ApprovalMode.deny_all)),
            (
                "--approval-mode",
                "auto_review",
                _assert_value("approval_mode", ApprovalMode.auto_review),
            ),
            ("--approval-mode", "phantom_mode", _assert_absent("approval_mode")),
        ]
        for flag, cli_value, assert_capture in cases:
            with self.subTest(flag=flag, cli_value=cli_value):
                captured.clear()
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        flag,
                        cli_value,
                    )
                assert_capture(captured)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_user_reviewer_modes_bypass_thread_turn(
        self, mock_codex: MagicMock
    ) -> None:
        """Custom user-reviewer modes are not in the SDK's ``ApprovalMode``
        enum, so the worker has to bypass ``Thread.turn(approval_mode=)``
        and post wire-level ``TurnStartParams`` directly: an on-request
        approval policy paired with ``ApprovalsReviewer.user`` routes every
        escalation to the client transport. Pin the wire call so a refactor
        cannot quietly downgrade these modes to one of the typed SDK values,
        or drop the explicit reviewer (which would let server-side routing
        send approvals to the auto-reviewer instead of the client)."""
        captured_params: dict[str, object] = {}
        codex_ctx = mock_codex.return_value.__enter__.return_value

        def _capture_turn_start(
            _thread_id: str, _input: object, *, params: object
        ) -> object:
            captured_params["input"] = _input
            captured_params["params"] = params
            return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

        codex_ctx._client.turn_start.side_effect = _capture_turn_start
        codex_ctx._client.next_turn_notification.return_value = _completed_event(
            "turn-1", TurnStatus.completed
        )
        codex_ctx.thread_resume.return_value = SimpleNamespace(
            id="thread-1", turn=MagicMock()
        )

        for mode in ("prompt_user", "approve_all"):
            with self.subTest(mode=mode):
                captured_params.clear()
                codex_ctx.thread_resume.return_value.turn.reset_mock()
                with (
                    tempfile.TemporaryDirectory() as raw,
                    override_settings(CODEX_EVENTS_DIR=Path(raw)),
                ):
                    image_path = codex_pool.input_attachments_dir() / "req" / "1.png"
                    image_path.parent.mkdir(parents=True)
                    image_path.write_bytes(b"image")
                    instance = self._make_instance(Path(raw))
                    instance.input_image_paths = [str(image_path)]
                    instance.input_attachment_paths = [str(image_path)]
                    instance.save(
                        update_fields=["input_image_paths", "input_attachment_paths"]
                    )
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        "--approval-mode",
                        mode,
                    )

                # ``Thread.turn`` (the typed SDK entry point) must NOT be
                # used — otherwise the call routes through ``ApprovalMode``
                # and the user-reviewer pairing is unreachable.
                codex_ctx.thread_resume.return_value.turn.assert_not_called()
                params = captured_params["params"]
                assert isinstance(params, TurnStartParams)
                wire_input = captured_params["input"]
                assert isinstance(wire_input, list)
                assert isinstance(wire_input[0], dict)
                assert isinstance(wire_input[1], dict)
                self.assertEqual(wire_input[0]["type"], "text")
                self.assertEqual(wire_input[0]["text"], "hi")
                self.assertEqual(wire_input[1]["type"], "localImage")
                self.assertEqual(wire_input[1]["path"], str(image_path))
                typed_input = params.input
                assert typed_input is not None
                self.assertEqual(typed_input[0].root.type, "text")
                self.assertEqual(typed_input[1].root.type, "localImage")
                self.assertEqual(getattr(typed_input[1].root, "path", None), str(image_path))
                # On-request approval policy + ``user`` reviewer means every
                # escalation is routed to the client transport. ``reviewer=None``
                # would defer to server-side routing and is NOT a safe substitute.
                approval_policy = params.approval_policy
                assert approval_policy is not None
                self.assertEqual(approval_policy.root, AskForApprovalValue.on_request)
                self.assertEqual(params.approvals_reviewer, ApprovalsReviewer.user)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_collaboration_modes_post_raw_turn_params(self, mock_codex: MagicMock) -> None:
        """Collaboration mode is exposed as a raw turn-start field, so these
        modes use the raw params path instead of the typed ``Thread.turn``."""
        captured_params: dict[str, object] = {}
        codex_ctx = mock_codex.return_value.__enter__.return_value

        def _capture_turn_start(
            _thread_id: str, _input: object, *, params: object
        ) -> object:
            captured_params["input"] = _input
            captured_params["params"] = params
            return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

        codex_ctx._client.turn_start.side_effect = _capture_turn_start
        codex_ctx._client.next_turn_notification.return_value = _completed_event(
            "turn-1", TurnStatus.completed
        )
        codex_ctx.thread_resume.return_value = SimpleNamespace(
            id="thread-1", turn=MagicMock()
        )

        cases = [
            (
                ["--plan-mode", "--model", "gpt-5.4"],
                {
                    "mode": "plan",
                    "settings": {
                        "developer_instructions": None,
                        "reasoning_effort": "medium",
                        "model": "gpt-5.4",
                    },
                },
            ),
            (
                [
                    "--collaboration-mode",
                    "default",
                    "--model",
                    "gpt-5.4",
                    "--reasoning-effort",
                    "high",
                ],
                {
                    "mode": "default",
                    "settings": {
                        "developer_instructions": _DEFAULT_COLLABORATION_INSTRUCTIONS,
                        "reasoning_effort": "high",
                        "model": "gpt-5.4",
                    },
                },
            ),
        ]
        for extra_args, expected in cases:
            with self.subTest(mode=expected["mode"]):
                captured_params.clear()
                codex_ctx.thread_resume.return_value.turn.reset_mock()
                with (
                    tempfile.TemporaryDirectory() as raw,
                    override_settings(CODEX_EVENTS_DIR=Path(raw)),
                ):
                    image_path = codex_pool.input_attachments_dir() / "req" / "1.png"
                    image_path.parent.mkdir(parents=True)
                    image_path.write_bytes(b"image")
                    instance = self._make_instance(Path(raw))
                    instance.input_image_paths = [str(image_path)]
                    instance.input_attachment_paths = [str(image_path)]
                    instance.save(
                        update_fields=["input_image_paths", "input_attachment_paths"]
                    )
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        *extra_args,
                    )

                codex_ctx.thread_resume.return_value.turn.assert_not_called()
                params = captured_params["params"]
                assert isinstance(params, dict)
                wire_input = captured_params["input"]
                params_input = params["input"]
                assert isinstance(wire_input, list)
                assert isinstance(params_input, list)
                for input_value in (wire_input, params_input):
                    assert isinstance(input_value[0], dict)
                    assert isinstance(input_value[1], dict)
                    self.assertEqual(input_value[0]["type"], "text")
                    self.assertEqual(input_value[0]["text"], "hi")
                    self.assertEqual(input_value[1]["type"], "localImage")
                    self.assertEqual(input_value[1]["path"], str(image_path))
                collaboration_mode = params["collaborationMode"]
                assert isinstance(collaboration_mode, dict)
                if expected["mode"] == "default":
                    settings = collaboration_mode["settings"]
                    assert isinstance(settings, dict)
                    instructions = settings["developer_instructions"]
                    assert isinstance(instructions, str)
                    self.assertIn("You are now in Default mode", instructions)
                    self.assertIn("previous instructions for other modes", instructions)
                self.assertEqual(collaboration_mode, expected)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_plan_mode_forwards_sandbox_and_approval_overrides(
        self, mock_codex: MagicMock
    ) -> None:
        captured_params: dict[str, object] = {}
        codex_ctx = mock_codex.return_value.__enter__.return_value

        def _capture_turn_start(
            _thread_id: str, _input: object, *, params: object
        ) -> object:
            captured_params["params"] = params
            return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

        codex_ctx._client.turn_start.side_effect = _capture_turn_start
        codex_ctx._client.next_turn_notification.return_value = _completed_event(
            "turn-1", TurnStatus.completed
        )
        codex_ctx.thread_resume.return_value = SimpleNamespace(
            id="thread-1", turn=MagicMock()
        )

        cases = [
            ("auto_review", "on-request", "auto_review"),
            ("deny_all", "never", None),
            ("prompt_user", "on-request", "user"),
            ("approve_all", "on-request", "user"),
        ]
        for mode, approval_policy, approvals_reviewer in cases:
            with self.subTest(mode=mode):
                captured_params.clear()
                codex_ctx.thread_resume.return_value.turn.reset_mock()
                with tempfile.TemporaryDirectory() as raw:
                    instance = self._make_instance(Path(raw))
                    call_command(
                        "codex_worker",
                        "--instance-id",
                        str(instance.pk),
                        "--plan-mode",
                        "--model",
                        "gpt-5.4",
                        "--sandbox-policy",
                        "workspaceWrite",
                        "--approval-mode",
                        mode,
                    )

                codex_ctx.thread_resume.return_value.turn.assert_not_called()
                params = captured_params["params"]
                assert isinstance(params, dict)
                self.assertEqual(params["approvalPolicy"], approval_policy)
                if approvals_reviewer is None:
                    self.assertNotIn("approvalsReviewer", params)
                else:
                    self.assertEqual(params["approvalsReviewer"], approvals_reviewer)
                sandbox_policy = params["sandboxPolicy"]
                assert isinstance(sandbox_policy, dict)
                self.assertEqual(sandbox_policy["type"], "workspaceWrite")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_marks_running_before_streaming(self, mock_codex: MagicMock) -> None:
        """The row flips to ``running`` before the first event so a slow
        codex initialization is visible to observers (UI / reconciliation)."""
        observed_status: dict[str, str] = {}

        def _capture_thread_resume(*_args: object, **_kwargs: object) -> object:
            observed_status["value"] = CodexInstance.objects.get(pk=instance.pk).status
            return _stub_thread_resume([_completed_event("turn-1", TurnStatus.completed)])

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = _capture_thread_resume

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        self.assertEqual(observed_status["value"], CodexInstance.STATUS_RUNNING)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_installs_interactive_approval_handler_before_streaming(
        self, mock_codex: MagicMock
    ) -> None:
        """The handler must be hooked onto ``_client._approval_handler``
        *before* ``thread_resume`` runs, otherwise an early in-stream
        approval request would still hit the SDK's default rubber-stamp
        handler instead of surfacing Hitch's interactive prompt."""
        observed_handler: dict[str, object] = {}
        codex_ctx = mock_codex.return_value.__enter__.return_value

        def _capture_thread_resume(*_args: object, **_kwargs: object) -> object:
            observed_handler["value"] = codex_ctx._client._approval_handler
            return _stub_thread_resume([_completed_event("turn-1", TurnStatus.completed)])

        codex_ctx.thread_resume.side_effect = _capture_thread_resume

        with tempfile.TemporaryDirectory() as raw:
            instance = self._make_instance(Path(raw))
            call_command("codex_worker", "--instance-id", str(instance.pk))

        handler = observed_handler["value"]
        # The handler is a closure (callable but not bound to the SDK's
        # default rubber-stamp method). Unrecognised methods short-circuit
        # to ``{}`` per the SDK's previous default-handler contract.
        assert callable(handler)
        self.assertEqual(handler("custom/method", None), {})


class ApprovalHandlerTests(TestCase):
    """``_make_approval_handler`` produces the closure the worker installs on
    ``AppServerClient._approval_handler``. The closure is called from the
    SDK's reader thread when codex escalates a command/file action.

    These tests exercise the closure directly rather than running a full
    worker turn — the surrounding worker plumbing is covered in
    ``CodexWorkerCommandTests``. Patching the polling sleep keeps the wait
    loop tight enough for unit tests.
    """

    def _make_instance(self) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=12345,
            thread_id="thread-approval",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

    def test_approve_all_handler_rubber_stamps_known_methods(self) -> None:
        """``approve_all`` must keep its "approve everything" promise even
        though we replaced the SDK's default handler. Crucially the wire
        value is the current app-server response value, ``accept``."""
        instance = self._make_instance()
        events: list[tuple[str, object]] = []

        def _record(method: str, payload: object) -> None:
            events.append((method, payload))

        handler = _make_approval_handler(
            instance=instance, write_event=_record, approval_mode="approve_all"
        )

        self.assertEqual(
            handler(
                "item/commandExecution/requestApproval",
                {"item": {"command": "ls"}},
            ),
            {"decision": "accept"},
        )
        self.assertEqual(
            handler("item/fileChange/requestApproval", {"item": {"changes": []}}),
            {"decision": "accept"},
        )
        # Unknown methods fall through to ``{}`` — the SDK uses that for
        # any server-to-client call we don't explicitly handle.
        self.assertEqual(handler("item/something/else", None), {})
        # Rubber-stamp mode emits no synthetic events; the UI doesn't
        # need an approval prompt to render in this mode.
        self.assertEqual(events, [])
        self.assertFalse(ApprovalRequest.objects.exists())

    def test_handler_routes_dynamic_tool_calls(self) -> None:
        instance = self._make_instance()
        handler = _make_approval_handler(
            instance=instance,
            write_event=lambda _method, _payload: None,
            approval_mode="prompt_user",
        )

        with (
            patch(
                "hitch.main.management.commands.codex_worker.is_dynamic_tool_call",
                return_value=True,
            ),
            patch(
                "hitch.main.management.commands.codex_worker.handle_dynamic_tool_call",
                return_value={"result": "ok"},
            ) as handle_dynamic_tool_call,
        ):
            result = handler("tool/dynamic", {"args": {"name": "value"}})

        self.assertEqual(result, {"result": "ok"})
        handle_dynamic_tool_call.assert_called_once()
        params, context = handle_dynamic_tool_call.call_args.args
        self.assertEqual(params, {"args": {"name": "value"}})
        self.assertEqual(context.cwd, "/repo")
        self.assertEqual(context.thread_id, "thread-approval")

    def test_handler_observes_live_approve_all_mode_change(self) -> None:
        instance = self._make_instance()
        events: list[tuple[str, object]] = []
        handler = _make_approval_handler(
            instance=instance,
            write_event=lambda method, payload: events.append((method, payload)),
            approval_mode="auto_review",
        )
        CodexInstance.objects.filter(pk=instance.pk).update(approval_mode="approve_all")

        self.assertEqual(
            handler(
                "item/commandExecution/requestApproval",
                {"item": {"command": "ls"}},
            ),
            {"decision": "accept"},
        )
        self.assertEqual(events, [])
        self.assertFalse(ApprovalRequest.objects.exists())

    def test_handler_observes_live_deny_all_mode_change(self) -> None:
        instance = self._make_instance()
        events: list[tuple[str, object]] = []
        handler = _make_approval_handler(
            instance=instance,
            write_event=lambda method, payload: events.append((method, payload)),
            approval_mode="prompt_user",
        )
        CodexInstance.objects.filter(pk=instance.pk).update(approval_mode="deny_all")

        self.assertEqual(
            handler(
                "item/commandExecution/requestApproval",
                {"item": {"command": "ls"}},
            ),
            {"decision": "decline"},
        )
        self.assertEqual(events, [])
        self.assertFalse(ApprovalRequest.objects.exists())

    def test_handler_rechecks_live_mode_after_creating_approval_row(self) -> None:
        def _recorder(events: list[tuple[str, object]]) -> Callable[[str, Any], None]:
            def _record(method: str, payload: Any) -> None:
                events.append((method, payload))

            return _record

        cases = [
            ("approve_all", "accept"),
            ("deny_all", "decline"),
        ]
        for mode, decision in cases:
            with self.subTest(mode=mode):
                instance = self._make_instance()
                events: list[tuple[str, object]] = []
                handler = _make_approval_handler(
                    instance=instance,
                    write_event=_recorder(events),
                    approval_mode="prompt_user",
                )

                with (
                    patch(
                        "hitch.main.management.commands.codex_worker."
                        "_current_approval_mode",
                        side_effect=["prompt_user", mode],
                    ),
                    patch(
                        "hitch.main.management.commands.codex_worker."
                        "_wait_for_decision"
                    ) as wait_for_decision,
                ):
                    result = handler(
                        "item/commandExecution/requestApproval",
                        {"item": {"command": "ls"}},
                    )

                self.assertEqual(result, {"decision": decision})
                wait_for_decision.assert_not_called()
                row = ApprovalRequest.objects.get(instance=instance)
                self.assertEqual(row.decision, decision)
                self.assertIsNotNone(row.decided_at)
                self.assertEqual(events, [])

    def test_record_live_approval_decision_preserves_existing_payload(self) -> None:
        instance = self._make_instance()
        payload = {"decision": "accept", "execPolicy": {"mode": "workspace-write"}}
        approval = ApprovalRequest.objects.create(
            instance=instance,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "ls"}},
            decision=ApprovalRequest.DECISION_ACCEPT,
            decision_payload=payload,
            decided_at=timezone.now(),
        )

        result = codex_worker_module._record_live_approval_decision(
            approval.pk,
            ApprovalRequest.DECISION_DECLINE,
        )

        self.assertEqual(result, payload)
        approval.refresh_from_db()
        self.assertEqual(approval.decision, ApprovalRequest.DECISION_ACCEPT)

    def test_record_live_approval_decision_handles_deleted_row(self) -> None:
        self.assertEqual(
            codex_worker_module._record_live_approval_decision(
                999_999,
                ApprovalRequest.DECISION_DECLINE,
            ),
            ApprovalRequest.DECISION_DECLINE,
        )

    def test_current_approval_mode_uses_fallback_for_missing_instance(self) -> None:
        instance = self._make_instance()
        pk = instance.pk
        instance.delete()

        self.assertEqual(
            codex_worker_module._current_approval_mode(
                instance=CodexInstance(pk=pk),
                fallback="prompt_user",
            ),
            "prompt_user",
        )

    def test_interactive_handler_creates_row_and_emits_events(self) -> None:
        """The interactive handler creates an ApprovalRequest row, emits an
        ``approval/requested`` event so the SSE stream surfaces the prompt,
        blocks until the row's ``decision`` is set, then emits an
        ``approval/resolved`` event with the chosen wire value.

        The polling wait is mocked because the production handler runs in
        the SDK reader thread while the view writes the decision from a
        request handler — sqlite under the test runner can't cleanly model
        that cross-thread race, so we exercise the orchestration directly
        and cover the polling loop in a separate test."""

        def _recorder(
            events: list[tuple[str, dict[str, Any]]],
        ) -> Callable[[str, Any], None]:
            def _record(method: str, payload: Any) -> None:
                assert isinstance(payload, dict)
                events.append((method, payload))

            return _record

        for approval_mode in ("auto_review", "prompt_user"):
            with self.subTest(approval_mode=approval_mode):
                instance = self._make_instance()
                events: list[tuple[str, dict[str, Any]]] = []

                handler = _make_approval_handler(
                    instance=instance,
                    write_event=_recorder(events),
                    approval_mode=approval_mode,
                )

                with patch(
                    "hitch.main.management.commands.codex_worker._wait_for_decision",
                    return_value="accept",
                ):
                    result = handler(
                        "item/commandExecution/requestApproval",
                        {"item": {"command": "rm -rf /"}},
                    )

                self.assertEqual(result, {"decision": "accept"})
                row = ApprovalRequest.objects.get(instance=instance)
                self.assertEqual(row.method, "item/commandExecution/requestApproval")
                self.assertEqual(row.params, {"item": {"command": "rm -rf /"}})
                # The pk is the link between the SSE event and the
                # ``POST /approval/<id>/`` URL the browser POSTs to.
                methods_to_payload = {m: p for m, p in events}
                self.assertIn("approval/requested", methods_to_payload)
                self.assertIn("approval/resolved", methods_to_payload)
                self.assertEqual(methods_to_payload["approval/requested"]["id"], row.pk)
                self.assertEqual(
                    methods_to_payload["approval/requested"]["method"], row.method
                )
                self.assertEqual(
                    methods_to_payload["approval/resolved"]["decision"], "accept"
                )

    def test_interactive_handler_round_trips_structured_decision(self) -> None:
        instance = self._make_instance()
        events: list[tuple[str, dict[str, Any]]] = []
        payload = {
            "acceptWithExecpolicyAmendment": {
                "execpolicy_amendment": ["just", "test"]
            }
        }

        def _record(method: str, event_payload: Any) -> None:
            assert isinstance(event_payload, dict)
            events.append((method, event_payload))

        handler = _make_approval_handler(
            instance=instance,
            write_event=_record,
            approval_mode="auto_review",
        )

        with patch(
            "hitch.main.management.commands.codex_worker._wait_for_decision",
            return_value=payload,
        ):
            result = handler(
                "item/commandExecution/requestApproval",
                {"item": {"command": "just test"}},
            )

        self.assertEqual(result, {"decision": payload})
        methods_to_payload = {method: event_payload for method, event_payload in events}
        self.assertEqual(
            methods_to_payload["approval/resolved"]["decision"], payload
        )

    def test_handler_creates_user_input_row_and_emits_events(self) -> None:
        """Plan-mode ``request_user_input`` calls use the same durable
        browser handoff shape as approvals, even in ``approve_all`` mode:
        there is no safe response to rubber-stamp because the request asks
        for structured human input."""

        def _recorder(
            events: list[tuple[str, dict[str, Any]]],
        ) -> Callable[[str, Any], None]:
            def _record(method: str, payload: Any) -> None:
                assert isinstance(payload, dict)
                events.append((method, payload))

            return _record

        params = {
            "questions": [
                {
                    "id": "trigger_surface",
                    "header": "Trigger",
                    "question": "How should the loop start?",
                    "options": [{"label": "Management command"}],
                }
            ]
        }
        response = {"answers": {"trigger_surface": "Management command"}}
        for approval_mode in ("auto_review", "approve_all"):
            for request_method in (
                "request_user_input",
                "requestUserInput",
                "item/tool/request_user_input",
                "item/tool/requestUserInput",
            ):
                with self.subTest(
                    approval_mode=approval_mode,
                    request_method=request_method,
                ):
                    instance = self._make_instance()
                    events: list[tuple[str, dict[str, Any]]] = []
                    handler = _make_approval_handler(
                        instance=instance,
                        write_event=_recorder(events),
                        approval_mode=approval_mode,
                    )

                    with patch(
                        "hitch.main.management.commands.codex_worker."
                        "_wait_for_user_input_response",
                        return_value=response,
                    ):
                        result = handler(request_method, params)

                    self.assertEqual(result, response)
                    row = UserInputRequest.objects.get(instance=instance)
                    self.assertEqual(row.method, request_method)
                    self.assertEqual(row.params, params)
                    methods_to_payload = {m: p for m, p in events}
                    self.assertIn("input/requested", methods_to_payload)
                    self.assertIn("input/resolved", methods_to_payload)
                    self.assertEqual(
                        methods_to_payload["input/requested"]["id"], row.pk
                    )
                    self.assertEqual(
                        methods_to_payload["input/requested"]["method"], row.method
                    )
                    self.assertEqual(
                        methods_to_payload["input/resolved"]["response"], response
                    )

    def test_handler_ignores_unrelated_request_user_input_substrings(self) -> None:
        instance = self._make_instance()
        events: list[tuple[str, object]] = []
        handler = _make_approval_handler(
            instance=instance,
            write_event=lambda m, p: events.append((m, p)),
            approval_mode="auto_review",
        )

        self.assertEqual(handler("custom/requestUserInputExtra", {}), {})
        self.assertFalse(UserInputRequest.objects.exists())
        self.assertEqual(events, [])

    def test_drain_steer_requests_splits_only_on_newline(self) -> None:
        """The control file is newline-delimited JSONL; the drain must split on
        ``\\n`` (matching its byte-offset accounting) and leave a trailing
        partial line for the next drain."""
        from hitch.main.management.commands import codex_worker

        instance = self._make_instance()
        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "control.jsonl"
            complete = (
                json.dumps({"op": "steer", "input": "first"}).encode()
                + b"\n"
                + json.dumps({"op": "steer", "input": "second"}).encode()
                + b"\n"
            )
            partial = json.dumps({"op": "steer", "input": "third"}).encode()
            control_path.write_bytes(complete + partial)

            steered: list[str] = []

            def _record_steer(turn: Any, text: str, **kw: Any) -> bool:
                steered.append(text)
                return True

            with patch.object(codex_worker, "_try_steer", side_effect=_record_steer):
                new_offset = codex_worker._drain_steer_requests(
                    MagicMock(),
                    instance=instance,
                    control_path=control_path,
                    control_offset=0,
                )

        self.assertEqual(steered, ["first", "second"])
        self.assertEqual(new_offset, len(complete))

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    def test_wait_for_decision_returns_recorded_decision(self) -> None:
        """Once the view records a decision on the row, the polling loop
        wakes up on the next interval and returns that wire value
        verbatim — the handler then forwards it into the SDK response."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/commandExecution/requestApproval",
            params={},
            decision="cancel",
        )

        self.assertEqual(_wait_for_decision(approval.pk), "cancel")

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    def test_wait_for_decision_returns_structured_decision_payload(self) -> None:
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        payload = {
            "acceptWithExecpolicyAmendment": {
                "execpolicy_amendment": ["uv", "run", "python"]
            }
        }
        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/commandExecution/requestApproval",
            params={},
            decision="accept",
            decision_payload=payload,
        )

        self.assertEqual(_wait_for_decision(approval.pk), payload)

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    def test_wait_for_decision_normalizes_legacy_recorded_decision(self) -> None:
        """A worker may observe an old-page POST that wrote pre-v2 decision
        strings. Normalize before answering app-server."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/commandExecution/requestApproval",
            params={},
            decision="approved",
        )

        self.assertEqual(_wait_for_decision(approval.pk), "accept")

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_WAIT_SECONDS", 0.02
    )
    def test_wait_for_decision_defaults_to_decline_on_timeout(self) -> None:
        """A stuck approval (browser tab closed, user away) must release the
        worker rather than hang the turn forever. The timeout writes
        ``decline`` to the row so the UI doesn't show a perpetually-pending
        prompt after the next page reload."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/fileChange/requestApproval",
            params={},
        )

        self.assertEqual(_wait_for_decision(approval.pk), "decline")
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "decline")
        self.assertIsNotNone(approval.decided_at)

    @patch("hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001)
    @patch("hitch.main.management.commands.codex_worker._cancel_requested", True)
    def test_wait_for_decision_declines_on_cancellation(self) -> None:
        """A Stop click sets ``_cancel_requested``; the main stream loop can't
        act on it while blocked awaiting this reply, so the wait itself must
        decline the pending action and return — letting codex unblock and the
        loop interrupt — instead of hanging until a second (SIGKILL) click."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/commandExecution/requestApproval",
            params={},
        )

        self.assertEqual(_wait_for_decision(approval.pk), "decline")
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "decline")
        self.assertIsNotNone(approval.decided_at)

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_WAIT_SECONDS", 0.0
    )
    def test_wait_for_decision_honours_user_pick_at_timeout_boundary(self) -> None:
        """When a user click lands in the window between the last empty
        read and the timeout's conditional UPDATE, the UPDATE matches
        zero rows. The handler must round-trip the user's recorded
        decision rather than overwriting it with ``decline`` (which
        would diverge the executed action from the click)."""
        from hitch.main.management.commands.codex_worker import _wait_for_decision

        approval = ApprovalRequest.objects.create(
            instance=self._make_instance(),
            method="item/commandExecution/requestApproval",
            params={},
        )

        # Simulate the race: the user POSTs the decision after the
        # initial values_list() read sees an empty decision but before
        # the timeout UPDATE runs. Patching ``values_list().get`` to
        # return ``""`` once and then handing control to the UPDATE
        # path with the real ``accept`` row already written is what
        # the production race looks like.
        original_values_list = ApprovalRequest.objects.values_list
        call_count = {"n": 0}

        def _values_list(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First read: row is still pending — flip it to
                # ``accept`` *after* we hand the empty value back, so
                # the timeout UPDATE will see ``decision != ""`` and
                # match zero rows.
                qs = original_values_list(*args, **kwargs)

                class _Wrap:
                    def get(self, **q: Any) -> tuple[str, Any]:
                        result: tuple[str, Any] = qs.get(**q)
                        ApprovalRequest.objects.filter(pk=approval.pk).update(
                            decision="accept"
                        )
                        return result

                return _Wrap()
            return original_values_list(*args, **kwargs)

        with patch.object(
            ApprovalRequest.objects, "values_list", side_effect=_values_list
        ):
            self.assertEqual(_wait_for_decision(approval.pk), "accept")
        approval.refresh_from_db()
        self.assertEqual(approval.decision, "accept")

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    def test_wait_for_user_input_response_returns_recorded_response(self) -> None:
        from hitch.main.management.commands.codex_worker import (
            _wait_for_user_input_response,
        )

        input_request = UserInputRequest.objects.create(
            instance=self._make_instance(),
            method="request_user_input",
            params={},
            response={"answers": {"scope": "UI"}},
        )

        self.assertEqual(
            _wait_for_user_input_response(input_request.pk),
            {"answers": {"scope": "UI"}},
        )

    def test_wait_for_user_input_response_defaults_to_empty_for_missing_row(self) -> None:
        from hitch.main.management.commands.codex_worker import (
            _wait_for_user_input_response,
        )

        self.assertEqual(_wait_for_user_input_response(999_999), {"answers": {}})

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001
    )
    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_WAIT_SECONDS", 0.02
    )
    def test_wait_for_user_input_response_defaults_to_empty_on_timeout(self) -> None:
        from hitch.main.management.commands.codex_worker import (
            _wait_for_user_input_response,
        )

        input_request = UserInputRequest.objects.create(
            instance=self._make_instance(),
            method="request_user_input",
            params={},
        )

        self.assertEqual(
            _wait_for_user_input_response(input_request.pk),
            {"answers": {}},
        )
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": {}})
        self.assertIsNotNone(input_request.responded_at)

    @patch("hitch.main.management.commands.codex_worker._APPROVAL_POLL_INTERVAL", 0.001)
    @patch("hitch.main.management.commands.codex_worker._cancel_requested", True)
    def test_wait_for_user_input_response_returns_empty_on_cancellation(self) -> None:
        """A Stop click must also release a worker blocked on a structured input
        request, recording the empty-answer fallback so codex unblocks and the
        main loop can interrupt rather than hanging until SIGKILL."""
        from hitch.main.management.commands.codex_worker import (
            _wait_for_user_input_response,
        )

        input_request = UserInputRequest.objects.create(
            instance=self._make_instance(),
            method="request_user_input",
            params={},
        )

        self.assertEqual(
            _wait_for_user_input_response(input_request.pk),
            {"answers": {}},
        )
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": {}})

    @patch(
        "hitch.main.management.commands.codex_worker._APPROVAL_WAIT_SECONDS", 0.0
    )
    def test_wait_for_user_input_response_honours_user_pick_at_timeout_boundary(
        self,
    ) -> None:
        """When the user's answer lands in the window between the last empty
        poll and the timeout's conditional default-write, the write matches
        zero rows. The handler must round-trip the user's recorded answer
        rather than returning the empty fallback -- otherwise codex acts on
        ``{"answers": {}}`` even though the user answered (and the browser
        already showed the answer as accepted). Mirrors
        ``test_wait_for_decision_honours_user_pick_at_timeout_boundary``."""
        from hitch.main.management.commands.codex_worker import (
            _wait_for_user_input_response,
        )

        # The user's answer is already persisted by the time the wait loop
        # reaches its timeout default-write: this is what the deadline-boundary
        # race looks like once the POST commits.
        input_request = UserInputRequest.objects.create(
            instance=self._make_instance(),
            method="request_user_input",
            params={},
            response={"answers": {"scope": "UI"}},
        )

        self.assertEqual(
            _wait_for_user_input_response(input_request.pk),
            {"answers": {"scope": "UI"}},
        )
        input_request.refresh_from_db()
        self.assertEqual(input_request.response, {"answers": {"scope": "UI"}})

    def test_interactive_handler_ignores_unknown_methods(self) -> None:
        """Approval methods we don't recognise (future SDK additions) must
        return ``{}`` without creating a stray row — the SDK treats ``{}``
        as "no opinion", which is what the previous default handler did."""
        instance = self._make_instance()
        events: list[tuple[str, object]] = []
        handler = _make_approval_handler(
            instance=instance,
            write_event=lambda m, p: events.append((m, p)),
            approval_mode="auto_review",
        )

        self.assertEqual(handler("custom/escalation", None), {})
        self.assertFalse(ApprovalRequest.objects.exists())
        self.assertEqual(events, [])


class WorkerCancellationTests(TestCase):
    """The worker's control paths: SIGTERM interrupt and SIGUSR1 steer.

    The stop endpoint signals the worker with SIGTERM; the worker turns
    that into the SDK's ``turn.interrupt()`` between stream events
    rather than dying abruptly. Active composer submissions append steer
    payloads to the worker's control file and nudge it with SIGUSR1; the
    worker control forwarder drains those payloads and calls
    ``turn.steer(...)``.
    """

    @override
    def setUp(self) -> None:
        # Module-level control state persists across tests; reset to a known
        # state so previous tests don't bleed into this one.
        codex_worker_module._cancel_requested = False
        codex_worker_module._steer_wakeup = None

    @override
    def tearDown(self) -> None:
        codex_worker_module._cancel_requested = False
        codex_worker_module._steer_wakeup = None

    def _make_instance(self, raw: str | Path) -> CodexInstance:
        return CodexInstance.objects.create(
            pid=4321,
            thread_id="t",
            cwd="/r",
            events_path=str(Path(raw) / "events.jsonl"),
            status=CodexInstance.STATUS_RUNNING,
        )

    def test_sigterm_handler_sets_flag(self) -> None:
        self.assertFalse(codex_worker_module._cancel_requested)
        codex_worker_module._on_sigterm(15, None)
        self.assertTrue(codex_worker_module._cancel_requested)

    def test_sigusr1_handler_wakes_control_forwarder(self) -> None:
        wakeup = threading.Event()
        codex_worker_module._steer_wakeup = wakeup

        self.assertFalse(wakeup.is_set())
        codex_worker_module._on_sigusr1(signal.SIGUSR1, None)
        self.assertTrue(wakeup.is_set())

    def test_try_interrupt_calls_sdk_and_reports_sent(self) -> None:
        turn = MagicMock()

        sent = codex_worker_module._try_interrupt(turn)

        self.assertTrue(sent)
        turn.interrupt.assert_called_once_with()

    def test_try_steer_calls_sdk_with_text_input_and_reports_sent(self) -> None:
        turn = MagicMock()

        sent = codex_worker_module._try_steer(turn, "also update docs")

        self.assertTrue(sent)
        turn.steer.assert_called_once()
        self.assertEqual(turn.steer.call_args.args[0].text, "also update docs")

    def test_try_steer_calls_sdk_with_text_and_images(self) -> None:
        turn = MagicMock()

        sent = codex_worker_module._try_steer(
            turn,
            "use this screenshot",
            input_image_paths=["/tmp/screen.png"],
        )

        self.assertTrue(sent)
        turn.steer.assert_called_once()
        input_items = turn.steer.call_args.args[0]
        assert isinstance(input_items, list)
        self.assertEqual(getattr(input_items[0], "text", None), "use this screenshot")
        self.assertEqual(getattr(input_items[1], "path", None), "/tmp/screen.png")

    def test_image_only_turn_input_omits_empty_text_item(self) -> None:
        input_items = codex_worker_module._turn_input(
            "",
            input_image_paths=["/tmp/screen.png"],
        )

        assert isinstance(input_items, list)
        self.assertEqual(len(input_items), 1)
        self.assertEqual(getattr(input_items[0], "path", None), "/tmp/screen.png")

    def test_image_only_typed_turn_input_omits_empty_text_item(self) -> None:
        input_items = codex_worker_module._typed_turn_input(
            "",
            input_image_paths=["/tmp/screen.png"],
        )

        self.assertEqual(len(input_items), 1)
        self.assertEqual(input_items[0].root.type, "localImage")
        self.assertEqual(getattr(input_items[0].root, "path", None), "/tmp/screen.png")

    def test_try_steer_reports_sdk_errors(self) -> None:
        turn = MagicMock()
        turn.steer.side_effect = RuntimeError("turn no longer accepts steer")

        sent = codex_worker_module._try_steer(turn, "also update docs")

        self.assertFalse(sent)
        turn.steer.assert_called_once()

    def test_drain_steer_requests_consumes_complete_jsonl_only(self) -> None:
        turn = MagicMock()
        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "worker.control.jsonl"
            control_path.write_bytes(
                b'{"op":"steer","input":"also run tests"}\n'
                b'{"op":"ignore","input":"ignored"}\n'
                b'{"op":"steer","input":"wait for newline"}'
            )

            offset = codex_worker_module._drain_steer_requests(
                turn,
                instance=self._make_instance(raw),
                control_path=control_path,
                control_offset=0,
            )

            turn.steer.assert_called_once()
            self.assertEqual(turn.steer.call_args.args[0].text, "also run tests")
            self.assertLess(offset, control_path.stat().st_size)

            with control_path.open("ab") as fh:
                fh.write(b"\n")
            offset = codex_worker_module._drain_steer_requests(
                turn,
                instance=self._make_instance(raw),
                control_path=control_path,
                control_offset=offset,
            )

            self.assertEqual(turn.steer.call_count, 2)
            self.assertEqual(turn.steer.call_args.args[0].text, "wait for newline")
            self.assertEqual(offset, control_path.stat().st_size)

    def test_drain_steer_requests_forwards_images(self) -> None:
        turn = MagicMock()
        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "worker.control.jsonl"
            control_path.write_text(
                json.dumps(
                    {
                        "op": "steer",
                        "input": "use this",
                        "inputImagePaths": ["/tmp/screen.png"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            offset = codex_worker_module._drain_steer_requests(
                turn,
                instance=self._make_instance(raw),
                control_path=control_path,
                control_offset=0,
            )
            self.assertEqual(offset, control_path.stat().st_size)

        input_items = turn.steer.call_args.args[0]
        assert isinstance(input_items, list)
        self.assertEqual(getattr(input_items[0], "text", None), "use this")
        self.assertEqual(getattr(input_items[1], "path", None), "/tmp/screen.png")

    def test_drain_steer_requests_forwards_image_only_input(self) -> None:
        turn = MagicMock()
        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "worker.control.jsonl"
            control_path.write_text(
                json.dumps(
                    {
                        "op": "steer",
                        "input": "",
                        "inputImagePaths": ["/tmp/screen.png"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            offset = codex_worker_module._drain_steer_requests(
                turn,
                instance=self._make_instance(raw),
                control_path=control_path,
                control_offset=0,
            )
            self.assertEqual(offset, control_path.stat().st_size)

        turn.steer.assert_called_once()
        input_items = turn.steer.call_args.args[0]
        assert isinstance(input_items, list)
        self.assertEqual(len(input_items), 1)
        self.assertEqual(getattr(input_items[0], "path", None), "/tmp/screen.png")

    def test_drain_steer_requests_skips_failed_steer_line(self) -> None:
        turn = MagicMock()
        turn.steer.side_effect = [RuntimeError("rejected"), None]
        first_line = b'{"op":"steer","input":"bad"}\n'
        retry_line = b'{"op":"steer","input":"retry me"}\n'
        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "worker.control.jsonl"
            control_path.write_bytes(first_line + retry_line)

            offset = codex_worker_module._drain_steer_requests(
                turn,
                instance=self._make_instance(raw),
                control_path=control_path,
                control_offset=0,
            )

            self.assertEqual(offset, len(first_line) + len(retry_line))
            self.assertEqual(turn.steer.call_count, 2)
        self.assertEqual(turn.steer.call_args.args[0].text, "retry me")

    def test_drain_steer_requests_missing_file_is_noop(self) -> None:
        turn = MagicMock()
        with tempfile.TemporaryDirectory() as raw:
            offset = codex_worker_module._drain_steer_requests(
                turn,
                instance=self._make_instance(raw),
                control_path=Path(raw) / "missing.control.jsonl",
                control_offset=0,
            )

        self.assertEqual(offset, 0)
        turn.steer.assert_not_called()

    def test_drain_steer_requests_ignores_malformed_json(self) -> None:
        turn = MagicMock()
        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "worker.control.jsonl"
            control_path.write_bytes(
                b"{not-json}\n"
                b'{"op":"steer","input":"valid after malformed"}\n'
            )

            offset = codex_worker_module._drain_steer_requests(
                turn,
                instance=self._make_instance(raw),
                control_path=control_path,
                control_offset=0,
            )
            expected_size = control_path.stat().st_size

        self.assertEqual(offset, expected_size)
        turn.steer.assert_called_once()
        self.assertEqual(
            turn.steer.call_args.args[0].text,
            "valid after malformed",
        )

    def test_drain_steer_requests_discards_failed_image_steer(self) -> None:
        turn = MagicMock()
        turn.steer.side_effect = RuntimeError("rejected")
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
                image_path = codex_pool.input_attachments_dir() / "steer" / "1.png"
                image_path.parent.mkdir(parents=True)
                image_path.write_bytes(b"image")
                instance = self._make_instance(raw)
                instance.input_attachment_paths = [str(image_path)]
                instance.save(update_fields=["input_attachment_paths"])
                control_path = Path(raw) / "worker.control.jsonl"
                control_path.write_text(
                    json.dumps(
                        {
                            "op": "steer",
                            "input": "use this",
                            "inputImagePaths": [str(image_path)],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                offset = codex_worker_module._drain_steer_requests(
                    turn,
                    instance=instance,
                    control_path=control_path,
                    control_offset=0,
                )

                self.assertEqual(offset, control_path.stat().st_size)
                self.assertFalse(image_path.exists())
                instance.refresh_from_db()
                self.assertEqual(instance.input_attachment_paths, [])

    def test_steer_forwarder_drains_once_after_stop(self) -> None:
        turn = MagicMock()
        initial_drain = threading.Event()
        original_drain = codex_worker_module._drain_steer_requests

        def drain_side_effect(
            turn_arg: Any,
            *,
            instance: CodexInstance,
            control_path: Path,
            control_offset: int,
        ) -> int:
            result = original_drain(
                turn_arg,
                instance=instance,
                control_path=control_path,
                control_offset=control_offset,
            )
            initial_drain.set()
            return result

        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "worker.control.jsonl"
            wakeup = threading.Event()
            stop = threading.Event()
            with (
                patch.object(codex_worker_module, "_STEER_CONTROL_POLL_INTERVAL", 60),
                patch.object(
                    codex_worker_module,
                    "_drain_steer_requests",
                    side_effect=drain_side_effect,
                ),
            ):
                forwarder = threading.Thread(
                    target=codex_worker_module._forward_steer_requests,
                    kwargs={
                        "turn": turn,
                        "instance": self._make_instance(raw),
                        "control_path": control_path,
                        "wakeup": wakeup,
                        "stop": stop,
                    },
                    daemon=True,
                )
                forwarder.start()
                self.assertTrue(initial_drain.wait(timeout=1))
                control_path.write_text(
                    json.dumps({"op": "steer", "input": "last chance"}) + "\n",
                    encoding="utf-8",
                )

                stop.set()
                wakeup.set()
                forwarder.join(timeout=1)

        self.assertFalse(forwarder.is_alive())
        turn.steer.assert_called_once()
        self.assertEqual(turn.steer.call_args.args[0].text, "last chance")

    def test_sigusr1_wakes_running_steer_forwarder(self) -> None:
        turn = MagicMock()
        initial_drain = threading.Event()
        original_drain = codex_worker_module._drain_steer_requests

        def drain_side_effect(
            turn_arg: Any,
            *,
            instance: CodexInstance,
            control_path: Path,
            control_offset: int,
        ) -> int:
            result = original_drain(
                turn_arg,
                instance=instance,
                control_path=control_path,
                control_offset=control_offset,
            )
            initial_drain.set()
            return result

        with tempfile.TemporaryDirectory() as raw:
            control_path = Path(raw) / "worker.control.jsonl"
            with (
                patch.object(codex_worker_module, "_STEER_CONTROL_POLL_INTERVAL", 60),
                patch.object(
                    codex_worker_module,
                    "_drain_steer_requests",
                    side_effect=drain_side_effect,
                ),
            ):
                forwarder = codex_worker_module._start_steer_control_forwarder(
                    turn,
                    instance=self._make_instance(raw),
                    control_path=control_path,
                )
                try:
                    self.assertTrue(initial_drain.wait(timeout=1))
                    control_path.write_text(
                        json.dumps({"op": "steer", "input": "wake up"}) + "\n",
                        encoding="utf-8",
                    )
                    codex_worker_module._on_sigusr1(signal.SIGUSR1, None)

                    deadline = time.monotonic() + 1
                    while turn.steer.call_count == 0 and time.monotonic() < deadline:
                        time.sleep(0.01)
                    turn.steer.assert_called_once()
                    self.assertEqual(turn.steer.call_args.args[0].text, "wake up")
                finally:
                    codex_worker_module._stop_steer_control_forwarder(forwarder)

    def test_try_interrupt_swallows_sdk_errors(self) -> None:
        # A failed SDK call (turn already done, transport hiccup) must
        # still report "sent" so the loop doesn't retry forever — the
        # user's escalation is SIGKILL via a second click, not a retry.
        turn = MagicMock()
        turn.interrupt.side_effect = RuntimeError("turn already terminated")

        sent = codex_worker_module._try_interrupt(turn)

        self.assertTrue(sent)
        turn.interrupt.assert_called_once_with()

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_interrupt_fires_when_flag_set_during_stream(
        self, mock_codex: MagicMock
    ) -> None:
        # Drive the stream loop through a flag-set transition: yield
        # one event, then arrange for the cancel flag to be observed
        # on the next iteration, then yield the turn/completed.
        # Verify exactly one ``turn.interrupt()`` call.
        first_event = SimpleNamespace(
            method="item/agentMessage/delta",
            payload=_FakePayload(detail="chunk"),
        )
        completion = _completed_event("turn-1", TurnStatus.completed)

        def gen() -> Iterator[Any]:
            yield first_event
            codex_worker_module._cancel_requested = True
            yield completion

        captured: dict[str, Any] = {}

        def thread_resume_side_effect(*_args: object, **_kwargs: object) -> object:
            turn_mock = MagicMock()
            turn_mock.id = "turn-1"
            turn_mock.stream.return_value = gen()
            captured["turn"] = turn_mock
            return SimpleNamespace(turn=lambda _input: turn_mock)

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = thread_resume_side_effect

        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                pid=12345,
                thread_id="thread-1",
                cwd="/repo",
                prompt="hi",
                events_path=str(Path(raw) / "events.jsonl"),
                status=CodexInstance.STATUS_STARTING,
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        captured["turn"].interrupt.assert_called_once_with()

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_queued_steer_file_drained_after_turn_handle_exists(
        self, mock_codex: MagicMock
    ) -> None:
        completion = _completed_event("turn-1", TurnStatus.completed)
        captured: dict[str, Any] = {}

        def thread_resume_side_effect(*_args: object, **_kwargs: object) -> object:
            turn_mock = MagicMock()
            turn_mock.id = "turn-1"
            turn_mock.stream.return_value = iter([completion])
            captured["turn"] = turn_mock
            return SimpleNamespace(turn=lambda _input: turn_mock)

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = thread_resume_side_effect

        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                pid=12345,
                thread_id="thread-1",
                cwd="/repo",
                prompt="hi",
                events_path=str(Path(raw) / "events.jsonl"),
                status=CodexInstance.STATUS_STARTING,
            )
            codex_pool.control_path_for(instance).write_text(
                json.dumps({"op": "steer", "input": "also check docs"}) + "\n",
                encoding="utf-8",
            )

            call_command("codex_worker", "--instance-id", str(instance.pk))

        captured["turn"].steer.assert_called_once()
        self.assertEqual(captured["turn"].steer.call_args.args[0].text, "also check docs")

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_interrupt_called_exactly_once_when_flag_stays_set(
        self, mock_codex: MagicMock
    ) -> None:
        # If multiple SIGTERMs arrive (or the flag was never cleared),
        # the loop must not re-send ``turn.interrupt()`` on every event.
        # The escalation lever is SIGKILL via a fresh click, not
        # spamming the SDK with cancellation requests.
        events = [
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=_FakePayload(detail="a"),
            ),
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=_FakePayload(detail="b"),
            ),
            _completed_event("turn-1", TurnStatus.completed),
        ]

        # Set the flag before the worker starts so it observes on the
        # initial pre-loop check; then more events arrive — interrupt
        # must not be re-called on each.
        codex_worker_module._cancel_requested = True

        captured: dict[str, Any] = {}

        def thread_resume_side_effect(*_args: object, **_kwargs: object) -> object:
            turn_mock = MagicMock()
            turn_mock.id = "turn-1"
            turn_mock.stream.return_value = iter(events)
            captured["turn"] = turn_mock
            return SimpleNamespace(turn=lambda _input: turn_mock)

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = thread_resume_side_effect

        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                pid=12345,
                thread_id="thread-1",
                cwd="/repo",
                prompt="hi",
                events_path=str(Path(raw) / "events.jsonl"),
                status=CodexInstance.STATUS_STARTING,
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        captured["turn"].interrupt.assert_called_once_with()

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_no_interrupt_when_flag_never_set(
        self, mock_codex: MagicMock
    ) -> None:
        # Sanity check: the normal turn flow doesn't call interrupt.
        completion = _completed_event("turn-1", TurnStatus.completed)

        captured: dict[str, Any] = {}

        def thread_resume_side_effect(*_args: object, **_kwargs: object) -> object:
            turn_mock = MagicMock()
            turn_mock.id = "turn-1"
            turn_mock.stream.return_value = iter([completion])
            captured["turn"] = turn_mock
            return SimpleNamespace(turn=lambda _input: turn_mock)

        codex_ctx = mock_codex.return_value.__enter__.return_value
        codex_ctx.thread_resume.side_effect = thread_resume_side_effect

        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                pid=12345,
                thread_id="thread-1",
                cwd="/repo",
                prompt="hi",
                events_path=str(Path(raw) / "events.jsonl"),
                status=CodexInstance.STATUS_STARTING,
            )
            call_command("codex_worker", "--instance-id", str(instance.pk))

        captured["turn"].interrupt.assert_not_called()


# A pid we know exists (this Python process). Tests that need it to represent
# a Codex worker patch ``worker_is_alive`` because worker liveness now verifies
# process identity, not just pid existence.
_LIVE_PID = os.getpid()


def _make_streaming_instance(
    events_path: str,
    *,
    thread_id: str = "thread-1",
    status: str = CodexInstance.STATUS_RUNNING,
    prompt: str = "do work",
    pid: int = 0,
) -> CodexInstance:
    return CodexInstance.objects.create(
        pid=pid,
        thread_id=thread_id,
        cwd="/repo",
        prompt=prompt,
        events_path=events_path,
        status=status,
    )


class StreamForInstanceTests(TestCase):
    """The SSE generator that re-emits a worker's JSONL events file frame-by-
    frame. Pairs with ``streaming.stream_for_instance`` / ``empty_stream``.
    """

    def test_idle_stream_emits_initial_heartbeat_and_recycles(self) -> None:
        # The "no active worker" path keeps the SSE channel open so the
        # session page's connection indicator can show ``connected, idle``.
        # Force the cap down so the loop returns promptly without an end
        # event — EventSource will reconnect transparently.
        with (
            patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001),
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
        ):
            frames = list(streaming.idle_stream("thread-none", baseline_id=None))
        self.assertEqual(frames[0], b"retry: 2000\n\n")
        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        self.assertGreaterEqual(len(heartbeats), 1)
        self.assertIn(b'"working": false', heartbeats[0])
        # No worker ever showed up, so the stream closes silently rather
        # than firing an ``end`` event that would force a page reload.
        self.assertFalse(any(f.startswith(b"event: end") for f in frames))

    def test_idle_stream_ends_when_worker_appears(self) -> None:
        # If a worker is spawned out-of-band (e.g. another tab), the idle
        # stream should fire ``event: end`` so the client reloads into the
        # live-streaming UI rather than waiting for the per-stream cap.
        # The baseline (``None``) reflects what the page saw at render
        # time; a fresh pk on the first poll proves an out-of-band turn
        # has landed since then.
        with (
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
            patch(
                "hitch.main.streaming.codex_pool.latest_id_for_thread",
                return_value=42,
            ),
        ):
            frames = list(streaming.idle_stream("thread-active", baseline_id=None))
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"active"', frames[-1])

    def test_idle_stream_ends_on_fast_completed_out_of_band_turn(self) -> None:
        # Reload-detection has to fire even for an out-of-band turn that
        # already finished by the time we look — pk tracking catches it
        # where a "still active?" check would have missed it. Page saw
        # baseline=7 at render; one poll later the DB shows pk=9 even
        # though no row is currently active.
        with (
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
            patch(
                "hitch.main.streaming.codex_pool.latest_id_for_thread",
                return_value=9,
            ),
        ):
            frames = list(
                streaming.idle_stream("thread-completed-fast", baseline_id=7)
            )
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"active"', frames[-1])

    def test_idle_stream_ends_when_demo_status_changes(self) -> None:
        SessionDemo.objects.create(
            thread_id="thread-demo",
            host="127.0.0.1",
            port=45678,
            status=SessionDemo.STATUS_REQUESTED,
        )
        baseline = streaming.demo_stream_token("thread-demo")
        SessionDemo.objects.filter(thread_id="thread-demo").update(
            status=SessionDemo.STATUS_ACTIVE
        )
        with (
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
            patch(
                "hitch.main.streaming.codex_pool.latest_id_for_thread",
                return_value=None,
            ),
        ):
            frames = list(
                streaming.idle_stream(
                    "thread-demo",
                    baseline_id=None,
                    demo_baseline=baseline,
                )
            )
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"demo"', frames[-1])

    @patch("hitch.main.streaming._HEARTBEAT_INTERVAL", 0.0)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.005)
    def test_idle_stream_resends_heartbeats_at_cadence(self) -> None:
        # With the heartbeat cadence collapsed to zero we should observe
        # multiple heartbeat frames before the per-stream cap closes the
        # stream — confirming the periodic refresh path actually runs.
        frames = list(streaming.idle_stream("thread-none", baseline_id=None))
        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        self.assertGreater(len(heartbeats), 1)
        for frame in heartbeats:
            self.assertIn(b'"working": false', frame)

    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    @patch("hitch.main.codex_pool.worker_is_alive", return_value=True)
    def test_system_workflow_stream_reports_working(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).write_text(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "hidden-thread",
                            "goal": {
                                "objective": "Review the diff",
                                "tokensUsed": 99,
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instance = _make_streaming_instance(
                events_path,
                thread_id="hidden-thread",
                pid=_LIVE_PID,
            )
            instance.purpose = CodexInstance.PURPOSE_SYSTEM_AGENT
            instance.workflow_id = workflow.pk
            instance.agent_kind = "pr_qa"
            instance.display_author = "QA agent"
            instance.save(
                update_fields=[
                    "purpose",
                    "workflow_id",
                    "agent_kind",
                    "display_author",
                ]
            )
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind="pr_qa",
                thread_id="hidden-thread",
                instance=instance,
                status=SystemAgentRun.STATUS_RUNNING,
            )

            frames = list(
                streaming.system_workflow_stream(
                    "thread-workflow", baseline_id=None, workflow_id=workflow.pk
                )
            )

        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        self.assertGreaterEqual(len(heartbeats), 1)
        self.assertIn(b'"working": true', heartbeats[0])
        self.assertIn(b'"statusText": "QA agent working...99 tokens"', heartbeats[0])

    def test_system_workflow_stream_uses_scoped_dead_reconciliation(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )

        with (
            patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.001),
            patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001),
            patch("hitch.main.streaming.codex_pool.reconcile_dead") as mock_global,
            patch(
                "hitch.main.streaming.codex_pool.reconcile_dead_for_workflow"
            ) as mock_scoped,
        ):
            frames = list(
                streaming.system_workflow_stream(
                    "thread-workflow", baseline_id=None, workflow_id=workflow.pk
                )
            )

        self.assertTrue(frames)
        mock_scoped.assert_any_call(workflow.pk, main_thread_id="thread-workflow")
        mock_global.assert_not_called()

    @patch("hitch.main.streaming._HEARTBEAT_INTERVAL", 0.0)
    @patch("hitch.main.streaming._IDLE_MAX_STREAM_SECONDS", 0.05)
    @patch("hitch.main.streaming._IDLE_POLL_INTERVAL", 0.001)
    def test_system_workflow_stream_resends_status_heartbeat(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )

        frames = list(
            streaming.system_workflow_stream(
                "thread-workflow", baseline_id=None, workflow_id=workflow.pk
            )
        )

        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        self.assertGreater(len(heartbeats), 1)
        self.assertIn(b'"statusText": "QA agent working..."', heartbeats[-1])

    def test_system_workflow_stream_ends_when_workflow_stops(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-workflow",
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step="blocked",
        )

        frames = list(
            streaming.system_workflow_stream(
                "thread-workflow", baseline_id=None, workflow_id=workflow.pk
            )
        )

        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"workflow"', frames[-1])

    def test_reload_stream_yields_immediate_end(self) -> None:
        # ``session_stream`` returns this when it detects the page is
        # stale (worker spawned/completed between page render and SSE
        # open). It needs to fire ``event: end`` immediately so the
        # client reloads — no heartbeats, no waiting on a poll.
        frames = list(streaming.reload_stream())
        self.assertEqual(frames[0], b"retry: 2000\n\n")
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"stale"', frames[-1])

    def test_streams_existing_lines_and_terminates_when_done(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"method": "item/started", "payload": {"item": {"id": "x"}}})
                    + "\n"
                )
                fh.write(json.dumps({"method": "turn/completed", "payload": {}}) + "\n")

            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_COMPLETED)
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 2)
        self.assertIn(b"item/started", data_frames[0])
        self.assertIn(b"turn/completed", data_frames[1])
        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertIn(b'"completed"', frames[-1])

    def test_initial_backlog_skips_completed_agent_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "method": "item/started",
                            "payload": {
                                "item": {
                                    "id": "msg-1",
                                    "text": "",
                                    "type": "agentMessage",
                                }
                            },
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "method": "item/agentMessage/delta",
                            "payload": {"itemId": "msg-1", "delta": "Hel"},
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "method": "item/agentMessage/delta",
                            "payload": {"itemId": "msg-1", "delta": "lo"},
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
                                    "id": "msg-1",
                                    "phase": "commentary",
                                    "text": "Hello",
                                    "type": "agentMessage",
                                }
                            },
                        }
                    )
                    + "\n"
                )

            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_COMPLETED
            )
            frames = list(streaming.stream_for_instance(instance))

        body = b"".join(frames)
        self.assertNotIn(b"item/agentMessage/delta", body)
        self.assertNotIn(b"item/started", body)
        self.assertIn(b"item/completed", body)
        self.assertIn(b"Hello", body)

    def test_initial_backlog_preserves_partial_agent_deltas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "method": "item/started",
                            "payload": {
                                "item": {
                                    "id": "msg-1",
                                    "text": "",
                                    "type": "agentMessage",
                                }
                            },
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "method": "item/agentMessage/delta",
                            "payload": {"itemId": "msg-1", "delta": "Hel"},
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "method": "item/agentMessage/delta",
                            "payload": {"itemId": "msg-1", "delta": "lo"},
                        }
                    )
                    + "\n"
                )

            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_COMPLETED
            )
            frames = list(streaming.stream_for_instance(instance))

        body = b"".join(frames)
        self.assertIn(b"item/started", body)
        self.assertIn(b'"delta": "Hel"', body)
        self.assertIn(b'"delta": "lo"', body)
        self.assertNotIn(b'"text":"Hello"', body)

    def test_initial_backlog_skips_only_method_output_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "method": "item/started",
                            "payload": {
                                "item": {
                                    "id": "cmd-1",
                                    "command": "rg outputDelta",
                                    "status": "inProgress",
                                    "type": "commandExecution",
                                }
                            },
                        }
                    )
                    + "\n"
                )
                fh.write(
                    json.dumps(
                        {
                            "method": "item/commandExecution/outputDelta",
                            "payload": {"itemId": "cmd-1", "delta": "ignored\n"},
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
                                    "id": "cmd-1",
                                    "command": "rg outputDelta",
                                    "status": "completed",
                                    "type": "commandExecution",
                                }
                            },
                        }
                    )
                    + "\n"
                )

            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_COMPLETED
            )
            frames = list(streaming.stream_for_instance(instance))

        body = b"".join(frames)
        self.assertIn(b"item/started", body)
        self.assertIn(b"item/completed", body)
        self.assertIn(b"rg outputDelta", body)
        self.assertNotIn(b"item/commandExecution/outputDelta", body)
        self.assertNotIn(b"ignored", body)

    def test_empty_initial_read_treats_next_agent_delta_as_live_tail(self) -> None:
        live = (
            json.dumps(
                {
                    "method": "item/started",
                    "payload": {
                        "item": {"id": "msg-1", "text": "", "type": "agentMessage"}
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "method": "item/agentMessage/delta",
                    "payload": {"itemId": "msg-1", "delta": "Hel"},
                }
            )
            + "\n"
        ).encode()
        fake_file = MagicMock()
        fake_file.__enter__.return_value = fake_file
        fake_file.__exit__.return_value = False
        fake_file.read.side_effect = [b"", live, b""]
        instance = _make_streaming_instance(
            "/tmp/hitch-test-empty-first-read.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
        )

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", return_value=fake_file),
        ):
            frames = list(streaming.stream_for_instance(instance))

        body = b"".join(frames)
        self.assertIn(b"item/started", body)
        self.assertIn(b'"delta": "Hel"', body)
        self.assertNotIn(b'"text":"Hel"', body)

    def test_live_tail_agent_delta_still_streams_after_initial_backlog(self) -> None:
        initial = (
            json.dumps(
                {
                    "method": "item/started",
                    "payload": {
                        "item": {"id": "msg-1", "text": "", "type": "agentMessage"}
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "method": "item/agentMessage/delta",
                    "payload": {"itemId": "msg-1", "delta": "old"},
                }
            )
            + "\n"
        ).encode()
        live = (
            json.dumps(
                {
                    "method": "item/agentMessage/delta",
                    "payload": {"itemId": "msg-1", "delta": "new"},
                }
            )
            + "\n"
        ).encode()
        fake_file = MagicMock()
        fake_file.__enter__.return_value = fake_file
        fake_file.__exit__.return_value = False
        fake_file.read.side_effect = [initial, b"", live, b""]
        instance = _make_streaming_instance(
            "/tmp/hitch-test-live-tail.jsonl", status=CodexInstance.STATUS_COMPLETED
        )

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", return_value=fake_file),
        ):
            frames = list(streaming.stream_for_instance(instance))

        body = b"".join(frames)
        self.assertIn(b"item/started", body)
        self.assertIn(b'"delta": "old"', body)
        self.assertIn(b'"delta": "new"', body)
        self.assertNotIn(b'"text":"old"', body)

    def test_streams_tolerate_partial_multibyte_trailing_line(self) -> None:
        # The worker writes line-buffered UTF-8; a reader can observe a flush
        # that ends mid-multibyte-character. A strict text-mode read raised
        # UnicodeDecodeError and dropped the connection with no end frame.
        # Binary buffering must hold the torn tail and still emit the complete
        # line that preceded it.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            complete = (
                json.dumps(
                    {"method": "item/started", "payload": {"item": {"id": "café"}}},
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            snowman = "☃".encode()  # 3 bytes; keep only the first
            with open(events_path, "wb") as fh:
                fh.write(complete)
                fh.write(b'{"method": "agent/delta", "payload": "' + snowman[:1])

            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_COMPLETED)
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertIn(b"item/started", data_frames[0])
        self.assertIn("café".encode(), data_frames[0])
        self.assertTrue(frames[-1].startswith(b"event: end"))

    def test_qa_instance_heartbeat_reports_goal_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).write_text(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "goal": {
                                "objective": "Review the diff",
                                "tokensUsed": 1200,
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            instance = _make_streaming_instance(
                events_path,
                status=CodexInstance.STATUS_COMPLETED,
            )
            instance.display_author = "QA agent"
            instance.save(update_fields=["display_author"])

            frames = list(streaming.stream_for_instance(instance))

        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        self.assertIn(b'"statusText": "QA agent working...1.2K tokens"', heartbeats[0])

    def test_qa_instance_status_text_falls_back_without_tokens(self) -> None:
        instance = _make_streaming_instance(
            "/tmp/hitch-test-missing-qa-progress.jsonl",
            status=CodexInstance.STATUS_COMPLETED,
        )
        instance.display_author = "QA agent"
        instance.save(update_fields=["display_author"])

        self.assertEqual(
            streaming.qa_agent_status_text_for_instance(instance),
            "QA agent working...",
        )

    def test_compact_token_count_formatter(self) -> None:
        self.assertEqual(streaming._format_compact_token_count(-1), "0")
        self.assertEqual(streaming._format_compact_token_count(999), "999")
        self.assertEqual(streaming._format_compact_token_count(1200), "1.2K")
        self.assertEqual(streaming._format_compact_token_count(1250), "1.3K")
        self.assertEqual(streaming._format_compact_token_count(10_500), "11K")
        self.assertEqual(streaming._format_compact_token_count(999_950), "1M")
        self.assertEqual(streaming._format_compact_token_count(13_000_000), "13M")
        self.assertEqual(streaming._format_compact_token_count(1_000_000_000), "1B")

    def test_system_workflow_status_text_handles_non_qa_workflow(self) -> None:
        workflow = cast(SystemWorkflow, SimpleNamespace(kind="other"))

        self.assertEqual(
            streaming.system_workflow_status_text(workflow),
            "Hitch system agent is working...",
        )

    def test_system_workflow_status_text_handles_spec_critic_classifying(self) -> None:
        workflow = cast(
            SystemWorkflow,
            SimpleNamespace(
                kind="spec_critic",
                step="spec_critic_classifying",
                state={},
            ),
        )

        self.assertEqual(
            streaming.system_workflow_status_text(workflow),
            "Spec Critic is reviewing the request...",
        )

    def test_system_workflow_status_text_handles_spec_critic_analyzing(self) -> None:
        workflow = cast(
            SystemWorkflow,
            SimpleNamespace(
                kind="spec_critic",
                step="spec_critic_analyzing",
                state={},
            ),
        )

        self.assertEqual(
            streaming.system_workflow_status_text(workflow),
            "Spec Critic is analyzing the request...",
        )

    def test_system_workflow_status_text_handles_qa_feedback_step(self) -> None:
        workflow = cast(
            SystemWorkflow,
            SimpleNamespace(
                kind=SystemWorkflow.KIND_PR_QA,
                step="feedback_running",
                state={},
            ),
        )

        self.assertEqual(
            streaming.system_workflow_status_text(workflow),
            "QA feedback agent is fixing feedback...",
        )

    def test_terminates_when_status_flips_to_failed(self) -> None:
        # A worker that ended with a failure status still flushes its events
        # file, but the end frame should carry the actual terminal status so
        # the client can surface the failure UI.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "item/started", "payload": {}}) + "\n")

            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_FAILED)
            frames = list(streaming.stream_for_instance(instance))

        self.assertIn(b'"failed"', frames[-1])

    def test_ignores_partial_trailing_line(self) -> None:
        # The worker is line-buffered, so a tailer that opens the file at the
        # exact moment a half-line is on disk must not emit a malformed JSON
        # frame to the client.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "item/started", "payload": {}}) + "\n")
                # Half a JSON object, no trailing newline.
                fh.write('{"method":"item/partial')

            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_COMPLETED)
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertIn(b"item/started", data_frames[0])
        self.assertNotIn(b"item/partial", b"".join(frames))

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.01)
    def test_missing_events_file_with_dead_worker_ends_promptly(self) -> None:
        # If the events file never appears but the worker process is also
        # gone, the tailer must bail rather than wait the full appearance
        # timeout. ``pid=99999999`` is well above the kernel's pid_max so
        # ``os.kill(pid, 0)`` raises ProcessLookupError → is_alive False.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "never-created.jsonl")
            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=99999999
            )
            frames = list(streaming.stream_for_instance(instance))

        self.assertTrue(frames[-1].startswith(b"event: end"))
        self.assertNotIn(b'"missing"', frames[-1])

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.005)
    @patch("hitch.main.streaming._FILE_APPEAR_TIMEOUT", 0.001)
    @patch("hitch.main.streaming.codex_pool.worker_is_alive", return_value=True)
    def test_appearance_timeout_when_file_never_arrives(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        # A live worker that's stuck before its first write should bail via
        # the appearance timeout rather than hanging the request handler.
        instance = _make_streaming_instance(
            "/tmp/hitch-test-never-created.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            pid=_LIVE_PID,
        )
        frames = list(streaming.stream_for_instance(instance))
        self.assertIn(b'"missing"', frames[-1])

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.001)
    @patch("hitch.main.streaming._HEARTBEAT_INTERVAL", 0.0)
    @patch("hitch.main.streaming._FILE_APPEAR_TIMEOUT", 0.02)
    @patch("hitch.main.streaming.codex_pool.worker_is_alive", return_value=True)
    def test_heartbeat_yielded_while_waiting_for_events_file(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        # The pre-file-open wait loop also has to refresh the heartbeat so
        # the page's connection indicator stays green during a worker's
        # startup latency, not just once the file is being tailed.
        instance = _make_streaming_instance(
            "/tmp/hitch-test-startup-heartbeat.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            pid=_LIVE_PID,
        )
        frames = list(streaming.stream_for_instance(instance))
        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        # One initial heartbeat plus at least one from the appearance wait.
        self.assertGreaterEqual(len(heartbeats), 2)
        for frame in heartbeats:
            self.assertIn(b'"working": true', frame)
        self.assertIn(b'"missing"', frames[-1])

    def test_emit_skips_blank_lines(self) -> None:
        # A stray blank line on the events file must not produce an empty
        # SSE ``data:`` frame.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "a", "payload": {}}) + "\n")
                fh.write("\n")
                fh.write(json.dumps({"method": "b", "payload": {}}) + "\n")
            instance = _make_streaming_instance(events_path, status=CodexInstance.STATUS_COMPLETED)
            frames = list(streaming.stream_for_instance(instance))
        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 2)

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.005)
    @patch("hitch.main.streaming._MAX_STREAM_SECONDS", 0.001)
    @patch("hitch.main.streaming.codex_pool.worker_is_alive", return_value=True)
    def test_read_loop_hits_stream_timeout(
        self, _mock_worker_alive: MagicMock
    ) -> None:
        # A live worker that goes silent for longer than the per-stream cap
        # must release the request thread; the browser's EventSource will
        # reconnect if the user is still on the page.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).touch()
            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=_LIVE_PID
            )
            frames = list(streaming.stream_for_instance(instance))
        self.assertIn(b'"timeout"', frames[-1])

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.01)
    @patch("hitch.main.streaming.codex_pool._pid_is_our_worker", return_value=False)
    @patch("hitch.main.streaming.codex_pool.is_alive", return_value=True)
    def test_running_instance_with_recycled_pid_terminates(
        self,
        mock_alive: MagicMock,
        mock_identity: MagicMock,
    ) -> None:
        # Generic pid checks can say "alive" after PID reuse. The stream must
        # still end because the process no longer matches this worker row.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).write_text(
                json.dumps({"method": "item/started", "payload": {}}) + "\n",
                encoding="utf-8",
            )
            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=4321
            )
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertTrue(frames[-1].startswith(b"event: end"))
        mock_identity.assert_called_once_with(4321, instance.pk)
        mock_alive.assert_not_called()

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.001)
    @patch("hitch.main.streaming._HEARTBEAT_INTERVAL", 0.0)
    def test_heartbeat_yielded_while_worker_is_idle(self) -> None:
        # Idle SSE connections get periodic heartbeat events so the page's
        # connection indicator can refresh and the channel stays open past
        # idle proxies. Force _is_done False on the first poll and True on
        # the second so we observe at least one heartbeat frame past the
        # initial one before the stream finishes cleanly.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            Path(events_path).touch()
            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=_LIVE_PID
            )

            done_calls = [0]

            def fake_is_done(_pk: int) -> bool:
                done_calls[0] += 1
                return done_calls[0] >= 2

            with patch("hitch.main.streaming._is_done", side_effect=fake_is_done):
                frames = list(streaming.stream_for_instance(instance))

        heartbeats = [f for f in frames if f.startswith(b"event: heartbeat")]
        # One initial heartbeat plus at least one from the idle poll loop.
        self.assertGreaterEqual(len(heartbeats), 2)
        for frame in heartbeats:
            self.assertIn(b'"working": true', frame)
        self.assertTrue(frames[-1].startswith(b"event: end"))

    def test_late_flush_after_status_flip_is_picked_up(self) -> None:
        # The worker's status transition and its final file write are not
        # atomic: a turn/completed event can land on disk *after* the row
        # already shows COMPLETED. The post-done re-read catches that line.
        fake_file = MagicMock()
        fake_file.__enter__.return_value = fake_file
        fake_file.__exit__.return_value = False
        fake_file.read.side_effect = [
            b"",
            (json.dumps({"method": "turn/completed", "payload": {}}) + "\n").encode("utf-8"),
            b"",
        ]

        instance = _make_streaming_instance(
            "/tmp/hitch-test-late.jsonl", status=CodexInstance.STATUS_COMPLETED
        )

        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "open", return_value=fake_file),
        ):
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertIn(b"turn/completed", data_frames[0])

    def test_is_done_treats_missing_instance_as_terminal(self) -> None:
        # A row deleted out from under the tailer (cleanup race) ends the
        # stream rather than crashing the generator with DoesNotExist.
        self.assertTrue(streaming._is_done(99999999))

    def test_current_status_returns_unknown_for_missing_instance(self) -> None:
        # Symmetric to _is_done: the end-frame status falls back to a
        # stable sentinel when the row is gone.
        self.assertEqual(streaming._current_status(99999999), "unknown")

    @patch("hitch.main.streaming._POLL_INTERVAL", 0.01)
    def test_running_instance_with_dead_pid_terminates(self) -> None:
        # The events file exists but the worker died before flipping its
        # status; ``is_alive`` short-circuits the otherwise-infinite read.
        with tempfile.TemporaryDirectory() as raw:
            events_path = str(Path(raw) / "events.jsonl")
            with open(events_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"method": "item/started", "payload": {}}) + "\n")

            instance = _make_streaming_instance(
                events_path, status=CodexInstance.STATUS_RUNNING, pid=99999999
            )
            frames = list(streaming.stream_for_instance(instance))

        data_frames = [f for f in frames if f.startswith(b"data: ")]
        self.assertEqual(len(data_frames), 1)
        self.assertTrue(frames[-1].startswith(b"event: end"))
