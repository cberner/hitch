"""Native goal controls, persistence, and continuation routing."""

import contextlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, override
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from openai_codex import Codex, CodexConfig, TurnHandle
from openai_codex.client import CodexClient
from openai_codex.errors import CodexError
from openai_codex.generated.v2_all import ThreadGoal, ThreadGoalUpdatedNotification, Turn, TurnStatus
from openai_codex.models import Notification

from hitch.main.management.commands import codex_worker
from hitch.main.models import ApprovalRequest, CodexInstance, SessionMetadata
from hitch.main.runtime import codex_pool, session_goals
from hitch.main.test.support import _setup_codex
from hitch.main.test.test_codex_subprocess import _turn_completed_event, _turn_started_event


def goal_snapshot(**kwargs: Any) -> dict[str, Any]:
    return {
        "threadId": "session", "objective": "Finish the proof", "status": "paused",
        "tokenBudget": 1000, "tokensUsed": 120, "timeUsedSeconds": 7200,
        "createdAt": 100, "updatedAt": 200, **kwargs,
    }


class SessionGoalTests(TestCase):
    @override
    def setUp(self) -> None:
        self.metadata = SessionMetadata.objects.create(thread_id="session", cwd="/repo")
        self.url = reverse("set_session_goal", args=["session"])
        self.goal: dict[str, Any] | None = goal_snapshot()
        patches: list[tuple[str, dict[str, Any]]] = [
            ("hitch.main.views.common._is_allowed_session_cwd", {"return_value": True}),
            ("hitch.main.runtime.session_goals.current_goal", {"side_effect": lambda _id: self.goal}),
            ("hitch.main.runtime.session_goals.prepare_home", {"return_value": Path("/tmp/goal-home")}),
            ("hitch.main.runtime.session_goals.acquire_home", {}),
            ("hitch.main.runtime.session_goals.compact_cleared_home", {}),
            ("hitch.main.runtime.reconciliation.reconcile_dead_for_thread", {}),
            ("hitch.main.views.common.Codex", {}),
            ("hitch.main.runtime.codex_pool.spawn_turn", {}),
        ]
        for target, kwargs in patches:
            patcher = patch(target, **kwargs)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith(".Codex"):
                self.native = _setup_codex(mocked)._client
                self.native._request_raw.return_value = {"goal": self.goal}
            elif target.endswith(".spawn_turn"):
                self.spawn = mocked
            elif target.endswith(".prepare_home"):
                self.prepare = mocked
            elif target.endswith(".acquire_home"):
                self.acquire = mocked

    def test_validation_and_read_only_never_mutate_or_start(self) -> None:
        self.assertEqual(self.client.get(self.url).status_code, 405)
        for data in (
            {"action": "unknown"}, {"action": "start", "objective": ""},
            {"action": "start", "objective": "x" * 4001},
            *({"action": "budget", "objective": "valid", "token_budget": b}
              for b in ["-1", "0", "1.5", "NaN", "９", "9007199254740992"]),
        ):
            self.assertEqual(self.client.post(self.url, data).status_code, 400)
        for field in ["is_hidden_system_session", "codex_archived"]:
            setattr(self.metadata, field, True)
            self.metadata.save()
            self.assertEqual(self.client.post(self.url, {"action": "resume"}).status_code, 409)
            setattr(self.metadata, field, False)
        self.native._request_raw.assert_not_called()
        self.spawn.assert_not_called()

    def test_budget_edit_preserves_status_and_resume_uses_session_settings(self) -> None:
        response = self.client.post(self.url, {"action": "budget", "objective": "Finish the proof", "token_budget": ""})
        self.assertEqual(response.status_code, 200)
        self.native._request_raw.assert_called_once_with("thread/goal/set", {
            "threadId": "session", "tokenBudget": None,
        })
        self.native.thread_resume.assert_not_called()
        self.spawn.assert_not_called()
        self.prepare.assert_called_once_with("session")
        self.acquire.return_value.release.assert_called_once()
        self.metadata.model, self.metadata.reasoning_effort = "chosen", "high"
        self.metadata.approval_mode = "prompt_user"
        self.metadata.save()
        response = self.client.post(self.url, {"action": "resume"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["started"])
        self.assertTrue(self.spawn.call_args.kwargs["resume_goal"])
        self.assertEqual(self.spawn.call_args.kwargs["model"], "chosen")
        self.assertEqual(self.spawn.call_args.kwargs["reasoning_effort"], "high")
        self.assertEqual(self.spawn.call_args.kwargs["approval_mode"], "prompt_user")
        self.assertEqual(self.client.post(self.url, {"action": "clear"}).status_code, 200)
        self.prepare.assert_called_with("session")
        self.native._request_raw.assert_called_with("thread/goal/clear", {"threadId": "session"})

    def test_start_persists_paused_before_worker_activation_and_rejects_exhausted_resume(self) -> None:
        self.goal = None
        response = self.client.post(self.url, {"action": "start", "objective": "My goal", "token_budget": "2000"})
        self.assertEqual(response.status_code, 200)
        self.native._request_raw.assert_called_once_with("thread/goal/set", {
            "threadId": "session", "objective": "My goal", "tokenBudget": 2000, "status": "paused",
        })
        self.spawn.assert_called_once()
        self.prepare.assert_called_once_with("session")
        self.spawn.reset_mock()
        for goal in [goal_snapshot(status="budgetLimited", tokensUsed=1000), goal_snapshot(status="complete")]:
            self.goal = goal
            self.assertEqual(self.client.post(self.url, {"action": "resume"}).status_code, 409)
        self.spawn.assert_not_called()

    @patch("hitch.main.runtime.session_goals.request_live_change")
    def test_running_changes_use_owning_worker_and_report_failure(self, live: MagicMock) -> None:
        instance = CodexInstance.objects.create(thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=42)
        live.return_value = self.goal
        self.assertEqual(self.client.post(self.url, {"action": "pause"}).status_code, 200)
        live.assert_called_once_with(instance, {"action": "pause"})
        live.side_effect = ValueError("Worker did not acknowledge")
        response = self.client.post(self.url, {"action": "clear"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("acknowledge", response.json()["error"])
        self.assertEqual(self.client.post(self.url, {"action": "resume"}).status_code, 409)
        self.native._request_raw.assert_not_called()
        self.spawn.assert_not_called()


class GoalRuntimeTests(TestCase):
    def test_continuations_keep_physical_ids_and_controls_target_current_turn(self) -> None:
        client = CodexClient()
        def goal(status: str) -> Notification:
            return Notification("thread/goal/updated", ThreadGoalUpdatedNotification(
                thread_id="session", goal=ThreadGoal.model_validate(goal_snapshot(status=status)),
            ))
        first = _turn_started_event(thread_id="session", turn_id="first")
        second = _turn_started_event(thread_id="session", turn_id="second")

        def start(*_args: Any) -> dict[str, Any]:
            for event in [goal("active"), first]:
                client._router.route_notification(event)
            return {"goal": goal_snapshot(status="active")}

        with patch.object(client, "_request_raw", side_effect=start):
            turn = session_goals.GoalTurn(client, "session")
        stream = turn.stream()
        self.assertEqual(next(stream), first)
        client._router.route_notification(_turn_completed_event(thread_id="session", turn_id="first"))
        self.assertEqual(next(stream), _turn_completed_event(thread_id="session", turn_id="first"))
        waiting = threading.Event()
        active_turn = turn.state.active_turn

        def wait_for_continuation() -> str | None:
            waiting.set()
            return active_turn()

        with (
            patch.object(client, "turn_steer") as steer,
            patch.object(client, "turn_interrupt") as interrupt,
            patch.object(type(turn.state), "active_turn", side_effect=wait_for_continuation),
            ThreadPoolExecutor() as executor,
        ):
            pending = executor.submit(turn.steer, "Keep going")
            try:
                self.assertTrue(waiting.wait(2))
                self.assertFalse(pending.done())
                steer.assert_not_called()
            finally:
                client._router.route_notification(second)
            pending.result(timeout=2)
            self.assertEqual(next(stream), second)
            self.assertEqual(steer.call_args.args[:2], ("session", "second"))
            with patch.object(client, "pause_goal") as pause:
                turn.interrupt()
                pause.assert_called_once_with("session")
                interrupt.assert_called_once_with("session", "second")
                interrupt.reset_mock()
                pause.side_effect = CodexError("No goal exists")
                turn.interrupt()
                interrupt.assert_called_once_with("session", "second")
        client._router.route_notification(goal("complete"))
        client._router.route_notification(_turn_completed_event(thread_id="session", turn_id="second"))
        self.assertEqual(list(stream), [_turn_completed_event(thread_id="session", turn_id="second")])
        self.assertFalse(client._router.has_goal("session"))

    def test_live_ack_and_expired_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=42,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            turn = MagicMock(thread_id="session")
            turn._client._request_raw.return_value = {"goal": goal_snapshot()}
            path = codex_pool.control_path_for(instance)

            def drain(*_args: Any) -> None:
                codex_worker._drain_steer_requests(turn, instance=instance, control_path=path, control_offset=0)

            with (
                patch.object(codex_pool, "_pid_is_instance_worker", return_value=True),
                patch("hitch.main.runtime.session_goals.os.kill", side_effect=drain),
                patch.object(codex_worker, "_goal_pause_requested", False),
            ):
                result = session_goals.request_live_change(instance, {"action": "pause"})
                self.assertTrue(codex_worker._goal_pause_requested)
            self.assertEqual(result, goal_snapshot())
            turn.interrupt.assert_called_once()
            turn._client._request_raw.reset_mock()
            offset = path.stat().st_size
            with path.open("a") as fh:
                fh.write(json.dumps({
                    "op": "goal", "id": "expired", "expiresAt": time.time() - 1, "change": {"action": "clear"},
                }) + "\n")
            codex_worker._drain_steer_requests(turn, instance=instance, control_path=path, control_offset=offset)
            turn._client._request_raw.assert_not_called()
            self.assertIn("expired", json.loads(path.read_text().splitlines()[-1])["error"])

    @patch("hitch.main.management.commands.codex_worker._run_turn")
    @patch("hitch.main.runtime.session_goals.current_goal", return_value=goal_snapshot())
    def test_only_explicit_goal_pauses_are_clean_and_all_interruptions_close_approvals(
        self, _goal: MagicMock, run: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            for pause, stop, resume in [(True, False, True), (False, True, False), (True, True, True)]:
                with self.subTest(pause=pause, stop=stop, resume=resume), patch.object(
                    codex_worker, "_cancel_requested", stop,
                ):
                    instance = CodexInstance.objects.create(
                        thread_id="session", status=CodexInstance.STATUS_STARTING, pid=42,
                        events_path=str(Path(raw) / "events.jsonl"),
                    )
                    approval = ApprovalRequest.objects.create(instance=instance, method="command", params={})

                    def interrupt(_pause: bool = pause, **_kwargs: Any) -> Turn:
                        codex_worker._goal_pause_requested = _pause
                        return Turn(id="turn", status=TurnStatus.interrupted, items=[])

                    run.side_effect = interrupt
                    call_command("codex_worker", "--instance-id", str(instance.pk), resume_goal=resume)
                    instance.refresh_from_db()
                    approval.refresh_from_db()
                    self.assertEqual(
                        instance.status, CodexInstance.STATUS_COMPLETED if pause and not stop
                        else CodexInstance.STATUS_FAILED,
                    )
                    self.assertEqual(instance.error, "" if pause and not stop else "turn ended with status interrupted")
                    self.assertIsNotNone(approval.decided_at)

    @patch("hitch.main.management.commands.codex_worker.Codex")
    def test_goal_worker_preserves_settings_and_waits_for_last_turn(self, native: MagicMock) -> None:
        class FixtureGoal(TurnHandle):
            def __init__(self, client: Any, thread_id: str) -> None:
                self._client, self.thread_id, self.id = client, thread_id, "first"

            @override
            def stream(self) -> Any:
                for turn_id in ["first", "second"]:
                    yield _turn_started_event(thread_id=self.thread_id, turn_id=turn_id)
                    yield _turn_completed_event(thread_id=self.thread_id, turn_id=turn_id)

        with tempfile.TemporaryDirectory() as raw, patch.object(session_goals, "GoalTurn", FixtureGoal):
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=42,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            with Path(instance.events_path).open("w") as events:
                final = codex_worker._run_turn(
                    instance=instance, prompt="", events_file=events, model="chosen", reasoning_effort="high",
                    sandbox_policy="readOnly", approval_mode="prompt_user", resume_goal=True,
                )
            assert final is not None
            self.assertEqual(final.id, "second")
            resume = native.return_value.__enter__.return_value.thread_resume
            self.assertEqual(resume.call_args.kwargs["model"], "chosen")
            self.assertEqual(resume.call_args.kwargs["config"], {
                "features.goals": True, "model_reasoning_effort": "high", "sandbox_mode": "read-only",
                "approval_policy": "on-request", "approvals_reviewer": "user",
            })

    def test_native_preservation_budget_accounting_and_completed_controls(self) -> None:
        with tempfile.TemporaryDirectory() as raw, self.settings(CODEX_SQLITE_HOME_BASE=Path(raw) / "homes"):
            root = Path(raw)
            (root / "codex").mkdir()
            worker_home = root / "homes" / "worker-0"
            env = {"CODEX_HOME": str(root / "codex"), "CODEX_SQLITE_HOME": str(worker_home)}
            with Codex(config=CodexConfig(env=env)) as codex:
                thread = codex.thread_start(cwd=raw)
                codex._client._request_raw("thread/goal/set", {
                    "threadId": thread.id, "objective": "  Keep the accounting\n", "status": "paused",
                    "tokenBudget": 1000,
                })
            with contextlib.closing(sqlite3.connect(worker_home / "goals_1.sqlite")) as db:
                db.execute("UPDATE thread_goals SET tokens_used=500, time_used_seconds=7200, objective=?",
                           ("  Keep the accounting\n",))
                db.commit()
            self.assertIsNone(session_goals.current_goal(thread.id))
            session_goals.preserve_worker_goal(thread.id, str(worker_home))
            home = session_goals.sqlite_home(thread.id)
            config = CodexConfig(env={**env, "CODEX_SQLITE_HOME": str(home)})
            with (
                Codex(config=config) as owner,
                Codex(config=CodexConfig(env={**env, "CODEX_SQLITE_HOME": str(root / "second")})) as second,
            ):
                owner.thread_resume(thread.id)
                with self.assertRaisesMessage(CodexError, "already has an active writer"):
                    second.thread_resume(thread.id)
            SessionMetadata.objects.create(thread_id=thread.id, cwd=raw)
            url = reverse("set_session_goal", args=[thread.id])
            with (
                patch.dict(os.environ, {"CODEX_HOME": str(root / "codex")}),
                patch("hitch.main.views.common._is_allowed_session_cwd", return_value=True),
                patch.object(codex_pool, "spawn_turn") as spawn,
            ):
                response = self.client.post(url, {
                    "action": "budget", "token_budget": "2000", "objective": "Must not replace the objective",
                })
                self.assertEqual(response.status_code, 200)
                updated = response.json()["goal"]
                self.assertEqual(updated["objective"], "  Keep the accounting\n")
                self.assertEqual(updated["tokensUsed"], 500)
                self.assertEqual(updated["timeUsedSeconds"], 7200)
                self.assertEqual(updated["tokenBudget"], 2000)
                self.assertEqual(updated["status"], "paused")
                self.assertEqual(session_goals.current_goal(thread.id), updated)
                with Codex(config=config) as codex:
                    codex._client._request_raw("thread/goal/set", {"threadId": thread.id, "status": "complete"})
                completed = session_goals.current_goal(thread.id)
                for action in ["pause", "resume", "budget", "start"]:
                    with self.subTest(action=action):
                        response = self.client.post(url, {"action": action, "objective": "New objective"})
                        self.assertEqual(response.status_code, 409)
                        self.assertEqual(session_goals.current_goal(thread.id), completed)
                spawn.assert_not_called()
                self.assertEqual(self.client.post(url, {"action": "clear"}).status_code, 200)
            self.assertEqual(list(home.iterdir()), [])
            self.assertIsNone(session_goals.current_goal(thread.id))
            self.assertEqual(session_goals.prepare_home(thread.id), home)

    @patch("hitch.main.management.commands.codex_worker._run_turn")
    @patch("hitch.main.management.commands.codex_worker.acquire_worker_sqlite_home")
    def test_goal_home_lease_rejects_overlapping_worker_without_fallback(
        self, pooled: MagicMock, run: MagicMock,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            self.settings(CODEX_SQLITE_HOME_BASE=Path(raw), TESTING=False),
            patch("hitch.main.runtime.disk_cleanup.run_finished_session_disk_cleanup"),
        ):
            home = session_goals.prepare_home("session")
            lease = session_goals.acquire_home("session")
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_STARTING, pid=42,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            try:
                with self.assertRaisesMessage(ValueError, "Another worker"):
                    call_command("codex_worker", "--instance-id", str(instance.pk))
                instance.refresh_from_db()
                self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
                run.assert_not_called()
                pooled.assert_not_called()
            finally:
                lease.release()
            with patch.object(
                session_goals, "compact_cleared_home", wraps=session_goals.compact_cleared_home,
            ) as compact:
                run.return_value = Turn(id="turn", status=TurnStatus.completed, items=[])
                instance.status = CodexInstance.STATUS_STARTING
                instance.save()
                call_command("codex_worker", "--instance-id", str(instance.pk))
                self.assertEqual(run.call_args.kwargs["sqlite_home"], str(home))
                compact.assert_called_once_with("session")
            session_goals.acquire_home("session").release()
