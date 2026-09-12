"""Native goal controls, persistence, and continuation routing."""

import contextlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, cast, override
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
        self.metadata.save()
        self.acquire.side_effect = ValueError("Another worker is preserving its goal")
        self.assertEqual(self.client.post(self.url, {"action": "budget", "token_budget": "2000"}).status_code, 409)
        self.prepare.assert_not_called()
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
        instance = CodexInstance(pk=1, thread_id="session")
        with (
            patch.object(client, "turn_steer") as steer,
            patch.object(client, "turn_interrupt") as interrupt,
            patch.object(client, "_request_raw", return_value={"status": "applied"}) as settings_update,
            patch.object(CodexInstance.objects, "filter"),
        ):
            with self.assertRaisesMessage(ValueError, "no running turn"):
                turn.steer("Keep going")
            self.assertFalse(codex_worker._apply_turn_model_settings(turn, instance, {
                "model": "chosen", "effort": "high",
            }))
            steer.assert_not_called()
            settings_update.assert_not_called()
            client._router.route_notification(second)
            turn.steer("Keep going")
            self.assertTrue(codex_worker._apply_turn_model_settings(
                turn, instance, {"model": "chosen", "effort": "high"},
            ))
            settings_update.assert_called_once_with("turn/settings/update", {
                "threadId": "session", "turnId": "second", "model": "chosen", "effort": "high",
            })
            self.assertEqual((instance.model, instance.reasoning_effort), ("chosen", "high"))
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
        client._router.route_notification(_turn_completed_event(thread_id="session", turn_id="second"))
        self.assertEqual(next(stream), _turn_completed_event(thread_id="session", turn_id="second"))
        with (
            patch.object(client, "pause_goal", side_effect=lambda _: client._router.route_notification(goal("paused"))),
            patch.object(client, "turn_interrupt") as interrupt,
        ):
            self.assertEqual(turn.interrupt().model_dump(), {})
            interrupt.assert_not_called()
        self.assertEqual(list(stream), [])
        self.assertFalse(client._router.has_goal("session"))
        with patch.object(client, "_request_raw") as update:
            self.assertFalse(codex_worker._apply_turn_model_settings(
                turn, instance, {"model": "new", "effort": "high"},
            ))
            update.assert_not_called()

    def test_control_loop_defers_messages_without_blocking_pause_or_reverting_models(self) -> None:
        def exercise(pause: bool) -> None:
            with self.subTest(pause=pause), tempfile.TemporaryDirectory() as raw:
                instance = CodexInstance.objects.create(
                    thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=42,
                    events_path=str(Path(raw) / "events.jsonl"),
                )
                turn = object.__new__(session_goals.GoalTurn)
                client, state = MagicMock(), MagicMock()
                turn.thread_id, turn._client, turn.state = "session", client, state
                state.current_turn.return_value = None
                state.is_finished.return_value = False
                path = codex_pool.control_path_for(instance)
                codex_pool._append_control_request(instance, {"op": "steer", "id": "message", "input": "Keep going"})
                stop, wakeup = threading.Event(), MagicMock()
                iterations = 0

                def wait(_timeout: float) -> None:
                    nonlocal iterations
                    iterations += 1
                    if iterations == 1:
                        client.turn_steer.assert_not_called()
                        self.assertIsNone(codex_pool._read_steer_ack(path, "message"))
                        if pause:
                            codex_pool._append_control_request(instance, {
                                "op": "goal", "id": "pause", "expiresAt": time.time() + 3,
                                "change": {"action": "pause"},
                            })
                        else:
                            state.current_turn.return_value = "second"
                    else:
                        stop.set()

                def saved_pair(_thread_id: str) -> tuple[str, str]:
                    # A save arrives after the continuation reads its old pair.
                    codex_pool._append_control_request(instance, {
                        "op": "model_settings", "id": "model", "model": "new", "effort": "low",
                    })
                    return "old", "high"

                def request(method: str, _params: Any) -> dict[str, Any]:
                    if method == "thread/goal/set":
                        state.is_finished.return_value = True
                    return {"goal": goal_snapshot(status="active"), "status": "applied"}

                wakeup.wait.side_effect = wait
                client._request_raw.side_effect = request
                with (
                    patch("hitch.main.sessions.model_settings.session_model_override", side_effect=saved_pair),
                    patch.object(codex_worker, "_goal_pause_requested", False),
                    patch.object(codex_worker, "discard_input_attachment_paths") as discard,
                ):
                    codex_worker._forward_steer_requests(
                        turn=turn, instance=instance, control_path=path, wakeup=wakeup, stop=stop,
                    )
                self.assertEqual(codex_pool._read_steer_ack(path, "message"), not pause)
                if pause:
                    reply = next(json.loads(line) for line in path.read_text().splitlines()
                                 if json.loads(line).get("op") == "goal_ack")
                    self.assertNotIn("error", reply)
                    client.turn_steer.assert_not_called()
                    client.turn_interrupt.assert_not_called()
                    discard.assert_called_once_with(instance, [])
                else:
                    client.turn_steer.assert_called_once()
                    self.assertEqual(client.turn_steer.call_args.args[:2], ("session", "second"))
                    calls = client._request_raw.call_args_list
                    self.assertEqual([call.args[1]["model"] for call in calls], ["old", "new"])
                    self.assertTrue(codex_pool._read_steer_ack(path, "model", op="model_settings_ack"))
                    instance.refresh_from_db()
                    self.assertEqual((instance.model, instance.reasoning_effort), ("new", "low"))

        for pause in [False, True]:
            exercise(pause)

    def test_live_ack_and_expired_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=42,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            turn = MagicMock(thread_id="session")
            current: dict[str, Any] | None = goal_snapshot(status="active")
            turn._client._request_raw.side_effect = lambda method, _params: {
                "goal": current if method == "thread/goal/get" else goal_snapshot(),
            }
            path = codex_pool.control_path_for(instance)
            offset = 0

            def drain(*_args: Any) -> None:
                codex_worker._drain_steer_requests(turn, instance=instance, control_path=path, control_offset=offset)

            with (
                patch.object(codex_pool, "_pid_is_instance_worker", return_value=True),
                patch("hitch.main.runtime.session_goals.os.kill", side_effect=drain),
                patch.object(codex_worker, "_goal_pause_requested", False),
            ):
                result = session_goals.request_live_change(instance, {"action": "pause"})
                self.assertTrue(codex_worker._goal_pause_requested)
            self.assertEqual(result, goal_snapshot())
            turn.interrupt.assert_called_once()
            for current in [goal_snapshot(), goal_snapshot(status="complete"), None]:
                with (
                    self.subTest(goal=current),
                    patch.object(codex_pool, "_pid_is_instance_worker", return_value=True),
                    patch("hitch.main.runtime.session_goals.os.kill", side_effect=drain),
                    patch.object(codex_worker, "_goal_pause_requested", False),
                ):
                    turn.interrupt.reset_mock()
                    turn._client._request_raw.reset_mock()
                    offset = path.stat().st_size
                    if current and current["status"] == "paused":
                        self.assertEqual(session_goals.request_live_change(instance, {"action": "pause"}), current)
                    else:
                        with self.assertRaisesMessage(ValueError, "complete" if current else "no longer"):
                            session_goals.request_live_change(instance, {"action": "pause"})
                        turn._client._request_raw.assert_called_once_with("thread/goal/get", {"threadId": "session"})
                    turn.interrupt.assert_not_called()
                    self.assertFalse(codex_worker._goal_pause_requested)
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
                self.state = MagicMock()
                self.state.current_turn.return_value = None

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

    @patch("hitch.main.management.commands.codex_worker.Codex")
    @patch("hitch.main.management.commands.codex_worker._start_turn")
    def test_ordinary_messages_pause_only_active_goals_before_loading(
        self, start: MagicMock, native: MagicMock,
    ) -> None:
        api = native.return_value.__enter__.return_value
        order = MagicMock()
        order.attach_mock(api._client.pause_goal, "pause")
        order.attach_mock(api.thread_resume, "load")
        order.attach_mock(start, "start")
        with tempfile.TemporaryDirectory() as raw:
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_RUNNING, pid=42,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            start.return_value.id = "ordinary"
            for status in [None, "active", "paused", "complete", "budgetLimited"]:
                with self.subTest(status=status), Path(instance.events_path).open("w") as events:
                    order.reset_mock()
                    api._client._request_raw.return_value = {"goal": goal_snapshot(status=status) if status else None}
                    start.return_value.stream.return_value = iter([
                        _turn_completed_event(thread_id="session", turn_id="ordinary"),
                    ])
                    final = codex_worker._run_turn(instance=instance, prompt="My message", events_file=events)
                    self.assertIsNotNone(final)
                    self.assertEqual([entry[0] for entry in order.mock_calls if entry[0] in {"pause", "load", "start"}],
                                     ["pause", "load", "start"] if status == "active" else ["load", "start"])
                    self.assertEqual(start.call_args.kwargs["prompt"], "My message")

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
            (home / "goals_1.sqlite").touch()
            self.assertIsNone(session_goals.current_goal(thread.id))
            with Codex(config=config) as codex:
                codex._client._request_raw("thread/goal/set", {
                    "threadId": thread.id, "objective": "Next goal", "status": "paused",
                })
            initialized = session_goals.current_goal(thread.id)
            assert initialized is not None
            self.assertEqual(initialized["objective"], "Next goal")
            connect = sqlite3.connect

            def removed_before_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
                (home / "goals_1.sqlite").unlink()
                return cast(sqlite3.Connection, connect(*args, **kwargs))

            with patch.object(sqlite3, "connect", side_effect=removed_before_connect):
                self.assertIsNone(session_goals.current_goal(thread.id))
            (home / "goals_1.sqlite").touch()
            with (
                patch.object(sqlite3, "connect", side_effect=sqlite3.OperationalError("Database is locked")),
                self.assertRaisesMessage(sqlite3.OperationalError, "Database is locked"),
            ):
                session_goals.current_goal(thread.id)

    @patch("hitch.main.management.commands.codex_worker._run_turn")
    @patch("hitch.main.management.commands.codex_worker.acquire_worker_sqlite_home")
    def test_goal_home_lease_waits_for_handoff_and_rejects_prolonged_overlap(
        self, pooled: MagicMock, run: MagicMock,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            self.settings(CODEX_SQLITE_HOME_BASE=Path(raw), TESTING=False),
            patch("hitch.main.runtime.disk_cleanup.run_finished_session_disk_cleanup"),
        ):
            home = session_goals.sqlite_home("session")
            self.assertFalse(home.exists())
            lease = session_goals.acquire_home("session")
            instance = CodexInstance.objects.create(
                thread_id="session", status=CodexInstance.STATUS_STARTING, pid=42,
                events_path=str(Path(raw) / "events.jsonl"),
            )
            try:
                with (
                    patch("hitch.main.runtime.session_goals.time.monotonic", side_effect=[0, 0, 11]),
                    patch("hitch.main.runtime.session_goals.time.sleep") as wait,
                    self.assertRaisesMessage(ValueError, "Another worker"),
                ):
                    call_command("codex_worker", "--instance-id", str(instance.pk))
                wait.assert_called_once_with(0.05)
                instance.refresh_from_db()
                self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
                run.assert_not_called()
                pooled.assert_not_called()
            finally:
                lease.release()
            lease = session_goals.acquire_home("session")
            def finish_preserving_goal(_seconds: float) -> None:
                session_goals.prepare_home("session")
                lease.release()

            try:
                with (
                    patch.object(
                        session_goals, "compact_cleared_home", wraps=session_goals.compact_cleared_home,
                    ) as compact,
                    patch("hitch.main.runtime.session_goals.time.sleep", side_effect=finish_preserving_goal) as wait,
                ):
                    run.return_value = Turn(id="turn", status=TurnStatus.completed, items=[])
                    instance.status = CodexInstance.STATUS_STARTING
                    instance.prompt = "Follow-up message"
                    instance.save()
                    call_command("codex_worker", "--instance-id", str(instance.pk))
                    wait.assert_called_once_with(0.05)
                    self.assertEqual(run.call_args.kwargs["sqlite_home"], str(home))
                    self.assertEqual(run.call_args.kwargs["prompt"], "Follow-up message")
                    pooled.assert_not_called()
                    compact.assert_called_once_with("session")
                    instance.refresh_from_db()
                    self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
            finally:
                lease.release()
            session_goals.acquire_home("session").release()
