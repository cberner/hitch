"""Per-session action endpoints: rename, archive, approval mode, demo."""


import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

from django.db import IntegrityError
from django.test import (
    TestCase,
    override_settings,
)
from django.urls import reverse
from openai_codex import Codex
from openai_codex.errors import InvalidRequestError

from hitch.main import demo, views
from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    CodexInstance,
    SessionDemo,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.test.support import (
    _seed_cookies,
    _setup_codex,
)
from hitch.main.test.views_helpers import (
    _WEB_SEARCH_COOKIE,
    _run_borrowed_with,
    _session,
)


class SetSessionNameViewTests(TestCase):
    @patch("hitch.main.views.common.Codex")
    def test_updates_name_and_response_shape(self, mock_codex: MagicMock) -> None:
        client = mock_codex.return_value.__enter__.return_value
        cases: list[tuple[str, dict[str, str], bool, int, str | None]] = [
            (
                "session redirect trims whitespace",
                {"name": "  New title  "},
                False,
                302,
                reverse("session", kwargs={"session_id": "abc"}),
            ),
            (
                "index redirect",
                {"name": "New title", "next": "index"},
                False,
                302,
                reverse("index"),
            ),
            (
                "ajax",
                {"name": "New title"},
                True,
                204,
                None,
            ),
        ]
        for label, data, ajax, status, location in cases:
            with self.subTest(label=label):
                client._client.thread_set_name.reset_mock()
                url = reverse("set_session_name", kwargs={"session_id": "abc"})
                if ajax:
                    response = self.client.post(
                        url,
                        data=data,
                        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
                    )
                else:
                    response = self.client.post(url, data=data)

                self.assertEqual(response.status_code, status)
                if location is None:
                    self.assertNotIn("Location", response.headers)
                else:
                    self.assertEqual(response.headers["Location"], location)
                client._client.thread_set_name.assert_called_once_with(
                    "abc", "New title"
                )

    @patch("hitch.main.views.common.Codex")
    def test_rejects_invalid_requests(self, mock_codex: MagicMock) -> None:
        # The form caps input client-side; the view enforces the same bounds
        # so a hand-crafted POST can't bypass them.
        cases: list[tuple[str, dict[str, str], str, int]] = [
            ("post", {"name": ""}, "empty", 400),
            ("post", {"name": "   "}, "whitespace only", 400),
            ("post", {"name": "x" * 201}, "over length cap", 400),
            ("get", {}, "method", 405),
        ]
        for method, data, label, status in cases:
            with self.subTest(label=label):
                url = reverse("set_session_name", kwargs={"session_id": "abc"})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_codex.assert_not_called()

class SetSessionApprovalModeViewTests(TestCase):
    def test_updates_and_resets_session_approval_mode(self) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "prompt_user"})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "abc"}),
        )
        metadata = SessionMetadata.objects.get(thread_id="abc")
        self.assertEqual(metadata.approval_mode, "prompt_user")

        response = self.client.post(url, data={"approval_mode": ""})

        self.assertEqual(response.status_code, 302)
        metadata.refresh_from_db()
        self.assertEqual(metadata.approval_mode, "")

    def test_archived_session_without_cached_cwd_returns_400(self) -> None:
        # The cwd fallback resumes the thread; the app-server raises
        # InvalidRequestError for archived/unknown threads -- an expected
        # state that must answer 400 rather than 500.
        with patch(
            "hitch.main.views.session_actions.app_server_pool.run_borrowed_op_with_retry",
            side_effect=InvalidRequestError(
                code=-32600, message="thread is archived"
            ),
        ):
            response = self.client.post(
                reverse("set_session_approval_mode", kwargs={"session_id": "abc"}),
                data={"approval_mode": "prompt_user"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(SessionMetadata.objects.filter(thread_id="abc").exists())

    def test_rejects_invalid_session_approval_mode(self) -> None:
        metadata = SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            approval_mode="deny_all",
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "phantom"})

        self.assertEqual(response.status_code, 400)
        metadata.refresh_from_db()
        self.assertEqual(metadata.approval_mode, "deny_all")

        response = self.client.get(url)

        self.assertEqual(response.status_code, 405)

    def test_approve_all_updates_live_instances_and_pending_approvals(self) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        pending = ApprovalRequest.objects.create(
            instance=running,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        completed = CodexInstance.objects.create(
            pid=2,
            thread_id="abc",
            cwd="/repo",
            prompt="done",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            approval_mode="auto_review",
        )
        completed_pending = ApprovalRequest.objects.create(
            instance=completed,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo test"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "approve_all"})

        self.assertEqual(response.status_code, 302)
        running.refresh_from_db()
        self.assertEqual(running.approval_mode, "approve_all")
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_ACCEPT)
        self.assertIsNotNone(pending.decided_at)
        completed.refresh_from_db()
        self.assertEqual(completed.approval_mode, "auto_review")
        completed_pending.refresh_from_db()
        self.assertEqual(completed_pending.decision, ApprovalRequest.DECISION_PENDING)

    def test_approve_all_does_not_rewrite_live_deny_all_instance(self) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="deny_all",
        )
        pending = ApprovalRequest.objects.create(
            instance=running,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "approve_all"})

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="abc")
        self.assertEqual(metadata.approval_mode, "approve_all")
        running.refresh_from_db()
        self.assertEqual(running.approval_mode, "deny_all")
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_PENDING)

    def test_approve_all_can_relax_live_editable_deny_all_instance(self) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="deny_all",
            approval_mode_live_editable=True,
        )
        pending = ApprovalRequest.objects.create(
            instance=running,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "approve_all"})

        self.assertEqual(response.status_code, 302)
        running.refresh_from_db()
        self.assertEqual(running.approval_mode, "approve_all")
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_ACCEPT)

    def test_deny_all_does_not_rewrite_fixed_auto_review_instance(self) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="auto_review",
        )
        pending = ApprovalRequest.objects.create(
            instance=running,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "deny_all"})

        self.assertEqual(response.status_code, 302)
        running.refresh_from_db()
        self.assertEqual(running.approval_mode, "auto_review")
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_PENDING)

    def test_auto_review_does_not_rewrite_live_editable_instance(self) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        pending = ApprovalRequest.objects.create(
            instance=running,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "auto_review"})

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="abc")
        self.assertEqual(metadata.approval_mode, "auto_review")
        running.refresh_from_db()
        self.assertEqual(running.approval_mode, "prompt_user")
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_PENDING)

    def test_deny_all_updates_live_instances_and_pending_approvals(self) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as events_file:
            events_path = events_file.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path=events_path,
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        pending = ApprovalRequest.objects.create(
            instance=running,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        Path(events_path).write_text(
            json.dumps(
                {
                    "method": "approval/requested",
                    "payload": {
                        "id": pending.pk,
                        "method": pending.method,
                        "params": pending.params,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": "deny_all"})

        self.assertEqual(response.status_code, 302)
        running.refresh_from_db()
        self.assertEqual(running.approval_mode, "deny_all")
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_DECLINE)
        self.assertIsNotNone(pending.decided_at)
        events = [
            json.loads(line)
            for line in Path(events_path).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(events[-1]["method"], "approval/resolved")
        self.assertEqual(
            events[-1]["payload"],
            {
                "id": pending.pk,
                "method": "item/commandExecution/requestApproval",
                "decision": ApprovalRequest.DECISION_DECLINE,
            },
        )

    @patch("hitch.main.views.common.codex_events.append_event", side_effect=OSError("full"))
    def test_live_pending_approval_append_failure_still_settles_row(
        self, mock_append_event: MagicMock
    ) -> None:
        SessionMetadata.objects.create(thread_id="abc", cwd="/repo")
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path="/tmp/hitch-test-events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="prompt_user",
            approval_mode_live_editable=True,
        )
        pending = ApprovalRequest.objects.create(
            instance=running,
            method="item/commandExecution/requestApproval",
            params={"item": {"command": "cargo bench"}},
            decision=ApprovalRequest.DECISION_PENDING,
        )
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        with patch("hitch.main.views.common.logger.warning") as warning:
            response = self.client.post(url, data={"approval_mode": "deny_all"})

        self.assertEqual(response.status_code, 302)
        pending.refresh_from_db()
        self.assertEqual(pending.decision, ApprovalRequest.DECISION_DECLINE)
        mock_append_event.assert_called_once()
        warning.assert_called_once()

    def test_live_pending_settlement_skips_rows_decided_by_race(self) -> None:
        class PendingQuery:
            def order_by(self, *_fields: str) -> "PendingQuery":
                return self

            def values(self, *_fields: str) -> list[dict[str, Any]]:
                return [
                    {
                        "pk": 1,
                        "method": "item/commandExecution/requestApproval",
                        "instance__events_path": "/tmp/hitch-test-events.jsonl",
                    }
                ]

        class UpdateQuery:
            def update(self, **_fields: Any) -> int:
                return 0

        with patch(
            "hitch.main.views.common.ApprovalRequest.objects.filter",
            side_effect=[PendingQuery(), UpdateQuery()],
        ):
            resolved_events = views._settle_live_pending_approval_requests(
                [1],
                ApprovalRequest.DECISION_ACCEPT,
            )

        self.assertEqual(resolved_events, [])

    def test_reset_session_approval_mode_applies_global_to_live_instance(self) -> None:
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            approval_mode="approve_all",
        )
        running = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            approval_mode="approve_all",
            approval_mode_live_editable=True,
        )
        _seed_cookies(self.client, hitch_approval_mode="prompt_user")
        url = reverse("set_session_approval_mode", kwargs={"session_id": "abc"})

        response = self.client.post(url, data={"approval_mode": ""})

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="abc")
        self.assertEqual(metadata.approval_mode, "")
        running.refresh_from_db()
        self.assertEqual(running.approval_mode, "prompt_user")

class SetSessionArchivedViewTests(TestCase):
    @patch("hitch.main.views.common.Codex")
    def test_updates_archive_state_and_response_shape(
        self, mock_codex: MagicMock
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        cases: list[
            tuple[str, dict[str, str], bool, int, str | None, str, bool]
        ] = [
            (
                "archive",
                {"archived": "true"},
                False,
                302,
                reverse("index"),
                "thread_archive",
                True,
            ),
            (
                "archive ajax",
                {"archived": "true"},
                True,
                204,
                None,
                "thread_archive",
                False,
            ),
            (
                "unarchive to session",
                {"archived": "false"},
                False,
                302,
                reverse("session", kwargs={"session_id": "abc"}),
                "thread_unarchive",
                True,
            ),
            (
                "unarchive to index",
                {"archived": "false", "next": "index"},
                False,
                302,
                reverse("index"),
                "thread_unarchive",
                False,
            ),
        ]
        for label, data, ajax, status, location, expected_call, seed_cache in cases:
            with self.subTest(label=label):
                ArchivedSessionTokenUsage.objects.all().delete()
                client.thread_archive.reset_mock()
                client.thread_unarchive.reset_mock()
                if seed_cache:
                    ArchivedSessionTokenUsage.objects.create(
                        thread_id="abc",
                        total_tokens=100,
                    )
                    ArchivedSessionTokenUsage.objects.create(
                        thread_id="abc-child",
                        total_tokens=200,
                    )

                url = reverse("set_session_archived", kwargs={"session_id": "abc"})
                if ajax:
                    response = self.client.post(
                        url,
                        data=data,
                        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
                    )
                else:
                    response = self.client.post(url, data=data)

                self.assertEqual(response.status_code, status)
                if location is None:
                    self.assertNotIn("Location", response.headers)
                else:
                    self.assertEqual(response.headers["Location"], location)
                if expected_call == "thread_archive":
                    client.thread_archive.assert_called_once_with("abc")
                    client.thread_unarchive.assert_not_called()
                else:
                    client.thread_unarchive.assert_called_once_with("abc")
                    client.thread_archive.assert_not_called()
                if seed_cache:
                    # The toggled session's cache is dropped because its
                    # rollout path moves when codex archives/unarchives it,
                    # but every other session's cache must survive — wiping
                    # the whole table forces /profile and /usage to re-parse
                    # every archived rollout file the next time they render.
                    self.assertFalse(
                        ArchivedSessionTokenUsage.objects.filter(
                            thread_id="abc"
                        ).exists()
                    )
                    self.assertTrue(
                        ArchivedSessionTokenUsage.objects.filter(
                            thread_id="abc-child"
                        ).exists()
                    )

    @patch("hitch.main.views.common.Codex")
    def test_archive_keeps_cached_usage_for_unrelated_sessions(
        self, mock_codex: MagicMock
    ) -> None:
        ArchivedSessionTokenUsage.objects.create(
            thread_id="abc", total_tokens=100
        )
        ArchivedSessionTokenUsage.objects.create(
            thread_id="other-1", total_tokens=200
        )
        ArchivedSessionTokenUsage.objects.create(
            thread_id="other-2", total_tokens=300
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(
            ArchivedSessionTokenUsage.objects.filter(thread_id="abc").exists()
        )
        other_totals = dict(
            ArchivedSessionTokenUsage.objects.filter(
                thread_id__in=["other-1", "other-2"]
            ).values_list("thread_id", "total_tokens")
        )
        self.assertEqual(other_totals, {"other-1": 200, "other-2": 300})

    @patch("hitch.main.demo.subprocess.run")
    @patch("hitch.main.views.common.Codex")
    def test_archive_cleans_up_active_demo_container(
        self, mock_codex: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = [
            SimpleNamespace(
                stdout=(
                    '[{"Config":{"Labels":{'
                    '"io.hitch.managed":"demo",'
                    '"io.hitch.session":"abc",'
                    '"io.hitch.demo_token":"token",'
                    '"io.hitch.container_name":"hitch-demo-abc-abcd"'
                    "}}}]"
                ),
                stderr="",
                returncode=0,
            ),
            SimpleNamespace(stdout="", stderr="", returncode=0),
            SimpleNamespace(stdout="[]", stderr="", returncode=0),
        ]
        SessionDemo.objects.create(
            thread_id="abc",
            host="127.0.0.1",
            port=45678,
            container_id="container-1",
            container_name="hitch-demo-abc-abcd",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(SessionDemo.objects.get(thread_id="abc").status, SessionDemo.STATUS_STOPPED)
        self.assertEqual(mock_run.call_args_list[1], call(
            ["podman", "rm", "-f", "container-1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ))
        mock_codex.return_value.__enter__.return_value.thread_archive.assert_called_once_with("abc")

    @patch("hitch.main.demo.subprocess.run")
    @patch("hitch.main.views.common.Codex")
    def test_failed_archive_does_not_clean_up_active_demo(
        self, mock_codex: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_codex.return_value.__enter__.return_value.thread_archive.side_effect = (
            RuntimeError("codex unavailable")
        )
        mock_run.side_effect = [
            SimpleNamespace(
                stdout=(
                    '[{"Config":{"Labels":{'
                    '"io.hitch.managed":"demo",'
                    '"io.hitch.session":"abc",'
                    '"io.hitch.demo_token":"token",'
                    '"io.hitch.container_name":"hitch-demo-abc-abcd"'
                    "}}}]"
                ),
                stderr="",
                returncode=0,
            ),
            SimpleNamespace(stdout="", stderr="", returncode=0),
            SimpleNamespace(stdout="[]", stderr="", returncode=0),
        ]
        SessionDemo.objects.create(
            thread_id="abc",
            host="127.0.0.1",
            port=45678,
            container_id="container-1",
            container_name="hitch-demo-abc-abcd",
            runtime="podman",
            status=SessionDemo.STATUS_ACTIVE,
            registration_token="token",
        )

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("set_session_archived", kwargs={"session_id": "abc"}),
                data={"archived": "true"},
            )

        mock_run.assert_not_called()
        self.assertEqual(
            SessionDemo.objects.get(thread_id="abc").status,
            SessionDemo.STATUS_ACTIVE,
        )

    @patch("hitch.main.runtime.codex_pool.cleanup_input_images_for_thread")
    @patch("hitch.main.views.common.Codex")
    def test_archive_keeps_retained_input_images_for_unarchive(
        self, mock_codex: MagicMock, mock_cleanup_images: MagicMock
    ) -> None:
        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_cleanup_images.assert_not_called()
        mock_codex.return_value.__enter__.return_value.thread_archive.assert_called_once_with(
            "abc"
        )

    @patch("hitch.main.views.common.Codex")
    def test_rejects_invalid_archive_requests(self, mock_codex: MagicMock) -> None:
        cases: list[tuple[str, dict[str, str], int]] = [
            ("post", {}, 400),
            ("post", {"archived": ""}, 400),
            ("post", {"archived": "yes"}, 400),
            ("get", {}, 405),
        ]
        for method, data, status in cases:
            with self.subTest(method=method, data=data):
                url = reverse("set_session_archived", kwargs={"session_id": "abc"})
                if method == "post":
                    response = self.client.post(url, data=data)
                else:
                    response = self.client.get(url)
                self.assertEqual(response.status_code, status)
        mock_codex.assert_not_called()

class StartSessionDemoViewTests(TestCase):
    @patch("hitch.main.demo.request_demo_start")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread")
    def test_rejects_start_while_system_workflow_is_active(
        self, mock_active_workflow: MagicMock, mock_request_demo: MagicMock
    ) -> None:
        mock_active_workflow.return_value = SimpleNamespace()

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "PR workflow is running for this session",
            status_code=400,
        )
        mock_request_demo.assert_not_called()

    @patch("hitch.main.demo.request_demo_start")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    def test_rejects_start_while_user_turn_is_active(
        self, _mock_active_workflow: MagicMock, mock_request_demo: MagicMock
    ) -> None:
        CodexInstance.objects.create(
            thread_id="abc",
            cwd="/repo",
            prompt="user turn",
            events_path="/tmp/events.jsonl",
            status=CodexInstance.STATUS_RUNNING,
            pid=123,
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "Codex is already working for this session",
            status_code=400,
        )
        mock_request_demo.assert_not_called()

    @override_settings(HITCH_DEMO_RUNTIME="docker")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.demo.request_demo_start")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    def test_rejects_unsupported_runtime_before_spawning_agent(
        self,
        _mock_active_workflow: MagicMock,
        mock_request_demo: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content, b"only podman demo runtime is supported")
        mock_request_demo.assert_not_called()
        mock_codex.assert_not_called()

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_requests_demo_agent_turn(
        self,
        mock_run_borrowed: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = SimpleNamespace(
            _client=SimpleNamespace(thread_resume=MagicMock())
        )
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        mock_run_borrowed.side_effect = _run_borrowed_with(client)
        spawned_instances: list[CodexInstance] = []

        def spawn_side_effect(**_kwargs: object) -> CodexInstance:
            instance = CodexInstance.objects.create(
                thread_id="abc",
                cwd="/repo",
                prompt="demo",
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_RUNNING,
                pid=123,
                agent_kind=demo.DEMO_AGENT_KIND,
            )
            spawned_instances.append(instance)
            return instance

        mock_spawn.side_effect = spawn_side_effect
        _seed_cookies(self.client, **{_WEB_SEARCH_COOKIE: "live"})

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("session", kwargs={"session_id": "abc"}))
        mock_spawn.assert_called_once()
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "abc")
        self.assertEqual(kwargs["cwd"], "/repo")
        self.assertEqual(kwargs["purpose"], CodexInstance.PURPOSE_SYSTEM_AGENT)
        self.assertEqual(kwargs["agent_kind"], demo.DEMO_AGENT_KIND)
        self.assertEqual(kwargs["web_search_mode"], "live")
        self.assertNotIn("output_schema", kwargs)
        self.assertIsNone(kwargs["user_message_index"])
        self.assertIn("Start an interactive web demo", kwargs["prompt"])
        self.assertIn("Registration token:", kwargs["prompt"])
        self.assertIn("io.hitch.managed=demo", kwargs["prompt"])
        self.assertIn("http://testserver/sessions/abc/demo/", kwargs["prompt"])
        client._client.thread_resume.assert_called_once_with("abc")
        mock_run_borrowed.assert_called_once()
        self.assertIs(mock_run_borrowed.call_args.args[0], Codex)
        self.assertEqual(
            mock_run_borrowed.call_args.kwargs,
            {"enable_memories": False},
        )
        session_demo = SessionDemo.objects.get(thread_id="abc")
        self.assertTrue(session_demo.registration_token)
        self.assertEqual(spawned_instances[0].agent_kind, demo.DEMO_AGENT_KIND)
        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        run = SystemAgentRun.objects.get(workflow=workflow)
        self.assertEqual(run.agent_kind, demo.DEMO_AGENT_KIND)
        self.assertEqual(run.thread_id, "abc")
        self.assertEqual(run.instance, spawned_instances[0])

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees")
    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_requests_demo_agent_turn_uses_managed_worktree_sandbox(
        self,
        mock_run_borrowed: MagicMock,
        _mock_discover: MagicMock,
        mock_managed_worktrees: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        worktree = "/repo-worktree"
        mock_managed_worktrees.return_value = [Path(worktree)]
        client = SimpleNamespace(
            _client=SimpleNamespace(thread_resume=MagicMock())
        )
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd=worktree, turns=[])
        )
        mock_run_borrowed.side_effect = _run_borrowed_with(client)

        def spawn_side_effect(**_kwargs: object) -> CodexInstance:
            return CodexInstance.objects.create(
                thread_id="abc",
                cwd=worktree,
                prompt="demo",
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_RUNNING,
                pid=123,
                agent_kind=demo.DEMO_AGENT_KIND,
            )

        mock_spawn.side_effect = spawn_side_effect

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_spawn.call_args.kwargs["sandbox_policy"], "workspaceWrite")

    @patch("hitch.main.demo.cleanup_demo_for_session")
    @patch("hitch.main.runtime.codex_pool.spawn_turn", side_effect=RuntimeError("spawn failed"))
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cleans_up_demo_when_worker_dispatch_fails(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        _mock_spawn: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        mock_cleanup.assert_called_once_with("abc")

    @patch("hitch.main.demo.cleanup_demo_for_session")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cleans_up_demo_when_workflow_state_save_fails(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        original_save = SystemWorkflow.save

        def save_side_effect(
            workflow: SystemWorkflow, *args: Any, **kwargs: Any
        ) -> None:
            if kwargs.get("update_fields") == ["state", "updated_at"]:
                raise RuntimeError("state save failed")
            original_save(workflow, *args, **kwargs)

        with patch.object(SystemWorkflow, "save", autospec=True) as mock_save:
            mock_save.side_effect = save_side_effect
            with self.assertRaisesRegex(RuntimeError, "state save failed"):
                self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)
        mock_spawn.assert_not_called()
        mock_cleanup.assert_called_once_with("abc")

    @patch("hitch.main.demo.cleanup_demo_for_session")
    @patch(
        "hitch.main.demo.start_demo_prompt_for",
        side_effect=RuntimeError("prompt failed"),
    )
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_cleans_up_demo_when_prompt_construction_fails(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_prompt: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        with self.assertRaisesRegex(RuntimeError, "prompt failed"):
            self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)
        mock_spawn.assert_not_called()
        mock_cleanup.assert_called_once_with("abc")

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_requests_demo_agent_turn_tolerates_existing_system_run(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        def spawn_side_effect(**kwargs: object) -> CodexInstance:
            workflow_id = cast(int, kwargs["workflow_id"])
            instance = CodexInstance.objects.create(
                thread_id="abc",
                cwd="/repo",
                prompt="demo",
                events_path="/tmp/events.jsonl",
                status=CodexInstance.STATUS_COMPLETED,
                pid=123,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                workflow_id=workflow_id,
                agent_kind=demo.DEMO_AGENT_KIND,
            )
            workflow = SystemWorkflow.objects.get(pk=workflow_id)
            SystemAgentRun.objects.create(
                workflow=workflow,
                agent_kind=demo.DEMO_AGENT_KIND,
                thread_id="abc",
                instance=instance,
                status=SystemAgentRun.STATUS_COMPLETED,
            )
            return instance

        mock_spawn.side_effect = spawn_side_effect

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(SystemAgentRun.objects.count(), 1)
        self.assertEqual(
            SystemAgentRun.objects.get().status,
            SystemAgentRun.STATUS_COMPLETED,
        )

    @patch("hitch.main.demo.cleanup_demo_for_session")
    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_duplicate_running_demo_workflow_rejects_without_mutating_owner(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_sweep: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        stale_workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "demo setup workflow is already running",
            status_code=400,
        )
        mock_codex.assert_not_called()
        mock_spawn.assert_not_called()
        mock_cleanup.assert_not_called()
        self.assertFalse(SessionDemo.objects.filter(thread_id="abc").exists())
        stale_workflow.refresh_from_db()
        self.assertEqual(stale_workflow.status, SystemWorkflow.STATUS_RUNNING)

    @patch("hitch.main.demo.request_demo_start")
    @patch("hitch.main.views.common.SystemWorkflow.objects.create")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_demo_workflow_integrity_error_rejects_before_mutating_demo_state(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        mock_create_workflow: MagicMock,
        mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        mock_create_workflow.side_effect = IntegrityError(
            "uniq_running_system_workflow"
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "demo setup workflow is already running",
            status_code=400,
        )
        mock_request_demo.assert_not_called()
        mock_spawn.assert_not_called()
        self.assertFalse(SessionDemo.objects.filter(thread_id="abc").exists())

    @patch("hitch.main.views.common.Codex")
    def test_system_sessions_lists_demo_run_without_hiding_user_session(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _session("thread-1", preview="User feature")
        _setup_codex(mock_codex, threads=[thread])
        workflow = SystemWorkflow.objects.create(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="thread-1",
            cwd="/repo",
            status=SystemWorkflow.STATUS_FAILED,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/repo",
            prompt="Start an interactive web demo",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=demo.DEMO_AGENT_KIND,
            display_author=demo.DEMO_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=demo.DEMO_AGENT_KIND,
            thread_id="thread-1",
            instance=instance,
            status=SystemAgentRun.STATUS_FAILED,
        )

        index_response = self.client.get(reverse("index"))
        system_response = self.client.get(reverse("system_sessions"))

        self.assertContains(index_response, "User feature")
        self.assertContains(system_response, "User feature")
        self.assertContains(system_response, "Demo agent")
        self.assertContains(
            system_response,
            reverse("system_session", kwargs={"session_id": "thread-1"}),
        )

    @patch("hitch.main.demo.request_demo_start")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_rejects_missing_cwd_before_starting_container(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="", turns=[])
        )

        response = self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "thread has no cwd", status_code=400)
        mock_request_demo.assert_not_called()

    @patch("hitch.main.demo.request_demo_start", side_effect=demo.DemoError("no podman"))
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_reports_demo_start_failure(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        _mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )
        original_save = SystemWorkflow.save

        def save_side_effect(
            workflow: SystemWorkflow, *args: Any, **kwargs: Any
        ) -> None:
            if kwargs.get("update_fields") == ["status", "updated_at"]:
                raise AssertionError("status failure should use queryset update")
            original_save(workflow, *args, **kwargs)

        with patch.object(SystemWorkflow, "save", autospec=True) as mock_save:
            mock_save.side_effect = save_side_effect
            response = self.client.post(
                reverse("start_session_demo", kwargs={"session_id": "abc"})
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content, b"no podman")
        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)

    @patch("hitch.main.demo.request_demo_start", side_effect=RuntimeError("boom"))
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_fails_workflow_when_demo_start_raises_unexpected_error(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
        _mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        workflow = SystemWorkflow.objects.get(
            kind=demo.DEMO_WORKFLOW_KIND,
            main_thread_id="abc",
        )
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_FAILED)
        mock_spawn.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_rejects_pending_demo_before_spawning_agent(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        SessionDemo.objects.create(
            thread_id="abc",
            host="127.0.0.1",
            port=3000,
            status=SessionDemo.STATUS_REQUESTED,
            registration_token="token",
        )
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo", turns=[])
        )

        response = self.client.post(reverse("start_session_demo", kwargs={"session_id": "abc"}))

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "demo setup is already running", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.demo.request_demo_start")
    @patch("hitch.main.workflows.system_agents.active_workflow_for_thread", return_value=None)
    @patch("hitch.main.worktrees.discover_managed_worktrees", return_value=[])
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_rejects_unallowed_cwd_before_starting_container(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        _mock_managed: MagicMock,
        _mock_workflow: MagicMock,
        mock_request_demo: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path("/repo")]
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/elsewhere", turns=[])
        )

        response = self.client.post(
            reverse("start_session_demo", kwargs={"session_id": "abc"})
        )

        self.assertEqual(response.status_code, 400)
        mock_request_demo.assert_not_called()

class RegisterSessionDemoViewTests(TestCase):
    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    @patch("hitch.main.demo._verify_registered_container_labels")
    def test_registers_active_demo_from_json(
        self, _mock_verify: MagicMock, _cleanup: MagicMock
    ) -> None:
        session_demo = demo.request_demo_start("abc")

        response = self.client.post(
            reverse("session_demo_register", kwargs={"session_id": "abc"}),
            data=json.dumps(
                {
                    "token": session_demo.registration_token,
                    "status": "active",
                    "container_name": "hitch-demo-abc-1234",
                    "container_id": "container123",
                    "host": "127.0.0.1",
                    "port": 45678,
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content)
        self.assertEqual(payload["status"], SessionDemo.STATUS_ACTIVE)
        session_demo.refresh_from_db()
        self.assertEqual(session_demo.container_name, "hitch-demo-abc-1234")
        self.assertEqual(session_demo.port, 45678)

    @patch("hitch.main.demo.cleanup_unregistered_demo_containers")
    def test_rejects_invalid_registration_token(self, _cleanup: MagicMock) -> None:
        demo.request_demo_start("abc")

        response = self.client.post(
            reverse("session_demo_register", kwargs={"session_id": "abc"}),
            data=json.dumps(
                {
                    "token": "bad",
                    "status": "preparing",
                    "container_name": "hitch-demo-abc-1234",
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"invalid demo registration token")

    def test_rejects_invalid_json(self) -> None:
        response = self.client.post(
            reverse("session_demo_register", kwargs={"session_id": "abc"}),
            data=b"{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"invalid JSON")
