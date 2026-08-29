"""Per-session action endpoints: rename, archive, and approval mode."""


import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.test import (
    TestCase,
)
from django.urls import reverse
from openai_codex.errors import InvalidRequestError

from hitch.main.models import (
    ApprovalRequest,
    ArchivedSessionTokenUsage,
    CodexInstance,
    SessionMetadata,
    SystemWorkflow,
)
from hitch.main.test.support import (
    _seed_cookies,
)
from hitch.main.views import common as common_views
from hitch.main.workflows import system_agents


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

    @patch("hitch.main.views.session_actions.session_index.update_cached_name")
    @patch("hitch.main.views.common.Codex")
    def test_archived_or_unknown_session_returns_400(
        self, mock_codex: MagicMock, mock_update: MagicMock
    ) -> None:
        # thread_set_name raises InvalidRequestError for archived/nonexistent
        # threads (e.g. a rename from a stale tab or right after archiving).
        # Answer 400 like the sibling endpoints rather than 500.
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_set_name.side_effect = InvalidRequestError(
            code=-32600, message="thread is archived"
        )

        response = self.client.post(
            reverse("set_session_name", kwargs={"session_id": "abc"}),
            data={"name": "New title"},
        )

        self.assertEqual(response.status_code, 400)
        # The rename never landed, so the cached name must not be updated.
        mock_update.assert_not_called()

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
            resolved_events = common_views._settle_live_pending_approval_requests(
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
    @patch(
        "hitch.main.views.session_actions._stored_rollout_path_for_thread",
        return_value=Path("/archived/rollout-abc.jsonl"),
    )
    @patch("hitch.main.views.common.Codex")
    def test_first_archive_preserves_resumed_thread_metadata(
        self, mock_codex: MagicMock, _mock_rollout_path: MagicMock
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(
                cwd="/repo",
                name="CLI session",
                preview="Implement the parser",
                created_at=1,
                updated_at=2,
                path="/active/rollout-abc.jsonl",
                thread_source="cli",
            )
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="abc")
        self.assertEqual(metadata.cwd, "/repo")
        self.assertEqual(metadata.codex_name, "CLI session")
        self.assertEqual(metadata.codex_display_title, "CLI session")
        self.assertEqual(metadata.codex_preview, "Implement the parser")
        self.assertEqual(metadata.codex_thread_source, "cli")
        self.assertEqual(metadata.codex_path, "/archived/rollout-abc.jsonl")
        self.assertTrue(metadata.codex_archived)

    @patch(
        "hitch.main.views.session_actions._stored_rollout_path_for_thread",
        return_value=None,
    )
    @patch("hitch.main.workflows.pr_qa.start_pr_qa_workflow")
    @patch("hitch.main.views.common.Codex")
    def test_unarchive_retries_auto_review_deferred_by_archive(
        self,
        mock_codex: MagicMock,
        mock_start: MagicMock,
        _mock_rollout_path: MagicMock,
    ) -> None:
        SessionMetadata.objects.create(
            thread_id="abc",
            cwd="/repo",
            codex_archived=True,
        )
        instance = CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="Implement it",
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            auto_qa_enabled=True,
            approval_mode="auto_review",
        )
        mock_start.side_effect = [
            system_agents.WorkflowStartBlockedByArchiveError(),
            MagicMock(),
        ]
        system_agents.on_codex_instance_finished(instance)
        instance.refresh_from_db()
        self.assertIsNone(instance.auto_qa_triggered_at)

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "false"},
        )

        self.assertEqual(response.status_code, 302)
        instance.refresh_from_db()
        self.assertIsNotNone(instance.auto_qa_triggered_at)
        self.assertEqual(mock_start.call_count, 2)
        self.assertTrue(mock_start.call_args.kwargs["lifecycle_lock_held"])
        mock_codex.return_value.__enter__.return_value.thread_unarchive.assert_called_once_with(
            "abc"
        )

    @patch(
        "hitch.main.views.session_actions._stored_rollout_path_for_thread",
        return_value=Path("/moved/rollout-abc.jsonl"),
    )
    @patch("hitch.main.views.common.Codex")
    def test_updates_archive_state_and_response_shape(
        self, mock_codex: MagicMock, _mock_rollout_path: MagicMock
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
                SessionMetadata.objects.update_or_create(
                    thread_id="abc",
                    defaults={"cwd": "/repo", "codex_path": "/stale/rollout.jsonl"},
                )
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
                self.assertEqual(
                    SessionMetadata.objects.get(thread_id="abc").codex_path,
                    "/moved/rollout-abc.jsonl",
                )
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
    def test_archive_conflict_preserves_active_workflow(
        self, mock_codex: MagicMock
    ) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="abc",
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_PR_PROMPT_RUNNING,
        )
        url = reverse("set_session_archived", kwargs={"session_id": "abc"})

        ajax_response = self.client.post(
            url,
            data={"archived": "true"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        html_response = self.client.post(url, data={"archived": "true"})

        self.assertEqual(ajax_response.status_code, 409)
        self.assertContains(
            ajax_response,
            "Stop the active workflow",
            status_code=409,
        )
        self.assertRedirects(
            html_response,
            reverse("session", kwargs={"session_id": "abc"}),
            fetch_redirect_response=False,
        )
        workflow.refresh_from_db()
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        mock_codex.assert_not_called()

    @patch(
        "hitch.main.views.session_actions.reconciliation.reconcile_dead_for_thread",
        return_value=0,
    )
    @patch("hitch.main.views.common.Codex")
    def test_archive_conflict_preserves_active_main_thread_turn(
        self, mock_codex: MagicMock, _mock_reconcile: MagicMock
    ) -> None:
        CodexInstance.objects.create(
            pid=1,
            thread_id="abc",
            cwd="/repo",
            prompt="Apply QA feedback",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 409)
        mock_codex.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_unarchive_bookkeeping_failure_restores_both_states(
        self, mock_codex: MagicMock
    ) -> None:
        SessionMetadata.objects.create(
            thread_id="abc",
            codex_archived=True,
            codex_path="/archived/rollout.jsonl",
        )
        client = mock_codex.return_value.__enter__.return_value

        with (
            patch(
                "hitch.main.views.session_actions.session_index.update_cached_archived",
                side_effect=RuntimeError("database unavailable"),
            ),
            self.assertRaisesMessage(RuntimeError, "database unavailable"),
        ):
            self.client.post(
                reverse("set_session_archived", kwargs={"session_id": "abc"}),
                data={"archived": "false"},
            )

        metadata = SessionMetadata.objects.get(thread_id="abc")
        self.assertTrue(metadata.codex_archived)
        self.assertEqual(metadata.codex_path, "/archived/rollout.jsonl")
        client.thread_unarchive.assert_called_once_with("abc")
        client.thread_archive.assert_called_once_with("abc")

    @patch(
        "hitch.main.views.session_actions._stored_rollout_path_for_thread",
        return_value=None,
    )
    @patch("hitch.main.views.common.Codex")
    def test_archive_keeps_usage_cache_when_moved_rollout_is_not_found(
        self, mock_codex: MagicMock, _mock_rollout_path: MagicMock
    ) -> None:
        SessionMetadata.objects.create(
            thread_id="abc", cwd="/repo", codex_path="/stale/rollout.jsonl"
        )
        ArchivedSessionTokenUsage.objects.create(
            thread_id="abc", total_tokens=100
        )

        response = self.client.post(
            reverse("set_session_archived", kwargs={"session_id": "abc"}),
            data={"archived": "true"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(SessionMetadata.objects.get(thread_id="abc").codex_path, "")
        self.assertTrue(
            ArchivedSessionTokenUsage.objects.filter(thread_id="abc").exists()
        )
        mock_codex.return_value.__enter__.return_value.thread_archive.assert_called_once_with(
            "abc"
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
