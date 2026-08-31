"""New-session form and start-flow tests."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.exceptions import SuspiciousOperation
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import HttpResponse
from django.test import (
    Client,
    RequestFactory,
    TestCase,
    override_settings,
)
from django.urls import reverse
from django.utils import timezone

from hitch.main import caches
from hitch.main import repos as repos_module
from hitch.main.models import (
    AutonomousGoal,
    Project,
    ProposedSession,
    SessionMetadata,
    UserSettings,
)
from hitch.main.runtime import input_images
from hitch.main.sessions import (
    agent_tasks,
    session_settings,
)
from hitch.main.test.support import (
    _cookie_value,
    _encode_extra_system_prompt,
    _make_model,
    _make_project,
    _seed_cookies,
    _setup_codex,
)
from hitch.main.test.views_helpers import (
    _AUTO_PR_COOKIE,
    _AUTO_QA_COOKIE,
    _ENABLE_MEMORIES_COOKIE,
    _EXTRA_SYSTEM_PROMPT_COOKIE,
    _LAST_SELECTED_REPO_COOKIE,
    _MODEL_COOKIE,
    _PNG_BYTES,
    _PR_PROMPT,
    _QA_PROMPT,
    _SANDBOX_COOKIE,
    _USE_WORKTREES_COOKIE,
    _WEB_SEARCH_COOKIE,
    _FailingUploadWriter,
    _UnreadableUpload,
)
from hitch.main.views import common as common_views
from hitch.main.views import messages as message_views
from hitch.main.views import new_session as new_session_views
from hitch.main.worktrees import (
    ManagedWorktree,
    WorktreeCleanupError,
    WorktreeCreationError,
)


class NewSessionViewTests(TestCase):
    REPO = "/home/user/proj"

    @override
    def setUp(self) -> None:
        super().setUp()
        self._clear_models_cache()
        self.addCleanup(self._clear_models_cache)

    @staticmethod
    def _clear_models_cache() -> None:
        with caches._MODELS_REFRESH_LOCK:
            caches._MODELS_CACHE_VALUE = {}
            caches._MODELS_CACHE_FETCHED_AT = {}
            caches._MODELS_REFRESH_IN_FLIGHT = set()

    def _assert_new_session_spawn(
        self,
        mock_spawn: MagicMock,
        *,
        cwd: str = REPO,
        prompt: str = "do thing",
        **overrides: Any,
    ) -> None:
        expected = {
            "cwd": cwd,
            "prompt": prompt,
            "developer_instructions": None,
            "model": None,
            "reasoning_effort": None,
            "sandbox_policy": None,
            "approval_mode": "auto_review",
        }
        expected.update(overrides)
        mock_spawn.assert_called_once_with(**expected)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_cold_empty_catalog_keeps_rendered_preferred_effort(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-cold-empty")
        caches._store_models_cache(enable_memories=False, models_data=[])
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "model": "",
                "reasoning_effort": "high",
                "rendered_model": "",
                "rendered_reasoning_effort": "high",
            },
        )

        self.assertEqual(response.status_code, 302)
        self._assert_new_session_spawn(
            mock_spawn,
            reasoning_effort="high",
        )

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_rejects_unsupported_model_effort_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(
            mock_codex,
            models=[_make_model("gpt-default", is_default=True)],
        )

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "model": "gpt-default",
                "reasoning_effort": "xhigh",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "reasoning effort 'xhigh' is not supported by model 'gpt-default'",
            status_code=400,
        )
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_rejects_unadvertised_effort_without_constraints(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        cases = ("unconstrained catalog", "empty catalog")

        for label in cases:
            with self.subTest(label=label):
                self._clear_models_cache()
                client = Client()
                model = _make_model("provider-model", is_default=True)
                model.supported_reasoning_efforts = []
                models = [model] if label == "unconstrained catalog" else []
                _setup_codex(mock_codex, models=models)
                if models:
                    caches._store_models_cache(
                        enable_memories=False,
                        models_data=models,
                    )
                _seed_cookies(
                    client,
                    **{
                        _MODEL_COOKIE: "provider-model",
                        "hitch_reasoning_effort": "provider-effort",
                    },
                )

                response = client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "do thing",
                        "cwd": self.REPO,
                        "model": "provider-model",
                        "reasoning_effort": "crafted-effort",
                    },
                )

                self.assertContains(
                    response,
                    "invalid reasoning effort",
                    status_code=400,
                )
                mock_spawn.assert_not_called()

    def test_input_image_upload_handler_rejects_limits_during_parse(self) -> None:
        with patch("hitch.main.runtime.input_images._INPUT_IMAGE_MAX_BYTES", 8):
            handler = input_images._InputImageLimitUploadHandler()
            with self.assertRaisesMessage(
                SuspiciousOperation,
                "image attachment is too large",
            ):
                handler.new_file(
                    "input_images",
                    "screen.png",
                    "image/png",
                    9,
                )

            handler = input_images._InputImageLimitUploadHandler()
            handler.new_file("input_images", "screen.png", "image/png", None)
            with self.assertRaisesMessage(
                SuspiciousOperation,
                "image attachment is too large",
            ):
                handler.receive_data_chunk(b"123456789", 0)

        handler = input_images._InputImageLimitUploadHandler()
        with self.assertRaisesMessage(
            SuspiciousOperation,
            "at most 4 image attachments are allowed",
        ):
            for index in range(5):
                handler.new_file(
                    "input_images",
                    f"screen-{index}.png",
                    "image/png",
                    1,
                )

    def test_input_image_request_size_cap_runs_before_parse(self) -> None:
        request = RequestFactory().post(reverse("new_session"), data={})
        request.META["CONTENT_LENGTH"] = str(input_images._INPUT_IMAGE_MAX_REQUEST_BYTES + 1)

        self.assertEqual(
            input_images._input_image_request_size_error(request),
            "image attachments are too large",
        )

    def test_input_image_upload_limiter_handles_cased_multipart_before_csrf(
        self,
    ) -> None:
        def view(request: Any) -> Any:
            self.assertIsInstance(
                request.upload_handlers[0],
                input_images._InputImageLimitUploadHandler,
            )
            return HttpResponse("ok")

        request = RequestFactory().generic("POST", reverse("new_session"), data=b"")
        request.content_type = "Multipart/form-data"
        request.META["CONTENT_TYPE"] = "Multipart/form-data; boundary=BOUNDARY"
        cast(Any, request)._dont_enforce_csrf_checks = True

        response = input_images._limit_input_image_uploads(view)(request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(getattr(new_session_views.new_session, "csrf_exempt", False))
        self.assertTrue(getattr(message_views.send_message, "csrf_exempt", False))

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_upload_limited_new_session_still_enforces_csrf(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        client = Client(enforce_csrf_checks=True)
        url = reverse("new_session")

        denied = client.post(
            url,
            data={
                "prompt": "Use this screenshot",
                "cwd": self.REPO,
                "input_images": SimpleUploadedFile("screen.png", _PNG_BYTES, content_type="image/png"),
            },
        )

        self.assertEqual(denied.status_code, 403)
        mock_spawn.assert_not_called()

        client.get(reverse("index"))
        token = client.cookies["csrftoken"].value
        allowed = client.post(
            url,
            data={
                "csrfmiddlewaretoken": token,
                "prompt": "Use this screenshot",
                "cwd": self.REPO,
                "input_images": SimpleUploadedFile("screen.png", _PNG_BYTES, content_type="image/png"),
            },
        )

        self.assertEqual(allowed.status_code, 302)
        mock_spawn.assert_called_once()

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_rejects_invalid_image_uploads_before_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
        cases: list[tuple[str, object, str]] = [
            (
                "too many",
                [SimpleUploadedFile(f"screen-{index}.png", _PNG_BYTES, content_type="image/png") for index in range(5)],
                "at most 4 image attachments are allowed",
            ),
            (
                "empty",
                SimpleUploadedFile("screen.png", b"", content_type="image/png"),
                "image attachment is empty",
            ),
            (
                "bad magic",
                SimpleUploadedFile("screen.png", b"not an image", content_type="image/png"),
                "image attachment must be PNG, JPEG, GIF, or WebP",
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            for label, upload, message in cases:
                with self.subTest(label=label):
                    mock_spawn.reset_mock()
                    response = self.client.post(
                        reverse("new_session"),
                        data={
                            "prompt": "Use the screenshot",
                            "cwd": self.REPO,
                            "input_images": upload,
                        },
                    )

                    self.assertContains(response, message, status_code=400)
                    mock_spawn.assert_not_called()
                    self.assertFalse((Path(raw) / "attachments").exists())

            with patch("hitch.main.runtime.input_images._INPUT_IMAGE_MAX_BYTES", len(_PNG_BYTES) - 1):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Use the screenshot",
                        "cwd": self.REPO,
                        "input_images": SimpleUploadedFile("screen.png", _PNG_BYTES, content_type="image/png"),
                    },
                )

            self.assertContains(response, "image attachment is too large", status_code=400)
            mock_spawn.assert_not_called()
            self.assertFalse((Path(raw) / "attachments").exists())

            with (
                patch(
                "os.fdopen",
                side_effect=lambda fd, *_args: _FailingUploadWriter(fd),
                ),
                self.assertLogs("hitch.main.views", level="ERROR"),
            ):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Use the screenshot",
                        "cwd": self.REPO,
                        "input_images": SimpleUploadedFile("screen.png", _PNG_BYTES, content_type="image/png"),
                    },
                )

            self.assertContains(
                response,
                "failed to save image attachment",
                status_code=400,
            )
            self.assertNotContains(response, "disk full", status_code=400)
            mock_spawn.assert_not_called()
            attachments = Path(raw) / "attachments"
            self.assertTrue(attachments.exists())
            self.assertEqual([path for path in attachments.rglob("*") if path.is_file()], [])

            real_fdopen = os.fdopen
            fdopen_calls = 0

            def fail_second_file(fd: int, *args: Any, **kwargs: Any) -> Any:
                nonlocal fdopen_calls
                fdopen_calls += 1
                if fdopen_calls == 2:
                    return _FailingUploadWriter(fd)
                return real_fdopen(fd, *args, **kwargs)

            with (
                patch(
                "os.fdopen",
                side_effect=fail_second_file,
                ),
                self.assertLogs("hitch.main.views", level="ERROR"),
            ):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Use the screenshots",
                        "cwd": self.REPO,
                        "input_images": [
                            SimpleUploadedFile("first.png", _PNG_BYTES, content_type="image/png"),
                            SimpleUploadedFile("second.png", _PNG_BYTES, content_type="image/png"),
                        ],
                    },
                )

            self.assertContains(
                response,
                "failed to save image attachment",
                status_code=400,
            )
            self.assertEqual([path for path in attachments.rglob("*") if path.is_file()], [])

    def test_image_upload_read_failure_returns_generic_error(self) -> None:
        with self.assertLogs("hitch.main.views", level="ERROR"):
            _extension, error = common_views._uploaded_input_image_extension(
                _UnreadableUpload("screen.png", _PNG_BYTES, content_type="image/png")
            )

        self.assertEqual(error, "failed to read image attachment")
        assert error is not None
        self.assertNotIn("/tmp/private", error)



    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.runtime.codex_pool.create_session_thread")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_rejects_review_task_images_before_side_effects(
        self,
        mock_discover: MagicMock,
        mock_codex: MagicMock,
        mock_create_thread: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        for prompt in ("/pr", "/pr-now", "/qa"):
            with self.subTest(prompt=prompt):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": prompt,
                        "cwd": self.REPO,
                        "input_images": SimpleUploadedFile(
                            "screen.png", _PNG_BYTES, content_type="image/png"
                        ),
                    },
                )

                self.assertContains(
                    response,
                    "image attachments are not supported for review or PR tasks",
                    status_code=400,
                )
                mock_create_thread.assert_not_called()
                mock_spawn.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.runtime.codex_pool.create_session_thread")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_pr_now_spawns_ordinary_publish_turn(
        self,
        mock_discover: MagicMock,
        mock_codex: MagicMock,
        mock_create_thread: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_thread.return_value = "pr-now-thread"
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/PR-NOW", "cwd": self.REPO, "plan_mode": "true"},
        )

        self.assertEqual(response.status_code, 302)
        mock_create_thread.assert_called_once_with(
            cwd=self.REPO,
            name=_PR_PROMPT,
            developer_instructions=None,
            model="gpt-5.4",
            enable_memories=False,
        )
        kwargs = mock_spawn.call_args.kwargs
        self.assertEqual(kwargs["thread_id"], "pr-now-thread")
        self.assertEqual(kwargs["prompt"], agent_tasks.publish_pr_task().prompt)
        self.assertEqual(kwargs["agent_kind"], agent_tasks.PR_PUBLISH_AGENT_KIND)
        self.assertNotIn("workflow_id", kwargs)

    @patch("hitch.main.views.common._save_posted_input_images")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_accept_losing_start_claim_redirects_without_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_save_images: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = _make_project(repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )
        mock_discover.return_value = [Path(self.REPO)]

        def reject_after_lookup(_request: Any) -> tuple[list[str], str | None]:
            rejected = ProposedSession.objects.filter(
                pk=proposal.pk,
                outcome_status=ProposedSession.OUTCOME_UNSET,
            ).update(
                outcome_status=ProposedSession.OUTCOME_REJECTED,
                outcome_notes="Resolved from another tab.",
                updated_at=timezone.now(),
            )
            self.assertEqual(rejected, 1)
            return [], None

        mock_save_images.side_effect = reject_after_lookup

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Go ahead and implement this proposed session.",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("inbox"))
        mock_spawn.assert_not_called()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertIsNone(proposal.accepted_session)

    @patch("hitch.main.views.common.goal_workflows.stop_running_autonomous_goal_stack_after_proposal_resolution")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_spawn_failure_resets_proposal_start_claim(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_stop_stack: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = _make_project(repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.side_effect = RuntimeError("worker failed")

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "Go ahead and implement this proposed session.",
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
            )

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )
        mock_stop_stack.assert_not_called()

    def test_new_session_finish_ignores_replaced_start_claim(self) -> None:
        project = _make_project(repo_path=self.REPO)
        claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
        old_claim = "2026-06-05T14:00:00+00:00"
        new_claim = "2026-06-05T14:45:00+00:00"
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: old_claim,
            },
        )
        ProposedSession.objects.filter(pk=proposal.pk).update(
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: new_claim,
            }
        )
        metadata = SessionMetadata.objects.create(
            thread_id="late-thread",
            cwd=self.REPO,
            project=project,
        )

        new_session_views._finish_new_session_proposal_start_claim(
            proposal,
            metadata,
            approved_snapshot="",
        )

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertEqual(proposal.outcome_metadata[claim_key], new_claim)

    def test_new_session_reset_ignores_replaced_start_claim(self) -> None:
        project = _make_project(repo_path=self.REPO)
        claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
        old_claim = "2026-06-05T14:00:00+00:00"
        new_claim = "2026-06-05T14:45:00+00:00"
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: old_claim,
            },
        )
        ProposedSession.objects.filter(pk=proposal.pk).update(
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                claim_key: new_claim,
            }
        )

        new_session_views._reset_new_session_proposal_start_claim(proposal)

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertEqual(proposal.outcome_metadata[claim_key], new_claim)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_accept_preserves_proposal_auto_qa_setting(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = _make_project(repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            outcome_metadata={
                "auto_pr_enabled": False,
                "auto_qa_enabled": True,
            },
        )
        AutonomousGoal.objects.filter(pk=goal.pk).update(auto_qa_enabled=False)
        prompt = "Go ahead and implement this proposed session."
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": prompt,
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
                "auto_qa": "false",
            },
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-xyz")
        self.assertTrue(metadata.auto_qa_enabled)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, metadata)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )
        self._assert_new_session_spawn(
            mock_spawn,
            prompt=agent_tasks.with_automatic_review_guidance(
                prompt,
                auto_pr_enabled=False,
                auto_qa_enabled=True,
            ),
            thread_name="Add parser coverage",
        )

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos", return_value=[])
    def test_upgrade_recovery_starts_fresh_with_stored_target_and_auto_pr(
        self,
        _mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        source = SessionMetadata.objects.create(
            thread_id="legacy-source",
            cwd=self.REPO,
            auto_pr_enabled=False,
        )
        proposal = ProposedSession.objects.create(
            source_session=source,
            title="Recovered request",
            prompt="Finish the interrupted request.",
            outcome_metadata={
                "resume_source_session": True,
                "auto_pr_enabled": True,
                "auto_qa_enabled": False,
            },
        )
        mock_spawn.return_value = SimpleNamespace(thread_id="fresh-recovery")

        with patch("hitch.main.repos.repo_root", return_value=Path(self.REPO)):
            get_response = self.client.get(
                reverse("new_session"),
                {"proposed_session": str(proposal.pk)},
            )
            post_response = self.client.post(
                reverse("new_session"),
                {
                    "prompt": proposal.prompt,
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                    "auto_pr": "false",
                },
            )

        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(post_response.status_code, 302)
        accepted = SessionMetadata.objects.get(thread_id="fresh-recovery")
        self.assertNotEqual(accepted.pk, source.pk)
        self.assertTrue(accepted.auto_pr_enabled)
        source.refresh_from_db()
        self.assertFalse(source.auto_pr_enabled)
        proposal.refresh_from_db()
        self.assertEqual(proposal.accepted_session, accepted)
        self._assert_new_session_spawn(
            mock_spawn,
            prompt=agent_tasks.with_automatic_review_guidance(
                proposal.prompt,
                auto_pr_enabled=True,
                auto_qa_enabled=False,
                pr_title=proposal.title,
            ),
            thread_name=proposal.title,
            agent_kind=agent_tasks.PR_PUBLISH_AGENT_KIND,
            user_message_index=0,
        )












    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos", return_value=[Path(REPO)])
    def test_new_session_rejects_invalid_proposed_session_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO)
        other_project = _make_project(name="Other", repo_path="/home/user/other")
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Improve docs",
            goal="Find useful docs increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add docs coverage",
        )
        resolved_order = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        resolved = ProposedSession.objects.create(
            autonomous_goal=resolved_order,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_REJECTED,
        )
        mock_discover.return_value = [Path(self.REPO), Path(other_project.repo_path)]
        _setup_codex(mock_codex, models=[])
        cases = [
            (
                "non-numeric id",
                {"cwd": self.REPO, "proposed_session": "not-a-number"},
                b"proposed session is required",
            ),
            (
                "zero id",
                {"cwd": self.REPO, "proposed_session": "0"},
                b"proposed session is required",
            ),
            (
                "missing id",
                {"cwd": self.REPO, "proposed_session": "999"},
                b"proposed session is required",
            ),
            (
                "posted project mismatch",
                {"project": str(project.pk), "proposed_session": str(proposal.pk)},
                b"proposed session does not match project",
            ),
            (
                "implicit cwd mismatch",
                {"cwd": self.REPO, "proposed_session": str(proposal.pk)},
                b"proposed session does not match project",
            ),
            (
                "bare repo mismatch",
                {
                    "project": session_settings._BARE_REPO_PROJECT_VALUE,
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
                b"proposed session does not match project",
            ),
            (
                "resolved proposal",
                {"project": str(project.pk), "proposed_session": str(resolved.pk)},
                b"proposed session is required",
            ),
        ]

        for label, data, message in cases:
            with self.subTest(label=label):
                response = self.client.post(
                    reverse("new_session"),
                    data={
                        "prompt": "Go ahead and implement this proposed session.",
                        **data,
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.content, message)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_remembers_selected_repo_in_account_settings(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user("dev@example.com", password="StrongPass123!")
        self.client.force_login(user)
        other_repo = "/home/user/other"
        mock_discover.return_value = [Path(self.REPO), Path(other_repo)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": other_repo},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.last_selected_repo, other_repo)
        self.assertEqual(_cookie_value(response, _LAST_SELECTED_REPO_COOKIE), other_repo)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_auto_pr_precedence_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        _setup_codex(mock_codex, models=[])
        cases: list[dict[str, Any]] = [
            {
                "name": "posted override enables bare repo",
                "post_auto_pr": "true",
                "expected": True,
            },
            {
                "name": "posted override disables global setting",
                "global_auto_pr": "true",
                "post_auto_pr": "false",
                "expected": False,
            },
            {
                "name": "project on sets default",
                "project_auto_pr_mode": Project.AUTO_PR_ON,
                "expected": True,
            },
            {
                "name": "project off overrides global setting",
                "global_auto_pr": "true",
                "project_auto_pr_mode": Project.AUTO_PR_OFF,
                "expected": False,
            },
            {
                "name": "posted override disables project setting",
                "project_auto_pr_mode": Project.AUTO_PR_ON,
                "post_auto_pr": "false",
                "expected": False,
            },
        ]

        for index, case in enumerate(cases):
            with self.subTest(case["name"]):
                client = Client()
                thread_id = f"thread-{index}"
                repo = f"{self.REPO}-{index}" if "project_auto_pr_mode" in case else self.REPO
                data = {"prompt": "do thing"}
                project_auto_pr_mode = case.get("project_auto_pr_mode")
                if project_auto_pr_mode is None:
                    data["cwd"] = repo
                else:
                    project = _make_project(
                        name=f"Hitch {index}",
                        repo_path=repo,
                        auto_pr_mode=project_auto_pr_mode,
                    )
                    data["project"] = str(project.pk)
                if "post_auto_pr" in case:
                    data["auto_pr"] = case["post_auto_pr"]
                if "global_auto_pr" in case:
                    _seed_cookies(client, **{_AUTO_PR_COOKIE: case["global_auto_pr"]})
                mock_discover.return_value = [Path(repo)]
                mock_spawn.return_value = SimpleNamespace(thread_id=thread_id)
                mock_spawn.reset_mock()

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                metadata = SessionMetadata.objects.get(thread_id=thread_id)
                self.assertEqual(metadata.auto_pr_enabled, case["expected"])
                expected_spawn: dict[str, Any] = {
                    "cwd": repo,
                    "prompt": agent_tasks.with_automatic_review_guidance(
                        "do thing",
                        auto_pr_enabled=case["expected"],
                        auto_qa_enabled=False,
                    ),
                    "developer_instructions": None,
                    "model": None,
                    "reasoning_effort": None,
                    "sandbox_policy": None,
                    "approval_mode": "auto_review",
                }
                if case["expected"]:
                    expected_spawn["agent_kind"] = agent_tasks.PR_PUBLISH_AGENT_KIND
                    expected_spawn["user_message_index"] = 0
                mock_spawn.assert_called_once_with(**expected_spawn)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_forwards_cookie_settings_to_spawn_matrix(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        codex = _setup_codex(mock_codex, models=[])
        cases: list[tuple[str, dict[str, str], dict[str, Any]]] = [
            (
                "extra prompt",
                {_EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt("  Always run focused tests.  ")},
                {"developer_instructions": "Always run focused tests."},
            ),
            (
                "model effort sandbox",
                {
                    "hitch_model": "gpt-5",
                    "hitch_reasoning_effort": "high",
                    "hitch_sandbox_policy": "workspaceWrite",
                },
                {
                    "model": "gpt-5",
                    "reasoning_effort": "high",
                    "sandbox_policy": "workspaceWrite",
                },
            ),
            ("memories", {_ENABLE_MEMORIES_COOKIE: "true"}, {"enable_memories": True}),
            (
                "web search",
                {_WEB_SEARCH_COOKIE: "live"},
                {"web_search_mode": "live"},
            ),
            (
                "deny all approval",
                {"hitch_approval_mode": "deny_all"},
                {"approval_mode": "deny_all"},
            ),
            (
                "prompt user approval",
                {"hitch_approval_mode": "prompt_user"},
                {"approval_mode": "prompt_user"},
            ),
        ]

        for index, (label, cookies, expected) in enumerate(cases):
            with self.subTest(label=label):
                self._clear_models_cache()
                client = Client()
                codex.models.return_value.data = (
                    [_make_model("gpt-5", is_default=True)] if "hitch_model" in cookies else []
                )
                mock_spawn.return_value = SimpleNamespace(thread_id=f"thread-{index}")
                mock_spawn.reset_mock()
                _seed_cookies(client, **cookies)

                response = client.post(
                    reverse("new_session"),
                    data={"prompt": "do thing", "cwd": self.REPO},
                )

                self.assertEqual(response.status_code, 302)
                self._assert_new_session_spawn(mock_spawn, **expected)

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_rejects_invalid_web_search_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "web_search_mode": "maybe",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "invalid web search setting", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_rejects_invalid_worktree_override(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "do thing",
                "cwd": self.REPO,
                "use_worktrees": "maybe",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "invalid worktree setting", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_plan_slash_command_allows_image_only_prompt(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-plan-image")
        _setup_codex(mock_codex, models=[_make_model("gpt-default", is_default=True)])

        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(CODEX_EVENTS_DIR=Path(raw)),
        ):
            response = self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "/plan",
                    "cwd": self.REPO,
                    "input_images": SimpleUploadedFile("screen.png", _PNG_BYTES, content_type="image/png"),
                },
            )

        self.assertEqual(response.status_code, 302)
        image_paths = mock_spawn.call_args.kwargs["input_image_paths"]
        self.assertEqual(len(image_paths), 1)
        self._assert_new_session_spawn(
            mock_spawn,
            prompt="",
            model="gpt-default",
            plan_mode=True,
            input_image_paths=image_paths,
        )

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.create_session_thread")
    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.repos.discover_repos")
    def test_new_session_review_and_pr_shortcuts_spawn_ordinary_turns(
        self,
        mock_discover: MagicMock,
        mock_create_worktree: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_turn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        codex = _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])
        cases: list[
            tuple[
                str,
                dict[str, str],
                dict[str, str],
                dict[str, Any],
                dict[str, Any],
            ]
        ] = [
            (
                "pr",
                {"prompt": "/PR", "cwd": self.REPO, "plan_mode": "true"},
                {
                    _MODEL_COOKIE: "gpt-5.4",
                    "hitch_reasoning_effort": "high",
                    _WEB_SEARCH_COOKIE: "live",
                    _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt("Use repo conventions."),
                },
                {
                    "name": _PR_PROMPT,
                    "developer_instructions": "Use repo conventions.",
                    "model": "gpt-5.4",
                    "web_search_mode": "live",
                },
                {
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "developer_instructions": "Use repo conventions.",
                    "web_search_mode": "live",
                },
            ),
            (
                "qa",
                {
                    "prompt": "/QA",
                    "cwd": self.REPO,
                    "plan_mode": "true",
                    "web_search_mode": "disabled",
                },
                {},
                {
                    "name": _QA_PROMPT,
                    "developer_instructions": None,
                    "model": "gpt-5.4",
                    "web_search_mode": "disabled",
                },
                {
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "developer_instructions": None,
                    "web_search_mode": "disabled",
                },
            ),
            (
                "pr uses selected repo when worktrees enabled",
                {"prompt": "/pr", "cwd": self.REPO},
                {_USE_WORKTREES_COOKIE: "true"},
                {"name": _PR_PROMPT, "developer_instructions": None, "model": None},
                {
                    "model": None,
                    "reasoning_effort": None,
                    "developer_instructions": None,
                },
            ),
        ]

        for index, (
            label,
            data,
            cookies,
            thread_kwargs,
            task_settings,
        ) in enumerate(cases):
            with self.subTest(label=label):
                self._clear_models_cache()
                client = Client()
                codex.models.return_value.data = (
                    [] if cookies.get(_USE_WORKTREES_COOKIE) == "true" else [_make_model("gpt-5.4", is_default=True)]
                )
                mock_create_thread.return_value = f"thread-{index}"
                mock_create_thread.reset_mock()
                mock_create_worktree.reset_mock()
                mock_turn.reset_mock()
                if cookies:
                    _seed_cookies(client, **cookies)

                response = client.post(reverse("new_session"), data=data)

                self.assertEqual(response.status_code, 302)
                mock_create_worktree.assert_not_called()
                mock_create_thread.assert_called_once_with(
                    cwd=self.REPO,
                    enable_memories=False,
                    **thread_kwargs,
                )
                task = agent_tasks.review_task(
                    prepare_pull_request=label != "qa"
                )
                mock_turn.assert_called_once_with(
                    thread_id=f"thread-{index}",
                    cwd=self.REPO,
                    prompt=task.prompt,
                    sandbox_policy=None,
                    approval_mode="auto_review",
                    enable_memories=False,
                    user_message_index=0,
                    agent_kind=task.agent_kind,
                    **task_settings,
                )

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.create_session_thread")
    @patch("hitch.main.repos.pull_default_branch_from_origin")
    def test_qa_with_auto_pull_enabled_preserves_dirty_checkout(
        self,
        mock_pull: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_turn: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO, auto_pull_enabled=True)
        mock_pull.side_effect = repos_module.AutoPullError("project repository has uncommitted changes")
        mock_create_thread.return_value = "dirty-qa-thread"
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "/qa", "project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        mock_pull.assert_not_called()
        mock_create_thread.assert_called_once()
        mock_turn.assert_called_once()

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.create_session_thread")
    @patch("hitch.main.repos.discover_repos")
    def test_coding_agent_proposal_qa_slash_does_not_persist_global_auto_review(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_turn: MagicMock,
    ) -> None:
        # A coding-agent proposal (no autonomous goal) leaves the inbox's
        # auto-review inputs empty, so the proposal did not request auto-review.
        # Accepting it via /qa with global auto-QA enabled must not persist
        # auto-QA on the session: only proposal-requested settings carry forward.
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[_make_model("gpt-5.4", is_default=True)])
        mock_create_thread.return_value = "coding-proposal-thread"
        project = _make_project(repo_path=self.REPO)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Tidy up logging",
        )
        self.assertIsNone(proposal.autonomous_goal)
        client = Client()
        _seed_cookies(client, **{_AUTO_QA_COOKIE: "true"})

        response = client.post(
            reverse("new_session"),
            data={
                "prompt": "/qa",
                "cwd": self.REPO,
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_turn.assert_called_once()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        metadata = SessionMetadata.objects.get(thread_id="coding-proposal-thread")
        self.assertFalse(metadata.auto_qa_enabled)
        self.assertFalse(metadata.auto_pr_enabled)

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.create_session_thread")
    @patch("hitch.main.repos.discover_repos")
    def test_proposal_qa_thread_create_failure_resets_start_claim(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_turn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
        project = _make_project(repo_path=self.REPO)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Tidy up logging",
        )
        mock_create_thread.side_effect = RuntimeError("thread failed")

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "/qa",
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
            )

        mock_turn.assert_not_called()
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.create_session_thread")
    @patch("hitch.main.repos.discover_repos")
    def test_proposal_qa_turn_start_failure_resets_start_claim(
        self,
        mock_discover: MagicMock,
        mock_create_thread: MagicMock,
        mock_codex: MagicMock,
        mock_turn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])
        project = _make_project(repo_path=self.REPO)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Tidy up logging",
        )
        mock_create_thread.return_value = "proposal-qa-thread"
        mock_turn.side_effect = RuntimeError("turn failed")

        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("new_session"),
                data={
                    "prompt": "/qa",
                    "cwd": self.REPO,
                    "proposed_session": str(proposal.pk),
                },
            )

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_plan_mode_returns_bad_request_when_model_cannot_be_resolved(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "make a migration plan",
                "cwd": self.REPO,
                "plan_mode": "true",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "plan mode requires a model", status_code=400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.repos.pull_default_branch_from_origin")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    def test_auto_pull_failure_blocks_new_session(
        self,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_pull: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO, auto_pull_enabled=True)
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})
        mock_pull.side_effect = repos_module.AutoPullError("project repository has uncommitted changes")

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 400)
        self.assertContains(
            response,
            "could not update project before session: project repository has uncommitted changes",
            status_code=400,
        )
        mock_create_worktree.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_managed_worktree_respects_explicit_sandbox_setting(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        worktree = Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12")
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = ManagedWorktree(
            path=worktree,
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(
            self.client,
            **{
                _USE_WORKTREES_COOKIE: "true",
                _SANDBOX_COOKIE: "readOnly",
            },
        )

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_spawn.call_args.kwargs["sandbox_policy"], "readOnly")

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    @patch("hitch.main.views.new_session.snapshot_worktree_to_commit")
    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    def test_legacy_ag_candidate_is_snapshotted_into_fresh_session(
        self,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_snapshot: MagicMock,
        mock_cleanup_candidate: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO, auto_pull_enabled=False)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve reliability",
            goal="Fix a meaningful reliability problem.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="legacy-candidate",
            cwd="/candidate",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Harden retries",
            candidate_session=candidate,
            outcome_metadata={
                "autonomous_goal_autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            },
        )
        snapshot = "b" * 40
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/legacy-snapshot"),
            branch="hitch/proj/legacy-snapshot",
            source_repo=Path(self.REPO),
        )
        mock_snapshot.return_value = snapshot
        mock_create_worktree.return_value = worktree
        mock_spawn.return_value = SimpleNamespace(thread_id="visible-thread")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Continue the proposed work.",
                "project": str(project.pk),
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_snapshot.assert_called_once_with(
            candidate.cwd,
            message="Snapshot legacy autonomous-goal proposal",
        )
        mock_create_worktree.assert_called_once_with(
            self.REPO, base_ref=snapshot
        )
        self.assertEqual(mock_spawn.call_args.kwargs["cwd"], str(worktree.path))
        spawned_prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn(
            f"Candidate log: {reverse('system_session', args=[candidate.thread_id])}",
            spawned_prompt,
        )
        self.assertIn(f"Approved snapshot: {snapshot}", spawned_prompt)
        self.assertIn(
            "candidate transcript is linked for reference and is not automatically loaded",
            spawned_prompt,
        )
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNotNone(proposal.accepted_session)
        assert proposal.accepted_session is not None
        self.assertEqual(proposal.accepted_session.thread_id, "visible-thread")
        self.assertNotEqual(proposal.accepted_session_id, candidate.pk)
        self.assertEqual(proposal.outcome_metadata["accepted_snapshot_sha"], snapshot)
        mock_cleanup_candidate.assert_called_once_with(candidate.cwd)

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    @patch("hitch.main.views.new_session.release_snapshot_commit_ref")
    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    def test_ag_snapshot_starts_fresh_visible_thread_and_worktree(
        self,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_release_snapshot: MagicMock,
        mock_cleanup_candidate: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO, auto_pull_enabled=False)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve reliability",
            goal="Fix a meaningful reliability problem.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )
        hidden_candidate = SessionMetadata.objects.create(
            thread_id="hidden-candidate",
            cwd="/candidate",
            project=project,
            is_hidden_system_session=True,
        )
        snapshot = "a" * 40
        snapshot_ref = (
            "refs/hitch/autonomous-goals/1/"
            "0123456789abcdef0123456789abcdef"
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Harden retries",
            candidate_session=hidden_candidate,
            outcome_metadata={
                "autonomous_goal_tool_protocol": True,
                "autonomous_goal_autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "approved_snapshot_sha": snapshot,
                "approved_snapshot_ref": snapshot_ref,
            },
        )
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/snapshot"),
            branch="hitch/proj/snapshot",
            source_repo=Path(self.REPO),
        )
        mock_create_worktree.return_value = worktree
        mock_spawn.return_value = SimpleNamespace(thread_id="visible-thread")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Continue the approved work.",
                "project": str(project.pk),
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_create_worktree.assert_called_once_with(
            self.REPO, base_ref=snapshot
        )
        self.assertEqual(mock_spawn.call_args.kwargs["cwd"], str(worktree.path))
        spawned_prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn(
            f"Candidate log: {reverse('system_session', args=[hidden_candidate.thread_id])}",
            spawned_prompt,
        )
        self.assertIn(f"Approved snapshot: {snapshot}", spawned_prompt)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNotNone(proposal.accepted_session)
        assert proposal.accepted_session is not None
        self.assertEqual(proposal.accepted_session.thread_id, "visible-thread")
        self.assertEqual(proposal.outcome_metadata["accepted_snapshot_sha"], snapshot)
        mock_release_snapshot.assert_called_once_with(self.REPO, snapshot_ref)
        mock_cleanup_candidate.assert_called_once_with(hidden_candidate.cwd)
        hidden_candidate.refresh_from_db()
        self.assertTrue(hidden_candidate.is_hidden_system_session)

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    @patch("hitch.main.views.new_session.release_snapshot_commit_ref")
    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    def test_ag_propose_only_snapshot_starts_in_selected_repository(
        self,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_release_snapshot: MagicMock,
        mock_cleanup_candidate: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO, auto_pull_enabled=False)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve reliability",
            goal="Find a meaningful reliability improvement.",
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="propose-only-candidate",
            cwd="/candidate",
            project=project,
            is_hidden_system_session=True,
        )
        snapshot_ref = (
            "refs/hitch/autonomous-goals/1/"
            "0123456789abcdef0123456789abcdef"
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Harden retries",
            candidate_session=candidate,
            outcome_metadata={
                "autonomous_goal_tool_protocol": True,
                "autonomous_goal_autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                "approved_snapshot_sha": "a" * 40,
                "approved_snapshot_ref": snapshot_ref,
            },
        )
        mock_spawn.return_value = SimpleNamespace(thread_id="visible-thread")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Investigate the proposed improvement.",
                "project": str(project.pk),
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_create_worktree.assert_not_called()
        self.assertEqual(mock_spawn.call_args.kwargs["cwd"], self.REPO)
        spawned_prompt = mock_spawn.call_args.kwargs["prompt"]
        self.assertIn(
            f"Candidate log: {reverse('system_session', args=[candidate.thread_id])}",
            spawned_prompt,
        )
        self.assertNotIn("Approved snapshot:", spawned_prompt)
        self.assertNotIn("starts from the approved snapshot", spawned_prompt)
        mock_release_snapshot.assert_called_once_with(self.REPO, snapshot_ref)
        mock_cleanup_candidate.assert_called_once_with(candidate.cwd)
        proposal.refresh_from_db()
        self.assertNotIn("accepted_snapshot_sha", proposal.outcome_metadata)

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_hidden_ag_candidate_cleanup_failure_is_nonfatal(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project(repo_path=self.REPO)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve reliability",
            goal="Fix a meaningful reliability problem.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="hidden-candidate",
            cwd="/candidate",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Harden retries",
            candidate_session=candidate,
            outcome_metadata={
                "autonomous_goal_tool_protocol": True,
                "autonomous_goal_autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            },
        )
        mock_cleanup.side_effect = WorktreeCleanupError("busy")

        with self.assertLogs("hitch.main.views.common", level="ERROR") as logs:
            new_session_views._cleanup_hidden_ag_candidate_worktree(proposal)

        mock_cleanup.assert_called_once_with(candidate.cwd)
        self.assertIn("failed to clean up hidden candidate worktree", logs.output[0])

    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    def test_ag_tool_proposal_requires_approved_snapshot(
        self,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO, auto_pull_enabled=False)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve reliability",
            goal="Fix a meaningful reliability problem.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Harden retries",
            outcome_metadata={
                "autonomous_goal_tool_protocol": True,
                "autonomous_goal_autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            },
        )
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Continue the approved work.",
                "project": str(project.pk),
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertContains(
            response,
            "approved autonomous-goal snapshot is missing or invalid",
            status_code=400,
        )
        mock_create_worktree.assert_not_called()
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    def test_ag_snapshot_worktree_creation_failure_is_reported(
        self,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        project = _make_project(repo_path=self.REPO, auto_pull_enabled=False)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve reliability",
            goal="Fix a meaningful reliability problem.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Harden retries",
            outcome_metadata={
                "autonomous_goal_tool_protocol": True,
                "autonomous_goal_autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "approved_snapshot_sha": "a" * 40,
            },
        )
        mock_create_worktree.side_effect = WorktreeCreationError("snapshot missing")
        _setup_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "Continue the approved work.",
                "project": str(project.pk),
                "proposed_session": str(proposal.pk),
            },
        )

        self.assertContains(response, "snapshot missing", status_code=400)
        mock_create_worktree.assert_called_once_with(
            self.REPO, base_ref="a" * 40
        )
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.cleanup_worktree")
    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_cleans_up_managed_worktree_when_upload_validation_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"),
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = worktree
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={
                "prompt": "use this screenshot",
                "cwd": self.REPO,
                "input_images": SimpleUploadedFile("screen.png", b"not an image", content_type="image/png"),
            },
        )

        self.assertContains(
            response,
            "image attachment must be PNG, JPEG, GIF, or WebP",
            status_code=400,
        )
        mock_cleanup.assert_called_once_with(worktree)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.common.cleanup_worktree")
    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_preserves_spawn_error_when_managed_worktree_cleanup_fails(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        worktree = ManagedWorktree(
            path=Path("/home/user/.hitch/worktrees/proj/20260516120000-abcdef12"),
            branch="hitch/proj/20260516120000-abcdef12",
            source_repo=Path(self.REPO),
        )
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.return_value = worktree
        mock_spawn.side_effect = RuntimeError("spawn failed")
        mock_cleanup.side_effect = WorktreeCleanupError("cleanup failed")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        with (
            self.assertLogs("hitch.main.views", level="ERROR") as logs,
            self.assertRaisesRegex(RuntimeError, "spawn failed"),
        ):
            self.client.post(
                reverse("new_session"),
                data={"prompt": "do thing", "cwd": self.REPO},
            )

        mock_cleanup.assert_called_once_with(worktree)
        self.assertIn("failed to clean up managed worktree", logs.output[0])

    @patch("hitch.main.views.common.create_worktree_for_session")
    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_reports_worktree_creation_failure(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
        mock_create_worktree: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self.REPO)]
        mock_create_worktree.side_effect = WorktreeCreationError("boom")
        _setup_codex(mock_codex, models=[])
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self.REPO},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"boom")
        mock_spawn.assert_not_called()

    @patch("hitch.main.runtime.codex_pool.spawn_new_session")
    @patch("hitch.main.repos.discover_repos")
    def test_rejects_invalid_input(self, mock_discover: MagicMock, mock_spawn: MagicMock) -> None:
        mock_discover.return_value = [Path(self.REPO)]

        cases: list[tuple[dict[str, str], str]] = [
            ({"prompt": "", "cwd": self.REPO}, "empty prompt"),
            ({"prompt": "hello", "cwd": ""}, "missing cwd"),
            ({"prompt": "hello", "cwd": "/etc"}, "cwd outside allowed list"),
        ]
        for data, label in cases:
            with self.subTest(label=label):
                mock_spawn.reset_mock()
                response = self.client.post(reverse("new_session"), data=data)
                self.assertEqual(response.status_code, 400)
                mock_spawn.assert_not_called()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_renders_get(self, mock_codex: MagicMock, mock_discover: MagicMock) -> None:
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        response = self.client.get(reverse("new_session"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Start a new session")
        self.assertContains(response, 'class="new-session-form"')
        self.assertContains(response, 'class="new-session-close"')
        self.assertContains(response, 'aria-label="Cancel new session"')
        self.assertContains(response, ">Cancel</a>", count=1)
