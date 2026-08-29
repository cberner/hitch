import base64
import re
import subprocess
from types import SimpleNamespace
from typing import cast, override
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from openai_codex.errors import MethodNotFoundError
from openai_codex.generated.v2_all import ReasoningEffort

from hitch.main import caches, context_processors
from hitch.main.models import GlobalSettings, UserSettings
from hitch.main.sessions import settings_cookies
from hitch.main.test.support import (
    _cookie_value,
    _make_project,
    _seed_cookies,
)
from hitch.main.workflows import pr_stage
from hitch.settings import common as common_settings

_MODEL_COOKIE = "hitch_model"
_EFFORT_COOKIE = "hitch_reasoning_effort"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_AUTO_PR_COOKIE = "hitch_auto_pr"
_AUTO_QA_COOKIE = "hitch_auto_qa"
_WEB_SEARCH_COOKIE = "hitch_web_search_mode"
_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_SELECTED_PROJECT_COOKIE = "hitch_selected_project_id"
_ENABLE_MEMORIES_COOKIE = "hitch_enable_memories"

# By default a test model accepts every enum value so tests that don't care
# about supported-effort filtering can stay terse; tests that exercise the
# narrowing logic pass an explicit ``supported_efforts``.
_ALL_EFFORT_VALUES = [e.value for e in ReasoningEffort]


class DatabaseSettingsTests(SimpleTestCase):
    def test_sqlite_waits_for_brief_writer_contention(self) -> None:
        database = common_settings.DATABASES["default"]
        options = cast(dict[str, object], database["OPTIONS"])

        self.assertEqual(options["timeout"], 60)
        self.assertEqual(options["transaction_mode"], "IMMEDIATE")
        init_command = str(options["init_command"])
        self.assertIn("PRAGMA journal_mode=WAL", init_command)
        self.assertIn("PRAGMA synchronous=NORMAL", init_command)
        self.assertIn("PRAGMA busy_timeout=60000", init_command)
        # Larger cache + mmap keep write transactions short so contended
        # writers release the single WAL write lock sooner.
        self.assertIn("PRAGMA cache_size=-65536", init_command)
        self.assertIn("PRAGMA mmap_size=268435456", init_command)
        # Django runs each ';'-separated statement on its own, so a stray
        # empty fragment would silently no-op; guard the list stays clean.
        statements = [stmt.strip() for stmt in init_command.split(";")]
        self.assertTrue(all(statements), init_command)


class StageCacheLockToleranceTests(SimpleTestCase):
    def test_best_effort_stage_cache_swallows_locked_database(self) -> None:
        """The session-detail render persists the derived-stage cache; a
        contended write lock must be skipped rather than 500 the page."""
        from django.db import OperationalError

        with patch(
            "hitch.main.workflows.pr_stage._update_cached_stage",
            side_effect=OperationalError("database is locked"),
        ):
            # Must not raise: a locked cache write is skipped, not surfaced.
            pr_stage._update_cached_stage_best_effort("thread-1", MagicMock(), 123)


def _model(
    model_id: str,
    *,
    is_default: bool = False,
    default_effort: str = "medium",
    display_name: str | None = None,
    supported_efforts: list[str] | None = None,
) -> SimpleNamespace:
    """Minimal Model stand-in shaped like ``openai_codex...Model``.

    Only the fields the reconcile helper, the template, and the
    update-settings validator touch are populated so a future regression
    that starts reading new fields surfaces as a clean ``AttributeError``
    rather than picking up silent MagicMock defaults.
    """
    efforts = supported_efforts if supported_efforts is not None else _ALL_EFFORT_VALUES
    return SimpleNamespace(
        id=model_id,
        display_name=display_name or model_id,
        is_default=is_default,
        default_reasoning_effort=SimpleNamespace(value=default_effort),
        supported_reasoning_efforts=[
            SimpleNamespace(reasoning_effort=SimpleNamespace(value=e), description=e)
            for e in efforts
        ],
    )


def _clear_models_cache() -> None:
    with caches._MODELS_REFRESH_LOCK:
        caches._MODELS_CACHE_VALUE = {}
        caches._MODELS_CACHE_FETCHED_AT = {}
        caches._MODELS_REFRESH_IN_FLIGHT = set()


def _clear_rate_limits_cache() -> None:
    with caches._RATE_LIMITS_REFRESH_LOCK:
        caches._RATE_LIMITS_CACHE_VALUE = None
        caches._RATE_LIMITS_CACHE_HAS_VALUE = False
        caches._RATE_LIMITS_CACHE_FETCHED_AT = None
        caches._RATE_LIMITS_REFRESH_ATTEMPTED_AT = None
        caches._RATE_LIMITS_REFRESH_IN_FLIGHT = False


def _seed_models_cache(
    models: list[SimpleNamespace], *, enable_memories: bool = False
) -> None:
    with caches._MODELS_REFRESH_LOCK:
        caches._MODELS_CACHE_VALUE[enable_memories] = list(models)
        caches._MODELS_CACHE_FETCHED_AT[enable_memories] = timezone.now()
        caches._MODELS_REFRESH_IN_FLIGHT.discard(enable_memories)


def _configure_codex(
    mock_codex: MagicMock,
    *,
    models: list[SimpleNamespace],
    threads: list[SimpleNamespace] | None = None,
    rate_limits: SimpleNamespace | BaseException | None = None,
) -> None:
    ctx = mock_codex.return_value.__enter__.return_value
    ctx.thread_list.return_value.data = threads or []
    ctx.models.return_value.data = models
    # The rate-limits endpoint is a raw JSON-RPC request, not a typed
    # client method. By default the helper raises MethodNotFound so the
    # view's fallback path is exercised; tests that care set an explicit
    # snapshot or pass an exception instance.
    if isinstance(rate_limits, BaseException):
        ctx._client.request.side_effect = rate_limits
    elif rate_limits is not None:
        ctx._client.request.return_value = SimpleNamespace(rate_limits=rate_limits)
    else:
        ctx._client.request.side_effect = MethodNotFoundError(
            -32601, "method not found", None
        )


def _rate_limit_snapshot(
    *,
    primary_used: int | None = None,
    secondary_used: int | None = None,
    primary_resets_at: int | None = None,
    secondary_resets_at: int | None = None,
    primary_window_mins: int | None = None,
    secondary_window_mins: int | None = None,
    limit_name: str | None = None,
    plan_type: str | None = None,
) -> SimpleNamespace:
    primary = (
        SimpleNamespace(
            used_percent=primary_used,
            resets_at=primary_resets_at,
            window_duration_mins=primary_window_mins,
        )
        if primary_used is not None
        else None
    )
    secondary = (
        SimpleNamespace(
            used_percent=secondary_used,
            resets_at=secondary_resets_at,
            window_duration_mins=secondary_window_mins,
        )
        if secondary_used is not None
        else None
    )
    return SimpleNamespace(
        primary=primary,
        secondary=secondary,
        limit_name=limit_name,
        plan_type=SimpleNamespace(value=plan_type) if plan_type else None,
    )


def _input_tag_containing(html: str, marker: str) -> str:
    marker_pos = html.index(marker)
    start = html.rfind("<input", 0, marker_pos)
    end = html.index(">", marker_pos) + 1
    return html[start:end]


def _extra_system_prompt_value(response: object) -> str:
    raw = _cookie_value(response, _EXTRA_SYSTEM_PROMPT_COOKIE)
    return base64.urlsafe_b64decode(raw.encode("ascii")).decode()


class ServerRevisionContextTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        context_processors.server_git_hash.cache_clear()
        super().setUp()

    @override
    def tearDown(self) -> None:
        context_processors.server_git_hash.cache_clear()
        super().tearDown()

    @override_settings(BASE_DIR="/srv/hitch")
    @patch("hitch.main.context_processors.subprocess.run")
    def test_server_revision_exposes_short_git_hash(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="abc123\n",
        )

        context = context_processors.server_revision(MagicMock())

        self.assertEqual(context, {"server_git_hash": "abc123"})
        mock_run.assert_called_once_with(
            ["git", "-C", "/srv/hitch", "rev-parse", "--short=6", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )

    @patch("hitch.main.context_processors.subprocess.run")
    def test_server_revision_is_blank_when_git_lookup_fails(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(128, "git")

        context = context_processors.server_revision(MagicMock())

        self.assertEqual(context, {"server_git_hash": ""})


class UsageModelCacheTests(SimpleTestCase):
    @override
    def tearDown(self) -> None:
        with caches._MODELS_REFRESH_LOCK:
            caches._MODELS_CACHE_VALUE = {}
            caches._MODELS_CACHE_FETCHED_AT = {}
            caches._MODELS_REFRESH_IN_FLIGHT = set()
        super().tearDown()

    def test_failed_initial_model_refresh_remains_retryable(self) -> None:
        with caches._MODELS_REFRESH_LOCK:
            caches._MODELS_CACHE_VALUE = {}
            caches._MODELS_CACHE_FETCHED_AT = {}
            caches._MODELS_REFRESH_IN_FLIGHT = {False}

        with (
            patch("hitch.main.runtime.codex_pool.app_server_config", return_value=object()),
            patch("hitch.main.caches.Codex", side_effect=RuntimeError("codex down")),
            patch("hitch.main.caches.logger.exception") as log_exception,
        ):
            caches._refresh_models_cache_best_effort(enable_memories=False)

        log_exception.assert_called_once()
        with caches._MODELS_REFRESH_LOCK:
            self.assertEqual(caches._MODELS_CACHE_VALUE, {})
            self.assertEqual(caches._MODELS_CACHE_FETCHED_AT, {})
            self.assertNotIn(False, caches._MODELS_REFRESH_IN_FLIGHT)
        self.assertTrue(caches._models_refresh_needed(enable_memories=False))

    @patch("hitch.main.runtime.codex_pool.app_server_config", return_value=object())
    @patch("hitch.main.caches.Codex")
    def test_successful_empty_model_refresh_marks_cache_fresh(
        self, mock_codex: MagicMock, _mock_config: MagicMock
    ) -> None:
        ctx = mock_codex.return_value.__enter__.return_value
        ctx.models.return_value.data = []

        caches._refresh_models_cache_best_effort(enable_memories=False)

        self.assertEqual(caches._cached_models_data(enable_memories=False), [])
        with caches._MODELS_REFRESH_LOCK:
            self.assertIn(False, caches._MODELS_CACHE_FETCHED_AT)
            self.assertNotIn(False, caches._MODELS_REFRESH_IN_FLIGHT)
        self.assertFalse(caches._models_refresh_needed(enable_memories=False))


class UsageRateLimitCacheTests(SimpleTestCase):
    @override
    def tearDown(self) -> None:
        with caches._RATE_LIMITS_REFRESH_LOCK:
            caches._RATE_LIMITS_CACHE_VALUE = None
            caches._RATE_LIMITS_CACHE_HAS_VALUE = False
            caches._RATE_LIMITS_CACHE_FETCHED_AT = None
            caches._RATE_LIMITS_REFRESH_ATTEMPTED_AT = None
            caches._RATE_LIMITS_REFRESH_IN_FLIGHT = False
        super().tearDown()

    def test_failed_cold_refresh_becomes_terminal_until_retry_ttl(self) -> None:
        with caches._RATE_LIMITS_REFRESH_LOCK:
            caches._RATE_LIMITS_REFRESH_IN_FLIGHT = True

        with (
            patch("hitch.main.caches.rate_limit.claim", return_value=True),
            patch(
                "hitch.main.caches.app_server_pool.borrow_codex",
                side_effect=RuntimeError("codex unavailable"),
            ),
            self.assertLogs("hitch.main.caches", level="ERROR"),
        ):
            caches._refresh_rate_limits_cache_best_effort(enable_memories=False)

        state = caches._rate_limits_for_usage_context(enable_memories=False)

        self.assertIsNone(state.rate_limits)
        self.assertFalse(state.refresh_pending)
        self.assertFalse(caches._rate_limits_refresh_needed())
        with caches._RATE_LIMITS_REFRESH_LOCK:
            self.assertIsNotNone(caches._RATE_LIMITS_REFRESH_ATTEMPTED_AT)

    def test_warm_rate_limit_thread_start_failure_backs_off(self) -> None:
        snapshot = {
            "windows": [],
            "limit_name": "Codex",
            "plan_type": "pro",
        }
        fetched_at = timezone.now() - caches._RATE_LIMITS_CACHE_TTL
        with caches._RATE_LIMITS_REFRESH_LOCK:
            caches._RATE_LIMITS_CACHE_VALUE = snapshot
            caches._RATE_LIMITS_CACHE_HAS_VALUE = True
            caches._RATE_LIMITS_CACHE_FETCHED_AT = fetched_at

        with (
            patch(
                "hitch.main.caches.threading.Thread",
                side_effect=RuntimeError("thread limit"),
            ),
            self.assertLogs("hitch.main.caches", level="ERROR"),
        ):
            caches._start_rate_limits_refresh_thread(enable_memories=False)

        self.assertEqual(caches._cached_rate_limits(), snapshot)
        with caches._RATE_LIMITS_REFRESH_LOCK:
            attempted_at = caches._RATE_LIMITS_REFRESH_ATTEMPTED_AT
        self.assertIsNotNone(attempted_at)
        assert attempted_at is not None
        self.assertGreater(attempted_at, fetched_at)
        self.assertFalse(caches._rate_limits_refresh_needed())
        with patch("hitch.main.caches.threading.Thread") as retry_thread:
            caches._start_rate_limits_refresh_thread(enable_memories=False)
        retry_thread.assert_not_called()


class SettingsPageRenderTests(TestCase):
    @override
    def setUp(self) -> None:
        _clear_models_cache()
        super().setUp()

    @override
    def tearDown(self) -> None:
        _clear_models_cache()
        super().tearDown()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_page_rounds_disk_usage_percent_to_input_step(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        GlobalSettings.objects.create(
            pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=35.55
        )
        _seed_models_cache([_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_not_called()
        self.assertContains(response, 'name="disk_usage_max_percent"')
        self.assertContains(response, 'value="35.5"')
        self.assertNotContains(response, 'value="35.55"')

    @staticmethod
    def _effort_option(body: str, value: str) -> str:
        match = re.search(rf'<option value="{re.escape(value)}"[^>]*>', body)
        assert match is not None, f"effort option {value!r} missing from settings page"
        return match.group(0)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_saved_archived_session_visibility_renders_index_toggle_checked(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(
            response, 'name="show_archived_sessions" value="true" checked'
        )
        body = response.content.decode()
        self.assertLess(
            body.index("Show archived"),
            body.index("No sessions found."),
        )

    @patch("hitch.main.caches.Codex")
    def test_usage_page_hides_rate_limits_when_unsupported(
        self, mock_codex: MagicMock
    ) -> None:
        """Local-dev (ollama) and older Codex builds reject the rate-limits
        method; the usage page must still render with an empty state."""
        _clear_rate_limits_cache()
        self.addCleanup(_clear_rate_limits_cache)
        _configure_codex(
            mock_codex,
            models=[],
            # explicit MethodNotFound is the typical signal from Codex when
            # the endpoint isn't wired in the current build.
            rate_limits=MethodNotFoundError(-32601, "method not found", None),
        )
        caches._refresh_rate_limits_cache_best_effort(enable_memories=False)

        with (
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-labelledby="quota-title"')
        self.assertContains(response, 'id="quota-title">Rate limits</h2>')
        self.assertContains(response, "Usage unavailable.")
        self.assertNotContains(response, "% remaining")

    @patch("hitch.main.caches.Codex")
    def test_usage_page_hides_rate_limits_on_unexpected_exception(
        self, mock_codex: MagicMock
    ) -> None:
        """Non-Codex exceptions (pydantic ValidationError on a malformed
        wire payload, transport hiccups not wrapped as CodexError) must
        also be swallowed so usage can show an empty state."""
        _clear_rate_limits_cache()
        self.addCleanup(_clear_rate_limits_cache)
        _configure_codex(
            mock_codex,
            models=[],
            rate_limits=ValueError("malformed payload"),
        )
        caches._refresh_rate_limits_cache_best_effort(enable_memories=False)

        with (
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-labelledby="quota-title"')
        self.assertContains(response, 'id="quota-title">Rate limits</h2>')
        self.assertContains(response, "Usage unavailable.")
        self.assertNotContains(response, "% remaining")

    @patch("hitch.main.caches.Codex")
    def test_usage_page_hides_rate_limits_when_both_windows_empty(
        self, mock_codex: MagicMock
    ) -> None:
        """An account that has no metered usage at all returns a snapshot
        with both windows unset; show the empty state under a valid heading."""
        _clear_rate_limits_cache()
        self.addCleanup(_clear_rate_limits_cache)
        _configure_codex(
            mock_codex,
            models=[],
            rate_limits=_rate_limit_snapshot(),
        )
        caches._refresh_rate_limits_cache_best_effort(enable_memories=False)

        with (
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-labelledby="quota-title"')
        self.assertContains(response, 'id="quota-title">Rate limits</h2>')
        self.assertContains(response, "Usage unavailable.")
        self.assertNotContains(response, "% remaining")


class ReconcileSettingsTests(TestCase):
    @override
    def setUp(self) -> None:
        _clear_models_cache()
        super().setUp()

    @override
    def tearDown(self) -> None:
        _clear_models_cache()
        super().tearDown()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_does_not_reset_when_codex_returns_no_models(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # No models from Codex means we have nothing to reconcile against;
        # leave the user's existing pick alone rather than blow it away.
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "previously-chosen", _EFFORT_COOKIE: "high"},
        )
        _configure_codex(mock_codex, models=[])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertNotIn(_MODEL_COOKIE, response.cookies)
        self.assertNotIn(_EFFORT_COOKIE, response.cookies)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_tampered_cookie_is_treated_as_missing(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A cookie whose signature doesn't verify (manual edit, key
        rotation) must not 500 the page; we treat it as absent and let
        reconcile reseed from Codex defaults."""
        self.client.cookies[_MODEL_COOKIE] = "not-a-signed-value"
        _seed_models_cache(
            [_model("gpt-5", is_default=True, default_effort="medium")]
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_not_called()
        self.assertContains(response, 'value="" selected')
        self.assertContains(response, 'value="high" selected')
        self.assertContains(response, 'value="gpt-5"')
        self.assertNotIn(_MODEL_COOKIE, response.cookies)

    def test_unexpected_signed_cookie_errors_are_not_hidden(self) -> None:
        request = MagicMock()
        request.COOKIES = {_MODEL_COOKIE: "signed"}
        request.get_signed_cookie.side_effect = RuntimeError("cookie backend failed")

        with self.assertRaisesRegex(RuntimeError, "cookie backend failed"):
            settings_cookies._valid_cookie_setting_updates(request)

        with self.assertRaisesRegex(RuntimeError, "cookie backend failed"):
            settings_cookies._read_cookie(request, _MODEL_COOKIE)


class UpdateSettingsViewTests(TestCase):
    @override
    def setUp(self) -> None:
        _clear_models_cache()
        super().setUp()

    @override
    def tearDown(self) -> None:
        _clear_models_cache()
        super().tearDown()

    @staticmethod
    def _effort_option(body: str, value: str) -> str:
        match = re.search(rf'<option value="{re.escape(value)}"[^>]*>', body)
        assert match is not None, f"effort option {value!r} missing from settings page"
        return match.group(0)

    @patch("hitch.main.sessions.session_settings.caches._schedule_models_refresh")
    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_cold_cache_guest_default_reconciles_to_supported_effort(
        self,
        mock_codex: MagicMock,
        _mock_discover: MagicMock,
        _mock_schedule: MagicMock,
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "gpt-current",
                    is_default=True,
                    default_effort="medium",
                    supported_efforts=["low", "medium"],
                )
            ],
        )

        get_response = self.client.get(reverse("update_settings"))
        post_response = self.client.post(
            reverse("update_settings"),
            data={"model": "", "reasoning_effort": "high"},
        )

        self.assertContains(get_response, 'value="high" selected')
        self.assertEqual(post_response.status_code, 302)
        self.assertEqual(_cookie_value(post_response, _MODEL_COOKIE), "gpt-current")
        self.assertEqual(_cookie_value(post_response, _EFFORT_COOKIE), "medium")

    @patch("hitch.main.sessions.session_settings.caches._schedule_models_refresh")
    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_unrelated_save_preserves_account_choice_missing_from_cache(
        self,
        mock_codex: MagicMock,
        _mock_discover: MagicMock,
        _mock_schedule: MagicMock,
    ) -> None:
        user = get_user_model().objects.create_user("dev@example.com")
        UserSettings.objects.create(
            user=user,
            model="still-live-model",
            reasoning_effort="high",
        )
        self.client.force_login(user)
        _seed_models_cache(
            [
                _model(
                    "cached-default",
                    is_default=True,
                    display_name="Cached Default",
                    default_effort="medium",
                )
            ]
        )
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "still-live-model",
                    is_default=True,
                    supported_efforts=["medium", "high"],
                )
            ],
        )

        get_response = self.client.get(reverse("update_settings"))
        post_response = self.client.post(
            reverse("update_settings"),
            data={
                "model": "still-live-model",
                "reasoning_effort": "high",
                "use_worktrees": "true",
            },
        )

        self.assertEqual(get_response.status_code, 200)
        self.assertContains(get_response, "Cached Default")
        self.assertContains(get_response, 'value="still-live-model" selected')
        self.assertEqual(post_response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.model, "still-live-model")
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertTrue(settings.use_worktrees)
        self.assertNotIn(_MODEL_COOKIE, get_response.cookies)
        self.assertNotIn(_EFFORT_COOKIE, get_response.cookies)

    @patch("hitch.main.sessions.session_settings.caches._schedule_models_refresh")
    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_unrelated_save_preserves_effort_missing_from_cached_model(
        self,
        mock_codex: MagicMock,
        _mock_discover: MagicMock,
        _mock_schedule: MagicMock,
    ) -> None:
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "saved-model", _EFFORT_COOKIE: "high"},
        )
        _seed_models_cache(
            [
                _model(
                    "saved-model",
                    is_default=True,
                    supported_efforts=["low", "medium"],
                )
            ]
        )
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "saved-model",
                    is_default=True,
                    supported_efforts=["medium", "high"],
                )
            ],
        )

        get_response = self.client.get(reverse("update_settings"))
        post_response = self.client.post(
            reverse("update_settings"),
            data={
                "model": "saved-model",
                "reasoning_effort": "high",
                "use_worktrees": "true",
            },
        )

        self.assertContains(get_response, 'value="saved-model" selected')
        self.assertContains(get_response, 'data-supported-efforts="high low medium"')
        self.assertContains(get_response, 'value="high" selected')
        self.assertEqual(post_response.status_code, 302)
        self.assertEqual(_cookie_value(post_response, _MODEL_COOKIE), "saved-model")
        self.assertEqual(_cookie_value(post_response, _EFFORT_COOKIE), "high")

    @patch("hitch.main.views.common.Codex")
    def test_saves_provider_advertised_effort_unknown_to_sdk_enum(
        self, mock_codex: MagicMock
    ) -> None:
        ctx = mock_codex.return_value.__enter__.return_value
        ctx._client._request_raw.return_value = {
            "data": [
                {
                    "id": "gpt-5.6",
                    "displayName": "GPT-5.6",
                    "isDefault": True,
                    "defaultReasoningEffort": "ultra",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "max", "description": "max"},
                        {"reasoningEffort": "ultra", "description": "ultra"},
                    ],
                }
            ]
        }
        ctx.models.side_effect = AssertionError("typed model list should not be used")

        response = self.client.post(
            reverse("update_settings"),
            data={"model": "gpt-5.6", "reasoning_effort": "ultra"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5.6")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "ultra")

    @patch("hitch.main.sessions.session_settings.caches._schedule_models_refresh")
    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_unrelated_save_reconciles_unchanged_narrowed_effort(
        self,
        mock_codex: MagicMock,
        _mock_discover: MagicMock,
        _mock_schedule: MagicMock,
    ) -> None:
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "current-model", _EFFORT_COOKIE: "high"},
        )
        _seed_models_cache(
            [_model("current-model", is_default=True, supported_efforts=["high"])]
        )
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "current-model",
                    is_default=True,
                    default_effort="medium",
                    supported_efforts=["low", "medium"],
                )
            ],
        )

        post_response = self.client.post(
            reverse("update_settings"),
            data={
                "model": "current-model",
                "reasoning_effort": "high",
                "use_worktrees": "true",
            },
        )

        self.assertEqual(post_response.status_code, 302)
        self.assertEqual(_cookie_value(post_response, _MODEL_COOKIE), "current-model")
        self.assertEqual(_cookie_value(post_response, _EFFORT_COOKIE), "medium")

    @patch("hitch.main.sessions.session_settings.caches._schedule_models_refresh")
    @patch("hitch.main.repos.discover_repos", return_value=[])
    @patch("hitch.main.views.common.Codex")
    def test_unrelated_save_preserves_provider_only_effort_when_fetch_fails(
        self,
        mock_codex: MagicMock,
        _mock_discover: MagicMock,
        _mock_schedule: MagicMock,
    ) -> None:
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "saved-model", _EFFORT_COOKIE: "ultra"},
        )
        _seed_models_cache(
            [_model("cached-default", is_default=True, supported_efforts=["medium"])]
        )
        ctx = mock_codex.return_value.__enter__.return_value
        ctx._client._request_raw.side_effect = ValueError("bad raw model list")
        ctx.models.side_effect = ValueError("bad typed model list")

        get_response = self.client.get(reverse("update_settings"))
        with patch("hitch.main.views.settings.logger.exception") as log_exception:
            post_response = self.client.post(
                reverse("update_settings"),
                data={
                    "model": "saved-model",
                    "reasoning_effort": "ultra",
                    "use_worktrees": "true",
                },
            )

        self.assertContains(get_response, 'value="saved-model" selected')
        self.assertEqual(post_response.status_code, 302)
        self.assertEqual(_cookie_value(post_response, _MODEL_COOKIE), "saved-model")
        self.assertEqual(_cookie_value(post_response, _EFFORT_COOKIE), "ultra")
        self.assertEqual(_cookie_value(post_response, _USE_WORKTREES_COOKIE), "true")
        log_exception.assert_called_once()

    @patch("hitch.main.views.common.Codex")
    def test_rejects_invalid_combinations(self, mock_codex: MagicMock) -> None:
        """Validator boundary cases: bad effort enum, oversized model value
        (would exceed the 4KB browser cookie cap and silently drop), unknown
        model id (would 500 ``thread_start``), effort not supported by the
        chosen model, and — when no model is posted — effort not supported
        by the default model the spawn helper will fall back to. All must
        return 400 without writing a cookie, so a previously-saved cookie
        survives."""
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "gpt-5",
                    is_default=True,
                    default_effort="medium",
                    supported_efforts=["low", "medium"],
                ),
                _model("other"),
            ],
        )

        cases = [
            ({"model": "gpt-5", "reasoning_effort": "ludicrous"}, "bad effort enum"),
            ({"model": "x" * 1024, "reasoning_effort": "medium"}, "oversized model"),
            ({"model": "phantom-model", "reasoning_effort": "medium"}, "unknown model"),
            ({"model": "gpt-5", "reasoning_effort": "xhigh"}, "effort unsupported by model"),
            ({"model": "", "reasoning_effort": "xhigh"}, "effort unsupported by default"),
        ]
        for data, label in cases:
            with self.subTest(label=label):
                _seed_cookies(self.client, **{_EFFORT_COOKIE: "low"})
                response = self.client.post(reverse("update_settings"), data=data)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn(_MODEL_COOKIE, response.cookies)
                self.assertNotIn(_EFFORT_COOKIE, response.cookies)

    def test_rejects_unsafe_next_url(self) -> None:
        response = self.client.post(
            reverse("update_settings"), data={"next": "https://example.invalid/"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("index"))

    def test_rejects_oversized_extra_system_prompt(self) -> None:
        _seed_cookies(
            self.client,
            **{_EXTRA_SYSTEM_PROMPT_COOKIE: "previous value"},
        )

        response = self.client.post(
            reverse("update_settings"),
            data={
                "model": "",
                "reasoning_effort": "",
                "extra_system_prompt": "x" * 2501,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertNotIn(_EXTRA_SYSTEM_PROMPT_COOKIE, response.cookies)

    def test_rejects_multibyte_prompt_that_would_overflow_the_cookie(self) -> None:
        """A prompt is stored as base64-of-UTF-8 inside a signed cookie. The
        2500-*character* cap doesn't bound that byte size, so a multibyte
        prompt well under the character limit can still produce a cookie past
        the browser's ~4KB per-cookie ceiling. The browser then silently drops
        the cookie, losing the setting even though the POST "succeeded" with a
        302. Such a prompt must be rejected up front rather than written out."""
        _seed_cookies(
            self.client, **{_EXTRA_SYSTEM_PROMPT_COOKIE: "previous value"}
        )
        # Hiragana costs 3 UTF-8 bytes/char: 2400 chars is under the 2500
        # character cap but base64-encodes to a ~9.6KB cookie.
        prompt = "あ" * 2400
        self.assertLessEqual(len(prompt), settings_cookies._EXTRA_SYSTEM_PROMPT_MAX_LEN)

        response = self.client.post(
            reverse("update_settings"),
            data={"model": "", "reasoning_effort": "", "extra_system_prompt": prompt},
        )

        self.assertEqual(response.status_code, 400)
        # No oversized cookie is written, so the prior value survives instead
        # of being clobbered by a value the browser would have discarded.
        self.assertNotIn(_EXTRA_SYSTEM_PROMPT_COOKIE, response.cookies)

    def test_rejects_unknown_optional_setting_values(self) -> None:
        cases: list[tuple[str, str, str, dict[str, str]]] = [
            (
                "sandbox policy",
                _SANDBOX_COOKIE,
                "readOnly",
                {"model": "", "reasoning_effort": "", "sandbox_policy": "evilMode"},
            ),
            (
                "archived visibility",
                _SHOW_ARCHIVED_COOKIE,
                "true",
                {"show_archived_sessions": "yes"},
            ),
            (
                "worktree setting",
                _USE_WORKTREES_COOKIE,
                "true",
                {"use_worktrees": "yes"},
            ),
            (
                "memories setting",
                _ENABLE_MEMORIES_COOKIE,
                "true",
                {"enable_memories": "yes"},
            ),
            (
                "auto-PR setting",
                _AUTO_PR_COOKIE,
                "true",
                {"auto_pr": "yes"},
            ),
            (
                "auto-QA setting",
                _AUTO_QA_COOKIE,
                "true",
                {"auto_qa": "yes"},
            ),
            (
                "web search setting",
                _WEB_SEARCH_COOKIE,
                "live",
                {"web_search_mode": "yes"},
            ),
            (
                "selected project",
                _SELECTED_PROJECT_COOKIE,
                "1",
                {"selected_project": "999"},
            ),
        ]
        for label, cookie, saved_value, data in cases:
            with self.subTest(label=label):
                client = Client()
                _seed_cookies(client, **{cookie: saved_value})

                response = client.post(reverse("update_settings"), data=data)

                self.assertEqual(response.status_code, 400)
                self.assertNotIn(cookie, response.cookies)

    def test_update_settings_allows_anonymous_disk_usage_global_setting(self) -> None:
        response = self.client.post(
            reverse("update_settings"),
            data={"disk_usage_max_percent": "35.5"},
        )

        self.assertEqual(response.status_code, 302)
        settings = GlobalSettings.objects.get(pk=GlobalSettings.SINGLETON_PK)
        self.assertEqual(settings.disk_usage_max_percent, 35.5)

    def test_staff_update_settings_updates_existing_disk_usage_global_setting(
        self,
    ) -> None:
        user = get_user_model().objects.create_user(
            "admin@example.com", password="StrongPass123!", is_staff=True
        )
        self.client.force_login(user)
        GlobalSettings.objects.create(
            pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=35.5
        )

        response = self.client.post(
            reverse("update_settings"),
            data={"disk_usage_max_percent": "42.3"},
        )

        self.assertEqual(response.status_code, 302)
        settings = GlobalSettings.objects.get(pk=GlobalSettings.SINGLETON_PK)
        self.assertEqual(settings.disk_usage_max_percent, 42.3)

    def test_staff_update_settings_skips_unchanged_initial_disk_usage_global_setting(
        self,
    ) -> None:
        user = get_user_model().objects.create_user(
            "admin@example.com", password="StrongPass123!", is_staff=True
        )
        self.client.force_login(user)
        GlobalSettings.objects.create(
            pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=42.3
        )

        response = self.client.post(
            reverse("update_settings"),
            data={
                "disk_usage_max_percent": "35.5",
                "initial_disk_usage_max_percent": "35.5",
            },
        )

        self.assertEqual(response.status_code, 302)
        settings = GlobalSettings.objects.get(pk=GlobalSettings.SINGLETON_PK)
        self.assertEqual(settings.disk_usage_max_percent, 42.3)

    def test_staff_update_settings_rejects_invalid_disk_usage_global_setting(
        self,
    ) -> None:
        user = get_user_model().objects.create_user(
            "admin@example.com", password="StrongPass123!", is_staff=True
        )
        self.client.force_login(user)
        cases = ["", "0", "0.05", "35.55", "100.1", "nan", "not-a-number"]
        for raw in cases:
            with self.subTest(raw=raw):
                response = self.client.post(
                    reverse("update_settings"),
                    data={"disk_usage_max_percent": raw},
                )

                self.assertEqual(response.status_code, 400)
                self.assertFalse(GlobalSettings.objects.exists())


class AuthenticatedWebSearchSettingsTests(TestCase):
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_account_web_search_setting_renders_selected(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user(
            "dev@example.com", password="StrongPass123!"
        )
        UserSettings.objects.create(user=user, web_search_mode="cached")
        self.client.force_login(user)
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'value="cached" selected')
        self.assertNotIn(_WEB_SEARCH_COOKIE, response.cookies)

    def test_login_imports_web_search_cookie_to_account(self) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user(
            "dev@example.com", password="StrongPass123!"
        )
        UserSettings.objects.create(user=user, web_search_mode="disabled")
        _seed_cookies(self.client, **{_WEB_SEARCH_COOKIE: "live"})

        response = self.client.post(
            reverse("login"),
            data={"username": "dev@example.com", "password": "StrongPass123!"},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.web_search_mode, "live")
        self.assertEqual(_cookie_value(response, _WEB_SEARCH_COOKIE), "live")


class UpdateArchivedSessionVisibilityViewTests(TestCase):
    def test_saves_visibility_to_signed_cookie(self) -> None:
        cases = [
            ({"show_archived_sessions": "true"}, "true"),
            ({}, "false"),
        ]
        for data, expected in cases:
            with self.subTest(expected=expected):
                response = self.client.post(
                    reverse("update_archived_session_visibility"), data=data
                )

                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.headers["Location"], reverse("index"))
                self.assertEqual(
                    _cookie_value(response, _SHOW_ARCHIVED_COOKIE), expected
                )

    def test_rejects_unknown_visibility(self) -> None:
        _seed_cookies(self.client, **{_SHOW_ARCHIVED_COOKIE: "true"})

        response = self.client.post(
            reverse("update_archived_session_visibility"),
            data={"show_archived_sessions": "yes"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertNotIn(_SHOW_ARCHIVED_COOKIE, response.cookies)


class SelectedProjectCookieImportTests(TestCase):
    def test_login_empty_selected_project_cookie_clears_account_project(self) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user("dev@example.com", password="StrongPass123!")
        project = _make_project()
        UserSettings.objects.create(user=user, selected_project=project)
        _seed_cookies(self.client, **{_SELECTED_PROJECT_COOKIE: ""})

        response = self.client.post(
            reverse("login"),
            data={"username": "dev@example.com", "password": "StrongPass123!"},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertIsNone(settings.selected_project)
        self.assertEqual(_cookie_value(response, _SELECTED_PROJECT_COOKIE), "")


class ApprovalModeSettingsTests(TestCase):
    @patch("hitch.main.views.common.Codex")
    def test_saves_approval_mode_to_signed_cookie(
        self, mock_codex: MagicMock
    ) -> None:
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        for mode in ("prompt_user", "deny_all", "approve_all"):
            with self.subTest(mode=mode):
                response = self.client.post(
                    reverse("update_settings"),
                    data={
                        "model": "gpt-5",
                        "reasoning_effort": "high",
                        "approval_mode": mode,
                    },
                )

                self.assertEqual(response.status_code, 302)
                self.assertEqual(_cookie_value(response, _APPROVAL_COOKIE), mode)

    def test_handles_unknown_and_empty_approval_mode(self) -> None:
        """A form post without the approval dropdown (e.g. an older client,
        a hand-crafted POST) must persist the safe default rather than an
        empty value the worker can't interpret. Unknown values are rejected
        without stomping the existing cookie."""
        cases: list[tuple[str, dict[str, str], dict[str, str], int, str | None]] = [
            (
                "unknown",
                {_APPROVAL_COOKIE: "deny_all"},
                {"model": "", "reasoning_effort": "", "approval_mode": "evilMode"},
                400,
                None,
            ),
            (
                "empty",
                {},
                {"model": "", "reasoning_effort": "", "approval_mode": ""},
                302,
                "auto_review",
            ),
        ]
        for label, seed, data, status, expected_cookie in cases:
            with self.subTest(label=label):
                client = Client()
                if seed:
                    _seed_cookies(client, **seed)

                response = client.post(reverse("update_settings"), data=data)

                self.assertEqual(response.status_code, status)
                if expected_cookie is None:
                    self.assertNotIn(_APPROVAL_COOKIE, response.cookies)
                else:
                    self.assertEqual(
                        _cookie_value(response, _APPROVAL_COOKIE),
                        expected_cookie,
                    )


class ApprovalModePageTests(TestCase):
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_saved_approval_renders_as_selected(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A mode persisted in the cookie must come back marked selected
        on the settings page — otherwise the dropdown silently rolls back
        to the safe default and the user assumes the pick was lost."""
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        for mode in ("prompt_user", "deny_all"):
            with self.subTest(mode=mode):
                _seed_cookies(self.client, **{_APPROVAL_COOKIE: mode})

                response = self.client.get(reverse("update_settings"))

                self.assertContains(response, f'value="{mode}" selected')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_unknown_approval_cookie_falls_back_to_safe_default(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A legacy/tampered cookie value must not render as a phantom
        option; the settings page snaps back to the safe default so the user has
        a coherent UI to recover from."""
        _seed_cookies(self.client, **{_APPROVAL_COOKIE: "phantomMode"})
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertNotContains(response, "phantomMode")
        self.assertContains(response, 'value="auto_review" selected')


class SandboxPolicyPageTests(TestCase):
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_saved_sandbox_renders_as_selected(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A policy persisted in the cookie must come back marked selected
        on the settings page — otherwise the dropdown silently rolls back to
        the empty default and the user assumes the pick was lost."""
        _seed_cookies(self.client, **{_SANDBOX_COOKIE: "dangerFullAccess"})
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertContains(response, 'value="dangerFullAccess" selected')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_unknown_sandbox_cookie_falls_back_to_empty(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A legacy/tampered cookie value must not render as a phantom
        selected option; the settings page snaps back to the empty "Codex
        default" state so the user has a coherent UI to recover from."""
        _seed_cookies(self.client, **{_SANDBOX_COOKIE: "phantomPolicy"})
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertNotContains(response, "phantomPolicy")
        self.assertContains(response, 'value="" selected')
