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
from hitch.main.models import GlobalSettings, Project, UserSettings
from hitch.main.sessions import settings_cookies
from hitch.main.test.support import (
    _cookie_value,
    _encode_extra_system_prompt,
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
_SPEC_CRITIC_COOKIE = "hitch_spec_critic"
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

    def test_best_effort_stage_cache_reraises_other_errors(self) -> None:
        from django.db import OperationalError

        with (
            patch(
                "hitch.main.workflows.pr_stage._update_cached_stage",
                side_effect=OperationalError("no such table: main_sessionmetadata"),
            ),
            self.assertRaises(OperationalError),
        ):
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


def _new_session_form_html(response: object) -> str:
    content: bytes = response.content  # type: ignore[attr-defined]
    body = content.decode()
    start = body.index('<form class="new-session-form"')
    end = body.index("</form>", start)
    return body[start:end]


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

    def test_model_cache_freshness_is_keyed_by_memories_mode(self) -> None:
        models = [_model("gpt-5", is_default=True)]
        with caches._MODELS_REFRESH_LOCK:
            caches._MODELS_CACHE_VALUE = {False: models}
            caches._MODELS_CACHE_FETCHED_AT = {False: timezone.now()}
            caches._MODELS_REFRESH_IN_FLIGHT = set()

        self.assertEqual(
            caches._cached_models_data(enable_memories=False),
            models,
        )
        self.assertEqual(caches._cached_models_data(enable_memories=True), [])
        self.assertFalse(caches._models_refresh_needed(enable_memories=False))
        self.assertTrue(caches._models_refresh_needed(enable_memories=True))


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

    def test_usage_state_reads_value_and_pending_from_one_snapshot(self) -> None:
        snapshot = {
            "windows": [],
            "limit_name": "Codex",
            "plan_type": "pro",
        }

        def complete_refresh(*, enable_memories: bool) -> None:
            self.assertFalse(enable_memories)
            with caches._RATE_LIMITS_REFRESH_LOCK:
                caches._RATE_LIMITS_CACHE_VALUE = snapshot
                caches._RATE_LIMITS_CACHE_HAS_VALUE = True
                caches._RATE_LIMITS_REFRESH_IN_FLIGHT = False

        with patch(
            "hitch.main.caches._schedule_rate_limits_refresh",
            side_effect=complete_refresh,
        ):
            state = caches._rate_limits_for_usage_context(enable_memories=False)

        self.assertEqual(state.rate_limits, snapshot)
        self.assertFalse(state.refresh_pending)

    def test_denied_cold_refresh_remains_eligible_to_retry(self) -> None:
        with caches._RATE_LIMITS_REFRESH_LOCK:
            caches._RATE_LIMITS_REFRESH_IN_FLIGHT = True

        with patch("hitch.main.caches.rate_limit.claim", return_value=False):
            caches._refresh_rate_limits_cache_best_effort(enable_memories=False)

        with (
            patch("hitch.main.caches._start_rate_limits_refresh_thread") as start_refresh,
            patch(
                "hitch.main.caches.transaction.on_commit",
                side_effect=lambda callback: callback(),
            ),
        ):
            state = caches._rate_limits_for_usage_context(enable_memories=False)

        self.assertIsNone(state.rate_limits)
        self.assertFalse(state.refresh_pending)
        self.assertTrue(caches._rate_limits_refresh_needed())
        start_refresh.assert_called_once_with(enable_memories=False)
        with caches._RATE_LIMITS_REFRESH_LOCK:
            self.assertIsNone(caches._RATE_LIMITS_REFRESH_ATTEMPTED_AT)

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

    def test_rate_limit_thread_start_failure_clears_pending_and_backs_off(self) -> None:
        with (
            patch(
                "hitch.main.caches.threading.Thread",
                side_effect=RuntimeError("thread limit"),
            ),
            self.assertLogs("hitch.main.caches", level="ERROR"),
        ):
            caches._start_rate_limits_refresh_thread(enable_memories=False)

        with caches._RATE_LIMITS_REFRESH_LOCK:
            self.assertFalse(caches._RATE_LIMITS_REFRESH_IN_FLIGHT)
            self.assertIsNotNone(caches._RATE_LIMITS_REFRESH_ATTEMPTED_AT)
        self.assertFalse(caches._rate_limits_refresh_needed())

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

    def test_empty_rate_limit_refresh_preserves_existing_snapshot(self) -> None:
        snapshot = {
            "windows": [
                {
                    "label": "Primary",
                    "used_percent": 30,
                    "remaining_percent": 70,
                    "resets_at": 1_700_000_000,
                    "window_duration_label": "5-hour",
                }
            ],
            "limit_name": None,
            "plan_type": "plus",
        }
        with caches._RATE_LIMITS_REFRESH_LOCK:
            caches._RATE_LIMITS_CACHE_VALUE = snapshot
            caches._RATE_LIMITS_CACHE_HAS_VALUE = True
            caches._RATE_LIMITS_CACHE_FETCHED_AT = None
            caches._RATE_LIMITS_REFRESH_IN_FLIGHT = True

        with (
            patch("hitch.main.caches.rate_limit.claim", return_value=True),
            patch("hitch.main.runtime.codex_pool.app_server_config", return_value=object()),
            patch("hitch.main.caches.Codex"),
            patch("hitch.main.caches._fetch_rate_limits", return_value=None),
        ):
            caches._refresh_rate_limits_cache_best_effort(enable_memories=False)

        self.assertEqual(caches._cached_rate_limits(), snapshot)
        with caches._RATE_LIMITS_REFRESH_LOCK:
            self.assertTrue(caches._RATE_LIMITS_CACHE_HAS_VALUE)
            self.assertIsNotNone(caches._RATE_LIMITS_CACHE_FETCHED_AT)
            self.assertFalse(caches._RATE_LIMITS_REFRESH_IN_FLIGHT)


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
    @patch("hitch.main.context_processors.server_git_hash", return_value="abc123")
    def test_page_lists_models_and_efforts(
        self, _mock_hash: MagicMock, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(
            reverse("update_settings"), {"next": reverse("usage")}
        )

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_called_once()
        self.assertContains(response, "data-nav-menu")
        self.assertContains(response, "data-nav-menu-open")
        self.assertContains(response, "data-nav-menu-panel")
        self.assertNotContains(response, "data-settings-dialog")
        self.assertContains(response, 'aria-label="Navigation menu"')
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(f'href="{reverse("new_session")}"', nav_html)
        self.assertIn('class="primary-nav-new-session"', nav_html)
        self.assertIn('aria-label="New session"', nav_html)
        self.assertNotIn(">new session<", nav_html)
        self.assertContains(response, f'href="{reverse("index")}"')
        self.assertContains(response, ">sessions<")
        body = response.content.decode()
        primary_nav = body[
            body.index('<div class="primary-nav-panel"') : body.index("</nav>")
        ]
        self.assertNotIn(reverse("system_sessions"), primary_nav)
        self.assertContains(response, ">settings<")
        self.assertContains(
            response, f'href="{reverse("update_settings")}" aria-current="page"'
        )
        self.assertNotIn("Server git hash", primary_nav)
        self.assertNotIn("abc123", primary_nav)
        self.assertNotIn(f'href="{reverse("usage")}"', primary_nav)
        self.assertIn(f'href="{reverse("profile")}"', primary_nav)
        self.assertIn(">anonymous</a>", primary_nav)
        self.assertContains(response, f'action="{reverse("update_settings")}"')
        self.assertContains(response, f'name="next" value="{reverse("usage")}"')
        self.assertContains(response, "GPT-5")
        # Spot-check a couple of reasoning-effort values so a future enum
        # rename can't quietly empty the dropdown.
        self.assertContains(response, 'value="minimal"')
        self.assertContains(response, 'value="high"')
        # The empty option is the only way the settings page surfaces the "let
        # Codex pick the model default" state; without it a user who once
        # set an explicit value has no UI path back to that state.
        self.assertContains(response, 'value=""')
        self.assertContains(response, "Model default")
        # Sandbox policy dropdown: surface the three policy variants plus
        # an empty "let Codex pick" option, so a future rename here can't
        # quietly drop the field from the page.
        self.assertContains(response, 'name="sandbox_policy"')
        self.assertContains(response, 'value="readOnly"')
        self.assertContains(response, 'value="workspaceWrite"')
        self.assertContains(response, 'value="dangerFullAccess"')
        # Approval mode dropdown: ``auto_review`` (safe default) is
        # pre-selected when no cookie has been written; the stricter
        # ``prompt_user``, ``deny_all``, and the rubber-stamp
        # ``approve_all`` (custom non-SDK modes) are opt-in. No empty
        # option — the worker always wants an explicit value.
        self.assertContains(response, 'name="approval_mode"')
        self.assertContains(response, 'value="auto_review" selected')
        self.assertContains(response, 'value="prompt_user"')
        self.assertContains(response, 'value="deny_all"')
        self.assertContains(response, 'value="approve_all"')
        self.assertNotContains(response, 'name="coding_agent"')
        self.assertContains(response, "Extra developer prompt")
        self.assertContains(response, 'name="extra_system_prompt"')
        self.assertContains(response, 'maxlength="2500"')
        self.assertContains(response, 'name="use_worktrees"')
        self.assertContains(response, "Use worktrees")
        self.assertContains(response, 'name="auto_pr"')
        self.assertContains(response, "Auto-PR")
        self.assertContains(response, 'name="auto_qa"')
        self.assertContains(response, "Auto-QA")
        self.assertContains(response, 'name="spec_critic"')
        self.assertContains(response, "Spec Critic preflight")
        self.assertContains(response, 'name="web_search_mode"')
        self.assertContains(response, "Web search")
        self.assertContains(response, 'value="disabled"')
        self.assertContains(response, 'value="cached"')
        self.assertContains(response, 'value="live"')
        self.assertContains(response, "Max Hitch disk usage (%)")
        self.assertContains(response, 'name="initial_disk_usage_max_percent"')
        self.assertContains(response, 'name="disk_usage_max_percent"')
        self.assertContains(response, 'name="selected_project"')
        self.assertContains(response, "All projects")
        self.assertContains(response, "Create project")
        self.assertContains(response, "Edit project")
        self.assertContains(response, "data-project-edit-dialog")
        self.assertContains(response, 'name="auto_pr_mode"')
        self.assertContains(response, "Follow global")
        self.assertContains(response, 'name="auto_pull"')
        self.assertContains(response, "Auto-pull")
        self.assertContains(response, "After the PR monitor sees a GitHub PR merge")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_page_renders_saved_disk_usage_percent(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        GlobalSettings.objects.create(
            pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=35.5
        )
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_called_once()
        self.assertContains(response, "Max Hitch disk usage (%)")
        self.assertContains(response, 'name="initial_disk_usage_max_percent"')
        self.assertContains(response, 'name="disk_usage_max_percent"')
        self.assertContains(response, 'value="35.5"')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_page_rounds_disk_usage_percent_to_input_step(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        GlobalSettings.objects.create(
            pk=GlobalSettings.SINGLETON_PK, disk_usage_max_percent=35.55
        )
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_called_once()
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
    def test_effort_dropdown_only_offers_efforts_the_model_supports(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """The selected model advertises which reasoning efforts it accepts,
        and ``update_settings`` returns a hard 400 for an unsupported (model,
        effort) pair. The dialog must therefore not present the efforts the
        current model rejects as plain, selectable options — otherwise a user
        picks one, hits Save, and the page bounces with a raw 400 instead of
        saving. Unsupported efforts are rendered ``hidden disabled`` so they
        can't be chosen, while every model option carries its supported set so
        the client can re-filter when the model selection changes."""
        _seed_cookies(
            self.client, **{_MODEL_COOKIE: "gpt-5-codex", _EFFORT_COOKIE: "low"}
        )
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "gpt-5-codex",
                    is_default=True,
                    default_effort="medium",
                    supported_efforts=["low", "medium"],
                ),
                _model(
                    "gpt-5",
                    default_effort="high",
                    supported_efforts=["low", "medium", "high", "xhigh"],
                ),
            ]
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_called_once()
        body = response.content.decode()
        # Efforts the selected model accepts stay selectable...
        for supported in ("low", "medium"):
            option = self._effort_option(body, supported)
            self.assertNotIn("hidden", option)
            self.assertNotIn("disabled", option)
        # ...and the ones it rejects are locked out so Save can't 400.
        for unsupported in ("high", "xhigh", "minimal"):
            option = self._effort_option(body, unsupported)
            self.assertIn("hidden", option)
            self.assertIn("disabled", option)
        # Each model exposes its supported efforts so the client can refilter
        # the dropdown the moment the user switches models.
        self.assertContains(response, 'data-supported-efforts="low medium"')
        self.assertContains(response, 'data-supported-efforts="high low medium xhigh"')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_effort_dropdown_offers_all_efforts_when_model_unconstrained(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A model that advertises no supported-effort constraint accepts any
        effort (matching ``_validate_model_and_effort_against_models``), so the dialog
        must keep every effort selectable rather than hiding them all."""
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, supported_efforts=[])],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_called_once()
        body = response.content.decode()
        for effort in (e.value for e in ReasoningEffort):
            option = self._effort_option(body, effort)
            self.assertNotIn("hidden", option)
            self.assertNotIn("disabled", option)

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_renders_primary_nav_menu_instead_of_back_link(
        self, mock_codex: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-nav-menu")
        self.assertContains(response, "data-nav-menu-open")
        self.assertContains(response, "data-nav-menu-panel")
        self.assertNotContains(response, "data-settings-dialog")
        self.assertContains(response, 'aria-label="Navigation menu"')
        self.assertContains(response, f'href="{reverse("index")}"')
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertNotIn(f'href="{reverse("usage")}"', nav_html)
        self.assertIn(f'href="{reverse("profile")}"', nav_html)
        self.assertIn(">anonymous</a>", nav_html)
        self.assertContains(response, ">settings<")
        self.assertContains(response, 'classList.add("primary-nav-js")')
        self.assertNotContains(response, "html:not(.js) .primary-nav-toggle")
        self.assertNotContains(response, 'class="back-link"')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_saved_cookie_settings_render_on_settings_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        cases = {
            "extra system prompt": (
                _EXTRA_SYSTEM_PROMPT_COOKIE,
                _encode_extra_system_prompt("Prefer small diffs."),
                "Prefer small diffs.",
            ),
            "worktrees": (
                _USE_WORKTREES_COOKIE,
                "true",
                'name="use_worktrees" value="true" checked',
            ),
            "auto PR": (
                _AUTO_PR_COOKIE,
                "true",
                'name="auto_pr" value="true" checked',
            ),
            "auto QA": (
                _AUTO_QA_COOKIE,
                "true",
                'name="auto_qa" value="true" checked',
            ),
            "spec critic": (
                _SPEC_CRITIC_COOKIE,
                "true",
                'name="spec_critic" value="true" checked',
            ),
            "web search": (
                _WEB_SEARCH_COOKIE,
                "live",
                'value="live" selected',
            ),
        }

        for name, (cookie, value, expected) in cases.items():
            with self.subTest(name=name):
                self.client.cookies.clear()
                _seed_cookies(self.client, **{cookie: value})

                response = self.client.get(reverse("update_settings"))

                self.assertContains(response, expected)

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

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_saved_auto_qa_setting_checks_new_session_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_AUTO_QA_COOKIE: "true"})
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("new_session"))
        form_html = _new_session_form_html(response)
        auto_qa_input = _input_tag_containing(
            form_html, "data-new-session-auto-qa"
        )

        self.assertIn('name="auto_qa"', auto_qa_input)
        self.assertIn("checked", auto_qa_input)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_auto_pr_takes_precedence_in_new_session_page(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(
            self.client,
            **{_AUTO_PR_COOKIE: "true", _AUTO_QA_COOKIE: "true"},
        )
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("new_session"))
        form_html = _new_session_form_html(response)
        auto_pr_input = _input_tag_containing(
            form_html, "data-new-session-auto-pr"
        )
        auto_qa_input = _input_tag_containing(
            form_html, "data-new-session-auto-qa"
        )

        self.assertIn("checked", auto_pr_input)
        self.assertNotIn("checked", auto_qa_input)

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_project_auto_pr_setting_renders_in_edit_dialog(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project(
            auto_pr_mode=Project.AUTO_PR_OFF,
            auto_pull_enabled=True,
        )
        _seed_cookies(self.client, **{_SELECTED_PROJECT_COOKIE: str(project.pk)})
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertContains(response, 'data-project-auto-pr-mode="off"')
        self.assertContains(response, 'data-project-auto-pull-enabled="true"')
        self.assertContains(response, 'value="off" selected')
        self.assertContains(response, 'checked data-project-edit-auto-pull')
        self.assertContains(response, 'data-project-edit-open')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_empty_effort_option_renders_selected_when_cookie_cleared(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A user who picks "Model default" (empty effort) and reloads must
        see that option as the selected one, otherwise the dropdown lies
        about the persisted state."""
        _seed_cookies(self.client, **{_MODEL_COOKIE: "gpt-5", _EFFORT_COOKIE: ""})
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, default_effort="medium")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertContains(response, 'value="" selected')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_dialog_captures_scroll_without_scrolling_session_list(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertContains(response, "overscroll-behavior: contain")
        self.assertContains(response, "overflow-y: auto")
        self.assertContains(response, "modal-scroll-locked")
        self.assertContains(response, 'document.querySelector("dialog[open]")')

    @patch("hitch.main.views.common.Codex")
    def test_usage_page_renders_rate_limit_windows(self, mock_codex: MagicMock) -> None:
        """When the cached account/rateLimits/read data has a snapshot, the
        usage page must render each present window so a user can see how
        much of their budget is left before kicking off a new turn."""
        rate_limits = {
            "windows": [
                {
                    "label": "Primary",
                    "used_percent": 30,
                    "remaining_percent": 70,
                    "resets_at": 1_700_000_000,
                    "window_duration_label": "5-hour",
                },
                {
                    "label": "Secondary",
                    "used_percent": 80,
                    "remaining_percent": 20,
                    "resets_at": 1_700_010_000,
                    "window_duration_label": "7-day",
                },
            ],
            "limit_name": None,
            "plan_type": "plus",
        }
        _configure_codex(
            mock_codex,
            models=[],
            rate_limits=_rate_limit_snapshot(
                primary_used=30,
                primary_resets_at=1_700_000_000,
                primary_window_mins=300,
                secondary_used=80,
                secondary_resets_at=1_700_010_000,
                secondary_window_mins=10_080,
                plan_type="plus",
            ),
        )

        with (
            patch(
                "hitch.main.caches._rate_limits_for_usage_context",
                return_value=caches._RateLimitsUsageState(rate_limits, False),
            ),
            patch("hitch.main.caches._start_models_refresh_thread"),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        # Both windows surface, with "remaining" framing rather than "used"
        # -- the page answers "how much budget do I have left?".
        self.assertContains(response, "Quota usage")
        self.assertContains(response, "Primary")
        self.assertContains(response, "70% remaining")
        self.assertContains(response, "Secondary")
        self.assertContains(response, "20% remaining")
        # Window duration and reset timestamp are surfaced; the timestamp
        # is rendered into a <time> element so the client-side script can
        # format it relative to the viewer.
        self.assertContains(response, "5-hour window")
        self.assertContains(response, "7-day window")
        self.assertNotContains(response, "-min window")
        self.assertContains(response, 'data-resets-at="1700000000"')
        # Plan label gives context for which plan the limits apply to.
        self.assertContains(response, "plus")

    def test_usage_page_reconciles_settings_from_cached_models(self) -> None:
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "stale-model", _EFFORT_COOKIE: "high"},
        )
        models = [_model("gpt-5", is_default=True, default_effort="medium")]

        with (
            patch("hitch.main.caches._cached_models_data", return_value=models),
            patch("hitch.main.caches._start_models_refresh_thread"),
            patch("hitch.main.caches._start_rate_limits_refresh_thread"),
            patch("hitch.main.views.common._start_usage_session_index_refresh_thread"),
        ):
            response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["current_model"], "gpt-5")
        self.assertEqual(response.context["current_effort"], "medium")
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "medium")
        self.assertEqual(
            response.context["model_options"],
            [
                {
                    "id": "gpt-5",
                    "display_name": "gpt-5",
                    "supported_efforts": " ".join(
                        sorted(e.value for e in ReasoningEffort)
                    ),
                }
            ],
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

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_renders_when_codex_offers_no_models(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # An empty model list is the pre-provider local-dev story; the settings
        # page must still render so the user sees the empty state instead of a 500.
        _configure_codex(mock_codex, models=[])
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Codex returned no models")


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
    def test_new_account_defaults_to_high(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        user = get_user_model().objects.create_user("dev@example.com")
        self.client.force_login(user)
        _configure_codex(
            mock_codex,
            models=[
                _model("other"),
                _model("gpt-5", is_default=True, default_effort="low"),
            ],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        # Hitch prefers high over the provider's low default and writes it
        # back so the next request has the application default in hand.
        mock_codex.assert_called_once()
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "high")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_guest_defaults_to_high(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, default_effort="low")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "high")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_guest_default_falls_back_when_high_is_unsupported(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "gpt-5-codex",
                    is_default=True,
                    default_effort="medium",
                    supported_efforts=["low", "medium"],
                )
            ],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "medium")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_resets_stale_model_to_current_default(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # The user had previously picked a model that the running Codex no
        # longer offers (e.g. provider swapped); the index render must snap
        # the cookie back to a valid choice.
        _seed_cookies(
            self.client, **{_MODEL_COOKIE: "removed-model", _EFFORT_COOKIE: "minimal"}
        )
        _seed_models_cache(
            [_model("gpt-5", is_default=True, default_effort="medium")]
        )
        self.addCleanup(_clear_models_cache)
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, default_effort="medium")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "medium")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_preserves_settings_when_still_valid(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(
            self.client, **{_MODEL_COOKIE: "gpt-5", _EFFORT_COOKIE: "low"}
        )
        _configure_codex(
            mock_codex,
            models=[
                _model("gpt-5", is_default=True, default_effort="medium"),
                _model("other"),
            ],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        # Saved values are still valid → no Set-Cookie on this response.
        mock_codex.assert_called_once()
        self.assertNotIn(_MODEL_COOKIE, response.cookies)
        self.assertNotIn(_EFFORT_COOKIE, response.cookies)
        self.assertContains(response, 'value="gpt-5"')
        self.assertContains(response, 'value="low" selected')

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_resets_effort_when_model_no_longer_supports_it(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A still-offered model may have its supported_reasoning_efforts
        narrowed under us (provider change, model retirement); the saved
        effort must snap to that model's default rather than ride a stale
        value into every future turn until the user notices."""
        _seed_cookies(
            self.client, **{_MODEL_COOKIE: "gpt-5", _EFFORT_COOKIE: "xhigh"}
        )
        _seed_models_cache(
            [
                _model(
                    "gpt-5",
                    is_default=True,
                    default_effort="medium",
                    supported_efforts=["low", "medium"],
                )
            ]
        )
        self.addCleanup(_clear_models_cache)
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "gpt-5",
                    is_default=True,
                    default_effort="medium",
                    supported_efforts=["low", "medium"],
                )
            ],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        # Model is preserved (no Set-Cookie for it), effort snaps to default.
        self.assertNotIn(_MODEL_COOKIE, response.cookies)
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "medium")

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
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, default_effort="medium")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        mock_codex.assert_called_once()
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")

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

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_get_uses_fresh_models_over_stale_cache(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_models_cache(
            [_model("cached-model", is_default=True, display_name="Cached Model")]
        )
        self.addCleanup(_clear_models_cache)
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5.6", is_default=True, display_name="GPT-5.6")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GPT-5.6")
        self.assertNotContains(response, "Cached Model")
        mock_codex.assert_called_once()

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_get_uses_raw_models_when_sdk_rejects_new_efforts(
        self, mock_codex: MagicMock, mock_discover: MagicMock
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
        mock_discover.return_value = []

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GPT-5.6")
        self.assertContains(response, 'data-supported-efforts="max ultra"')
        body = response.content.decode()
        for effort in ("max", "ultra"):
            option = self._effort_option(body, effort)
            self.assertNotIn("hidden", option)
            self.assertNotIn("disabled", option)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5.6")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "ultra")
        cached_models = caches._cached_models_data(enable_memories=False)
        self.assertEqual([model.id for model in cached_models], ["gpt-5.6"])
        self.assertEqual(
            [
                option.reasoning_effort.value
                for option in cached_models[0].supported_reasoning_efforts
            ],
            ["max", "ultra"],
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_get_falls_back_to_cached_models_when_fresh_fetch_fails(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_models_cache(
            [
                _model(
                    "cached-model",
                    is_default=True,
                    display_name="Cached Model",
                    default_effort="medium",
                )
            ]
        )
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "retired-model", _EFFORT_COOKIE: "ultra"},
        )
        ctx = mock_codex.return_value.__enter__.return_value
        ctx._client._request_raw.side_effect = ValueError("bad raw model list")
        ctx.models.side_effect = ValueError("bad typed model list")
        mock_discover.return_value = []

        with patch("hitch.main.views.settings.logger.exception") as log_exception:
            response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cached Model")
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "cached-model")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "medium")
        log_exception.assert_called_once()

    @patch("hitch.main.views.common.Codex")
    def test_saves_model_and_effort_to_signed_cookies(
        self, mock_codex: MagicMock
    ) -> None:
        """A valid POST persists the model and effort to signed cookies,
        redirects back to the index, and uses cookie attributes that
        outlive a session and survive cross-page form submits (Lax)."""
        _configure_codex(
            mock_codex,
            models=[
                _model("gpt-5", is_default=True, supported_efforts=["medium", "high"])
            ],
        )
        response = self.client.post(
            reverse("update_settings"),
            data={"model": "gpt-5", "reasoning_effort": "high"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("index"))
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "high")
        morsel = response.cookies[_MODEL_COOKIE]
        self.assertGreaterEqual(int(morsel["max-age"]), 30 * 24 * 60 * 60)
        self.assertEqual(morsel["samesite"], "Lax")

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

    @patch("hitch.main.views.common.Codex")
    def test_post_validates_against_fresh_models_over_stale_cache(
        self, mock_codex: MagicMock
    ) -> None:
        _seed_models_cache(
            [
                _model(
                    "cached-model",
                    is_default=True,
                    supported_efforts=["medium", "high"],
                )
            ]
        )
        self.addCleanup(_clear_models_cache)
        _configure_codex(
            mock_codex,
            models=[
                _model(
                    "gpt-5.6",
                    is_default=True,
                    supported_efforts=["medium", "high"],
                )
            ],
        )

        response = self.client.post(
            reverse("update_settings"),
            data={"model": "gpt-5.6", "reasoning_effort": "high"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5.6")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "high")
        mock_codex.assert_called_once()
        self.assertEqual(
            [model.id for model in caches._cached_models_data(enable_memories=False)],
            ["gpt-5.6"],
        )

    @patch("hitch.main.views.common.Codex")
    def test_post_validates_against_cached_models_when_fresh_fetch_fails(
        self, mock_codex: MagicMock
    ) -> None:
        _seed_models_cache(
            [
                _model(
                    "gpt-5",
                    is_default=True,
                    supported_efforts=["low", "medium"],
                )
            ]
        )
        ctx = mock_codex.return_value.__enter__.return_value
        ctx._client._request_raw.side_effect = ValueError("bad raw model list")
        ctx.models.side_effect = ValueError("bad typed model list")

        with patch("hitch.main.views.settings.logger.exception") as log_exception:
            response = self.client.post(
                reverse("update_settings"),
                data={"model": "gpt-5", "reasoning_effort": "ultra"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content.decode(), "invalid reasoning effort")
        self.assertNotIn(_MODEL_COOKIE, response.cookies)
        self.assertNotIn(_EFFORT_COOKIE, response.cookies)
        log_exception.assert_called_once()

    @patch("hitch.main.views.common.Codex")
    def test_post_falls_back_to_codex_when_model_cache_empty(
        self, mock_codex: MagicMock
    ) -> None:
        _clear_models_cache()
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, supported_efforts=["medium"])],
        )

        response = self.client.post(
            reverse("update_settings"),
            data={"model": "gpt-5", "reasoning_effort": "medium"},
        )

        self.assertEqual(response.status_code, 302)
        mock_codex.assert_called_once()

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

    @patch("hitch.main.views.common.Codex")
    def test_accepts_anything_when_codex_offers_no_models(
        self, mock_codex: MagicMock
    ) -> None:
        # No models means we can't validate; trust the POST so a transient
        # Codex outage doesn't lock the user out of changing settings.
        _configure_codex(mock_codex, models=[])
        response = self.client.post(
            reverse("update_settings"),
            data={"model": "anything", "reasoning_effort": "high"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "anything")

    def test_allows_empty_effort(self) -> None:
        # Empty effort = "let Codex choose"; the update endpoint must accept
        # it so a user can revert a previous explicit pick. With both fields
        # empty there's nothing to validate against Codex, so the endpoint
        # stays cheap (no app-server roundtrip).
        response = self.client.post(
            reverse("update_settings"),
            data={"model": "", "reasoning_effort": ""},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "")

    def test_redirects_to_safe_next_url(self) -> None:
        session_url = reverse("session", kwargs={"session_id": "abc"})

        response = self.client.post(
            reverse("update_settings"), data={"next": session_url}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], session_url)

    def test_rejects_unsafe_next_url(self) -> None:
        response = self.client.post(
            reverse("update_settings"), data={"next": "https://example.invalid/"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("index"))

    def test_saves_extra_system_prompt_to_signed_cookie(self) -> None:
        response = self.client.post(
            reverse("update_settings"),
            data={
                "model": "",
                "reasoning_effort": "",
                "extra_system_prompt": "  Prefer focused tests.\nKeep diffs small.  ",
            },
        )

        self.assertEqual(response.status_code, 302)
        raw_cookie_value = _cookie_value(response, _EXTRA_SYSTEM_PROMPT_COOKIE)
        self.assertNotIn("\n", raw_cookie_value)
        self.assertEqual(
            _extra_system_prompt_value(response),
            "Prefer focused tests.\nKeep diffs small.",
        )

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

    def test_accepted_prompt_cookie_stays_within_browser_limit(self) -> None:
        """Every prompt the server accepts must fit in the cookie it then
        writes — ASCII at the character cap and a sizable multibyte prompt
        both round-trip without crossing the byte budget."""
        for label, prompt in (
            ("ascii at cap", "a" * settings_cookies._EXTRA_SYSTEM_PROMPT_MAX_LEN),
            ("multibyte", "あ" * 800),
        ):
            with self.subTest(label=label):
                response = Client().post(
                    reverse("update_settings"),
                    data={
                        "model": "",
                        "reasoning_effort": "",
                        "extra_system_prompt": prompt,
                    },
                )

                self.assertEqual(response.status_code, 302)
                morsel = response.cookies[_EXTRA_SYSTEM_PROMPT_COOKIE]
                name_value = morsel.output(header="").split(";")[0].strip()
                self.assertLessEqual(
                    len(name_value.encode()), settings_cookies._COOKIE_MAX_VALUE_BYTES
                )
                # The value still round-trips intact under the byte budget.
                self.assertEqual(_extra_system_prompt_value(response), prompt)

    @patch("hitch.main.views.common.Codex")
    def test_get_renders_settings_page(self, mock_codex: MagicMock) -> None:
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])

        response = self.client.get(reverse("update_settings"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<h1>Settings</h1>")

    @patch("hitch.main.views.common.Codex")
    def test_saves_optional_signed_cookie_settings(
        self, mock_codex: MagicMock
    ) -> None:
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        project = _make_project()
        cases: list[tuple[str, dict[str, str], dict[str, str], str, str]] = [
            (
                "sandbox policy",
                {
                    "model": "gpt-5",
                    "reasoning_effort": "high",
                    "sandbox_policy": "workspaceWrite",
                },
                {},
                _SANDBOX_COOKIE,
                "workspaceWrite",
            ),
            (
                "empty sandbox policy",
                {"model": "", "reasoning_effort": "", "sandbox_policy": ""},
                {},
                _SANDBOX_COOKIE,
                "",
            ),
            (
                "show archived enabled",
                {"show_archived_sessions": "true"},
                {},
                _SHOW_ARCHIVED_COOKIE,
                "true",
            ),
            (
                "show archived disabled",
                {"show_archived_sessions": ""},
                {},
                _SHOW_ARCHIVED_COOKIE,
                "false",
            ),
            (
                "show archived preserved",
                {},
                {_SHOW_ARCHIVED_COOKIE: "true"},
                _SHOW_ARCHIVED_COOKIE,
                "true",
            ),
            (
                "worktrees enabled",
                {"use_worktrees": "true"},
                {},
                _USE_WORKTREES_COOKIE,
                "true",
            ),
            ("worktrees disabled", {}, {}, _USE_WORKTREES_COOKIE, "false"),
            (
                "auto-PR enabled",
                {"auto_pr": "true"},
                {},
                _AUTO_PR_COOKIE,
                "true",
            ),
            ("auto-PR disabled", {}, {}, _AUTO_PR_COOKIE, "false"),
            (
                "auto-QA enabled",
                {"auto_qa": "true"},
                {},
                _AUTO_QA_COOKIE,
                "true",
            ),
            ("auto-QA disabled", {}, {}, _AUTO_QA_COOKIE, "false"),
            (
                "Spec Critic enabled",
                {"spec_critic": "true"},
                {},
                _SPEC_CRITIC_COOKIE,
                "true",
            ),
            ("Spec Critic disabled", {}, {}, _SPEC_CRITIC_COOKIE, "false"),
            (
                "memories enabled",
                {"enable_memories": "true"},
                {},
                _ENABLE_MEMORIES_COOKIE,
                "true",
            ),
            ("memories disabled", {}, {}, _ENABLE_MEMORIES_COOKIE, "false"),
            (
                "web search live",
                {"web_search_mode": "live"},
                {},
                _WEB_SEARCH_COOKIE,
                "live",
            ),
            (
                "web search default",
                {"web_search_mode": ""},
                {_WEB_SEARCH_COOKIE: "live"},
                _WEB_SEARCH_COOKIE,
                "",
            ),
            (
                "selected project",
                {"selected_project": str(project.pk)},
                {},
                _SELECTED_PROJECT_COOKIE,
                str(project.pk),
            ),
            (
                "all projects",
                {"selected_project": ""},
                {_SELECTED_PROJECT_COOKIE: str(project.pk)},
                _SELECTED_PROJECT_COOKIE,
                "",
            ),
        ]
        for label, data, seed, cookie, expected in cases:
            with self.subTest(label=label):
                client = Client()
                if seed:
                    _seed_cookies(client, **seed)

                response = client.post(reverse("update_settings"), data=data)

                self.assertEqual(response.status_code, 302)
                self.assertEqual(_cookie_value(response, cookie), expected)

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
                "Spec Critic setting",
                _SPEC_CRITIC_COOKIE,
                "true",
                {"spec_critic": "yes"},
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

    def test_update_settings_allows_non_staff_disk_usage_global_setting(self) -> None:
        user = get_user_model().objects.create_user(
            "dev@example.com", password="StrongPass123!"
        )
        self.client.force_login(user)

        response = self.client.post(
            reverse("update_settings"),
            data={"disk_usage_max_percent": "35.5"},
        )

        self.assertEqual(response.status_code, 302)
        settings = GlobalSettings.objects.get(pk=GlobalSettings.SINGLETON_PK)
        self.assertEqual(settings.disk_usage_max_percent, 35.5)

    def test_staff_update_settings_saves_disk_usage_global_setting(self) -> None:
        user = get_user_model().objects.create_user(
            "admin@example.com", password="StrongPass123!", is_staff=True
        )
        self.client.force_login(user)

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

    def test_staff_update_settings_updates_changed_initial_disk_usage_global_setting(
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
            data={
                "disk_usage_max_percent": "42.3",
                "initial_disk_usage_max_percent": "35.5",
            },
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
        self.assertEqual(_cookie_value(response, _WEB_SEARCH_COOKIE), "cached")

    def test_update_settings_saves_web_search_to_account(self) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user(
            "dev@example.com", password="StrongPass123!"
        )
        UserSettings.objects.create(user=user, web_search_mode="disabled")
        self.client.force_login(user)

        response = self.client.post(
            reverse("update_settings"),
            data={"web_search_mode": "live"},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.web_search_mode, "live")
        self.assertEqual(_cookie_value(response, _WEB_SEARCH_COOKIE), "live")

    def test_account_prompt_too_big_for_cookie_still_saves_to_db(self) -> None:
        """For a logged-in user the prompt's source of truth is the DB; the
        cookie is only a mirror. A prompt under the character cap but over the
        cookie byte budget must still save to the account instead of being
        rejected — the cookie-overflow guard is for the anonymous, cookie-only
        path, not the DB-backed one."""
        user_model = get_user_model()
        user = user_model.objects.create_user(
            "dev@example.com", password="StrongPass123!"
        )
        UserSettings.objects.create(user=user)
        self.client.force_login(user)
        # Under the 2500-character cap, but its base64 cookie would blow past
        # the browser limit (the value an anonymous POST would be rejected for).
        prompt = "あ" * 2400
        self.assertLessEqual(len(prompt), settings_cookies._EXTRA_SYSTEM_PROMPT_MAX_LEN)
        self.assertFalse(settings_cookies._extra_system_prompt_cookie_fits(prompt))

        response = self.client.post(
            reverse("update_settings"),
            data={"model": "", "reasoning_effort": "", "extra_system_prompt": prompt},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            UserSettings.objects.get(user=user).extra_system_prompt, prompt
        )

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

    def test_preserves_other_cookie_settings(self) -> None:
        _seed_cookies(
            self.client,
            **{
                _MODEL_COOKIE: "gpt-5",
                _EFFORT_COOKIE: "high",
                _SANDBOX_COOKIE: "readOnly",
                _APPROVAL_COOKIE: "deny_all",
                _USE_WORKTREES_COOKIE: "true",
            },
        )

        response = self.client.post(reverse("update_archived_session_visibility"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "high")
        self.assertEqual(_cookie_value(response, _SANDBOX_COOKIE), "readOnly")
        self.assertEqual(_cookie_value(response, _APPROVAL_COOKIE), "deny_all")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _SHOW_ARCHIVED_COOKIE), "false")


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

    def test_login_stale_selected_project_cookie_clears_account_project(self) -> None:
        user_model = get_user_model()
        user = user_model.objects.create_user("dev@example.com", password="StrongPass123!")
        project = _make_project()
        stale = _make_project(name="Old", repo_path="/old")
        stale_id = stale.pk
        stale.delete()
        UserSettings.objects.create(user=user, selected_project=project)
        _seed_cookies(self.client, **{_SELECTED_PROJECT_COOKIE: str(stale_id)})

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
