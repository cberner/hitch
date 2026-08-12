from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from hitch.main.models import UserSettings
from hitch.main.test.support import (
    _cookie_value,
    _decode_extra_system_prompt,
    _encode_extra_system_prompt,
    _seed_cookies,
    _setup_codex,
)
from hitch.main.test.support import (
    _make_model as _model,
)

_MODEL_COOKIE = "hitch_model"
_EFFORT_COOKIE = "hitch_reasoning_effort"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_AUTO_PR_COOKIE = "hitch_auto_pr"
_AUTO_QA_COOKIE = "hitch_auto_qa"
_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_LAST_SELECTED_REPO_COOKIE = "hitch_last_selected_repo"
_ENABLE_MEMORIES_COOKIE = "hitch_enable_memories"


def _make_user(username: str = "dev@example.com", password: str = "StrongPass123!") -> Any:
    user_model = get_user_model()
    return user_model.objects.create_user(username=username, password=password)


class AuthViewTests(TestCase):
    def test_register_accepts_email_shaped_username_and_logs_user_in(self) -> None:
        response = self.client.post(
            reverse("register"),
            data={
                "username": "dev@example.com",
                "password1": "StrongPass123!",
                "password2": "StrongPass123!",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("index"))
        user_model = get_user_model()
        user = user_model.objects.get(username="dev@example.com")
        self.assertEqual(int(self.client.session["_auth_user_id"]), user.pk)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.reasoning_effort, "high")

    def test_login_imports_anonymous_cookie_settings_to_account(self) -> None:
        user = _make_user()
        _seed_cookies(
            self.client,
            **{
                _MODEL_COOKIE: "gpt-5",
                _EFFORT_COOKIE: "high",
                _SANDBOX_COOKIE: "workspaceWrite",
                _APPROVAL_COOKIE: "deny_all",
                _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt(
                    "Prefer focused tests."
                ),
                _USE_WORKTREES_COOKIE: "true",
                _AUTO_PR_COOKIE: "true",
                _AUTO_QA_COOKIE: "true",
                _SHOW_ARCHIVED_COOKIE: "true",
                _LAST_SELECTED_REPO_COOKIE: "/home/user/proj",
                _ENABLE_MEMORIES_COOKIE: "true",
            },
        )

        response = self.client.post(
            reverse("login"),
            data={"username": "dev@example.com", "password": "StrongPass123!"},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.model, "gpt-5")
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertEqual(settings.sandbox_policy, "workspaceWrite")
        self.assertEqual(settings.approval_mode, "deny_all")
        self.assertEqual(settings.extra_system_prompt, "Prefer focused tests.")
        self.assertTrue(settings.use_worktrees)
        self.assertTrue(settings.auto_pr_enabled)
        self.assertTrue(settings.auto_qa_enabled)
        self.assertTrue(settings.show_archived_sessions)
        self.assertEqual(settings.last_selected_repo, "/home/user/proj")
        self.assertTrue(settings.enable_memories)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_PR_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_QA_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _ENABLE_MEMORIES_COOKIE), "true")
        self.assertEqual(
            _cookie_value(response, _LAST_SELECTED_REPO_COOKIE), "/home/user/proj"
        )
        self.assertEqual(
            _decode_extra_system_prompt(
                _cookie_value(response, _EXTRA_SYSTEM_PROMPT_COOKIE)
            ),
            "Prefer focused tests.",
        )

    def test_login_imports_provider_advertised_effort_unknown_to_sdk_enum(self) -> None:
        user = _make_user()
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "gpt-5.6", _EFFORT_COOKIE: "ultra"},
        )

        response = self.client.post(
            reverse("login"),
            data={"username": "dev@example.com", "password": "StrongPass123!"},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.model, "gpt-5.6")
        self.assertEqual(settings.reasoning_effort, "ultra")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "ultra")

    def test_login_without_settings_cookies_preserves_db_settings(self) -> None:
        user = _make_user()
        UserSettings.objects.create(
            user=user,
            model="stored-model",
            reasoning_effort="low",
            sandbox_policy="readOnly",
            approval_mode="deny_all",
            extra_system_prompt="Stored prompt.",
            use_worktrees=True,
            auto_pr_enabled=True,
            auto_qa_enabled=True,
            show_archived_sessions=True,
            last_selected_repo="/home/user/stored",
        )

        response = self.client.post(
            reverse("login"),
            data={"username": "dev@example.com", "password": "StrongPass123!"},
        )

        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.model, "stored-model")
        self.assertEqual(settings.reasoning_effort, "low")
        self.assertEqual(settings.sandbox_policy, "readOnly")
        self.assertTrue(settings.use_worktrees)
        self.assertTrue(settings.auto_pr_enabled)
        self.assertTrue(settings.auto_qa_enabled)
        self.assertTrue(settings.show_archived_sessions)
        self.assertEqual(settings.last_selected_repo, "/home/user/stored")
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "stored-model")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_PR_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_QA_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _SHOW_ARCHIVED_COOKIE), "true")
        self.assertEqual(
            _cookie_value(response, _LAST_SELECTED_REPO_COOKIE), "/home/user/stored"
        )

    def test_login_preserves_account_settings_over_stale_browser_cookies(self) -> None:
        user = _make_user()
        UserSettings.objects.create(
            user=user,
            model="gpt-5.6-sol",
            reasoning_effort="xhigh",
        )
        _seed_cookies(
            self.client,
            **{_MODEL_COOKIE: "gpt-5.5", _EFFORT_COOKIE: "low"},
        )

        response = self.client.post(
            reverse("login"),
            data={"username": "dev@example.com", "password": "StrongPass123!"},
        )

        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.model, "gpt-5.6-sol")
        self.assertEqual(settings.reasoning_effort, "xhigh")
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5.6-sol")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "xhigh")

    def test_logout_mirrors_account_settings_to_cookies_for_guest_mode(self) -> None:
        user = _make_user()
        UserSettings.objects.create(
            user=user,
            model="stored-model",
            sandbox_policy="workspaceWrite",
            approval_mode="deny_all",
            use_worktrees=True,
            auto_pr_enabled=True,
            auto_qa_enabled=True,
            last_selected_repo="/home/user/stored",
        )
        self.client.force_login(user)

        response = self.client.post(reverse("logout"))

        self.assertEqual(response.status_code, 302)
        self.assertNotIn("_auth_user_id", self.client.session)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "stored-model")
        self.assertEqual(_cookie_value(response, _SANDBOX_COOKIE), "workspaceWrite")
        self.assertEqual(_cookie_value(response, _APPROVAL_COOKIE), "deny_all")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_PR_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_QA_COOKIE), "true")
        self.assertEqual(
            _cookie_value(response, _LAST_SELECTED_REPO_COOKIE), "/home/user/stored"
        )
        self.assertEqual(_cookie_value(response, _ENABLE_MEMORIES_COOKIE), "false")

    @patch("hitch.main.views.common.Codex")
    def test_profile_renders_anonymous_user_with_usage(
        self, mock_codex: MagicMock
    ) -> None:
        _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(f'href="{reverse("profile")}"', nav_html)
        self.assertIn('aria-current="page"', nav_html)
        self.assertIn(">anonymous</a>", nav_html)
        self.assertContains(response, "anonymous")
        self.assertContains(response, "Signed out")
        self.assertContains(response, "Token usage")
        self.assertContains(response, "Quota usage")
        self.assertContains(response, f'href="{reverse("login")}"')
        self.assertContains(response, f'href="{reverse("register")}"')
        self.assertNotContains(response, f'action="{reverse("logout")}"')

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.context_processors.server_git_hash", return_value="abc123")
    def test_profile_hides_server_git_hash_for_anonymous_user(
        self, _mock_hash: MagicMock, mock_codex: MagicMock
    ) -> None:
        _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "anonymous")
        self.assertNotContains(response, "Server git hash")
        self.assertNotContains(response, "abc123")

    @patch("hitch.main.views.common.Codex")
    @patch("hitch.main.context_processors.server_git_hash", return_value="abc123")
    def test_profile_renders_logout_form_for_authenticated_user(
        self, _mock_hash: MagicMock, mock_codex: MagicMock
    ) -> None:
        user = _make_user()
        self.client.force_login(user)
        _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"))

        body = response.content.decode()
        self.assertContains(response, "dev@example.com")
        self.assertContains(response, "Signed in")
        self.assertContains(response, "Token usage")
        self.assertContains(response, "Quota usage")
        self.assertContains(response, f'action="{reverse("logout")}"')
        self.assertContains(response, ">Log out</button>")
        self.assertContains(response, "Server git hash")
        self.assertContains(response, ">abc123</code>")
        self.assertLess(
            body.index('<section class="profile-panel"'),
            body.index('<section class="usage-section"'),
        )
        self.assertLess(
            body.index('<section class="usage-section"'),
            body.index('<form class="profile-logout-form"'),
        )
        self.assertLess(
            body.index('<form class="profile-logout-form"'),
            body.index('<p class="profile-revision"'),
        )

    @patch("hitch.main.views.common._usage_context", side_effect=RuntimeError("codex down"))
    def test_profile_renders_account_controls_when_usage_context_fails(
        self, mock_usage_context: MagicMock
    ) -> None:
        user = _make_user()
        self.client.force_login(user)

        response = self.client.get(reverse("profile"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "dev@example.com")
        self.assertContains(response, f'action="{reverse("logout")}"')
        self.assertContains(response, "All sessions usage unavailable.")
        self.assertContains(response, "Usage unavailable.")
        mock_usage_context.assert_called_once()


class NukeCodexViewTests(TestCase):
    @patch("hitch.main.views.common.Codex")
    def test_profile_renders_nuke_button_for_authenticated_user(
        self, mock_codex: MagicMock
    ) -> None:
        self.client.force_login(_make_user())
        _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"))

        self.assertContains(response, f'action="{reverse("nuke_codex")}"')
        self.assertContains(response, ">Nuke Codex instances</button>")
        # No confirmation line until the action has run.
        self.assertNotContains(response, "Killed ")

    @patch("hitch.main.views.common.Codex")
    def test_profile_hides_nuke_button_for_anonymous_user(
        self, mock_codex: MagicMock
    ) -> None:
        _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"))

        self.assertNotContains(response, "Nuke Codex instances")

    @patch("hitch.main.views.common.Codex")
    def test_profile_shows_killed_count_after_nuke(
        self, mock_codex: MagicMock
    ) -> None:
        self.client.force_login(_make_user())
        _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"), {"nuked": "1"})

        self.assertContains(response, "Killed 1 Codex app server.")

    @patch("hitch.main.views.common.Codex")
    def test_profile_ignores_malformed_nuked_param(
        self, mock_codex: MagicMock
    ) -> None:
        self.client.force_login(_make_user())
        _setup_codex(mock_codex)

        response = self.client.get(reverse("profile"), {"nuked": "lots"})

        self.assertNotContains(response, "Killed ")

    @patch("hitch.main.runtime.reconciliation.nuke_codex_app_servers", return_value=3)
    def test_nuke_kills_app_servers_and_redirects_with_count(
        self, mock_nuke: MagicMock
    ) -> None:
        self.client.force_login(_make_user())

        response = self.client.post(reverse("nuke_codex"))

        mock_nuke.assert_called_once_with()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], f"{reverse('profile')}?nuked=3")

    @patch("hitch.main.runtime.reconciliation.nuke_codex_app_servers")
    def test_nuke_requires_authentication(self, mock_nuke: MagicMock) -> None:
        response = self.client.post(reverse("nuke_codex"))

        self.assertEqual(response.status_code, 403)
        mock_nuke.assert_not_called()

    def test_nuke_rejects_get(self) -> None:
        self.client.force_login(_make_user())

        response = self.client.get(reverse("nuke_codex"))

        self.assertEqual(response.status_code, 405)


class AuthenticatedSettingsTests(TestCase):
    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_index_places_profile_link_in_primary_nav(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        user = _make_user()
        self.client.force_login(user)
        _setup_codex(mock_codex)
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        profile_pos = nav_html.index(f'href="{reverse("profile")}"')
        self.assertLess(nav_html.index(">settings</a>"), profile_pos)
        self.assertIn('class="primary-nav-account"', nav_html)
        self.assertIn(">dev@example.com</a>", nav_html)
        self.assertNotIn(f'href="{reverse("usage")}"', nav_html)
        self.assertNotIn(reverse("logout"), nav_html)
        self.assertNotIn("Server git hash", nav_html)
        self.assertNotIn("account-label", body)

    def test_update_settings_writes_database_and_cookie_mirror(self) -> None:
        user = _make_user()
        self.client.force_login(user)

        response = self.client.post(
            reverse("update_settings"),
            data={
                "model": "",
                "reasoning_effort": "",
                "sandbox_policy": "readOnly",
                "approval_mode": "deny_all",
                "extra_system_prompt": "  Keep it small.  ",
                "use_worktrees": "true",
                "auto_pr": "true",
                "auto_qa": "true",
                "show_archived_sessions": "true",
                "enable_memories": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.sandbox_policy, "readOnly")
        self.assertEqual(settings.approval_mode, "deny_all")
        self.assertEqual(settings.extra_system_prompt, "Keep it small.")
        self.assertTrue(settings.use_worktrees)
        self.assertTrue(settings.auto_pr_enabled)
        self.assertTrue(settings.auto_qa_enabled)
        self.assertTrue(settings.show_archived_sessions)
        self.assertTrue(settings.enable_memories)
        self.assertEqual(_cookie_value(response, _SANDBOX_COOKIE), "readOnly")
        self.assertEqual(_cookie_value(response, _APPROVAL_COOKIE), "deny_all")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_PR_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _AUTO_QA_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _ENABLE_MEMORIES_COOKIE), "true")

    def test_archived_visibility_update_preserves_other_account_settings(self) -> None:
        user = _make_user()
        self.client.force_login(user)
        UserSettings.objects.create(
            user=user,
            model="gpt-5",
            reasoning_effort="high",
            sandbox_policy="dangerFullAccess",
            approval_mode="deny_all",
            extra_system_prompt="Keep it small.",
            use_worktrees=True,
            auto_pr_enabled=True,
            auto_qa_enabled=True,
            show_archived_sessions=False,
            last_selected_repo="/home/user/proj",
        )

        response = self.client.post(
            reverse("update_archived_session_visibility"),
            data={"show_archived_sessions": "true"},
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.model, "gpt-5")
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertEqual(settings.sandbox_policy, "dangerFullAccess")
        self.assertEqual(settings.approval_mode, "deny_all")
        self.assertEqual(settings.extra_system_prompt, "Keep it small.")
        self.assertTrue(settings.use_worktrees)
        self.assertTrue(settings.auto_pr_enabled)
        self.assertTrue(settings.auto_qa_enabled)
        self.assertTrue(settings.show_archived_sessions)
        self.assertEqual(settings.last_selected_repo, "/home/user/proj")
        self.assertEqual(_cookie_value(response, _SHOW_ARCHIVED_COOKIE), "true")

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_prefers_account_settings_over_conflicting_cookies(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        user = _make_user()
        UserSettings.objects.create(
            user=user,
            model="gpt-5",
            reasoning_effort="high",
            sandbox_policy="dangerFullAccess",
            approval_mode="deny_all",
            use_worktrees=True,
            auto_pr_enabled=True,
            auto_qa_enabled=True,
            last_selected_repo="/home/user/account",
            enable_memories=True,
        )
        self.client.force_login(user)
        _seed_cookies(
            self.client,
            **{
                _MODEL_COOKIE: "other",
                _EFFORT_COOKIE: "low",
                _SANDBOX_COOKIE: "workspaceWrite",
                _APPROVAL_COOKIE: "approve_all",
                _USE_WORKTREES_COOKIE: "false",
                _AUTO_PR_COOKIE: "false",
                _AUTO_QA_COOKIE: "false",
                _LAST_SELECTED_REPO_COOKIE: "/home/user/cookie",
                _ENABLE_MEMORIES_COOKIE: "false",
            },
        )
        models = [_model("gpt-5", is_default=True)]
        _setup_codex(mock_codex, models=models)
        mock_discover.return_value = [Path("/home/user/account"), Path("/home/user/cookie")]

        with (
            patch("hitch.main.caches._cached_models_data", return_value=models),
            patch("hitch.main.caches._start_models_refresh_thread"),
        ):
            new_session_response = self.client.get(reverse("new_session"))
            settings_response = self.client.get(reverse("update_settings"))

        self.assertContains(settings_response, 'value="gpt-5" selected')
        self.assertContains(settings_response, 'value="high" selected')
        self.assertContains(settings_response, 'value="dangerFullAccess" selected')
        self.assertContains(settings_response, 'value="deny_all" selected')
        self.assertContains(
            settings_response, 'name="use_worktrees" value="true" checked'
        )
        self.assertContains(settings_response, 'name="auto_pr" value="true" checked')
        self.assertContains(settings_response, 'name="auto_qa" value="true" checked')
        self.assertContains(
            settings_response, 'name="enable_memories" value="true" checked'
        )
        self.assertContains(
            new_session_response, 'value="/home/user/account" selected'
        )
        self.assertEqual(_cookie_value(new_session_response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(
            _cookie_value(new_session_response, _SANDBOX_COOKIE), "dangerFullAccess"
        )
        self.assertEqual(
            _cookie_value(new_session_response, _USE_WORKTREES_COOKIE), "true"
        )
        self.assertEqual(_cookie_value(new_session_response, _AUTO_PR_COOKIE), "true")
        self.assertEqual(_cookie_value(new_session_response, _AUTO_QA_COOKIE), "true")
        self.assertEqual(
            _cookie_value(new_session_response, _LAST_SELECTED_REPO_COOKIE),
            "/home/user/account",
        )
        self.assertEqual(
            _cookie_value(new_session_response, _ENABLE_MEMORIES_COOKIE), "true"
        )

    @patch("hitch.main.repos.discover_repos")
    @patch("hitch.main.runtime.codex_pool.spawn_turn")
    @patch("hitch.main.views.common.Codex")
    def test_send_message_uses_account_settings_without_cookies(
        self,
        mock_codex: MagicMock,
        mock_spawn: MagicMock,
        mock_discover: MagicMock,
    ) -> None:
        user = _make_user()
        UserSettings.objects.create(
            user=user,
            sandbox_policy="workspaceWrite",
            approval_mode="deny_all",
            enable_memories=True,
        )
        self.client.force_login(user)
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=SimpleNamespace(cwd="/repo")
        )
        mock_discover.return_value = [Path("/repo")]

        response = self.client.post(
            reverse("send_message", kwargs={"session_id": "abc"}),
            data={"prompt": "follow-up"},
        )

        self.assertEqual(response.status_code, 302)
        mock_spawn.assert_called_once_with(
            thread_id="abc",
            cwd="/repo",
            prompt="follow-up",
            sandbox_policy="workspaceWrite",
            approval_mode="deny_all",
            enable_memories=True,
        )


class ConfirmFormTests(TestCase):
    """The shared data-confirm wirer replaces inline onsubmit confirms.

    Regression guard for J13: a form[data-confirm] confirms before submitting,
    and declining cancels the submit.
    """

    @patch("hitch.main.views.common.Codex")
    def test_data_confirm_blocks_submit_when_declined(
        self, mock_codex: MagicMock
    ) -> None:
        self.client.force_login(_make_user())
        _setup_codex(mock_codex)
        html = self.client.get(reverse("profile")).content.decode()
        self.assertIn('data-confirm="Force-kill', html)

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        dispatch = """
            (accept) => {
                window.confirm = () => accept;
                const form = document.querySelector("form[data-confirm]");
                const event = new Event("submit", { cancelable: true, bubbles: true });
                form.dispatchEvent(event);
                return event.defaultPrevented;
            }
        """
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                # Declining the confirm cancels the submit.
                self.assertTrue(page.evaluate(dispatch, False))
                # Accepting it lets the submit proceed.
                self.assertFalse(page.evaluate(dispatch, True))
            finally:
                browser.close()
