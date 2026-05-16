import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core import signing
from django.test import Client, TestCase
from django.urls import reverse
from openai_codex.errors import MethodNotFoundError

from hitch.main.models import UserSettings

_MODEL_COOKIE = "hitch_model"
_EFFORT_COOKIE = "hitch_reasoning_effort"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"


def _sign(name: str, value: str) -> str:
    return signing.get_cookie_signer(salt=name).sign(value)


def _seed_cookies(client: Client, **values: str) -> None:
    for name, value in values.items():
        client.cookies[name] = _sign(name, value)


def _cookie_value(response: object, name: str) -> str:
    raw = response.cookies[name].value  # type: ignore[attr-defined]
    return signing.get_cookie_signer(salt=name).unsign(raw)


def _encode_extra_system_prompt(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode("ascii")


def _decode_extra_system_prompt(value: str) -> str:
    return base64.urlsafe_b64decode(value.encode("ascii")).decode()


def _make_user(username: str = "dev@example.com", password: str = "StrongPass123!") -> Any:
    user_model = get_user_model()
    return user_model.objects.create_user(username=username, password=password)


def _setup_codex(mock_codex: MagicMock, *, models: list[Any] | None = None) -> None:
    ctx = mock_codex.return_value.__enter__.return_value
    ctx.thread_list.return_value.data = []
    ctx.models.return_value.data = models or []
    ctx._client.request.side_effect = MethodNotFoundError(
        -32601, "method not found", None
    )


def _model(model_id: str, *, is_default: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        id=model_id,
        display_name=model_id,
        is_default=is_default,
        default_reasoning_effort=SimpleNamespace(value="medium"),
        supported_reasoning_efforts=[
            SimpleNamespace(reasoning_effort=SimpleNamespace(value=v), description=v)
            for v in ("low", "medium", "high")
        ],
    )


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
        self.assertTrue(UserSettings.objects.filter(user=user).exists())

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
                _SHOW_ARCHIVED_COOKIE: "true",
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
        self.assertTrue(settings.show_archived_sessions)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")
        self.assertEqual(
            _decode_extra_system_prompt(
                _cookie_value(response, _EXTRA_SYSTEM_PROMPT_COOKIE)
            ),
            "Prefer focused tests.",
        )

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
            show_archived_sessions=True,
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
        self.assertTrue(settings.show_archived_sessions)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "stored-model")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")
        self.assertEqual(_cookie_value(response, _SHOW_ARCHIVED_COOKIE), "true")

    def test_logout_mirrors_account_settings_to_cookies_for_guest_mode(self) -> None:
        user = _make_user()
        UserSettings.objects.create(
            user=user,
            model="stored-model",
            sandbox_policy="workspaceWrite",
            approval_mode="deny_all",
            use_worktrees=True,
        )
        self.client.force_login(user)

        response = self.client.post(reverse("logout"))

        self.assertEqual(response.status_code, 302)
        self.assertNotIn("_auth_user_id", self.client.session)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "stored-model")
        self.assertEqual(_cookie_value(response, _SANDBOX_COOKIE), "workspaceWrite")
        self.assertEqual(_cookie_value(response, _APPROVAL_COOKIE), "deny_all")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")


class AuthenticatedSettingsTests(TestCase):
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
                "show_archived_sessions": "true",
            },
        )

        self.assertEqual(response.status_code, 302)
        settings = UserSettings.objects.get(user=user)
        self.assertEqual(settings.sandbox_policy, "readOnly")
        self.assertEqual(settings.approval_mode, "deny_all")
        self.assertEqual(settings.extra_system_prompt, "Keep it small.")
        self.assertTrue(settings.use_worktrees)
        self.assertTrue(settings.show_archived_sessions)
        self.assertEqual(_cookie_value(response, _SANDBOX_COOKIE), "readOnly")
        self.assertEqual(_cookie_value(response, _APPROVAL_COOKIE), "deny_all")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_index_prefers_account_settings_over_conflicting_cookies(
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
            },
        )
        _setup_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, 'value="gpt-5" selected')
        self.assertContains(response, 'value="high" selected')
        self.assertContains(response, 'value="dangerFullAccess" selected')
        self.assertContains(response, 'value="deny_all" selected')
        self.assertContains(response, 'name="use_worktrees" value="true" checked')
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _SANDBOX_COOKIE), "dangerFullAccess")
        self.assertEqual(_cookie_value(response, _USE_WORKTREES_COOKIE), "true")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.codex_pool.spawn_turn")
    @patch("hitch.main.views.Codex")
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
        )
