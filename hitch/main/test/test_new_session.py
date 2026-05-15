from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.core import signing
from django.test import TestCase
from django.urls import reverse


def _make_model(model_id: str, *, is_default: bool = False) -> SimpleNamespace:
    """Minimal Model stand-in for the codex.models() round-trip."""
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


def _stub_codex(mock_codex: MagicMock, *, models: list[SimpleNamespace] | None = None) -> None:
    """Configure the Codex mock for the pre-spawn reconcile call."""
    ctx = mock_codex.return_value.__enter__.return_value
    ctx.models.return_value.data = models or []


class NewSessionViewTests(TestCase):
    def _allowed_repo(self) -> str:
        return "/home/user/proj"

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_spawns_worker_and_redirects(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        # No models from Codex → reconcile is a no-op, session stays empty,
        # ``spawn_new_session`` sees the original ``model=None, effort=None``
        # contract.
        _stub_codex(mock_codex, models=[])

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "Refactor the login flow", "cwd": self._allowed_repo()},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            response.headers["Location"],
            reverse("session", kwargs={"session_id": "thread-xyz"}),
        )
        mock_spawn.assert_called_once_with(
            cwd=self._allowed_repo(),
            prompt="Refactor the login flow",
            model=None,
            reasoning_effort=None,
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_forwards_settings_session_to_spawn(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        """The dialog's picks ride into ``spawn_new_session`` via the
        request's session — this is how the signed-cookie-backed settings
        reach Codex without ever touching server-side storage."""
        mock_discover.return_value = [Path(self._allowed_repo())]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        # Codex still offers "gpt-5" with "high" in its supported list, so
        # reconciliation is a no-op and the saved pair flows through.
        _stub_codex(mock_codex, models=[_make_model("gpt-5", is_default=True)])
        self.client.cookies["hitch_model"] = signing.get_cookie_signer(
            salt="hitch_model"
        ).sign("gpt-5")
        self.client.cookies["hitch_reasoning_effort"] = signing.get_cookie_signer(
            salt="hitch_reasoning_effort"
        ).sign("high")

        self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self._allowed_repo()},
        )

        mock_spawn.assert_called_once_with(
            cwd=self._allowed_repo(),
            prompt="do thing",
            model="gpt-5",
            reasoning_effort="high",
        )

    @patch("hitch.main.views.Codex")
    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_reconciles_stale_model_before_spawning(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
        mock_codex: MagicMock,
    ) -> None:
        """A long-lived tab can POST with a session that names a model the
        running Codex no longer offers; this is where the reconcile catches
        it so ``thread_start(model=...)`` doesn't get a stale id."""
        mock_discover.return_value = [Path(self._allowed_repo())]
        mock_spawn.return_value = SimpleNamespace(thread_id="thread-xyz")
        _stub_codex(mock_codex, models=[_make_model("gpt-5", is_default=True)])
        self.client.cookies["hitch_model"] = signing.get_cookie_signer(
            salt="hitch_model"
        ).sign("ancient-model")
        self.client.cookies["hitch_reasoning_effort"] = signing.get_cookie_signer(
            salt="hitch_reasoning_effort"
        ).sign("low")

        self.client.post(
            reverse("new_session"),
            data={"prompt": "do thing", "cwd": self._allowed_repo()},
        )

        # The reconcile snapped the stale model back to the provider
        # default plus its default effort before spawn was called.
        mock_spawn.assert_called_once_with(
            cwd=self._allowed_repo(),
            prompt="do thing",
            model="gpt-5",
            reasoning_effort="medium",
        )

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_missing_prompt(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "", "cwd": self._allowed_repo()},
        )

        self.assertEqual(response.status_code, 400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_missing_cwd(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "hello", "cwd": ""},
        )

        self.assertEqual(response.status_code, 400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.codex_pool.spawn_new_session")
    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_unknown_cwd(
        self,
        mock_discover: MagicMock,
        mock_spawn: MagicMock,
    ) -> None:
        mock_discover.return_value = [Path(self._allowed_repo())]

        response = self.client.post(
            reverse("new_session"),
            data={"prompt": "hello", "cwd": "/etc"},
        )

        self.assertEqual(response.status_code, 400)
        mock_spawn.assert_not_called()

    @patch("hitch.main.views.discover_repos")
    def test_new_session_rejects_get(self, mock_discover: MagicMock) -> None:
        mock_discover.return_value = []
        response = self.client.get(reverse("new_session"))
        self.assertEqual(response.status_code, 405)
