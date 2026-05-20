import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.core import signing
from django.test import Client, TestCase
from django.urls import reverse
from openai_codex.errors import MethodNotFoundError
from openai_codex.generated.v2_all import ReasoningEffort

_MODEL_COOKIE = "hitch_model"
_EFFORT_COOKIE = "hitch_reasoning_effort"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_ENABLE_MEMORIES_COOKIE = "hitch_enable_memories"

# By default a test model accepts every enum value so tests that don't care
# about supported-effort filtering can stay terse; tests that exercise the
# narrowing logic pass an explicit ``supported_efforts``.
_ALL_EFFORT_VALUES = [e.value for e in ReasoningEffort]


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


def _signer(name: str) -> signing.Signer:
    return signing.get_cookie_signer(salt=name)


def _seed_cookies(client: Client, **values: str) -> None:
    for name, value in values.items():
        client.cookies[name] = _signer(name).sign(value)


def _cookie_value(response: object, name: str) -> str:
    """Pull a signed cookie's plaintext value out of a TestClient response."""
    raw = response.cookies[name].value  # type: ignore[attr-defined]
    return _signer(name).unsign(raw)


def _encode_extra_system_prompt(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode("ascii")


def _extra_system_prompt_value(response: object) -> str:
    raw = _cookie_value(response, _EXTRA_SYSTEM_PROMPT_COOKIE)
    return base64.urlsafe_b64decode(raw.encode("ascii")).decode()


class SettingsDialogRenderTests(TestCase):
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_dialog_lists_models_and_efforts(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-nav-menu")
        self.assertContains(response, "data-nav-menu-open")
        self.assertContains(response, "data-nav-menu-panel")
        self.assertContains(response, "data-settings-dialog")
        self.assertContains(response, 'aria-label="Navigation menu"')
        self.assertContains(response, f'href="{reverse("index")}"')
        self.assertContains(response, ">sessions<")
        self.assertContains(response, ">settings<")
        self.assertContains(response, f'href="{reverse("usage")}"')
        self.assertContains(response, ">usage<")
        self.assertContains(response, "GPT-5")
        # Spot-check a couple of reasoning-effort values so a future enum
        # rename can't quietly empty the dropdown.
        self.assertContains(response, 'value="minimal"')
        self.assertContains(response, 'value="high"')
        # The empty option is the only way the dialog surfaces the "let
        # Codex pick the model default" state; without it a user who once
        # set an explicit value has no UI path back to that state.
        self.assertContains(response, 'value=""')
        self.assertContains(response, "Model default")
        # Sandbox policy dropdown: surface the three policy variants plus
        # an empty "let Codex pick" option, so a future rename here can't
        # quietly drop the field from the dialog.
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
        self.assertContains(response, "Extra developer prompt")
        self.assertContains(response, 'name="extra_system_prompt"')
        self.assertContains(response, 'maxlength="2500"')
        self.assertContains(
            response, f'action="{reverse("update_archived_session_visibility")}"'
        )
        self.assertContains(response, "data-archived-visibility-form")
        self.assertContains(response, "data-archived-visibility-submit")
        self.assertContains(
            response,
            "requestSubmit(archivedVisibilityForm, archivedVisibilitySubmit)",
        )
        self.assertContains(response, "Show archived")
        self.assertNotContains(response, "Show archived sessions")
        self.assertContains(response, 'name="use_worktrees"')
        self.assertContains(response, "Use worktrees")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_saved_extra_system_prompt_renders_in_dialog(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(
            self.client,
            **{
                _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt(
                    "Prefer small diffs."
                )
            },
        )
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, "Prefer small diffs.")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_saved_worktree_setting_renders_checked(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _seed_cookies(self.client, **{_USE_WORKTREES_COOKIE: "true"})
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, 'name="use_worktrees" value="true" checked')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

        response = self.client.get(reverse("index"))

        self.assertContains(response, 'value="" selected')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_dialog_captures_scroll_without_scrolling_session_list(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, display_name="GPT-5")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, "overscroll-behavior: contain")
        self.assertContains(response, "overflow-y: auto")
        self.assertContains(response, "modal-scroll-locked")
        self.assertContains(response, 'document.querySelector("dialog[open]")')

    @patch("hitch.main.views.Codex")
    def test_usage_page_renders_rate_limit_windows(self, mock_codex: MagicMock) -> None:
        """When the account/rateLimits/read call returns a snapshot, the
        usage page must render each present window so a user can see how
        much of their budget is left before kicking off a new turn."""
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

    @patch("hitch.main.views.Codex")
    def test_usage_page_hides_rate_limits_when_unsupported(
        self, mock_codex: MagicMock
    ) -> None:
        """Local-dev (ollama) and older Codex builds reject the rate-limits
        method; the usage page must still render with an empty state."""
        _configure_codex(
            mock_codex,
            models=[],
            # explicit MethodNotFound is the typical signal from Codex when
            # the endpoint isn't wired in the current build.
            rate_limits=MethodNotFoundError(-32601, "method not found", None),
        )

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Usage unavailable.")
        self.assertNotContains(response, "% remaining")

    @patch("hitch.main.views.Codex")
    def test_usage_page_hides_rate_limits_on_unexpected_exception(
        self, mock_codex: MagicMock
    ) -> None:
        """Non-Codex exceptions (pydantic ValidationError on a malformed
        wire payload, transport hiccups not wrapped as AppServerError) must
        also be swallowed so usage can show an empty state."""
        _configure_codex(
            mock_codex,
            models=[],
            rate_limits=ValueError("malformed payload"),
        )

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Usage unavailable.")
        self.assertNotContains(response, "% remaining")

    @patch("hitch.main.views.Codex")
    def test_usage_page_hides_rate_limits_when_both_windows_empty(
        self, mock_codex: MagicMock
    ) -> None:
        """An account that has no metered usage at all returns a snapshot
        with both windows unset; show the empty state rather than render an
        empty header."""
        _configure_codex(
            mock_codex,
            models=[],
            rate_limits=_rate_limit_snapshot(),
        )

        response = self.client.get(reverse("usage"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Usage unavailable.")
        self.assertNotContains(response, "% remaining")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_renders_when_codex_offers_no_models(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # An empty model list is the pre-provider local-dev story; the dialog
        # must still render so the user sees the empty state instead of a 500.
        _configure_codex(mock_codex, models=[])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Codex returned no models")


class ReconcileSettingsTests(TestCase):
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_seeds_defaults_when_no_cookies(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        _configure_codex(
            mock_codex,
            models=[
                _model("other"),
                _model("gpt-5", is_default=True, default_effort="high"),
            ],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        # Defaults are written back to the browser so the next request has
        # them in hand — the "reset on server start based on what Codex
        # provides" behavior expressed through signed cookies.
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "high")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_resets_stale_model_to_current_default(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        # The user had previously picked a model that the running Codex no
        # longer offers (e.g. provider swapped); the index render must snap
        # the cookie back to a valid choice.
        _seed_cookies(
            self.client, **{_MODEL_COOKIE: "removed-model", _EFFORT_COOKIE: "minimal"}
        )
        _configure_codex(
            mock_codex,
            models=[_model("gpt-5", is_default=True, default_effort="medium")],
        )
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")
        self.assertEqual(_cookie_value(response, _EFFORT_COOKIE), "medium")

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

        response = self.client.get(reverse("index"))

        # Saved values are still valid → no Set-Cookie on this response.
        self.assertNotIn(_MODEL_COOKIE, response.cookies)
        self.assertNotIn(_EFFORT_COOKIE, response.cookies)
        self.assertContains(response, 'value="gpt-5"')
        self.assertContains(response, 'value="low" selected')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
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

        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(_cookie_value(response, _MODEL_COOKIE), "gpt-5")


class UpdateSettingsViewTests(TestCase):
    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
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

    def test_rejects_get(self) -> None:
        response = self.client.get(reverse("update_settings"))
        self.assertEqual(response.status_code, 405)

    @patch("hitch.main.views.Codex")
    def test_saves_optional_signed_cookie_settings(
        self, mock_codex: MagicMock
    ) -> None:
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
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
                "memories enabled",
                {"enable_memories": "true"},
                {},
                _ENABLE_MEMORIES_COOKIE,
                "true",
            ),
            ("memories disabled", {}, {}, _ENABLE_MEMORIES_COOKIE, "false"),
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
        ]
        for label, cookie, saved_value, data in cases:
            with self.subTest(label=label):
                client = Client()
                _seed_cookies(client, **{cookie: saved_value})

                response = client.post(reverse("update_settings"), data=data)

                self.assertEqual(response.status_code, 400)
                self.assertNotIn(cookie, response.cookies)


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


class ApprovalModeSettingsTests(TestCase):
    @patch("hitch.main.views.Codex")
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


class ApprovalModeDialogTests(TestCase):
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_saved_approval_renders_as_selected(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A mode persisted in the cookie must come back marked selected
        on the next render — otherwise the dropdown silently rolls back
        to the safe default and the user assumes the pick was lost."""
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        for mode in ("prompt_user", "deny_all"):
            with self.subTest(mode=mode):
                _seed_cookies(self.client, **{_APPROVAL_COOKIE: mode})

                response = self.client.get(reverse("index"))

                self.assertContains(response, f'value="{mode}" selected')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_unknown_approval_cookie_falls_back_to_safe_default(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A legacy/tampered cookie value must not render as a phantom
        option; the dialog snaps back to the safe default so the user has
        a coherent UI to recover from."""
        _seed_cookies(self.client, **{_APPROVAL_COOKIE: "phantomMode"})
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertNotContains(response, "phantomMode")
        self.assertContains(response, 'value="auto_review" selected')


class SandboxPolicyDialogTests(TestCase):
    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_saved_sandbox_renders_as_selected(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A policy persisted in the cookie must come back marked selected
        on the next render — otherwise the dropdown silently rolls back to
        the empty default and the user assumes the pick was lost."""
        _seed_cookies(self.client, **{_SANDBOX_COOKIE: "dangerFullAccess"})
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertContains(response, 'value="dangerFullAccess" selected')

    @patch("hitch.main.views.discover_repos")
    @patch("hitch.main.views.Codex")
    def test_unknown_sandbox_cookie_falls_back_to_empty(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        """A legacy/tampered cookie value must not render as a phantom
        selected option; the dialog snaps back to the empty "Codex
        default" state so the user has a coherent UI to recover from."""
        _seed_cookies(self.client, **{_SANDBOX_COOKIE: "phantomPolicy"})
        _configure_codex(mock_codex, models=[_model("gpt-5", is_default=True)])
        mock_discover.return_value = []

        response = self.client.get(reverse("index"))

        self.assertNotContains(response, "phantomPolicy")
        self.assertContains(response, 'value="" selected')
