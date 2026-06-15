import base64
import binascii
import json
from collections.abc import Callable
from typing import Any, NamedTuple

from django.core import signing
from django.http import HttpRequest, HttpResponse
from openai_codex.generated.v2_all import ReasoningEffort

from hitch.main import coding_agents
from hitch.main.models import ApprovalRequest, AutonomousGoal, Project

# Upper bound for ``CodexInstance.pk`` validation. The project sets
# ``DEFAULT_AUTO_FIELD = BigAutoField``, which is a signed 64-bit
# integer column. A POST'd value larger than this otherwise reaches
# the ORM and surfaces as a backend-specific OverflowError/DataError
# from ``objects.get`` — a 500 for what should be a clean 400.
_MAX_BIGAUTOFIELD = 2**63 - 1


class SettingsValues(NamedTuple):
    model: str
    reasoning_effort: str
    sandbox_policy: str
    approval_mode: str
    coding_agent: str
    extra_system_prompt: str
    use_worktrees: bool
    auto_pr_enabled: bool
    auto_qa_enabled: bool
    spec_critic_enabled: bool
    web_search_mode: str
    show_archived_sessions: bool
    last_selected_repo: str
    selected_project_id: int | None
    visible_session_project_ids: tuple[int, ...] | None
    show_no_project_sessions: bool
    enable_memories: bool
    provider: str = coding_agents.PROVIDER_CODEX


class SessionProjectVisibility(NamedTuple):
    project_ids: frozenset[int] | None
    include_no_project: bool


class ResolvedSettings(NamedTuple):
    values: SettingsValues
    cookie_updates: dict[str, str]


_SANDBOX_POLICY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("readOnly", "Read only"),
    ("workspaceWrite", "Workspace write"),
    ("dangerFullAccess", "Danger - full access"),
)
_VALID_SANDBOX_POLICIES = {value for value, _ in _SANDBOX_POLICY_OPTIONS}
_MANAGED_WORKTREE_DEFAULT_SANDBOX_POLICY = "workspaceWrite"

# Approval modes the dialog offers. ``auto_review`` and ``deny_all`` map 1:1
# onto the SDK's ``ApprovalMode`` enum. The ``prompt_user`` and
# ``approve_all`` modes are custom worker modes that force escalations to
# the client transport with the ``user`` reviewer; the worker either surfaces
# a browser prompt or rubber-stamps the request depending on the selected
# mode. ``auto_review`` is also the SDK's own default; keeping it first here
# makes it the safe default the dialog selects when no cookie has been
# written yet.
_APPROVAL_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto_review", "Auto review (default)"),
    ("prompt_user", "Always prompt for approval"),
    ("deny_all", "Deny all escalations"),
    ("approve_all", "Approve all (dangerous)"),
)
_VALID_APPROVAL_MODES = {value for value, _ in _APPROVAL_MODE_OPTIONS}
# Safe default: ``auto_review`` matches the SDK default and keeps an
# automated reviewer in the loop. Tampered/legacy cookie values fall back
# to this rather than to ``deny_all``, which would silently block the
# agent from ever escalating.
_DEFAULT_APPROVAL_MODE = "auto_review"
_PROMPT_USER_MODE = "prompt_user"
_APPROVE_ALL_MODE = "approve_all"
_DENY_ALL_MODE = "deny_all"
_LIVE_HANDLER_APPROVAL_MODES = frozenset(
    {_PROMPT_USER_MODE, _APPROVE_ALL_MODE, _DENY_ALL_MODE}
)
# Non-interactive live modes must unblock workers already waiting on a row.
_LIVE_PENDING_APPROVAL_DECISIONS_BY_MODE = {
    _APPROVE_ALL_MODE: ApprovalRequest.DECISION_ACCEPT,
    _DENY_ALL_MODE: ApprovalRequest.DECISION_DECLINE,
}

_WEB_SEARCH_MODE_OPTIONS = AutonomousGoal.WEB_SEARCH_CHOICES
_VALID_WEB_SEARCH_MODES = {
    value for value, _label in _WEB_SEARCH_MODE_OPTIONS if value
}

# Dedicated signed cookies for the settings dialog. Kept separate from
# Django's session cookie so the (non-revocable, long-lived) settings
# state never rides alongside an auth session — admin auth is allowed to
# stay DB-backed and therefore revocable on logout.
_MODEL_COOKIE = "hitch_model"
_EFFORT_COOKIE = "hitch_reasoning_effort"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_PROVIDER_COOKIE = "hitch_provider"
_CODING_AGENT_COOKIE = "hitch_coding_agent"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_AUTO_PR_COOKIE = "hitch_auto_pr"
_AUTO_QA_COOKIE = "hitch_auto_qa"
_SPEC_CRITIC_COOKIE = "hitch_spec_critic"
_WEB_SEARCH_COOKIE = "hitch_web_search_mode"
_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"
_LAST_SELECTED_REPO_COOKIE = "hitch_last_selected_repo"
_SELECTED_PROJECT_COOKIE = "hitch_selected_project_id"
_VISIBLE_SESSION_PROJECTS_COOKIE = "hitch_visible_session_project_ids"
_SHOW_NO_PROJECT_SESSIONS_COOKIE = "hitch_show_no_project_sessions"
_ENABLE_MEMORIES_COOKIE = "hitch_enable_memories"

# Roughly one year. Long enough that a user's pick survives across
# sessions without ever needing a manual revisit; short enough that the
# browser eventually evicts a stale value if the user stops using the app.
_COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# Cap on the posted model id so a crafted oversized POST can't push the
# cookie past the browser's 4KB limit (which would cause the browser to
# silently drop the cookie). Real Codex model ids are tens of chars; 256
# is comfortably more than that without leaving room for abuse.
_MODEL_MAX_LEN = 256

# UX-facing character cap on the developer prompt; mirrored by the textarea's
# ``maxlength``. This bounds the input length but is NOT the real guard against
# overflowing a cookie: the prompt is base64-encoded (and the whole cookie then
# signed), and a single multibyte character costs several UTF-8 bytes, so a
# prompt comfortably under this character count can still produce a cookie far
# past the browser limit. ``_extra_system_prompt_cookie_fits`` enforces the byte
# budget below; both checks run on save.
_EXTRA_SYSTEM_PROMPT_MAX_LEN = 2500

# RFC 6265 only guarantees ~4096 bytes per cookie (name + value + attributes).
# When a signed cookie crosses that line the browser silently drops it, so a
# setting we persist solely in a cookie (anonymous users) would vanish on the
# next request even though the save "succeeded". Budget conservatively under
# 4096 to leave room for the Max-Age/Path/SameSite attributes attached in
# ``_apply_cookie_updates``.
_COOKIE_MAX_VALUE_BYTES = 4000

_LAST_SELECTED_REPO_MAX_LEN = 4096


def _effective_coding_agent(settings: SettingsValues) -> str:
    if settings.coding_agent in coding_agents.VALID_CODING_AGENTS:
        return settings.coding_agent
    return coding_agents.DEFAULT_CODING_AGENT


def _effective_provider(settings: SettingsValues) -> str:
    if settings.provider in coding_agents.VALID_PROVIDERS:
        return settings.provider
    return coding_agents.DEFAULT_PROVIDER


def _option_label(
    options: tuple[tuple[str, str], ...], value: str, *, default: str | None = None
) -> str:
    if not value and default is not None:
        return default
    return next((label for option_value, label in options if option_value == value), value)


def _web_search_mode_label(value: str) -> str:
    return _option_label(_WEB_SEARCH_MODE_OPTIONS, value)


def _valid_web_search_mode_or_default(value: str) -> str:
    return value if value in _VALID_WEB_SEARCH_MODES else ""


def _settings_cookie_updates(values: SettingsValues) -> dict[str, str]:
    return {
        spec.cookie: spec.to_cookie(getattr(values, spec.field))
        for spec in _SETTING_SPECS
    }


def _valid_cookie_setting_updates(
    request: HttpRequest,
) -> dict[str, str | bool | int | list[int] | None]:
    updates: dict[str, str | bool | int | list[int] | None] = {}
    for spec in _SETTING_SPECS:
        raw = _read_signed_cookie_if_present(request, spec.cookie)
        if raw is None:
            continue
        value = spec.import_value(raw)
        if value is not _SKIP_IMPORT:
            updates[spec.field] = value
    return updates


def _read_signed_cookie_if_present(request: HttpRequest, name: str) -> str | None:
    if name not in request.COOKIES:
        return None
    try:
        value = request.get_signed_cookie(name)
    except Exception:
        return None
    return (value or "").strip()


def _encode_visible_session_project_ids_cookie(values: tuple[int, ...] | None) -> str:
    if values is None:
        return ""
    return json.dumps(list(values), separators=(",", ":"))


def _decode_visible_session_project_ids_cookie(value: str) -> object:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _valid_visible_session_project_ids(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    project_ids: list[int] = []
    seen: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        if item < 1 or item > _MAX_BIGAUTOFIELD or item in seen:
            continue
        seen.add(item)
        project_ids.append(item)
    return tuple(project_ids)


def _valid_selected_project_id(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        project_id = int(raw)
    except ValueError:
        return None
    if project_id < 1 or project_id > _MAX_BIGAUTOFIELD:
        return None
    return project_id if Project.objects.filter(pk=project_id).exists() else None


def _read_cookie(request: HttpRequest, name: str) -> str:
    """Read a signed cookie; return ``""`` on missing or tampered value.

    ``request.get_signed_cookie`` raises on a missing/invalid signature;
    we treat both as "no value" so a corrupt cookie just falls through to
    the reconcile path instead of 500ing the index render.
    """
    try:
        value = request.get_signed_cookie(name, default="")
    except Exception:
        return ""
    return (value or "").strip()


def _decode_extra_system_prompt_value(encoded: str) -> str:
    try:
        decoded = base64.urlsafe_b64decode(encoded.encode("ascii")).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        # Fallback for pre-encoding local cookies from development builds.
        decoded = encoded
    return decoded.strip()


def _encode_extra_system_prompt_cookie(value: str) -> str:
    if not value:
        return ""
    return base64.urlsafe_b64encode(value.encode()).decode("ascii")


class _SettingSpec(NamedTuple):
    """One settings field, declared once.

    Owns every representation of the field: the ``SettingsValues`` attribute
    (which doubles as the ``UserSettings`` column name), the signed cookie,
    and the codecs between them. The cookie write, the anonymous cookie
    read, the login-time cookie import, and the UserSettings load/save are
    all derived from ``_SETTING_SPECS``, so adding a setting means adding
    one spec here (plus the model column and the settings-dialog form
    handling) instead of hand-editing five parallel enumerations.

    ``import_value`` validates a raw cookie for the login-time import and
    returns a model-ready value, or ``_SKIP_IMPORT`` to drop the update
    (matching the historical per-field skip-vs-coerce semantics).
    """

    field: str
    cookie: str
    to_cookie: Callable[[Any], str]
    from_cookie: Callable[[str], Any]
    import_value: Callable[[str], Any]
    to_model: Callable[[Any], Any] = lambda value: value
    from_model: Callable[[Any], Any] = lambda value: value


_SKIP_IMPORT = object()


def _bool_to_cookie(value: Any) -> str:
    return "true" if value else "false"


def _bool_import(raw: str) -> Any:
    return raw == "true" if raw in ("true", "false") else _SKIP_IMPORT


def _bool_spec(field: str, cookie: str, *, default_true: bool = False) -> _SettingSpec:
    return _SettingSpec(
        field=field,
        cookie=cookie,
        to_cookie=_bool_to_cookie,
        from_cookie=(
            (lambda raw: raw != "false") if default_true else (lambda raw: raw == "true")
        ),
        import_value=_bool_import,
    )


def _import_extra_system_prompt(raw: str) -> Any:
    decoded = _decode_extra_system_prompt_value(raw)
    return decoded if len(decoded) <= _EXTRA_SYSTEM_PROMPT_MAX_LEN else _SKIP_IMPORT


def _import_visible_session_project_ids(raw: str) -> Any:
    valid = _valid_visible_session_project_ids(
        _decode_visible_session_project_ids_cookie(raw)
    )
    return list(valid) if valid is not None else None


_SETTING_SPECS: tuple[_SettingSpec, ...] = (
    _SettingSpec(
        "model",
        _MODEL_COOKIE,
        to_cookie=str,
        from_cookie=str,
        import_value=lambda raw: raw if len(raw) <= _MODEL_MAX_LEN else _SKIP_IMPORT,
    ),
    _SettingSpec(
        "reasoning_effort",
        _EFFORT_COOKIE,
        to_cookie=str,
        from_cookie=str,
        import_value=lambda raw: (
            raw
            if not raw or raw in {effort.value for effort in ReasoningEffort}
            else _SKIP_IMPORT
        ),
    ),
    _SettingSpec(
        "sandbox_policy",
        _SANDBOX_COOKIE,
        to_cookie=str,
        from_cookie=str,
        import_value=lambda raw: raw if raw in _VALID_SANDBOX_POLICIES else "",
    ),
    _SettingSpec(
        "approval_mode",
        _APPROVAL_COOKIE,
        to_cookie=str,
        from_cookie=str,
        import_value=lambda raw: (
            raw if raw in _VALID_APPROVAL_MODES else _DEFAULT_APPROVAL_MODE
        ),
    ),
    _SettingSpec(
        "provider",
        _PROVIDER_COOKIE,
        to_cookie=lambda value: (
            value
            if value in coding_agents.VALID_PROVIDERS
            else coding_agents.DEFAULT_PROVIDER
        ),
        from_cookie=str,
        import_value=lambda raw: (
            raw
            if raw in coding_agents.VALID_PROVIDERS
            else coding_agents.DEFAULT_PROVIDER
        ),
    ),
    _SettingSpec(
        "coding_agent",
        _CODING_AGENT_COOKIE,
        to_cookie=lambda value: (
            value
            if value in coding_agents.VALID_CODING_AGENTS
            else coding_agents.DEFAULT_CODING_AGENT
        ),
        from_cookie=str,
        import_value=lambda raw: (
            raw
            if raw in coding_agents.VALID_CODING_AGENTS
            else coding_agents.DEFAULT_CODING_AGENT
        ),
    ),
    _SettingSpec(
        "extra_system_prompt",
        _EXTRA_SYSTEM_PROMPT_COOKIE,
        to_cookie=_encode_extra_system_prompt_cookie,
        from_cookie=_decode_extra_system_prompt_value,
        import_value=_import_extra_system_prompt,
    ),
    _bool_spec("use_worktrees", _USE_WORKTREES_COOKIE),
    _bool_spec("auto_pr_enabled", _AUTO_PR_COOKIE),
    _bool_spec("auto_qa_enabled", _AUTO_QA_COOKIE),
    _bool_spec("spec_critic_enabled", _SPEC_CRITIC_COOKIE),
    _SettingSpec(
        "web_search_mode",
        _WEB_SEARCH_COOKIE,
        to_cookie=str,
        from_cookie=str,
        import_value=lambda raw: raw if raw in _VALID_WEB_SEARCH_MODES else "",
    ),
    _bool_spec("show_archived_sessions", _SHOW_ARCHIVED_COOKIE),
    _SettingSpec(
        "last_selected_repo",
        _LAST_SELECTED_REPO_COOKIE,
        to_cookie=str,
        from_cookie=str,
        import_value=lambda raw: (
            raw if len(raw) <= _LAST_SELECTED_REPO_MAX_LEN else _SKIP_IMPORT
        ),
    ),
    _SettingSpec(
        "selected_project_id",
        _SELECTED_PROJECT_COOKIE,
        to_cookie=lambda value: str(value) if value is not None else "",
        from_cookie=_valid_selected_project_id,
        import_value=_valid_selected_project_id,
    ),
    _SettingSpec(
        "visible_session_project_ids",
        _VISIBLE_SESSION_PROJECTS_COOKIE,
        to_cookie=_encode_visible_session_project_ids_cookie,
        from_cookie=lambda raw: _valid_visible_session_project_ids(
            _decode_visible_session_project_ids_cookie(raw)
        ),
        import_value=_import_visible_session_project_ids,
        to_model=lambda value: list(value) if value is not None else None,
        from_model=_valid_visible_session_project_ids,
    ),
    _bool_spec(
        "show_no_project_sessions",
        _SHOW_NO_PROJECT_SESSIONS_COOKIE,
        default_true=True,
    ),
    _bool_spec("enable_memories", _ENABLE_MEMORIES_COOKIE),
)

# Completeness guarantee: every SettingsValues field has exactly one spec.
# Forgetting to register a new setting fails at import time instead of
# silently dropping the field from one of the derived code paths.
assert {spec.field for spec in _SETTING_SPECS} == set(SettingsValues._fields), (
    "settings registry out of sync with SettingsValues"
)



def _signed_cookie_fits(name: str, value: str) -> bool:
    """Return whether the signed ``name=value`` cookie stays within the limit.

    Mirrors how ``_apply_cookie_updates`` writes the cookie (Django signs with
    a per-name salt) so the size we check is the size the browser would see.
    A cookie over the limit is silently dropped client-side, taking the value
    with it, so callers reject the input before it reaches that state.
    """
    signed = signing.get_cookie_signer(salt=name).sign(value)
    return len(f"{name}={signed}".encode()) <= _COOKIE_MAX_VALUE_BYTES


def _extra_system_prompt_cookie_fits(value: str) -> bool:
    return _signed_cookie_fits(
        _EXTRA_SYSTEM_PROMPT_COOKIE, _encode_extra_system_prompt_cookie(value)
    )


def _visible_session_project_ids_cookie_fits(values: tuple[int, ...]) -> bool:
    return _signed_cookie_fits(
        _VISIBLE_SESSION_PROJECTS_COOKIE,
        _encode_visible_session_project_ids_cookie(values),
    )


def _apply_cookie_updates(response: HttpResponse, updates: dict[str, str]) -> None:
    for name, value in updates.items():
        response.set_signed_cookie(
            name, value, max_age=_COOKIE_MAX_AGE, samesite="Lax"
        )
