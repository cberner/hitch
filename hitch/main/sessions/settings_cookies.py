import base64
import binascii
import json
from typing import NamedTuple

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
        _MODEL_COOKIE: values.model,
        _EFFORT_COOKIE: values.reasoning_effort,
        _SANDBOX_COOKIE: values.sandbox_policy,
        _APPROVAL_COOKIE: values.approval_mode,
        _CODING_AGENT_COOKIE: _effective_coding_agent(values),
        _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt_cookie(
            values.extra_system_prompt
        ),
        _USE_WORKTREES_COOKIE: "true" if values.use_worktrees else "false",
        _AUTO_PR_COOKIE: "true" if values.auto_pr_enabled else "false",
        _AUTO_QA_COOKIE: "true" if values.auto_qa_enabled else "false",
        _SPEC_CRITIC_COOKIE: "true" if values.spec_critic_enabled else "false",
        _WEB_SEARCH_COOKIE: values.web_search_mode,
        _SHOW_ARCHIVED_COOKIE: "true" if values.show_archived_sessions else "false",
        _LAST_SELECTED_REPO_COOKIE: values.last_selected_repo,
        _SELECTED_PROJECT_COOKIE: (
            str(values.selected_project_id) if values.selected_project_id is not None else ""
        ),
        _VISIBLE_SESSION_PROJECTS_COOKIE: _encode_visible_session_project_ids_cookie(
            values.visible_session_project_ids
        ),
        _SHOW_NO_PROJECT_SESSIONS_COOKIE: (
            "true" if values.show_no_project_sessions else "false"
        ),
        _ENABLE_MEMORIES_COOKIE: "true" if values.enable_memories else "false",
    }


def _valid_cookie_setting_updates(
    request: HttpRequest,
) -> dict[str, str | bool | int | list[int] | None]:
    updates: dict[str, str | bool | int | list[int] | None] = {}
    model = _read_signed_cookie_if_present(request, _MODEL_COOKIE)
    if model is not None and len(model) <= _MODEL_MAX_LEN:
        updates["model"] = model
    effort = _read_signed_cookie_if_present(request, _EFFORT_COOKIE)
    if effort is not None and (not effort or effort in {e.value for e in ReasoningEffort}):
        updates["reasoning_effort"] = effort
    sandbox = _read_signed_cookie_if_present(request, _SANDBOX_COOKIE)
    if sandbox is not None:
        updates["sandbox_policy"] = sandbox if sandbox in _VALID_SANDBOX_POLICIES else ""
    approval = _read_signed_cookie_if_present(request, _APPROVAL_COOKIE)
    if approval is not None:
        updates["approval_mode"] = (
            approval if approval in _VALID_APPROVAL_MODES else _DEFAULT_APPROVAL_MODE
        )
    coding_agent = _read_signed_cookie_if_present(request, _CODING_AGENT_COOKIE)
    if coding_agent is not None:
        updates["coding_agent"] = (
            coding_agent
            if coding_agent in coding_agents.VALID_CODING_AGENTS
            else coding_agents.DEFAULT_CODING_AGENT
        )
    extra_prompt = _read_signed_cookie_if_present(request, _EXTRA_SYSTEM_PROMPT_COOKIE)
    if extra_prompt is not None:
        decoded = _decode_extra_system_prompt_value(extra_prompt)
        if len(decoded) <= _EXTRA_SYSTEM_PROMPT_MAX_LEN:
            updates["extra_system_prompt"] = decoded
    show_archived = _read_signed_cookie_if_present(request, _SHOW_ARCHIVED_COOKIE)
    if show_archived in {"true", "false"}:
        updates["show_archived_sessions"] = show_archived == "true"
    use_worktrees = _read_signed_cookie_if_present(request, _USE_WORKTREES_COOKIE)
    if use_worktrees in {"true", "false"}:
        updates["use_worktrees"] = use_worktrees == "true"
    auto_pr = _read_signed_cookie_if_present(request, _AUTO_PR_COOKIE)
    if auto_pr in {"true", "false"}:
        updates["auto_pr_enabled"] = auto_pr == "true"
    auto_qa = _read_signed_cookie_if_present(request, _AUTO_QA_COOKIE)
    if auto_qa in {"true", "false"}:
        updates["auto_qa_enabled"] = auto_qa == "true"
    spec_critic = _read_signed_cookie_if_present(request, _SPEC_CRITIC_COOKIE)
    if spec_critic in {"true", "false"}:
        updates["spec_critic_enabled"] = spec_critic == "true"
    web_search = _read_signed_cookie_if_present(request, _WEB_SEARCH_COOKIE)
    if web_search is not None:
        updates["web_search_mode"] = (
            web_search if web_search in _VALID_WEB_SEARCH_MODES else ""
        )
    last_selected_repo = _read_signed_cookie_if_present(
        request, _LAST_SELECTED_REPO_COOKIE
    )
    if (
        last_selected_repo is not None
        and len(last_selected_repo) <= _LAST_SELECTED_REPO_MAX_LEN
    ):
        updates["last_selected_repo"] = last_selected_repo
    selected_project_raw = _read_signed_cookie_if_present(request, _SELECTED_PROJECT_COOKIE)
    if selected_project_raw is not None:
        updates["selected_project_id"] = _valid_selected_project_id(selected_project_raw)
    visible_projects_raw = _read_signed_cookie_if_present(
        request, _VISIBLE_SESSION_PROJECTS_COOKIE
    )
    if visible_projects_raw is not None:
        visible_project_ids = _valid_visible_session_project_ids(
            _decode_visible_session_project_ids_cookie(visible_projects_raw)
        )
        updates["visible_session_project_ids"] = (
            list(visible_project_ids) if visible_project_ids is not None else None
        )
    show_no_project = _read_signed_cookie_if_present(
        request, _SHOW_NO_PROJECT_SESSIONS_COOKIE
    )
    if show_no_project in {"true", "false"}:
        updates["show_no_project_sessions"] = show_no_project == "true"
    enable_memories = _read_signed_cookie_if_present(request, _ENABLE_MEMORIES_COOKIE)
    if enable_memories in {"true", "false"}:
        updates["enable_memories"] = enable_memories == "true"
    return updates


def _read_signed_cookie_if_present(request: HttpRequest, name: str) -> str | None:
    if name not in request.COOKIES:
        return None
    try:
        value = request.get_signed_cookie(name)
    except Exception:
        return None
    return (value or "").strip()


def _read_selected_project_cookie(request: HttpRequest) -> int | None:
    return _valid_selected_project_id(
        _read_signed_cookie_if_present(request, _SELECTED_PROJECT_COOKIE)
    )


def _read_visible_session_project_ids_cookie(
    request: HttpRequest,
) -> tuple[int, ...] | None:
    raw = _read_signed_cookie_if_present(request, _VISIBLE_SESSION_PROJECTS_COOKIE)
    if raw is None:
        return None
    return _valid_visible_session_project_ids(
        _decode_visible_session_project_ids_cookie(raw)
    )


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


def _read_extra_system_prompt_cookie(request: HttpRequest) -> str:
    encoded = _read_cookie(request, _EXTRA_SYSTEM_PROMPT_COOKIE)
    if not encoded:
        return ""
    return _decode_extra_system_prompt_value(encoded)


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
