import base64
import binascii
import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlencode

from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    StreamingHttpResponse,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_http_methods
from openai_codex import AppServerConfig, AppServerError, Codex
from openai_codex.generated.v2_all import (
    GetAccountRateLimitsResponse,
    RateLimitSnapshot,
    ReasoningEffort,
)

from hitch.main import codex_events, codex_pool, rollout, streaming
from hitch.main.diffs import build_worktree_diff
from hitch.main.formatting import looks_like_markdown, render_markdown
from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    UserInputRequest,
    UserSettings,
)
from hitch.main.repos import discover_repos
from hitch.main.worktrees import (
    WorktreeCleanupError,
    WorktreeCreationError,
    cleanup_worktree,
    create_worktree_for_session,
    discover_managed_worktrees,
)

logger = logging.getLogger(__name__)


class SettingsValues(NamedTuple):
    model: str
    reasoning_effort: str
    sandbox_policy: str
    approval_mode: str
    extra_system_prompt: str
    use_worktrees: bool
    show_archived_sessions: bool

# Sandbox-policy variants offered in the settings dialog. Stored as the
# SandboxPolicy ``type`` discriminator string so the cookie value can map
# 1:1 onto a constructed SandboxPolicy in the worker without any further
# translation. We omit ``externalSandbox`` because it requires a host
# sandbox the in-process worker doesn't speak; the three values below are
# the union of variants the codex CLI ships out of the box.
_SANDBOX_POLICY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("readOnly", "Read only"),
    ("workspaceWrite", "Workspace write"),
    ("dangerFullAccess", "Danger - full access"),
)
_VALID_SANDBOX_POLICIES = {value for value, _ in _SANDBOX_POLICY_OPTIONS}

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

# Dedicated signed cookies for the settings dialog. Kept separate from
# Django's session cookie so the (non-revocable, long-lived) settings
# state never rides alongside an auth session — admin auth is allowed to
# stay DB-backed and therefore revocable on logout.
_MODEL_COOKIE = "hitch_model"
_EFFORT_COOKIE = "hitch_reasoning_effort"
_SANDBOX_COOKIE = "hitch_sandbox_policy"
_APPROVAL_COOKIE = "hitch_approval_mode"
_EXTRA_SYSTEM_PROMPT_COOKIE = "hitch_extra_system_prompt"
_USE_WORKTREES_COOKIE = "hitch_use_worktrees"
_SHOW_ARCHIVED_COOKIE = "hitch_show_archived_sessions"

# Roughly one year. Long enough that a user's pick survives across
# sessions without ever needing a manual revisit; short enough that the
# browser eventually evicts a stale value if the user stops using the app.
_COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# Cap on the posted model id so a crafted oversized POST can't push the
# cookie past the browser's 4KB limit (which would cause the browser to
# silently drop the cookie). Real Codex model ids are tens of chars; 256
# is comfortably more than that without leaving room for abuse.
_MODEL_MAX_LEN = 256

# The prompt is base64-encoded inside a signed cookie, so keep enough room for
# encoding/signing overhead and browser cookie limits.
_EXTRA_SYSTEM_PROMPT_MAX_LEN = 2500

# Upper bound on what we render inline as a session's title. Codex does not
# generate its own thread summaries, so for unnamed threads `Thread.preview`
# (the full first user message) is what we get; that is often paragraphs
# long and would overflow the list rows without a clip.
_DISPLAY_TITLE_MAX_LEN = 80
_ARCHIVED_SESSIONS_DIR = "archived_sessions"
_MINUTES_PER_HOUR = 60
_MINUTES_PER_DAY = 24 * _MINUTES_PER_HOUR

# Server-side cap on user-supplied thread names. Matches the `maxlength` we
# set on the edit form so a client without HTML validation cannot push an
# unbounded blob through.
_NAME_MAX_LEN = 200

# Upper bound for ``CodexInstance.pk`` validation. The project sets
# ``DEFAULT_AUTO_FIELD = BigAutoField``, which is a signed 64-bit
# integer column. A POST'd value larger than this otherwise reaches
# the ORM and surfaces as a backend-specific OverflowError/DataError
# from ``objects.get`` — a 500 for what should be a clean 400.
_MAX_BIGAUTOFIELD = 2**63 - 1
_PLAN_SLASH_COMMAND = "/plan"
_PR_SLASH_COMMAND = "/pr"
_PR_SLASH_PROMPT = "Do a thorough review of the diff. Clean it up, and then open a PR"
_PLAN_MODE_REASONING_EFFORT = ReasoningEffort.medium.value

# Friendly labels for non-message thread item types. Anything not in this map
# falls back to the raw type tag so we never silently drop an item from the UI.
_NON_MESSAGE_LABELS = {
    "commandExecution": "Command",
    "mcpToolCall": "MCP tool call",
    "dynamicToolCall": "Tool call",
    "fileChange": "File change",
    "webSearch": "Web search",
    "collabAgentToolCall": "Collab agent call",
    "imageView": "Image view",
    "imageGeneration": "Image generation",
    "reasoning": "Reasoning",
    "plan": "Plan",
    "hookPrompt": "Hook prompt",
    "enteredReviewMode": "Entered review mode",
    "exitedReviewMode": "Exited review mode",
    "contextCompaction": "Context compaction",
}


def index(request: HttpRequest) -> HttpResponse:
    # Sweep workers whose pid is gone: a Popen that crashed before a worker
    # could record its terminal status (or a row stuck in ``starting``)
    # otherwise stays pending forever, since we don't run a periodic task.
    codex_pool.reconcile_dead()
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        models_data = list(codex.models().data)
        (
            current_model,
            current_effort,
            current_sandbox,
            current_approval,
            current_extra_system_prompt,
            current_use_worktrees,
            current_show_archived_sessions,
            cookie_updates,
        ) = _resolved_settings(request, models_data)
        threads = list(codex.thread_list().data)
        if current_show_archived_sessions:
            threads.extend(codex.thread_list(archived=True).data)
        rate_limits = _fetch_rate_limits(codex)
    threads = sorted(threads, key=lambda s: s.updated_at, reverse=True)
    sessions = []
    for thread in threads:
        is_archived = _thread_is_archived(thread)
        if is_archived and not current_show_archived_sessions:
            continue
        sessions.append(
            {
                "id": thread.id,
                "cwd": thread.cwd,
                "updated_at": thread.updated_at,
                "display_title": _display_title(thread),
                "is_archived": is_archived,
            }
        )
    repos = [str(p) for p in discover_repos()]
    model_options = [
        {"id": m.id, "display_name": m.display_name} for m in models_data
    ]
    effort_options = [effort.value for effort in ReasoningEffort]
    sandbox_options = [
        {"id": value, "display_name": label}
        for value, label in _SANDBOX_POLICY_OPTIONS
    ]
    approval_options = [
        {"id": value, "display_name": label}
        for value, label in _APPROVAL_MODE_OPTIONS
    ]
    response = render(
        request,
        "index.html",
        {
            "sessions": sessions,
            "repos": repos,
            "new_session_url": reverse("new_session"),
            "settings_url": reverse("update_settings"),
            "login_url": reverse("login"),
            "logout_url": reverse("logout"),
            "register_url": reverse("register"),
            "model_options": model_options,
            "effort_options": effort_options,
            "sandbox_options": sandbox_options,
            "approval_options": approval_options,
            "current_model": current_model,
            "current_effort": current_effort,
            "current_sandbox": current_sandbox,
            "current_approval": current_approval,
            "current_extra_system_prompt": current_extra_system_prompt,
            "extra_system_prompt_max_len": _EXTRA_SYSTEM_PROMPT_MAX_LEN,
            "current_use_worktrees": current_use_worktrees,
            "current_show_archived_sessions": current_show_archived_sessions,
            "rate_limits": rate_limits,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response


def session(request: HttpRequest, session_id: str) -> HttpResponse:
    # Sweep stuck workers before reading status: a worker that died without
    # writing a terminal status would otherwise leave the page in "streaming"
    # mode forever, since the EventSource wouldn't reach an end event.
    codex_pool.reconcile_dead()
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    settings = _stored_settings(request)
    with Codex(config=config) as codex:
        # ``thread/read`` only works for threads already loaded into the
        # app-server's in-memory map. Each request spawns a fresh app-server
        # subprocess, so newly-created threads (or any thread persisted by a
        # different worker) need ``thread/resume`` to read them off disk.
        # The resume response already carries the full thread including turns,
        # so a follow-up ``thread/read`` would just be a redundant round-trip.
        resumed = codex._client.thread_resume(session_id)
        thread = resumed.thread
        plan_model = _plan_mode_model(codex, resumed, settings)
    is_archived = _thread_is_archived(thread)
    entries = list(_entries_for(thread))
    name_value = getattr(thread, "name", None) or ""
    active_instance = _active_instance_for(session_id)
    # While a worker is running, drop the entries that belong to its
    # in-progress turn — the SSE stream replays them from byte 0 of the
    # events file, so leaving the rollout-rendered copy in place would
    # double up every entry in the live DOM. The page reload on stream end
    # restores the canonical view.
    entries = _trim_in_progress_turn(entries, active_instance)
    token_usage = _token_usage_for(thread)
    goal_objective = codex_events.latest_goal_for_thread(session_id)
    diff_view = build_worktree_diff(_thread_cwd(thread))
    return render(
        request,
        "session.html",
        {
            "thread": thread,
            "entries": entries,
            "display_title": _display_title(thread),
            "name_value": name_value,
            "name_max_len": _NAME_MAX_LEN,
            "set_name_url": reverse("set_session_name", kwargs={"session_id": session_id}),
            "set_archived_url": reverse(
                "set_session_archived", kwargs={"session_id": session_id}
            ),
            "is_archived": is_archived,
            "send_message_url": reverse("send_message", kwargs={"session_id": session_id}),
            "stop_url": reverse("stop_session", kwargs={"session_id": session_id}),
            # Pin the stream to the specific worker shown on this page
            # so a newer turn starting between render and EventSource
            # connect can't divert the live view away from the worker
            # the Stop button is wired to.
            "stream_url": _stream_url_for(session_id, active_instance),
            # The JS swaps the trailing ``0`` for the real ApprovalRequest
            # pk on each POST. Templating the URL server-side (rather than
            # building it in JS from a base path) keeps Django's URL
            # resolver authoritative even when the route changes.
            "approval_url_template": reverse(
                "resolve_approval", kwargs={"approval_id": 0}
            ),
            "input_url_template": reverse(
                "resolve_input_request", kwargs={"input_id": 0}
            ),
            "active_worker": active_instance is not None,
            # Carried into the Stop button so the click targets the
            # specific worker the page is streaming, not "whichever
            # worker is latest at click time" — overlapping turns can
            # stack two active workers on the same thread.
            "active_instance": active_instance,
            # The in-progress turn is trimmed from ``entries`` above, so the
            # user wouldn't see their own message at all without a pending
            # bubble while the stream catches up.
            "pending_user_prompt": _pending_user_prompt(active_instance),
            "token_usage": token_usage,
            "next_message_config": _next_message_config(settings, resumed, plan_model),
            "goal_objective": goal_objective,
            "diff_view": diff_view,
        },
    )


@require_http_methods(["GET", "POST"])
def register(request: HttpRequest) -> HttpResponse:
    if _authenticated_user(request) is not None:
        return redirect("index")
    form: Any
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            _import_cookie_settings_to_user(request, user)
            auth_login(request, user)
            response = redirect("index")
            _apply_cookie_updates(
                response, _settings_cookie_updates(_stored_settings(request))
            )
            return response
    else:
        form = UserCreationForm()
    return render(
        request,
        "register.html",
        {"form": form, "login_url": reverse("login")},
    )


@require_http_methods(["GET", "POST"])
def login(request: HttpRequest) -> HttpResponse:
    if _authenticated_user(request) is not None:
        return redirect("index")
    next_url = _safe_next_url(request)
    form: Any
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            _import_cookie_settings_to_user(request, user)
            auth_login(request, user)
            response = redirect(next_url or "index")
            _apply_cookie_updates(
                response, _settings_cookie_updates(_stored_settings(request))
            )
            return response
    else:
        form = AuthenticationForm(request)
    return render(
        request,
        "login.html",
        {
            "form": form,
            "next": next_url,
            "register_url": reverse("register"),
        },
    )


@require_http_methods(["POST"])
def logout(request: HttpRequest) -> HttpResponse:
    values = _stored_settings(request) if _authenticated_user(request) is not None else None
    auth_logout(request)
    response = redirect("index")
    if values is not None:
        _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


def _thread_is_archived(thread: Any) -> bool:
    """Return whether Codex resumed this thread from archived rollout storage."""
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return False
    return _ARCHIVED_SESSIONS_DIR in Path(path).parts


def _token_usage_for(thread: Any) -> dict[str, str] | None:
    """Return formatted input/cached/output token counts, or None.

    Token usage is only persisted in the on-disk rollout file (as
    ``TokenCount`` event_msg entries); the SDK ``Thread`` does not carry it.
    Returns None when the thread has no rollout path or the rollout contains
    no token_count event yet — the template hides the section in that case.
    """
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return None
    rollout_path = Path(path)
    if not rollout_path.is_file():
        return None
    usage = rollout.latest_token_usage(rollout_path)
    if usage is None:
        return None
    formatted = {
        "input": f"{usage['input_tokens']:,}",
        "cached": f"{usage['cached_input_tokens']:,}",
        "output": f"{usage['output_tokens']:,}",
    }
    context_tokens = usage.get("context_tokens", 0)
    context_window = usage.get("model_context_window", 0)
    if context_tokens > 0 and context_window > 0:
        percent = round((context_tokens / context_window) * 100)
        percent = min(100, max(0, percent))
        formatted.update(
            {
                "context": f"{percent}%",
                "context_title": f"{context_tokens:,} of {context_window:,} tokens in current context",
            }
        )
    return formatted


def _next_message_config(
    settings: SettingsValues, resumed: Any, plan_model: str | None
) -> list[dict[str, str]]:
    """Return the settings that will govern the next submitted message."""
    model = _string_value(getattr(resumed, "model", None))
    reasoning = _string_value(getattr(resumed, "reasoning_effort", None))
    plan_model_value = plan_model or "Unknown"
    sandbox_value = _option_label(
        _SANDBOX_POLICY_OPTIONS,
        _effective_sandbox_policy(settings),
        default="Codex default",
    )
    approval_value = _option_label(
        _APPROVAL_MODE_OPTIONS, _effective_approval_mode(settings)
    )
    return [
        {"label": "model", "value": model or "Unknown", "plan_value": plan_model_value},
        {
            "label": "reasoning",
            "value": reasoning or "Model default",
            "plan_value": _PLAN_MODE_REASONING_EFFORT,
        },
        {
            "label": "sandbox",
            "value": sandbox_value,
            "plan_value": sandbox_value,
        },
        {
            "label": "approval",
            "value": approval_value,
            "plan_value": approval_value,
        },
    ]


def _effective_sandbox_policy(settings: SettingsValues) -> str:
    sandbox_policy = settings.sandbox_policy
    if sandbox_policy and sandbox_policy not in _VALID_SANDBOX_POLICIES:
        return ""
    return sandbox_policy


def _effective_approval_mode(settings: SettingsValues) -> str:
    if settings.approval_mode not in _VALID_APPROVAL_MODES:
        return _DEFAULT_APPROVAL_MODE
    return settings.approval_mode


def _string_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return raw.strip() if isinstance(raw, str) else ""


def _option_label(
    options: tuple[tuple[str, str], ...], value: str, *, default: str | None = None
) -> str:
    if not value and default is not None:
        return default
    return next((label for option_value, label in options if option_value == value), value)


def _active_instance_for(session_id: str) -> CodexInstance | None:
    """Return the latest *active* CodexInstance for ``session_id``, or None.

    Selecting on status first (rather than picking the newest row and
    checking its status) means a quickly-terminal newer row doesn't mask
    an older worker that's still mid-turn — ``send_message`` can stack
    workers, so the page must stay in streaming mode as long as any one
    of them is alive.
    """
    return codex_pool.latest_active_for_thread(session_id)


def _trim_in_progress_turn(
    entries: list[dict[str, Any]], active: CodexInstance | None
) -> list[dict[str, Any]]:
    """Drop the in-progress turn's entries from the tail of ``entries``.

    The SSE stream re-emits every event from the start of the worker's
    events file, including the user message and any agent / tool items
    the rollout has already captured. Without this trim those entries
    render twice on the live page — once from the server-side rollout
    pass, once by the streaming JS that can't dedupe against DOM nodes
    it didn't create.

    The in-progress turn is identified by the most recent user-message
    entry whose text matches the active worker's prompt; anything from
    that point onward is owned by the stream until the turn ends.
    """
    if active is None or not active.prompt:
        return entries
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.get("kind") == "user" and entry.get("text") == active.prompt:
            return entries[:i]
    return entries


def _pending_user_prompt(active: CodexInstance | None) -> str:
    """Surface the active worker's prompt as a pending user bubble.

    Pairs with ``_trim_in_progress_turn``: that helper strips the
    in-progress turn from the rollout-rendered entries (so the stream
    owns rendering it), which means the user wouldn't see their own
    message at all between Send and the first stream event without this
    placeholder. The streaming JS removes the bubble as soon as the
    real ``userMessage`` event lands.
    """
    if active is None or not active.prompt:
        return ""
    return active.prompt


@require_http_methods(["GET"])
def session_stream(request: HttpRequest, session_id: str) -> StreamingHttpResponse:
    """SSE endpoint that mirrors the active worker's events file to the browser.

    When no worker is active the connection stays open emitting heartbeat
    frames so the page's connection indicator can show ``connected, idle``,
    and reloads itself when a worker is later spawned out-of-band.

    The page passes its render-time view of the session state on the URL
    (``baseline`` = latest ``CodexInstance.pk``, ``active`` = the active
    worker's pk if any). If either differs from what the database shows
    when SSE opens, the page is by definition stale (e.g. a worker was
    spawned/completed in the gap, or has already finished by the time
    the browser opens SSE) — we force an immediate reload so the DOM
    matches reality before any item events start flowing.
    """
    baseline_param = request.GET.get("baseline", "")
    active_param = request.GET.get("active", "")
    current_latest = codex_pool.latest_id_for_thread(session_id)
    current_latest_str = str(current_latest) if current_latest is not None else ""
    active = _active_instance_for(session_id)
    current_active_str = str(active.pk) if active is not None else ""

    if baseline_param != current_latest_str or active_param != current_active_str:
        response = StreamingHttpResponse(
            streaming.reload_stream(), content_type="text/event-stream"
        )
    elif active is None:
        response = StreamingHttpResponse(
            streaming.idle_stream(session_id, current_latest),
            content_type="text/event-stream",
        )
    else:
        response = StreamingHttpResponse(
            streaming.stream_for_instance(active), content_type="text/event-stream"
        )
    # Discourage proxies from buffering: SSE depends on every frame reaching
    # the client immediately, not coalesced into a single response body.
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _stream_url_for(session_id: str, active_instance: CodexInstance | None) -> str:
    """Build the SSE URL for the session view, tagging it with the page's
    render-time view of the session state.

    Pairs with ``session_stream``: those query params are how the SSE
    endpoint detects that the page was rendered against a stale DB view
    (worker spawned/completed in the gap) and forces a reload. The
    params are always emitted (even when empty) so a malformed direct
    hit without them is treated as ``no state known`` and routes to the
    reload path on any non-empty DB state.
    """
    baseline_id = codex_pool.latest_id_for_thread(session_id)
    active_id = active_instance.pk if active_instance is not None else None
    qs = urlencode(
        {
            "baseline": str(baseline_id) if baseline_id is not None else "",
            "active": str(active_id) if active_id is not None else "",
        }
    )
    return f"{reverse('session_stream', kwargs={'session_id': session_id})}?{qs}"


def _display_title(thread: Any) -> str:
    """Return a short, single-line title for a thread.

    Falls back through `name` -> first line of `preview` -> `id`, clipping
    to ``_DISPLAY_TITLE_MAX_LEN`` so a long auto-fallback preview cannot
    overflow the row. Threads without any usable text degrade to the id
    rather than to a blank link.
    """
    name = getattr(thread, "name", None)
    candidate = name.strip() if isinstance(name, str) else ""
    if not candidate:
        preview = getattr(thread, "preview", None) or ""
        candidate = preview.split("\n", 1)[0].strip()
    if not candidate:
        return getattr(thread, "id", "") or ""
    if len(candidate) > _DISPLAY_TITLE_MAX_LEN:
        return candidate[:_DISPLAY_TITLE_MAX_LEN].rstrip() + "..."
    return candidate


def _entries_for(thread: Any) -> Iterator[dict[str, Any]]:
    """Prefer the on-disk rollout so commandExecution rows surface.

    ``thread/read`` rebuilds turns through codex's Limited-mode persistence
    filter, which drops every commandExecution item. When ``Thread.path``
    points at a rollout file we can read, parse it ourselves to recover the
    dropped entries; otherwise (ephemeral threads, unreadable paths, parser
    failures, or an empty rollout) fall back to rebuilding from
    ``Thread.turns`` so the page is never empty just because the rollout
    layer misbehaved.
    """
    flat = _entries_from_rollout(thread)
    if flat is not None:
        yield from _collapse_flat_entries(flat)
        return
    yield from _render_entries(thread)


def _entries_from_rollout(thread: Any) -> list[dict[str, Any]] | None:
    """Materialise entries from the on-disk rollout, or return None to fall back.

    Returning ``None`` (vs. an empty list) is what triggers the SDK fallback;
    an empty list is treated as "the rollout exists and is genuinely empty,"
    matching the behaviour of an empty ``Thread.turns``.
    """
    path = getattr(thread, "path", None)
    if not isinstance(path, str) or not path:
        return None
    rollout_path = Path(path)
    if not rollout_path.is_file():
        logger.warning("thread.path %s is not a readable file; falling back to SDK turns", path)
        return None
    try:
        entries = list(rollout.iter_entries(rollout_path))
    except Exception:
        logger.exception("failed to parse rollout %s; falling back to SDK turns", path)
        return None
    # If the rollout reconstructs no conversation at all but the SDK has
    # turns, prefer the SDK output — that combination almost always means
    # the rollout schema drifted under us (renamed event tag, new wrapper)
    # so our parser silently skipped the user/agent messages even though
    # they are present on disk. A rollout with only tool-call entries is
    # treated the same way: the SDK path may know how to surface the user
    # request, and we'd rather render the conversation without commands
    # than render commands without the conversation. A truly empty parse
    # against an equally empty Thread.turns falls through so the page can
    # show its empty-state placeholder.
    if getattr(thread, "turns", None) and not any(
        entry["kind"] in ("user", "agent") for entry in entries
    ):
        logger.warning(
            "rollout %s yielded no user/agent entries; falling back to SDK turns", path
        )
        return None
    return entries


def _resolved_settings(
    request: HttpRequest, models_data: list[Any]
) -> tuple[str, str, str, str, str, bool, bool, dict[str, str]]:
    """Read the dialog state from storage and reconcile against Codex.

    Returns ``(model, effort, sandbox_policy, approval_mode,
    extra_system_prompt, use_worktrees, show_archived_sessions,
    cookie_updates)``.
    ``cookie_updates`` is a dict of cookie-name → new-value pairs the caller
    must persist on the response (via ``_apply_cookie_updates``) so the
    corrected state takes effect on the next request.

    Two stale-state cases handled here:
      1. The saved model id is no longer offered → snap to the provider's
         default model *and* that model's default effort, since the
         supported-effort set can differ between providers.
      2. The model is still offered but its
         ``supported_reasoning_efforts`` has narrowed under us so the
         saved effort no longer fits → snap effort to that model's
         default while leaving the model alone.

    Authenticated users read from ``UserSettings`` and get a full cookie
    mirror back on each resolution. Anonymous users continue to read and
    write the signed cookies directly.

    Empty ``models_data`` (transport hiccup, mock in tests) means we can't
    validate model compatibility; return the saved values untouched.

    Sandbox is validated against our own static enum rather than Codex's
    model list (it's not a model-scoped setting), so a tampered/legacy
    cookie value falls through to the empty "model default" state.

    Approval mode is validated against our own static enum and falls back
    to ``_DEFAULT_APPROVAL_MODE`` (a safe default with an automated
    reviewer in the loop) when the cookie is missing or invalid, so the
    UI is never left in an ambiguous "no policy picked" state.
    """
    saved = _stored_settings(request)
    saved_sandbox = saved.sandbox_policy
    saved_approval = saved.approval_mode
    if saved_sandbox and saved_sandbox not in _VALID_SANDBOX_POLICIES:
        saved_sandbox = ""
    if saved_approval not in _VALID_APPROVAL_MODES:
        saved_approval = _DEFAULT_APPROVAL_MODE
    saved = saved._replace(
        sandbox_policy=saved_sandbox,
        approval_mode=saved_approval,
    )
    if not models_data:
        return _resolved_settings_result(request, saved, {})

    valid_ids = {m.id for m in models_data}
    if saved.model and saved.model in valid_ids:
        model_obj = next(m for m in models_data if m.id == saved.model)
        if saved.reasoning_effort:
            supported = _supported_effort_values(model_obj)
            if supported and saved.reasoning_effort not in supported:
                new_effort = _model_default_effort(model_obj)
                return _resolved_settings_result(
                    request,
                    saved._replace(reasoning_effort=new_effort),
                    {_EFFORT_COOKIE: new_effort},
                )
        return _resolved_settings_result(request, saved, {})

    default_model = next((m for m in models_data if m.is_default), models_data[0])
    new_effort = _model_default_effort(default_model)
    return _resolved_settings_result(
        request,
        saved._replace(model=default_model.id, reasoning_effort=new_effort),
        {_MODEL_COOKIE: default_model.id, _EFFORT_COOKIE: new_effort},
    )


def _resolved_settings_result(
    request: HttpRequest, values: SettingsValues, cookie_updates: dict[str, str]
) -> tuple[str, str, str, str, str, bool, bool, dict[str, str]]:
    user = _authenticated_user(request)
    if user is not None:
        _save_user_settings(user, values)
        cookie_updates = _settings_cookie_updates(values)
    return (
        values.model,
        values.reasoning_effort,
        values.sandbox_policy,
        values.approval_mode,
        values.extra_system_prompt,
        values.use_worktrees,
        values.show_archived_sessions,
        cookie_updates,
    )


def _authenticated_user(request: HttpRequest) -> Any | None:
    user = request.user
    return user if user.is_authenticated else None


def _stored_settings(request: HttpRequest) -> SettingsValues:
    user = _authenticated_user(request)
    if user is not None:
        return _settings_values_for_user(_settings_for_user(user))
    return SettingsValues(
        model=_read_cookie(request, _MODEL_COOKIE),
        reasoning_effort=_read_cookie(request, _EFFORT_COOKIE),
        sandbox_policy=_read_cookie(request, _SANDBOX_COOKIE),
        approval_mode=_read_cookie(request, _APPROVAL_COOKIE),
        extra_system_prompt=_read_extra_system_prompt_cookie(request),
        use_worktrees=_read_cookie(request, _USE_WORKTREES_COOKIE) == "true",
        show_archived_sessions=_read_cookie(request, _SHOW_ARCHIVED_COOKIE) == "true",
    )


def _settings_for_user(user: Any) -> UserSettings:
    settings, _created = UserSettings.objects.get_or_create(user=user)
    return settings


def _settings_values_for_user(settings: UserSettings) -> SettingsValues:
    return SettingsValues(
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
        sandbox_policy=settings.sandbox_policy,
        approval_mode=settings.approval_mode,
        extra_system_prompt=settings.extra_system_prompt,
        use_worktrees=settings.use_worktrees,
        show_archived_sessions=settings.show_archived_sessions,
    )


def _save_user_settings(user: Any, values: SettingsValues) -> UserSettings:
    settings = _settings_for_user(user)
    updates: list[str] = []
    for field, value in (
        ("model", values.model),
        ("reasoning_effort", values.reasoning_effort),
        ("sandbox_policy", values.sandbox_policy),
        ("approval_mode", values.approval_mode),
        ("extra_system_prompt", values.extra_system_prompt),
        ("use_worktrees", values.use_worktrees),
        ("show_archived_sessions", values.show_archived_sessions),
    ):
        if getattr(settings, field) != value:
            setattr(settings, field, value)
            updates.append(field)
    if updates:
        settings.save(update_fields=[*updates, "updated_at"])
    return settings


def _settings_cookie_updates(values: SettingsValues) -> dict[str, str]:
    return {
        _MODEL_COOKIE: values.model,
        _EFFORT_COOKIE: values.reasoning_effort,
        _SANDBOX_COOKIE: values.sandbox_policy,
        _APPROVAL_COOKIE: values.approval_mode,
        _EXTRA_SYSTEM_PROMPT_COOKIE: _encode_extra_system_prompt_cookie(
            values.extra_system_prompt
        ),
        _USE_WORKTREES_COOKIE: "true" if values.use_worktrees else "false",
        _SHOW_ARCHIVED_COOKIE: "true" if values.show_archived_sessions else "false",
    }


def _import_cookie_settings_to_user(request: HttpRequest, user: Any) -> UserSettings:
    settings = _settings_for_user(user)
    updates: list[str] = []
    for field, value in _valid_cookie_setting_updates(request).items():
        if getattr(settings, field) != value:
            setattr(settings, field, value)
            updates.append(field)
    if updates:
        settings.save(update_fields=[*updates, "updated_at"])
    return settings


def _valid_cookie_setting_updates(request: HttpRequest) -> dict[str, str | bool]:
    updates: dict[str, str | bool] = {}
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
    return updates


def _read_signed_cookie_if_present(request: HttpRequest, name: str) -> str | None:
    if name not in request.COOKIES:
        return None
    try:
        value = request.get_signed_cookie(name)
    except Exception:
        return None
    return (value or "").strip()


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


def _apply_cookie_updates(response: HttpResponse, updates: dict[str, str]) -> None:
    for name, value in updates.items():
        response.set_signed_cookie(
            name, value, max_age=_COOKIE_MAX_AGE, samesite="Lax"
        )


def _safe_next_url(request: HttpRequest) -> str:
    candidate = request.POST.get("next", "").strip() or request.GET.get(
        "next", ""
    ).strip()
    if not candidate:
        return ""
    if url_has_allowed_host_and_scheme(
        candidate,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return candidate
    return ""


def _fetch_rate_limits(codex: Codex) -> dict[str, Any] | None:
    """Fetch the account/rateLimits/read snapshot, or None if unavailable.

    The endpoint is meaningful only when Codex is talking to a real OpenAI
    account; local-dev (no auth, custom provider via ollama) and older
    Codex builds will fail with MethodNotFound or an auth error. The
    settings dialog must still render in those modes, so any failure here
    swallows into None and the rate-limits section is omitted.
    """
    try:
        response = codex._client.request(
            "account/rateLimits/read",
            None,
            response_model=GetAccountRateLimitsResponse,
        )
    except AppServerError:
        return None
    except Exception:
        logger.exception("failed to fetch account rate limits; omitting from settings dialog")
        return None
    return _format_rate_limit_snapshot(response.rate_limits)


def _format_rate_limit_snapshot(snapshot: RateLimitSnapshot) -> dict[str, Any] | None:
    """Project a RateLimitSnapshot into a template-friendly dict.

    Returns None when neither the primary nor secondary window is set; the
    template hides the section entirely in that case rather than render an
    empty header. ``used_percent`` from Codex describes consumption, so we
    expose ``remaining_percent`` (the more intuitive framing for a user
    looking at their remaining budget) alongside it.
    """
    windows: list[dict[str, Any]] = []
    for label, window in (("Primary", snapshot.primary), ("Secondary", snapshot.secondary)):
        if window is None:
            continue
        used = max(0, min(100, window.used_percent))
        windows.append(
            {
                "label": label,
                "used_percent": used,
                "remaining_percent": 100 - used,
                "resets_at": window.resets_at,
                "window_duration_label": _format_window_duration(window.window_duration_mins),
            }
        )
    if not windows:
        return None
    plan_type = snapshot.plan_type.value if snapshot.plan_type is not None else None
    return {
        "windows": windows,
        "limit_name": snapshot.limit_name,
        "plan_type": plan_type,
    }


def _format_window_duration(window_duration_mins: int | None) -> str | None:
    if window_duration_mins is None or window_duration_mins <= 0:
        return None
    if window_duration_mins % _MINUTES_PER_DAY == 0:
        days = window_duration_mins // _MINUTES_PER_DAY
        return f"{days}-day"
    if window_duration_mins % _MINUTES_PER_HOUR == 0:
        hours = window_duration_mins // _MINUTES_PER_HOUR
        return f"{hours}-hour"
    return f"{window_duration_mins}-min"


def _supported_effort_values(model_obj: Any) -> set[str]:
    """Return the set of effort enum string values ``model_obj`` accepts."""
    return {
        getattr(opt.reasoning_effort, "value", str(opt.reasoning_effort))
        for opt in (getattr(model_obj, "supported_reasoning_efforts", None) or [])
    }


def _model_default_effort(model_obj: Any) -> str:
    default = getattr(model_obj, "default_reasoning_effort", None)
    if default is None:
        return ""
    return getattr(default, "value", str(default))


@require_http_methods(["POST"])
def update_settings(request: HttpRequest) -> HttpResponse:
    model = request.POST.get("model", "").strip()
    effort = request.POST.get("reasoning_effort", "").strip()
    sandbox = request.POST.get("sandbox_policy", "").strip()
    approval = request.POST.get("approval_mode", "").strip()
    extra_system_prompt = request.POST.get("extra_system_prompt", "").strip()
    use_worktrees = request.POST.get("use_worktrees", "").strip()
    show_archived = request.POST.get("show_archived_sessions", "").strip()
    if len(model) > _MODEL_MAX_LEN:
        return HttpResponseBadRequest("model id is too long")
    if len(extra_system_prompt) > _EXTRA_SYSTEM_PROMPT_MAX_LEN:
        return HttpResponseBadRequest("extra system prompt is too long")
    valid_efforts = {e.value for e in ReasoningEffort}
    if effort and effort not in valid_efforts:
        return HttpResponseBadRequest("invalid reasoning effort")
    if sandbox and sandbox not in _VALID_SANDBOX_POLICIES:
        return HttpResponseBadRequest("invalid sandbox policy")
    # Approval mode always carries one of the dialog's values. An empty
    # form post is treated as "user picked nothing", which we snap to the
    # safe default.
    if approval and approval not in _VALID_APPROVAL_MODES:
        return HttpResponseBadRequest("invalid approval mode")
    if not approval:
        approval = _DEFAULT_APPROVAL_MODE
    if use_worktrees not in {"", "true"}:
        return HttpResponseBadRequest("invalid worktree setting")
    use_worktrees = "true" if use_worktrees == "true" else "false"
    if show_archived not in {"", "true"}:
        return HttpResponseBadRequest("invalid archived sessions visibility")
    show_archived = "true" if show_archived == "true" else "false"
    if model or effort:
        # Cross-check the posted (model, effort) pair against what Codex
        # actually offers so a malformed POST (typo, stale model id, effort
        # the chosen model doesn't support) gets a clean 400 instead of
        # quietly poisoning every subsequent turn at runtime.
        config = AppServerConfig(codex_bin=shutil.which("codex"))
        with Codex(config=config) as codex:
            models_data = list(codex.models().data)
        compat_error = _validate_settings_against_models(model, effort, models_data)
        if compat_error:
            return HttpResponseBadRequest(compat_error)
    values = SettingsValues(
        model=model,
        reasoning_effort=effort,
        sandbox_policy=sandbox,
        approval_mode=approval,
        extra_system_prompt=extra_system_prompt,
        use_worktrees=use_worktrees == "true",
        show_archived_sessions=show_archived == "true",
    )
    user = _authenticated_user(request)
    if user is not None:
        _save_user_settings(user, values)
    response = redirect("index")
    _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response


def _validate_settings_against_models(
    model: str, effort: str, models_data: list[Any]
) -> str | None:
    """Return an error message for an invalid (model, effort) pair, or None.

    Empty ``models_data`` (transport hiccup, pre-provider state, mock in
    tests) means we can't validate; trust the caller in that case so a
    temporary Codex outage doesn't block the user from saving.

    When ``model`` is blank the effort is checked against the provider's
    default model — the one Codex will fall back to inside ``new_session``
    — so an empty model can't quietly bypass the supported-effort check.
    """
    if not models_data:
        return None
    valid_ids = {m.id for m in models_data}
    if model and model not in valid_ids:
        return f"model {model!r} is not available"
    if effort:
        effective = (
            next((m for m in models_data if m.id == model), None)
            if model
            else next((m for m in models_data if m.is_default), models_data[0])
        )
        if effective is not None:
            supported = _supported_effort_values(effective)
            if supported and effort not in supported:
                return (
                    f"reasoning effort {effort!r} is not supported by "
                    f"model {effective.id!r}"
                )
    return None


@require_http_methods(["POST"])
def set_session_name(request: HttpRequest, session_id: str) -> HttpResponse:
    name = request.POST.get("name", "").strip()
    if not name:
        return HttpResponseBadRequest("name is required")
    if len(name) > _NAME_MAX_LEN:
        return HttpResponseBadRequest("name is too long")
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        codex._client.thread_set_name(session_id, name)
    return redirect("session", session_id=session_id)


@require_http_methods(["POST"])
def set_session_archived(request: HttpRequest, session_id: str) -> HttpResponse:
    archived = request.POST.get("archived", "").strip()
    if archived not in {"true", "false"}:
        return HttpResponseBadRequest("archived must be true or false")
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        if archived == "true":
            codex.thread_archive(session_id)
        else:
            codex.thread_unarchive(session_id)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return HttpResponse(status=204)
    if archived == "true":
        return redirect("index")
    return redirect("session", session_id=session_id)


@require_http_methods(["POST"])
def send_message(request: HttpRequest, session_id: str) -> HttpResponse:
    prompt, plan_mode = _message_prompt_and_plan_mode(request)
    if not prompt:
        return HttpResponseBadRequest("prompt is required")
    settings = _stored_settings(request)
    raw_active = request.POST.get("active_instance", "").strip()
    if raw_active:
        instance_id, error = _parse_instance_id(raw_active)
        if error is not None or instance_id is None:
            return HttpResponseBadRequest(error or "invalid instance id")
        steered = codex_pool.steer_instance(
            instance_id,
            expected_thread_id=session_id,
            prompt=prompt,
        )
        if steered is not None:
            return redirect("session", session_id=session_id)
    else:
        active_instance = codex_pool.latest_active_for_thread(session_id)
        if active_instance is not None:
            steered = codex_pool.steer_instance(
                active_instance.pk,
                expected_thread_id=session_id,
                prompt=prompt,
            )
            if steered is not None:
                return redirect("session", session_id=session_id)
    # If steering is unavailable or races a terminal worker, preserve the
    # submitted prompt by treating it as an ordinary follow-up turn.
    # ``raw_active`` posts still do not retarget a different active worker.
    # ``Thread.cwd`` is an ``AbsolutePathBuf`` pydantic RootModel, so unwrap
    # ``.root`` to get the underlying string the worker subprocess expects;
    # also accept a plain str so a future SDK schema change does not break us.
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        resumed = codex._client.thread_resume(session_id)
        thread = resumed.thread
        plan_model = _plan_mode_model(codex, resumed, settings) if plan_mode else None
    cwd = _thread_cwd(thread)
    if not cwd:
        return HttpResponseBadRequest("thread has no cwd")
    # The session list surfaces every thread the app-server knows about, not
    # just those created via ``new_session``, so the resumed ``cwd`` is not
    # automatically inside the discover_repos() allowlist. Re-validate before
    # spawning so a follow-up cannot run a worker in an unintended directory.
    if cwd not in _allowed_session_cwds():
        return HttpResponseBadRequest("thread cwd is not an allowed repository")
    # Sandbox policy and approval mode are applied per-turn rather than
    # persisted on the thread, so follow-up messages have to re-forward
    # the cookies or every turn after the first silently reverts to Codex
    # defaults — which breaks multi-turn workflows that depend on
    # elevated permissions or stricter escalation handling.
    sandbox_policy = _effective_sandbox_policy(settings)
    approval_mode = _effective_approval_mode(settings)
    spawn_kwargs: dict[str, Any] = {
        "thread_id": session_id,
        "cwd": cwd,
        "prompt": prompt,
        "sandbox_policy": sandbox_policy or None,
        "approval_mode": approval_mode,
    }
    if plan_mode:
        if not plan_model:
            return HttpResponseBadRequest("plan mode requires a model")
        spawn_kwargs["model"] = plan_model
        spawn_kwargs["plan_mode"] = True
    codex_pool.spawn_turn(**spawn_kwargs)
    return redirect("session", session_id=session_id)


def _message_prompt_and_plan_mode(request: HttpRequest) -> tuple[str, bool]:
    prompt = request.POST.get("prompt", "").strip()
    plan_mode = request.POST.get("plan_mode", "").strip().lower() == "true"
    parts = prompt.split(maxsplit=1)
    if not parts:
        return prompt, plan_mode
    command = parts[0].lower()
    if command == _PLAN_SLASH_COMMAND:
        return (parts[1].strip() if len(parts) > 1 else ""), True
    if command == _PR_SLASH_COMMAND:
        return _PR_SLASH_PROMPT, False
    return prompt, plan_mode


def _plan_mode_model(codex: Codex, resumed: Any, settings: SettingsValues) -> str | None:
    resumed_model = getattr(resumed, "model", "")
    if isinstance(resumed_model, str) and resumed_model.strip():
        return resumed_model.strip()
    models_data = _models_for_plan_mode_fallback(codex)
    if settings.model:
        valid_ids = {m.id for m in models_data}
        if not valid_ids or settings.model in valid_ids:
            return settings.model
    default_model = next((m for m in models_data if m.is_default), None)
    if default_model is None and models_data:
        default_model = models_data[0]
    model_id = getattr(default_model, "id", "") if default_model is not None else ""
    return model_id if isinstance(model_id, str) and model_id else None


def _models_for_plan_mode_fallback(codex: Codex) -> list[Any]:
    try:
        return list(codex.models().data)
    except Exception:
        logger.exception("failed to fetch models for plan mode fallback")
        return []


def _thread_cwd(thread: Any) -> str | None:
    raw = getattr(thread, "cwd", None)
    if isinstance(raw, str):
        return raw or None
    root = getattr(raw, "root", None)
    return root if isinstance(root, str) and root else None


def _allowed_session_cwds() -> set[str]:
    return {str(p) for p in [*discover_repos(), *discover_managed_worktrees()]}


def _parse_instance_id(raw: str) -> tuple[int | None, str | None]:
    try:
        instance_id = int(raw)
    except ValueError:
        return None, "invalid instance id"
    # Cross-check against the column type up front so a tampered value past
    # the BigAutoField range can't leak a backend-specific OverflowError or
    # DataError out as a 500 from ``objects.get``.
    if instance_id < 1 or instance_id > _MAX_BIGAUTOFIELD:
        return None, "instance id out of range"
    return instance_id, None


# Decisions the approval endpoint accepts. The wire string the worker writes
# back to codex's app-server is taken straight from this set, so the constants
# must stay aligned with app-server's approval response schema
# (``accept`` / ``decline`` / ``cancel``). UI labels live in the template;
# this layer only validates the wire value.
_VALID_APPROVAL_DECISIONS = frozenset(
    {
        ApprovalRequest.DECISION_ACCEPT,
        ApprovalRequest.DECISION_DECLINE,
        ApprovalRequest.DECISION_CANCEL,
    }
)


@require_http_methods(["POST"])
def resolve_approval(request: HttpRequest, approval_id: int) -> HttpResponse:
    """Record the user's decision on a pending command/file approval.

    The worker's polling loop wakes on the row update and answers the
    SDK's JSON-RPC request with the recorded ``decision``. The response is
    intentionally minimal (200 with the recorded decision string) so the
    browser-side fetch can surface success without parsing JSON.

    Returns 409 if the approval has already been resolved — racing two
    clicks shouldn't silently overwrite an earlier choice that the worker
    has already returned to codex.
    """
    raw_decision = request.POST.get("decision", "").strip()
    decision = ApprovalRequest.normalize_decision(raw_decision)
    if decision not in _VALID_APPROVAL_DECISIONS:
        return HttpResponseBadRequest("invalid decision")
    try:
        approval = ApprovalRequest.objects.get(pk=approval_id)
    except ApprovalRequest.DoesNotExist:
        return HttpResponse("approval not found", status=404)
    if approval.decision:
        return HttpResponse("approval already resolved", status=409)
    # Filter on ``decision=""`` so two concurrent POSTs can't both succeed
    # in flipping the row away from pending.
    updated = ApprovalRequest.objects.filter(pk=approval_id, decision="").update(
        decision=decision,
        decided_at=timezone.now(),
    )
    if not updated:
        return HttpResponse("approval already resolved", status=409)
    return HttpResponse(decision, content_type="text/plain")


@require_http_methods(["POST"])
def resolve_input_request(request: HttpRequest, input_id: int) -> HttpResponse:
    raw_answers = request.POST.get("answers", "").strip()
    try:
        parsed = json.loads(raw_answers) if raw_answers else {}
    except json.JSONDecodeError:
        return HttpResponseBadRequest("invalid answers")
    if not isinstance(parsed, dict):
        return HttpResponseBadRequest("invalid answers")
    answers: dict[str, Any] = {}
    for key, value in parsed.items():
        key = key.strip()
        if isinstance(value, str):
            value = value.strip()
        if key:
            answers[key] = value
    response: dict[str, Any] = {"answers": answers}
    try:
        input_request = UserInputRequest.objects.get(pk=input_id)
    except UserInputRequest.DoesNotExist:
        return HttpResponse("input request not found", status=404)
    if input_request.response is not None:
        return HttpResponse("input request already resolved", status=409)
    updated = UserInputRequest.objects.filter(pk=input_id, response__isnull=True).update(
        response=response,
        responded_at=timezone.now(),
    )
    if not updated:
        return HttpResponse("input request already resolved", status=409)
    return HttpResponse(json.dumps(response), content_type="application/json")


@require_http_methods(["POST"])
def stop_session(request: HttpRequest, session_id: str) -> HttpResponse:
    """Interrupt the in-progress turn for ``session_id``.

    The Stop button posts the active worker's id (as ``instance``) so a
    stale tab can't accidentally abort a newer overlapping worker the
    user can't see. When the form value is missing (older cached page,
    direct POST) we fall back to "latest active worker for this thread".

    No-ops cleanly when no worker is active so a double-click after the
    turn already finished still lands on the session page rather than 404.
    """
    raw = request.POST.get("instance", "").strip()
    if raw:
        instance_id, error = _parse_instance_id(raw)
        if error is not None or instance_id is None:
            return HttpResponseBadRequest(error or "invalid instance id")
        codex_pool.interrupt_instance(instance_id, expected_thread_id=session_id)
    else:
        codex_pool.interrupt_active(session_id)
    return redirect("session", session_id=session_id)


@require_http_methods(["POST"])
def new_session(request: HttpRequest) -> HttpResponse:
    prompt, plan_mode = _message_prompt_and_plan_mode(request)
    cwd = request.POST.get("cwd", "").strip()
    if not prompt:
        return HttpResponseBadRequest("prompt is required")
    if not cwd:
        return HttpResponseBadRequest("cwd is required")
    # Restrict cwd to a discovered repo so an arbitrary path can't be injected
    # via the form post.
    allowed = {str(p) for p in discover_repos()}
    if cwd not in allowed:
        return HttpResponseBadRequest("cwd must be a discovered repository")

    # Re-reconcile the cookies against Codex's current model list before
    # spawning. A long-lived tab might still be carrying a model the index
    # render would have snapped away from; without this, a stale value
    # would ride straight into ``thread_start(model=...)`` and 500 the
    # new-session click.
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        models_data = list(codex.models().data)
    (
        model,
        reasoning_effort,
        sandbox_policy,
        approval_mode,
        extra_system_prompt,
        use_worktrees,
        _show_archived_sessions,
        cookie_updates,
    ) = _resolved_settings(request, models_data)
    if plan_mode and not model:
        return HttpResponseBadRequest("plan mode requires a model")

    session_cwd = cwd
    managed_worktree = None
    if use_worktrees:
        try:
            managed_worktree = create_worktree_for_session(cwd)
        except WorktreeCreationError as exc:
            return HttpResponseBadRequest(str(exc))
        session_cwd = str(managed_worktree.path)

    # Detach a worker subprocess so the initial turn keeps running past a
    # Django restart. The thread itself is created synchronously to give the
    # caller a stable id to redirect to.
    spawn_kwargs: dict[str, Any] = {
        "cwd": session_cwd,
        "prompt": prompt,
        "developer_instructions": extra_system_prompt or None,
        "model": model or None,
        "reasoning_effort": reasoning_effort or None,
        "sandbox_policy": sandbox_policy or None,
        "approval_mode": approval_mode,
    }
    if plan_mode:
        spawn_kwargs["plan_mode"] = True
    try:
        instance = codex_pool.spawn_new_session(**spawn_kwargs)
    except Exception:
        if managed_worktree is not None:
            try:
                cleanup_worktree(managed_worktree)
            except WorktreeCleanupError:
                logger.exception(
                    "failed to clean up managed worktree %s", managed_worktree.path
                )
        raise
    response = redirect("session", session_id=instance.thread_id)
    _apply_cookie_updates(response, cookie_updates)
    return response


def _collapse_flat_entries(flat: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Apply the same intermediate-collapsing as ``_render_entries`` to the
    flat per-item entries produced by the rollout parser.

    Turn boundaries are detected via the user kind because the rollout file
    is a chronological log without per-item turn ids exposed at the entry
    level. This matches the SDK's `handle_user_message` fallback for streams
    that did not open turns explicitly.
    """
    turn: list[dict[str, Any]] = []
    for entry in flat:
        if entry["kind"] == "user" and turn:
            yield from _emit_collapsed_turn(turn)
            turn = []
        turn.append(entry)
    if turn:
        yield from _emit_collapsed_turn(turn)


def _emit_collapsed_turn(turn: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    final_idx = _final_agent_idx_in_flat(turn)
    intermediate: list[dict[str, Any]] = []
    for i, entry in enumerate(turn):
        if i == final_idx:
            if intermediate:
                yield _make_intermediate_entry(intermediate)
                intermediate = []
            yield _finalize_agent_entry(_strip_phase(entry))
        elif entry["kind"] == "user":
            # `_collapse_flat_entries` splits on every user past the first, so
            # any user reaching this branch is the leading entry of the turn
            # and intermediate is empty.
            yield entry
        elif entry["kind"] == "agent":
            intermediate.append({**_strip_phase(entry), "kind": "thinking"})
        elif entry["kind"] in {"approval_declined", "plan"}:
            if intermediate:
                yield _make_intermediate_entry(intermediate)
                intermediate = []
            yield entry
        else:
            intermediate.append(entry)
    if intermediate:
        yield _make_intermediate_entry(intermediate)


def _final_agent_idx_in_flat(entries: list[dict[str, Any]]) -> int:
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry["kind"] == "agent" and entry.get("phase") == "final_answer":
            return i
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry["kind"] != "agent":
            continue
        if entry.get("phase") == "commentary":
            continue
        return i
    return -1


def _strip_phase(entry: dict[str, Any]) -> dict[str, Any]:
    """Return ``entry`` minus its ``phase`` key.

    Called on every agent entry just before it leaves the collapsing pass so
    the template never sees the internal phase marker. The rollout parser
    sets ``phase`` on every agent entry (to ``None`` when absent), so this
    function never has to consult the dict before stripping.
    """
    return {k: v for k, v in entry.items() if k != "phase"}


def _finalize_agent_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Annotate a turn's final agent entry with rendered-markdown HTML.

    Only the final agent message of a turn is ever fed through this
    function, so collapsed "thinking" messages stay plain-text. A failure
    to recognise the text as markdown leaves the entry untouched, and the
    template falls back to the plain-text body.
    """
    text = entry.get("text")
    if isinstance(text, str) and looks_like_markdown(text):
        entry["html"] = render_markdown(text)
    return entry


def _render_entries(thread: Any) -> Iterator[dict[str, Any]]:
    """Walk every turn's items in order, surfacing the user message and the
    final agent reply as top-level entries and folding everything else
    (intermediate agent commentary plus every tool-call variant) into a
    single collapsible "intermediate" entry per run so long sessions don't
    bury the actual answer behind dozens of tool rows.

    The SDK marks final responses with MessagePhase.final_answer when known;
    for sessions where phase is unset (older data or an in-progress turn)
    the last agentMessage in the turn is treated as final. Each entry
    carries the turn's started_at timestamp; per-item timestamps are not
    exposed by the SDK.
    """
    for turn in thread.turns:
        timestamp = getattr(turn, "started_at", None)
        items = [thread_item.root for thread_item in turn.items]
        final_idx = _find_final_agent_idx(items)
        intermediate: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            if i == final_idx:
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                yield _finalize_agent_entry(
                    {"kind": "agent", "text": item.text, "timestamp": timestamp}
                )
            elif item.type == "userMessage":
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                yield {
                    "kind": "user",
                    "text": _user_message_text(item),
                    "timestamp": timestamp,
                }
            elif item.type == "agentMessage":
                intermediate.append(
                    {"kind": "thinking", "text": item.text, "timestamp": timestamp}
                )
            elif item.type == "plan":
                if intermediate:
                    yield _make_intermediate_entry(intermediate)
                    intermediate = []
                yield {
                    "kind": "plan",
                    "text": getattr(item, "text", "") or "",
                    "timestamp": timestamp,
                }
            else:
                intermediate.append(_make_tool_call_entry(item, timestamp))

        if intermediate:
            yield _make_intermediate_entry(intermediate)


def _find_final_agent_idx(items: list[Any]) -> int:
    """Index of the agent message to display as this turn's final response, or
    -1 if there is no agent message that could be the final.

    An explicit MessagePhase.final_answer always wins (the last one if
    multiple). Otherwise the last agent message whose phase is not
    MessagePhase.commentary is treated as final — i.e. unset phases (older
    sessions / in-progress turns) are eligible to be the final, but explicit
    commentary never is.
    """
    for i in range(len(items) - 1, -1, -1):
        item = items[i]
        if item.type == "agentMessage" and _phase_value(item) == "final_answer":
            return i
    for i in range(len(items) - 1, -1, -1):
        item = items[i]
        if item.type != "agentMessage":
            continue
        if _phase_value(item) == "commentary":
            continue
        return i
    return -1


def _phase_value(item: Any) -> str | None:
    # The SDK normally deserializes `phase` into a MessagePhase enum instance
    # (with `.value`), but accept a raw wire string too so this stays robust
    # against thread data that bypasses pydantic deserialization.
    phase = getattr(item, "phase", None)
    if phase is None:
        return None
    if isinstance(phase, str):
        return phase
    value = getattr(phase, "value", None)
    return value if isinstance(value, str) else None


def _make_tool_call_entry(item: Any, timestamp: Any) -> dict[str, Any]:
    item_type = item.type
    return {
        "kind": "tool_call",
        "type": item_type,
        "label": _NON_MESSAGE_LABELS.get(item_type, item_type),
        "detail": _tool_call_detail(item, item_type),
        "status": _tool_call_status(item),
        "timestamp": timestamp,
    }


def _make_intermediate_entry(items: list[dict[str, Any]]) -> dict[str, Any]:
    thinking_count = sum(1 for e in items if e["kind"] == "thinking")
    tool_call_count = sum(1 for e in items if e["kind"] == "tool_call")
    return {
        "kind": "intermediate",
        "thinking_count": thinking_count,
        "tool_call_count": tool_call_count,
        "items": items,
    }


def _user_message_text(item: Any) -> str:
    parts: list[str] = []
    for input_item in item.content:
        inner = input_item.root
        match inner.type:
            case "text":
                parts.append(inner.text)
            case "mention":
                parts.append(f"@{inner.name}")
            case "skill":
                parts.append(f"/{inner.name}")
            case "image":
                parts.append("[image]")
            case "localImage":
                parts.append(f"[image: {inner.path}]")
    return "\n".join(parts)


def _tool_call_detail(item: Any, item_type: str) -> str:
    """Return a short, human-readable description of a tool-call item.

    Returns an empty string for item types that do not carry useful inline
    detail; the label alone is enough to surface them in the UI.
    """
    match item_type:
        case "commandExecution":
            return getattr(item, "command", "") or ""
        case "mcpToolCall":
            return f"{item.server} / {item.tool}"
        case "dynamicToolCall":
            namespace = getattr(item, "namespace", None)
            return f"{namespace}::{item.tool}" if namespace else item.tool
        case "fileChange":
            paths = [str(change.path) for change in getattr(item, "changes", []) or []]
            if not paths:
                return ""
            if len(paths) == 1:
                return paths[0]
            return f"{paths[0]} (+{len(paths) - 1} more)"
        case "webSearch":
            return getattr(item, "query", "") or ""
        case "plan":
            text = getattr(item, "text", "") or ""
            return text.split("\n", 1)[0]
        case "imageView":
            return getattr(item, "path", "") or ""
        case "imageGeneration":
            return (
                getattr(item, "revised_prompt", None)
                or getattr(item, "saved_path", None)
                or ""
            )
        case "collabAgentToolCall":
            tool = getattr(item, "tool", None)
            tool_name = getattr(tool, "value", None) or (tool if isinstance(tool, str) else "")
            receivers = getattr(item, "receiver_thread_ids", None) or []
            if receivers:
                return f"{tool_name} → {receivers[0]}"
            return str(tool_name)
        case _:
            return ""


def _tool_call_status(item: Any) -> str | None:
    """Return a non-success status string (e.g. ``failed``) or None.

    Hides ``completed`` so the common case stays uncluttered; surfaces
    in-progress, failed, and declined so unusual outcomes are visible.
    """
    status = getattr(item, "status", None)
    if status is None:
        return None
    value = getattr(status, "value", status)
    if not isinstance(value, str) or value == "completed":
        return None
    return value
