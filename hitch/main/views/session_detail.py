"""The session detail page, SSE stream, and intermediate-entry endpoint."""
from typing import Any

from django.core import signing
from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    StreamingHttpResponse,
)
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from hitch.main import demo
from hitch.main.runtime import codex_pool, reconciliation, rollout, streaming
from hitch.main.runtime.rollout_state import (
    _rollout_file_state_from_value,
    _RolloutFileState,
)
from hitch.main.sessions.entry_render import (
    collapse_flat_entries,
)
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_entry_display import (
    _active_history_user_identity,
    _active_instance_for,
    _apply_qa_approval_messages,
    _apply_system_authors,
    _demo_agent_prompts,
    _filter_demo_agent_entries,
    _trim_in_progress_turn,
)
from hitch.main.sessions.session_resume import (
    _entries_include_transcript,
    _rollout_path_for_session_detail,
    _session_detail_metadata,
)
from hitch.main.sessions.system_agent_summary import (
    _system_agent_run_for_thread,
)
from hitch.main.views import common
from hitch.main.workflows import system_agents


def session(request: HttpRequest, session_id: str) -> HttpResponse:
    return common._render_session_detail(request, session_id)


@require_http_methods(["GET"])
def session_history(request: HttpRequest, session_id: str) -> HttpResponse:
    """Return one older, bounded conversation-preview fragment."""
    metadata = _session_detail_metadata(session_id)
    rollout_path = _rollout_path_for_session_detail(session_id, metadata)
    rollout_state = _rollout_file_state_from_value(
        str(rollout_path) if rollout_path is not None else None
    )
    if rollout_state is None:
        raise Http404("session not found")
    try:
        before_offset = int(request.GET.get("before", ""))
        raw_record_end = request.GET.get("record_end", "")
        partial_record_end = int(raw_record_end) if raw_record_end else None
    except ValueError as exc:
        raise Http404("history page not found") from exc
    raw_demo_context = request.GET.get("demo_context", "")
    show_demo_entries = _session_intermediate_allows_demo_entries(
        session_id, raw_demo_context
    )
    hidden_demo_prompts = (
        frozenset() if show_demo_entries else _demo_agent_prompts(session_id)
    )
    active_instance = _active_instance_for(session_id)
    active_user_identity = _active_history_user_identity(active_instance)
    page = rollout.session_history_page(
        rollout_state.path,
        before_offset=before_offset,
        partial_record_end=partial_record_end,
        message_target=common._SESSION_HISTORY_MESSAGE_TARGET,
        hidden_user_prompts=hidden_demo_prompts,
        active_user_identity=active_user_identity,
    )
    if page is None:
        raise Http404("history page not found")
    newer_turn_continues = request.GET.get("newer_turn") == "continued"
    leading_turn_continues = bool(
        page.has_older
        and page.flat_entries
        and page.flat_entries[0].get("kind") != "user"
    )
    entries = list(
        collapse_flat_entries(
            list(page.flat_entries),
            leading_turn_continues=leading_turn_continues,
            trailing_turn_continues=newer_turn_continues,
        )
    )
    entries = _filter_demo_agent_entries(
        entries,
        session_id,
        initial_user_text=page.leading_user_text,
        hidden_prompts=hidden_demo_prompts,
    )
    entries = _trim_in_progress_turn(
        entries,
        active_instance,
        active_turn_unresolved=page.active_turn_unresolved,
    )
    response = render(
        request,
        "_session_history_page.html",
        {
            "entries": entries,
            "history_next_url": (
                common._session_history_url(
                    session_id,
                    before_offset=page.start_offset,
                    partial_record_end=page.partial_record_end,
                    newer_turn_continues=(
                        page.flat_entries[0].get("kind") != "user"
                        if page.flat_entries
                        else newer_turn_continues
                    ),
                    demo_context=raw_demo_context if show_demo_entries else "",
                )
                if page.has_older
                else ""
            ),
        },
    )
    return common._prevent_stale_cache(response)


def _cached_intermediate_detail(
    *,
    session_id: str,
    rollout_state: _RolloutFileState,
    hide_demo_agent_entries: bool,
    entry_index: int,
) -> dict[str, Any] | None:
    key = common._intermediate_detail_cache_key(
        session_id=session_id,
        rollout_state=rollout_state,
        hide_demo_agent_entries=hide_demo_agent_entries,
        entry_index=entry_index,
    )
    with common._INTERMEDIATE_DETAIL_CACHE_LOCK:
        entry = common._INTERMEDIATE_DETAIL_CACHE.get(key)
        if entry is not None:
            common._INTERMEDIATE_DETAIL_CACHE.move_to_end(key)
        return entry

@require_http_methods(["GET"])
def session_intermediate(
    request: HttpRequest, session_id: str, entry_index: int
) -> HttpResponse:
    if entry_index < 0:
        raise Http404("intermediate entry not found")
    hide_demo_agent_entries = not _session_intermediate_allows_demo_entries(
        session_id, request.GET.get("demo_context", "")
    )
    entry = _rollout_intermediate_entry_for_detail(
        session_id,
        entry_index=entry_index,
        hide_demo_agent_entries=hide_demo_agent_entries,
    )
    response = render(request, "_session_intermediate_body.html", {"entry": entry})
    # The body depends on the current rollout contents; with no validators a
    # browser may heuristically cache this lazily-fetched fragment and show a
    # stale block after the rollout entry changes.
    return common._prevent_stale_cache(response)

def _session_intermediate_allows_demo_entries(
    session_id: str, raw_context: str | None
) -> bool:
    if not raw_context:
        return False
    try:
        context = signing.loads(
            raw_context,
            salt=common._SESSION_INTERMEDIATE_DEMO_CONTEXT_SALT,
        )
    except signing.BadSignature:
        return False
    if not isinstance(context, dict) or context.get("session_id") != session_id:
        return False
    run_id = context.get("run_id")
    if not isinstance(run_id, int) or run_id <= 0:
        return False
    run = _system_agent_run_for_thread(session_id, run_id=run_id)
    return run is not None and run.agent_kind == demo.DEMO_AGENT_KIND

def _rollout_intermediate_entry_for_detail(
    session_id: str, *, entry_index: int, hide_demo_agent_entries: bool
) -> dict[str, Any]:
    metadata = _session_detail_metadata(session_id)
    if metadata is None:
        raise Http404("session not found")
    rollout_state = _rollout_file_state_from_value(metadata.codex_path)
    if rollout_state is None:
        raise Http404("session not found")
    cached = _cached_intermediate_detail(
        session_id=session_id,
        rollout_state=rollout_state,
        hide_demo_agent_entries=hide_demo_agent_entries,
        entry_index=entry_index,
    )
    if cached is not None:
        return cached
    try:
        rollout_data = rollout.session_detail_data(rollout_state.path)
    except Exception as exc:
        common.logger.exception(
            "failed to parse rollout %s for intermediate detail", rollout_state.path
        )
        raise Http404("intermediate entry not found") from exc
    if rollout_data is None:
        raise Http404("session not found")
    entries = list(collapse_flat_entries(list(rollout_data.flat_entries)))
    if not _entries_include_transcript(entries):
        raise Http404("session not found")
    entries = _apply_system_authors(entries, session_id)
    entries = _apply_qa_approval_messages(entries, session_id)
    if hide_demo_agent_entries:
        entries = _filter_demo_agent_entries(entries, session_id)
    if entry_index >= len(entries):
        raise Http404("intermediate entry not found")
    entry = entries[entry_index]
    if entry.get("kind") != "intermediate":
        raise Http404("intermediate entry not found")
    common._cache_intermediate_detail(
        session_id=session_id,
        rollout_state=rollout_state,
        hide_demo_agent_entries=hide_demo_agent_entries,
        entry_index=entry_index,
        entry=entry,
    )
    return entry

@require_http_methods(["GET"])
def session_stream(request: HttpRequest, session_id: str) -> StreamingHttpResponse:
    """SSE endpoint that mirrors the active worker's events file to the browser.

    When no worker is active the connection emits one heartbeat and asks the
    client to reconnect after a short delay. This shows ``connected, idle``
    without consuming a blocking WSGI thread for the lifetime of every tab.

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
    workflow_param = request.GET.get("workflow", "")
    steering_param = request.GET.get("steering", "")
    demo_param = request.GET.get("demo", "")
    active = _active_instance_for(session_id)
    active_workflow = system_agents.active_workflow_for_thread(
        session_id,
        reconcile=False,
    )
    # Idle clients reconnect frequently to detect state changes. Keep unchanged
    # idle probes read-only; active streams still reconcile before routing so a
    # dead worker or workflow cannot leave the page stuck in working state.
    if active is not None or active_workflow is not None:
        reconciliation.reconcile_dead_for_thread(session_id)
        reconciliation.reconcile_dead_if_due()
        active = _active_instance_for(session_id)
        active_workflow = system_agents.active_workflow_for_thread(session_id)

    current_latest = codex_pool.latest_id_for_thread(session_id)
    current_latest_str = str(current_latest) if current_latest is not None else ""
    current_active_str = str(active.pk) if active is not None else ""
    current_workflow_str = str(active_workflow.pk) if active_workflow is not None else ""
    current_steering_revision = streaming.workflow_steering_revision(active_workflow)
    current_steering_str = (
        str(current_steering_revision) if active_workflow is not None else ""
    )
    current_demo_str = streaming.demo_stream_token(session_id)

    if (
        baseline_param != current_latest_str
        or active_param != current_active_str
        or workflow_param != current_workflow_str
        or steering_param != current_steering_str
        or demo_param != current_demo_str
    ):
        response = StreamingHttpResponse(
            streaming.reload_stream(), content_type="text/event-stream"
        )
    elif active is not None:
        response = StreamingHttpResponse(
            streaming.capacity_limited_stream(
                streaming.stream_for_instance(
                    active,
                    demo_baseline=current_demo_str,
                    steering_revision=(
                        current_steering_revision
                        if active_workflow is not None
                        else None
                    ),
                ),
            ),
            content_type="text/event-stream",
        )
    elif active_workflow is not None:
        response = StreamingHttpResponse(
            streaming.capacity_limited_stream(
                streaming.system_workflow_stream(
                    session_id,
                    current_latest,
                    active_workflow.pk,
                    current_steering_revision,
                )
            ),
            content_type="text/event-stream",
        )
    else:
        response = StreamingHttpResponse(
            streaming.idle_stream(),
            content_type="text/event-stream",
        )
    # Discourage proxies from buffering: SSE depends on every frame reaching
    # the client immediately, not coalesced into a single response body.
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
