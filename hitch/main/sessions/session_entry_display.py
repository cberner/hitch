"""Session-detail entry/transcript display helpers.

Pure code-movement extraction from ``views.py``: builds the rendered
entry list for a session (rollout vs. SDK fallback and system author
tagging) and active-worker helpers the detail page and SSE view consume.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.db.models.functions import Coalesce
from django.urls import reverse
from django.utils import timezone

from hitch.main.goals.autonomous_goal_proposal_stack import (
    AUTONOMOUS_GOAL_ACCEPTED_SNAPSHOT_METADATA_KEY,
    AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY,
    AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY,
)
from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    ProposedSession,
)
from hitch.main.runtime import codex_events, codex_pool, rollout
from hitch.main.sessions import session_index
from hitch.main.sessions.entry_render import (
    collapse_flat_entries,
    render_entries,
)

logger = logging.getLogger(__name__)

# Upper bound on what we render inline as a session's title. Codex does not
# generate its own thread summaries, so for unnamed threads `Thread.preview`
# (the full first user message) is what we get; that is often paragraphs
# long and would overflow the list rows without a clip.


def _accepted_proposal_context(session_id: str) -> dict[str, str] | None:
    proposal = (
        ProposedSession.objects.select_related("candidate_session")
        .filter(
            accepted_session__thread_id=session_id,
            autonomous_goal_id__isnull=False,
            candidate_session__isnull=False,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        .order_by("-updated_at", "-pk")
        .first()
    )
    if proposal is None or proposal.candidate_session is None:
        return None
    context = {
        "candidate_log_url": reverse(
            "system_session",
            kwargs={"session_id": proposal.candidate_session.thread_id},
        )
    }
    metadata = (
        proposal.outcome_metadata
        if isinstance(proposal.outcome_metadata, dict)
        else {}
    )
    snapshot = metadata.get(AUTONOMOUS_GOAL_ACCEPTED_SNAPSHOT_METADATA_KEY)
    if not isinstance(snapshot, str) and (
        metadata.get(AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY) is True
    ):
        # Accepted tool-protocol proposals created before final-snapshot
        # provenance was persisted used their pre-created approved snapshot.
        snapshot = metadata.get(AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY)
    snapshot_text = snapshot.strip() if isinstance(snapshot, str) else ""
    if (
        metadata.get("autonomous_goal_autonomy")
        != AutonomousGoal.AUTONOMY_PROPOSE_ONLY
        and 40 <= len(snapshot_text) <= 64
        and all(character in "0123456789abcdef" for character in snapshot_text)
    ):
        context["approved_snapshot"] = snapshot_text
    return context


def _attach_accepted_proposal_context(
    entries: list[dict[str, Any]], context: dict[str, str]
) -> None:
    for entry in entries:
        if entry.get("kind") == "user":
            entry["accepted_proposal_context"] = context
            return


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
    entries: list[dict[str, Any]],
    active: CodexInstance | None,
    *,
    active_turn_unresolved: bool = False,
    active_stream_owns_turn: bool = True,
) -> list[dict[str, Any]]:
    """Drop the in-progress turn's entries from the tail of ``entries``.

    The SSE stream re-emits every event from the start of the worker's
    events file, including the user message and any agent / tool items
    the rollout has already captured. Without this trim those entries
    render twice on the live page — once from the server-side rollout
    pass, once by the streaming JS that can't dedupe against DOM nodes
    it didn't create.

    The in-progress turn is identified by the most recent user-message entry
    whose text matches the active worker's original prompt plus its initial
    image markers. Mid-turn steer images live in the attachment ledger but do
    not change this identity. When the rollout is the page's fallback transcript
    owner, preserve these entries instead of trimming them for SSE replay.
    """
    if active is not None and not active_stream_owns_turn:
        return entries
    if active_turn_unresolved and active is not None:
        return []
    active_turn_start = _active_turn_start_index(entries, active)
    if active_turn_start is not None:
        return entries[:active_turn_start]
    return entries


_STREAM_USER_MESSAGE_MARKER = b'"userMessage"'
_ACTIVE_STREAM_CLAIM_GRACE = timedelta(seconds=30)


def _active_stream_owns_turn(active: CodexInstance | None) -> bool:
    """Return whether SSE replay owns or is still claiming the active turn.

    A newly started worker gets a short grace period for its first user event
    to arrive. After that, only the original user item claims transcript
    ownership; later steering messages do not.
    """
    if active is None:
        return False
    if active.events_path:
        try:
            with Path(active.events_path).open("rb") as events:
                if _event_log_contains_original_user(events, active):
                    return True
        except OSError:
            pass
    return bool(active.started_at and active.started_at > timezone.now() - _ACTIVE_STREAM_CLAIM_GRACE)


def _entries_include_active_turn(entries: list[dict[str, Any]], active: CodexInstance | None) -> bool:
    """Return whether rollout-rendered entries contain the active boundary."""
    return _active_turn_start_index(entries, active) is not None


def _mark_active_history_user_entries(entries: list[dict[str, Any]], active: CodexInstance | None) -> None:
    """Mark the current active user boundary using prompt and timestamp."""
    identity = _active_history_user_identity(active)
    if identity is None:
        return
    for entry in entries:
        if entry.get("kind") != "user":
            continue
        timestamp = entry.get("timestamp")
        entry["_hitch_active_user"] = bool(
            isinstance(timestamp, int | float)
            and not isinstance(timestamp, bool)
            and timestamp >= identity.started_at
            and entry.get("text") == identity.text
        )


def _active_turn_start_index(entries: list[dict[str, Any]], active: CodexInstance | None) -> int | None:
    active_text = _active_user_message_text(active)
    if not active_text:
        return None
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        active_marker = entry.get("_hitch_active_user")
        if entry.get("kind") == "user" and (
            active_marker is True or ("_hitch_active_user" not in entry and entry.get("text") == active_text)
        ):
            return index
    return None


def _event_log_contains_original_user(lines: Iterable[bytes], active: CodexInstance) -> bool:
    """Identify this worker's submitted user item, not a later steer."""
    expected_client_id = f"hitch-instance-{active.pk}"
    first_user_item: dict[str, Any] | None = None
    for line in lines:
        if _STREAM_USER_MESSAGE_MARKER not in line:
            continue
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        item = _event_user_item(event)
        if item is None:
            continue
        if first_user_item is None:
            first_user_item = item
        client_id = item.get("clientId", item.get("client_id"))
        if client_id == expected_client_id:
            return True

    if first_user_item is None:
        return False
    client_id = first_user_item.get("clientId", first_user_item.get("client_id"))
    return client_id is None and _event_user_message_text(first_user_item) == (_active_user_message_text(active))


def _event_user_item(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict) or event.get("method") not in {
        "item/started",
        "item/completed",
    }:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    item = payload.get("item")
    if not isinstance(item, dict) or item.get("type") != "userMessage":
        return None
    return item


def _event_user_message_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for value in content:
        if not isinstance(value, dict):
            continue
        item_type = value.get("type")
        if item_type == "text":
            text = value.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif item_type == "mention":
            name = value.get("name")
            if isinstance(name, str):
                parts.append(f"@{name}")
        elif item_type == "skill":
            name = value.get("name")
            if isinstance(name, str):
                parts.append(f"/{name}")
        elif item_type in {"image", "localImage"}:
            parts.append("[image]")
    return "\n".join(parts)


def _show_active_worker_transcript(active: CodexInstance | None) -> bool:
    return active is not None


def _pending_user_prompt(active: CodexInstance | None) -> str:
    """Surface the active worker's prompt as a pending user bubble.

    Pairs with ``_trim_in_progress_turn``: that helper strips the
    in-progress turn from the rollout-rendered entries (so the stream
    owns rendering it), which means the user wouldn't see their own
    message at all between Send and the first stream event without this
    placeholder. The streaming JS removes the bubble as soon as the
    real ``userMessage`` event lands.
    """
    if active is None:
        return ""
    return _active_user_message_text(active)


def _active_user_message_text(active: CodexInstance | None) -> str:
    if active is None:
        return ""
    parts: list[str] = []
    if active.prompt:
        parts.append(active.prompt)
    parts.extend("[image]" for _path in _normalized_json_string_list(active.input_image_paths))
    return "\n".join(parts)


def _active_history_user_identity(
    active: CodexInstance | None,
) -> rollout.SessionHistoryUserIdentity | None:
    if active is None:
        return None
    return rollout.SessionHistoryUserIdentity(
        text=_active_user_message_text(active),
        prompt=active.prompt,
        started_at=active.started_at.timestamp(),
    )


def _normalized_json_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _task_plan_context(
    snapshot: codex_events.TaskPlanSnapshot | None,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    current = _current_task_text(snapshot.steps) or snapshot.explanation
    return {
        "visible": bool(current or snapshot.steps),
        "current": current,
        "explanation": snapshot.explanation,
        "recorded_at": snapshot.order[0],
        "event_seq": snapshot.order[1],
        "fallback_order": snapshot.order[2],
        "steps": [{"step": step.step, "status": step.status} for step in snapshot.steps],
    }


def _current_task_text(steps: tuple[codex_events.TaskPlanStep, ...]) -> str:
    for status in ("inProgress", "pending"):
        for step in steps:
            if step.status == status:
                return step.step
    return steps[-1].step if steps else ""


def _pending_user_author(active: CodexInstance | None) -> str:
    if active is None:
        return ""
    return active.display_author if active.purpose == CodexInstance.PURPOSE_SYSTEM_FEEDBACK else ""


def _pending_user_timestamp(active: CodexInstance | None) -> int:
    if active is None:
        return 0
    return int(active.started_at.timestamp())


def _active_worker_status_text(active: CodexInstance | None) -> str:
    return ""


def _latest_user_turn_failure(session_id: str) -> dict[str, Any] | None:
    """Return display data when the latest user-turn lifecycle event is a failure."""
    latest = (
        CodexInstance.objects.filter(
            thread_id=session_id,
            purpose=CodexInstance.PURPOSE_USER,
        )
        .only("status", "error", "started_at", "ended_at")
        # User turns may overlap. A later-ending older turn must supersede a
        # newer-started turn that already finished, while a newly started active
        # turn should supersede failures that ended before it began.
        .order_by(Coalesce("ended_at", "started_at").desc(), "-pk")
        .first()
    )
    if latest is None or latest.status != CodexInstance.STATUS_FAILED:
        return None
    timestamp = latest.ended_at or latest.started_at
    return {
        "message": latest.error.strip() or "The agent turn ended without an error message.",
        "timestamp": int(timestamp.timestamp()),
    }


def _apply_system_authors(entries: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    system_authors: dict[int, str] = {
        user_message_index: author
        for user_message_index, author in CodexInstance.objects.filter(
            thread_id=session_id,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            user_message_index__isnull=False,
        ).values_list("user_message_index", "display_author")
        if isinstance(user_message_index, int) and author
    }
    if not system_authors:
        return entries
    user_message_index = 0
    for entry in entries:
        user_message_index = _apply_system_author(entry, system_authors, user_message_index)
    return entries


def _apply_system_author(entry: dict[str, Any], system_authors: dict[int, str], user_message_index: int) -> int:
    if entry.get("kind") == "user":
        author = system_authors.get(user_message_index)
        if author:
            entry["display_author"] = author
        return user_message_index + 1
    if entry.get("kind") == "intermediate":
        for item in entry.get("items", []):
            user_message_index = _apply_system_author(item, system_authors, user_message_index)
    return user_message_index


def _display_title(thread: Any) -> str:
    """Return a short, single-line title for a thread.

    Falls back through `name` -> first line of `preview` -> `id`, clipping
    so a long auto-fallback preview cannot overflow the row. Threads
    without any usable text degrade to the id rather than to a blank link.
    One rule shared with the session index (and the optimistic rename in
    the index page JS): see ``session_index.display_title_for``.
    """
    return session_index.display_title_for(
        thread_id=getattr(thread, "id", "") or "",
        name=getattr(thread, "name", None),
        preview=getattr(thread, "preview", None) or "",
    )


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
    entries, _rollout_backed = _entries_for_with_source(thread)
    yield from entries


def _entries_for_with_source(
    thread: Any, *, fallback_rollout_path: str | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """Return rendered entries and whether a rollout successfully backed them."""
    flat = _entries_from_rollout(thread, fallback_rollout_path=fallback_rollout_path)
    if flat is not None:
        return list(collapse_flat_entries(flat)), True
    return list(render_entries(thread)), False


def _entries_from_rollout(thread: Any, *, fallback_rollout_path: str | None = None) -> list[dict[str, Any]] | None:
    """Materialise entries from the on-disk rollout, or return None to fall back.

    Returning ``None`` (vs. an empty list) is what triggers the SDK fallback;
    an empty list is treated as "the rollout exists and is genuinely empty,"
    matching the behaviour of an empty ``Thread.turns``.
    """
    path = getattr(thread, "path", None) or fallback_rollout_path
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
    if getattr(thread, "turns", None) and not any(entry["kind"] in ("user", "agent") for entry in entries):
        logger.warning("rollout %s yielded no user/agent entries; falling back to SDK turns", path)
        return None
    return entries
