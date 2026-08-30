"""Tail a Codex worker's JSONL events file and yield Server-Sent Events.

The detached worker writes one ``{"method": ..., "payload": ...}`` JSON object
per line into ``CodexInstance.events_path``. This module re-reads that file
incrementally and re-emits each line as an SSE ``data:`` frame, ending with a
named ``end`` event once the worker's CodexInstance row transitions to a
terminal status (or the worker process dies without reporting one).

The generator is driven by a Django ``StreamingHttpResponse`` and intentionally
blocks on a short sleep when the file has no new bytes — this is a single-user
dev tool, so holding one request-handler thread per active turn is acceptable.

The session page also probes this stream when no worker is active so its
connection indicator can show a live ``connected`` state. ``idle_stream``
serves that case with a heartbeat followed by a client-directed reconnect.
Keeping idle connections short prevents dormant tabs from occupying every
blocking WSGI request thread.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

from django.db import close_old_connections

from hitch.main.models import CodexInstance
from hitch.main.runtime import codex_pool, reconciliation
from hitch.main.runtime.db import run_ignoring_database_locks

# Cadence at which we re-poll the events file when it has no new bytes. Short
# enough that streamed deltas surface in near-real-time; long enough not to
# pin a CPU when a turn is mostly waiting on the model.
_POLL_INTERVAL = 0.2

# Cadence for the named ``heartbeat`` event the client uses to drive its
# connection-status indicator. The indicator enters a recoverable reconnecting
# state if no heartbeat (or other frame) lands within roughly two intervals,
# so this should stay comfortably shorter than the visible stale threshold.
_HEARTBEAT_INTERVAL = 3.0

# Hard ceiling on how long a single stream connection stays open. Without this
# a hung worker (or a row stuck in ``running`` past reconciliation) would hold
# a Django request-handler thread indefinitely. Browsers will reconnect via
# EventSource if the user is still on the page.
_MAX_STREAM_SECONDS = 60 * 30

# Idle streams (no active worker) recycle more aggressively so a tab left open
# for hours doesn't pin a request-handler thread. The client handles the named
# reconnect event without treating the intentional recycle as transport loss.
_IDLE_RECONNECT_MILLISECONDS = 5000

# The installed service has 64 request threads. Bound long-lived worker streams
# below that so ordinary pages retain capacity even when many active sessions
# are open. Idle streams do not consume these slots.
_ACTIVE_STREAM_SLOTS = threading.BoundedSemaphore(48)

_COMPACTABLE_TEXT_DELTA_METHODS = frozenset(
    {
        "item/agentMessage/delta",
        "item/plan/delta",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
    }
)

# Upper bound on how long we wait for the events file to appear before giving
# up. ``_spawn_worker`` creates the row before launching the subprocess, so on
# a healthy host the file shows up within a fraction of a second; this caps
# the case where the subprocess never started writing.
_FILE_APPEAR_TIMEOUT = 30.0



def stream_for_instance(
    instance: CodexInstance,
) -> Iterator[bytes]:
    """Yield SSE frames (as bytes) for a single CodexInstance.

    Always ends with a named ``end`` event so the client can stop its
    EventSource explicitly rather than relying on the connection close.
    """
    yield b"retry: 2000\n\n"
    yield _heartbeat_frame(
        working=True,
        status_text="",
    )

    path = Path(instance.events_path)
    started = time.monotonic()
    last_heartbeat = time.monotonic()
    while not path.exists():
        if _is_done(instance.pk):
            yield _end_frame(_current_status(instance.pk))
            return
        if time.monotonic() - started > _FILE_APPEAR_TIMEOUT:
            yield _end_frame("missing")
            return
        if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
            yield _heartbeat_frame(
                working=True,
                status_text="",
            )
            last_heartbeat = time.monotonic()
        time.sleep(_POLL_INTERVAL)

    # Read in binary and split on the newline byte: the worker writes
    # line-buffered UTF-8, so a concurrent reader can observe a flush that ends
    # mid-multibyte-character. A strict text-mode read would raise
    # UnicodeDecodeError on that torn prefix, abort the generator, and drop the
    # connection with no terminating frame; on EventSource reconnect the stream
    # re-reads from byte 0 and replays additive deltas. Buffering raw bytes and
    # forwarding only complete lines keeps the torn tail until the worker
    # finishes writing it.
    with path.open("rb") as fh:
        buffer = b""
        initial_backlog = True
        deadline = time.monotonic() + _MAX_STREAM_SECONDS
        while True:
            chunk = fh.read()
            if chunk:
                buffer += chunk
                if initial_backlog:
                    buffer = yield from _emit_initial_backlog(buffer)
                    initial_backlog = False
                else:
                    buffer = yield from _emit_complete_lines(buffer)
                continue

            if initial_backlog:
                initial_backlog = False

            done = _is_done(instance.pk)
            if done:
                # Drain anything the worker flushed after we last read so the
                # final agent message / turn/completed event isn't dropped on
                # the close race.
                chunk = fh.read()
                if chunk:
                    buffer += chunk
                if initial_backlog:
                    buffer = yield from _emit_initial_backlog(buffer)
                    initial_backlog = False
                else:
                    buffer = yield from _emit_complete_lines(buffer)
                yield _end_frame(_current_status(instance.pk))
                return

            if time.monotonic() > deadline:
                yield _end_frame("timeout")
                return

            if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
                yield _heartbeat_frame(
                    working=True,
                    status_text="",
                )
                last_heartbeat = time.monotonic()
            time.sleep(_POLL_INTERVAL)


def capacity_limited_stream(stream: Iterator[bytes]) -> Iterator[bytes]:
    """Yield an active stream while preserving capacity for ordinary requests."""
    if not _ACTIVE_STREAM_SLOTS.acquire(blocking=False):
        yield f"retry: {_IDLE_RECONNECT_MILLISECONDS}\n\n".encode()
        yield _reconnect_frame()
        return
    try:
        yield from stream
    finally:
        _ACTIVE_STREAM_SLOTS.release()


def idle_stream() -> Iterator[bytes]:
    """Send one idle heartbeat and direct the client to reconnect shortly.

    ``session_stream`` validates the page's worker baseline on every
    connection. Reconnecting therefore retains out-of-band change detection
    without holding one blocking WSGI thread per idle browser tab.
    """
    yield f"retry: {_IDLE_RECONNECT_MILLISECONDS}\n\n".encode()
    yield _heartbeat_frame(working=False)
    yield _reconnect_frame()


def reload_stream() -> Iterator[bytes]:
    """Immediate ``event: end`` so the client reloads.

    Used when ``session_stream`` detects that the page was rendered
    against a state that no longer matches the database — e.g. a worker
    was spawned or completed between page render and SSE open. The
    reload re-runs the session view so the DOM matches reality (live-
    root present when needed, pending bubble cleared, etc.) before any
    streamed item events start arriving.
    """
    yield b"retry: 2000\n\n"
    yield _end_frame("stale")


def _reconnect_frame() -> bytes:
    return (
        "event: reconnect\n"
        f"data: {{\"afterMs\": {_IDLE_RECONNECT_MILLISECONDS}}}\n\n"
    ).encode()


def _emit_complete_lines(buffer: bytes) -> Generator[bytes, None, bytes]:
    """Emit one SSE frame per newline-terminated line; return the trailing
    partial line so the caller can fold the next read into it.

    Operates on bytes: a complete (newline-terminated) line the worker wrote is
    always well-formed UTF-8, so the raw bytes are forwarded as-is. An
    unterminated trailing fragment — which may end mid-multibyte-character if it
    was observed during a partial flush — is returned untouched and completed by
    the next read instead of being decoded eagerly.
    """
    lines, trailing = _split_complete_lines(buffer)
    for line in lines:
        yield _data_frame(line)
    return trailing


def _emit_initial_backlog(buffer: bytes) -> Generator[bytes, None, bytes]:
    """Emit already-written worker events without replaying completed deltas.

    A session page can attach to a long-running worker after thousands of events
    already exist, or reconnect to the same worker while keeping its current DOM.
    The stream can compact only items that reached ``item/completed`` inside the
    replay window: incomplete items must keep their original started/delta
    sequence because those deltas may be missed live updates for an existing DOM
    node.
    """
    lines, trailing = _split_complete_lines(buffer)
    methods = [_event_line_method(line) for line in lines]
    events: list[dict[str, Any] | None] = [
        _decode_backlog_event_line(method, line)
        for method, line in zip(methods, lines, strict=True)
    ]
    completed_item_ids: set[str] = set()
    latest_diff_index = next(
        (
            index
            for index in range(len(methods) - 1, -1, -1)
            if methods[index] == "turn/diff/updated"
        ),
        None,
    )

    for event in events:
        if event is None:
            continue
        method = event.get("method")
        payload = event.get("payload")
        if not isinstance(method, str) or not isinstance(payload, dict):
            continue
        if method == "item/completed":
            item_id = _event_item_id(payload)
            if item_id:
                completed_item_ids.add(item_id)

    for index, (line, method, event) in enumerate(
        zip(lines, methods, events, strict=True)
    ):
        # The session JS does not render command output deltas, and backlog
        # copies can be megabytes each; completed tool summaries arrive via
        # item/completed. Check only the event method so command text like
        # "rg outputDelta" does not make the owning item disappear.
        if _is_item_output_delta(method):
            continue
        # Each diff update is a full snapshot. Replaying superseded snapshots
        # makes reconnect responses grow with every edit while conveying no
        # additional state to the browser.
        if method == "turn/diff/updated" and index != latest_diff_index:
            continue
        if event is None:
            yield _data_frame(line)
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            yield _data_frame(line)
            continue
        if method == "item/started":
            item_id = _event_item_id(payload)
            if item_id in completed_item_ids:
                continue
        elif method in _COMPACTABLE_TEXT_DELTA_METHODS:
            item_id = _event_payload_item_id(payload)
            if item_id in completed_item_ids:
                continue
        yield _data_frame(line)
    return trailing


def _split_complete_lines(buffer: bytes) -> tuple[list[bytes], bytes]:
    if not buffer:
        return [], b""
    parts = buffer.split(b"\n")
    trailing = parts.pop()
    return [line.strip() for line in parts if line.strip()], trailing


def _event_line_method(line: bytes) -> str:
    offset = 0
    while offset < len(line) and line[offset] in b" \t\r\n":
        offset += 1
    if offset >= len(line) or line[offset] != ord("{"):
        return ""
    offset += 1
    while offset < len(line) and line[offset] in b" \t\r\n":
        offset += 1
    prefix = b'"method"'
    if not line.startswith(prefix, offset):
        return ""
    offset += len(prefix)
    while offset < len(line) and line[offset] in b" \t\r\n":
        offset += 1
    if offset >= len(line) or line[offset] != ord(":"):
        return ""
    offset += 1
    while offset < len(line) and line[offset] in b" \t\r\n":
        offset += 1
    if offset >= len(line) or line[offset] != ord('"'):
        return ""
    offset += 1
    end = line.find(b'"', offset)
    if end < 0:
        return ""
    try:
        return line[offset:end].decode("ascii")
    except UnicodeDecodeError:
        return ""


def _decode_backlog_event_line(method: str, line: bytes) -> dict[str, Any] | None:
    if (
        method not in _COMPACTABLE_TEXT_DELTA_METHODS
        and not (
            method in {"item/started", "item/completed"}
            and _has_text_item_type(line)
        )
    ):
        return None
    return _decode_event_line(line)


def _decode_event_line(line: bytes) -> dict[str, Any] | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _data_frame(line: bytes) -> bytes:
    return b"data: " + line + b"\n\n"


def _event_item_id(payload: dict[str, Any]) -> str:
    item = payload.get("item")
    if not isinstance(item, dict):
        return ""
    item_id = item.get("id")
    return item_id if isinstance(item_id, str) else ""


def _event_payload_item_id(payload: dict[str, Any]) -> str:
    item_id = payload.get("itemId")
    return item_id if isinstance(item_id, str) else ""


def _is_item_output_delta(method: str) -> bool:
    return method.startswith("item/") and method.endswith("/outputDelta")


def _has_text_item_type(line: bytes) -> bool:
    return (
        b'"type": "agentMessage"' in line
        or b'"type":"agentMessage"' in line
        or b'"type": "plan"' in line
        or b'"type":"plan"' in line
        or b'"type": "reasoning"' in line
        or b'"type":"reasoning"' in line
    )


def _end_frame(status: str) -> bytes:
    return b"event: end\ndata: " + json.dumps({"status": status}).encode("utf-8") + b"\n\n"


def _heartbeat_frame(
    *,
    working: bool,
    status_text: str = "",
) -> bytes:
    payload_data: dict[str, Any] = {"working": working}
    if status_text:
        payload_data["statusText"] = status_text
    payload = json.dumps(payload_data).encode("utf-8")
    return b"event: heartbeat\ndata: " + payload + b"\n\n"


def _is_done(instance_id: int) -> bool:
    """Treat an instance as done once its status is terminal *or* its worker
    process is no longer alive. The latter catches workers that crashed
    before recording a status — we'd otherwise tail forever.
    """
    try:
        try:
            instance = CodexInstance.objects.get(pk=instance_id)
        except CodexInstance.DoesNotExist:
            return True
    finally:
        close_old_connections()
    if instance.status in (CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED):
        return True
    if bool(instance.pid) and not codex_pool.worker_is_alive(instance):
        _reconcile_dead_for_thread(instance.thread_id)
        return True
    return False


def _current_status(instance_id: int) -> str:
    try:
        try:
            return CodexInstance.objects.values_list("status", flat=True).get(
                pk=instance_id
            )
        except CodexInstance.DoesNotExist:
            return "unknown"
    finally:
        close_old_connections()


def _reconcile_dead_for_thread(session_id: str) -> None:
    try:
        run_ignoring_database_locks(
            lambda: reconciliation.reconcile_dead_for_thread(session_id),
            description="stream dead-worker reconcile",
        )
    finally:
        close_old_connections()
