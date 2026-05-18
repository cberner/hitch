"""Tail a Codex worker's JSONL events file and yield Server-Sent Events.

The detached worker writes one ``{"method": ..., "payload": ...}`` JSON object
per line into ``CodexInstance.events_path``. This module re-reads that file
incrementally and re-emits each line as an SSE ``data:`` frame, ending with a
named ``end`` event once the worker's CodexInstance row transitions to a
terminal status (or the worker process dies without reporting one).

The generator is driven by a Django ``StreamingHttpResponse`` and intentionally
blocks on a short sleep when the file has no new bytes — this is a single-user
dev tool, so holding one request-handler thread per active turn is acceptable.

The session page also subscribes to this stream when no worker is active so
its connection indicator can show a live ``connected`` state. ``idle_stream``
serves that case: it stays open emitting ``heartbeat`` events and ends with a
reload signal as soon as a worker for the session shows up (e.g. spawned from
another tab).
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Iterator
from pathlib import Path

from hitch.main import codex_pool
from hitch.main.models import CodexInstance, SystemWorkflow

# Cadence at which we re-poll the events file when it has no new bytes. Short
# enough that streamed deltas surface in near-real-time; long enough not to
# pin a CPU when a turn is mostly waiting on the model.
_POLL_INTERVAL = 0.2

# Cadence at which idle_stream re-checks the database for a newly-spawned
# worker. Only used when no worker is active for the session, so it doesn't
# need to be as snappy as ``_POLL_INTERVAL``.
_IDLE_POLL_INTERVAL = 1.0

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
# for hours doesn't pin a request-handler thread for the full 30-minute cap.
# EventSource transparently reconnects after the response closes.
_IDLE_MAX_STREAM_SECONDS = 5 * 60

# Upper bound on how long we wait for the events file to appear before giving
# up. ``_spawn_worker`` creates the row before launching the subprocess, so on
# a healthy host the file shows up within a fraction of a second; this caps
# the case where the subprocess never started writing.
_FILE_APPEAR_TIMEOUT = 30.0


def stream_for_instance(instance: CodexInstance) -> Iterator[bytes]:
    """Yield SSE frames (as bytes) for a single CodexInstance.

    Always ends with a named ``end`` event so the client can stop its
    EventSource explicitly rather than relying on the connection close.
    """
    yield b"retry: 2000\n\n"
    yield _heartbeat_frame(working=True)

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
            yield _heartbeat_frame(working=True)
            last_heartbeat = time.monotonic()
        time.sleep(_POLL_INTERVAL)

    with path.open("r", encoding="utf-8") as fh:
        buffer = ""
        deadline = time.monotonic() + _MAX_STREAM_SECONDS
        while True:
            chunk = fh.read()
            if chunk:
                buffer += chunk
                buffer = yield from _emit_complete_lines(buffer)
                continue

            done = _is_done(instance.pk)
            if done:
                # Drain anything the worker flushed after we last read so the
                # final agent message / turn/completed event isn't dropped on
                # the close race.
                chunk = fh.read()
                if chunk:
                    buffer += chunk
                buffer = yield from _emit_complete_lines(buffer)
                yield _end_frame(_current_status(instance.pk))
                return

            if time.monotonic() > deadline:
                yield _end_frame("timeout")
                return

            if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
                yield _heartbeat_frame(working=True)
                last_heartbeat = time.monotonic()
            time.sleep(_POLL_INTERVAL)


def idle_stream(session_id: str, baseline_id: int | None) -> Iterator[bytes]:
    """Long-running stream for a session with no active worker.

    Keeps the SSE channel open so the page's connection indicator can show
    a healthy ``connected, idle`` state, and watches for a new worker
    spawned out-of-band (e.g. from another tab) so the page reloads itself
    into the live-streaming UI as soon as one appears.

    ``baseline_id`` is the highest ``CodexInstance.pk`` the page knew
    about when it rendered (passed by the view, not resampled here). The
    caller has already verified that the page-render state matches the
    current DB state — so any later change to the latest pk is by
    definition a new out-of-band turn that the page hasn't seen yet.
    Keying off this baseline rather than "is anything currently active"
    catches fast turns that start and complete between two polls.

    Closes without an ``end`` event when the per-stream cap is hit so
    EventSource transparently reconnects rather than triggering a page
    reload on every recycle.
    """
    yield b"retry: 2000\n\n"
    yield _heartbeat_frame(working=False)
    deadline = time.monotonic() + _IDLE_MAX_STREAM_SECONDS
    last_heartbeat = time.monotonic()
    while True:
        if codex_pool.latest_id_for_thread(session_id) != baseline_id:
            # A worker showed up after the page rendered (still running,
            # or already terminal from a fast turn). End the stream so the
            # client reloads and re-renders with the live UI.
            yield _end_frame("active")
            return
        if time.monotonic() > deadline:
            return
        if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
            yield _heartbeat_frame(working=False)
            last_heartbeat = time.monotonic()
        time.sleep(_IDLE_POLL_INTERVAL)


def system_workflow_stream(
    session_id: str, baseline_id: int | None, workflow_id: int
) -> Iterator[bytes]:
    """Heartbeat stream while a hidden system workflow owns the main thread."""
    yield b"retry: 2000\n\n"
    yield _heartbeat_frame(working=True)
    deadline = time.monotonic() + _IDLE_MAX_STREAM_SECONDS
    last_heartbeat = time.monotonic()
    while True:
        if codex_pool.latest_id_for_thread(session_id) != baseline_id:
            yield _end_frame("active")
            return
        if not SystemWorkflow.objects.filter(
            pk=workflow_id,
            main_thread_id=session_id,
            status=SystemWorkflow.STATUS_RUNNING,
        ).exists():
            yield _end_frame("workflow")
            return
        if time.monotonic() > deadline:
            return
        if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
            yield _heartbeat_frame(working=True)
            last_heartbeat = time.monotonic()
        time.sleep(_IDLE_POLL_INTERVAL)


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


def _emit_complete_lines(buffer: str) -> Generator[bytes, None, str]:
    """Emit one SSE frame per newline-terminated line; return the trailing
    partial line so the caller can fold the next read into it.
    """
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line:
            continue
        yield b"data: " + line.encode("utf-8") + b"\n\n"
    return buffer


def _end_frame(status: str) -> bytes:
    return b"event: end\ndata: " + json.dumps({"status": status}).encode("utf-8") + b"\n\n"


def _heartbeat_frame(*, working: bool) -> bytes:
    payload = json.dumps({"working": working}).encode("utf-8")
    return b"event: heartbeat\ndata: " + payload + b"\n\n"


def _is_done(instance_id: int) -> bool:
    """Treat an instance as done once its status is terminal *or* its worker
    process is no longer alive. The latter catches workers that crashed
    before recording a status — we'd otherwise tail forever.
    """
    try:
        instance = CodexInstance.objects.get(pk=instance_id)
    except CodexInstance.DoesNotExist:
        return True
    if instance.status in (CodexInstance.STATUS_COMPLETED, CodexInstance.STATUS_FAILED):
        return True
    return bool(instance.pid) and not codex_pool.is_alive(instance.pid)


def _current_status(instance_id: int) -> str:
    try:
        return CodexInstance.objects.values_list("status", flat=True).get(pk=instance_id)
    except CodexInstance.DoesNotExist:
        return "unknown"
