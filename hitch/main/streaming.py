"""Tail a Codex worker's JSONL events file and yield Server-Sent Events.

The detached worker writes one ``{"method": ..., "payload": ...}`` JSON object
per line into ``CodexInstance.events_path``. This module re-reads that file
incrementally and re-emits each line as an SSE ``data:`` frame, ending with a
named ``end`` event once the worker's CodexInstance row transitions to a
terminal status (or the worker process dies without reporting one).

The generator is driven by a Django ``StreamingHttpResponse`` and intentionally
blocks on a short sleep when the file has no new bytes — this is a single-user
dev tool, so holding one request-handler thread per active turn is acceptable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Iterator
from pathlib import Path

from hitch.main import codex_pool
from hitch.main.models import CodexInstance

# Cadence at which we re-poll the events file when it has no new bytes. Short
# enough that streamed deltas surface in near-real-time; long enough not to
# pin a CPU when a turn is mostly waiting on the model.
_POLL_INTERVAL = 0.2

# Hard ceiling on how long a single stream connection stays open. Without this
# a hung worker (or a row stuck in ``running`` past reconciliation) would hold
# a Django request-handler thread indefinitely. Browsers will reconnect via
# EventSource if the user is still on the page.
_MAX_STREAM_SECONDS = 60 * 30

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

    path = Path(instance.events_path)
    started = time.monotonic()
    while not path.exists():
        if _is_done(instance.pk):
            yield _end_frame(_current_status(instance.pk))
            return
        if time.monotonic() - started > _FILE_APPEAR_TIMEOUT:
            yield _end_frame("missing")
            return
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

            # SSE heartbeat: a comment line keeps the connection alive past
            # idle proxies/load-balancers without surfacing as an event on
            # the client side.
            yield b": keepalive\n\n"
            time.sleep(_POLL_INTERVAL)


def empty_stream() -> Iterator[bytes]:
    """Stream that immediately closes — used when there is no active worker."""
    yield b"retry: 2000\n\n"
    yield _end_frame("inactive")


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
