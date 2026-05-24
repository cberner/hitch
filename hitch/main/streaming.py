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
from typing import Any

from hitch.main import codex_events, codex_pool
from hitch.main.models import (
    CodexInstance,
    SessionDemo,
    SystemAgentRun,
    SystemWorkflow,
    UserInputRequest,
)

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

_QA_AGENT_DISPLAY_AUTHOR = "QA agent"
_QA_PANEL_DISPLAY_AUTHOR = "QA panel"
_QA_AGENT_KIND = "pr_qa"
_PR_MONITOR_AGENT_KIND = "pr_followup_monitor"
_STEP_QA_RUNNING = "qa_running"
_STEP_FEEDBACK_RUNNING = "feedback_running"
_STEP_PR_PROMPT_RUNNING = "pr_prompt_running"
_STEP_PR_MONITORING = "pr_monitoring"
_STEP_PR_FEEDBACK_RUNNING = "pr_feedback_running"
_STEP_SPEC_CRITIC_ANALYZING = "spec_critic_analyzing"
_STEP_SPEC_CRITIC_CLARIFYING = "spec_critic_clarifying"
_STEP_SPEC_CRITIC_SYNTHESIZING = "spec_critic_synthesizing"
_QA_PANEL_SYNTHESIZER_AGENT_KIND = "pr_qa_panel_synthesizer"
_QA_PANEL_AGENT_KIND_PREFIX = "pr_qa_"
_SPEC_CRITIC_WORKFLOW_KIND = "spec_critic"
_COMPACT_TOKEN_UNITS = (
    (1_000_000_000, "B"),
    (1_000_000, "M"),
    (1_000, "K"),
)


def stream_for_instance(
    instance: CodexInstance, *, demo_baseline: str | None = None
) -> Iterator[bytes]:
    """Yield SSE frames (as bytes) for a single CodexInstance.

    Always ends with a named ``end`` event so the client can stop its
    EventSource explicitly rather than relying on the connection close.
    """
    yield b"retry: 2000\n\n"
    yield _heartbeat_frame(
        working=True,
        status_text=qa_agent_status_text_for_instance(instance),
        workflow=_workflow_for_instance(instance),
    )

    path = Path(instance.events_path)
    started = time.monotonic()
    last_heartbeat = time.monotonic()
    while not path.exists():
        if _demo_changed(instance.thread_id, demo_baseline):
            yield _end_frame("demo")
            return
        if _is_done(instance.pk):
            yield _end_frame(_current_status(instance.pk))
            return
        if time.monotonic() - started > _FILE_APPEAR_TIMEOUT:
            yield _end_frame("missing")
            return
        if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
            yield _heartbeat_frame(
                working=True,
                status_text=qa_agent_status_text_for_instance(instance),
                workflow=_workflow_for_instance(instance),
            )
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
                if _demo_changed(instance.thread_id, demo_baseline):
                    yield _end_frame("demo")
                    return
                continue

            if _demo_changed(instance.thread_id, demo_baseline):
                yield _end_frame("demo")
                return

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
                yield _heartbeat_frame(
                    working=True,
                    status_text=qa_agent_status_text_for_instance(instance),
                    workflow=_workflow_for_instance(instance),
                )
                last_heartbeat = time.monotonic()
            time.sleep(_POLL_INTERVAL)


def idle_stream(
    session_id: str,
    baseline_id: int | None,
    demo_baseline: str = "",
) -> Iterator[bytes]:
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
        if demo_stream_token(session_id) != demo_baseline:
            yield _end_frame("demo")
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
    workflow = _running_system_workflow(session_id, workflow_id)
    yield _heartbeat_frame(
        working=True,
        status_text=system_workflow_status_text(workflow),
        workflow=workflow,
    )
    deadline = time.monotonic() + _IDLE_MAX_STREAM_SECONDS
    last_heartbeat = time.monotonic()
    seen_inputs: dict[int, str] = {}
    if workflow is not None:
        yield from _workflow_input_request_frames(workflow.pk, seen_inputs)
    while True:
        if codex_pool.latest_id_for_thread(session_id) != baseline_id:
            yield _end_frame("active")
            return
        workflow = _running_system_workflow(session_id, workflow_id)
        if workflow is None:
            yield _end_frame("workflow")
            return
        yield from _workflow_input_request_frames(workflow.pk, seen_inputs)
        if time.monotonic() > deadline:
            return
        if time.monotonic() - last_heartbeat >= _HEARTBEAT_INTERVAL:
            yield _heartbeat_frame(
                working=True,
                status_text=system_workflow_status_text(workflow),
                workflow=workflow,
            )
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


def demo_stream_token(session_id: str) -> str:
    row = (
        SessionDemo.objects.filter(thread_id=session_id)
        .values_list("status", "updated_at")
        .first()
    )
    if row is None:
        return ""
    status, updated_at = row
    timestamp = updated_at.timestamp() if updated_at is not None else 0
    return f"{status}:{timestamp}"


def _demo_changed(session_id: str, demo_baseline: str | None) -> bool:
    return demo_baseline is not None and demo_stream_token(session_id) != demo_baseline


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


def _message_frame(method: str, payload: dict[str, object]) -> bytes:
    event = {"method": method, "payload": payload}
    return b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"


def _workflow_input_request_frames(
    workflow_id: int, seen: dict[int, str]
) -> Iterator[bytes]:
    requests = (
        UserInputRequest.objects.filter(
            instance__system_agent_runs__workflow_id=workflow_id
        )
        .order_by("created_at", "id")
        .values("id", "method", "params", "response")
    )
    for row in requests:
        request_id = row["id"]
        if not isinstance(request_id, int):
            continue
        method = row.get("method") if isinstance(row.get("method"), str) else ""
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        response = row.get("response")
        prior = seen.get(request_id)
        if prior is None:
            yield _message_frame(
                "input/requested",
                {"id": request_id, "method": method, "params": params},
            )
        marker = json.dumps(response, sort_keys=True) if response is not None else ""
        if marker and prior != marker:
            yield _message_frame(
                "input/resolved",
                {
                    "id": request_id,
                    "method": method,
                    "response": response if isinstance(response, dict) else {},
                },
            )
        seen[request_id] = marker


def _heartbeat_frame(
    *,
    working: bool,
    status_text: str = "",
    workflow: SystemWorkflow | None = None,
) -> bytes:
    payload_data: dict[str, Any] = {"working": working}
    if status_text:
        payload_data["statusText"] = status_text
    if workflow is not None and workflow.kind == SystemWorkflow.KIND_PR_QA:
        progress = pr_workflow_progress(workflow)
        payload_data["prWorkflowProgress"] = progress
    payload = json.dumps(payload_data).encode("utf-8")
    return b"event: heartbeat\ndata: " + payload + b"\n\n"


def pr_workflow_progress(workflow: SystemWorkflow | None) -> list[dict[str, str]]:
    if workflow is None or workflow.kind != SystemWorkflow.KIND_PR_QA:
        return []
    raw_gates = workflow.state.get("pr_gates") if isinstance(workflow.state, dict) else []
    if not isinstance(raw_gates, list) or not raw_gates:
        return []
    progress: list[dict[str, str]] = []
    for raw in raw_gates:
        if not isinstance(raw, dict):
            continue
        key = raw.get("key")
        label = raw.get("label")
        status = raw.get("status")
        summary = raw.get("summary")
        if not isinstance(key, str) or not isinstance(label, str):
            continue
        if status not in {"passed", "blocked", "pending", "checking", "stopped"}:
            status = "pending"
        progress.append(
            {
                "key": key,
                "label": label,
                "status": status,
                "statusLabel": _pr_gate_status_label(status),
                "summary": summary if isinstance(summary, str) else "",
            }
        )
    return progress


def _pr_gate_status_label(status: str) -> str:
    return {
        "blocked": "Blocked",
        "checking": "Checking",
        "passed": "Passed",
        "pending": "Pending",
        "stopped": "Stopped",
    }.get(status, "Pending")


def qa_agent_status_text_for_instance(instance: CodexInstance | None) -> str:
    if instance is None:
        return ""
    if instance.agent_kind == _PR_MONITOR_AGENT_KIND:
        tokens_used = codex_events.latest_goal_tokens_for_instance(instance)
        if tokens_used is None:
            return "PR follow-up agent working..."
        return (
            "PR follow-up agent working..."
            f"{_format_compact_token_count(tokens_used)} tokens"
        )
    if not _is_qa_agent_instance(instance):
        return ""
    tokens_used = codex_events.latest_goal_tokens_for_instance(instance)
    if tokens_used is None:
        return "QA agent working..."
    return f"QA agent working...{_format_compact_token_count(tokens_used)} tokens"


def system_workflow_status_text(workflow: SystemWorkflow | None) -> str:
    if workflow is None:
        return ""
    if workflow.kind == _SPEC_CRITIC_WORKFLOW_KIND:
        if workflow.step == _STEP_SPEC_CRITIC_CLARIFYING:
            return "Spec Critic is waiting for clarification..."
        if workflow.step == _STEP_SPEC_CRITIC_SYNTHESIZING:
            return "Spec Critic is synthesizing the brief..."
        if workflow.step == _STEP_SPEC_CRITIC_ANALYZING:
            return "Spec Critic is reviewing the request..."
        return "Spec Critic is preparing the implementation..."
    if workflow.kind != SystemWorkflow.KIND_PR_QA:
        return "Hitch system agent is working..."
    if workflow.step == _STEP_FEEDBACK_RUNNING:
        return "QA feedback agent is fixing feedback..."
    if workflow.step == _STEP_PR_PROMPT_RUNNING:
        return "PR agent is opening and following up..."
    if workflow.step == _STEP_PR_FEEDBACK_RUNNING:
        return "PR follow-up agent is fixing feedback..."
    if workflow.step == _STEP_PR_MONITORING:
        instance = _running_system_agent_instance(workflow.pk)
        if instance is not None and instance.agent_kind == _PR_MONITOR_AGENT_KIND:
            return "PR monitor is checking GitHub..."
        return "PR monitor is waiting..."
    if (
        workflow.step == _STEP_QA_RUNNING
        and workflow.state.get("qa_panel_enabled") is True
    ):
        return _qa_panel_status_text(workflow)
    instance = _running_system_agent_instance(workflow.pk)
    status_text = qa_agent_status_text_for_instance(instance)
    return status_text or "QA agent working..."


def _is_qa_agent_instance(instance: CodexInstance) -> bool:
    return (
        instance.display_author == _QA_AGENT_DISPLAY_AUTHOR
        or instance.display_author == _QA_PANEL_DISPLAY_AUTHOR
        or instance.agent_kind == _QA_AGENT_KIND
        or instance.agent_kind == _QA_PANEL_SYNTHESIZER_AGENT_KIND
        or instance.agent_kind.startswith(_QA_PANEL_AGENT_KIND_PREFIX)
    )


def _qa_panel_status_text(workflow: SystemWorkflow) -> str:
    runs = list(
        SystemAgentRun.objects.filter(workflow=workflow)
        .select_related("instance")
        .order_by("created_at", "id")
    )
    synthesizer = next(
        (
            run
            for run in reversed(runs)
            if run.agent_kind == _QA_PANEL_SYNTHESIZER_AGENT_KIND
            and run.status == SystemAgentRun.STATUS_RUNNING
            and _system_agent_run_iteration(run) == workflow.iteration
        ),
        None,
    )
    if synthesizer is not None:
        tokens = codex_events.latest_goal_tokens_for_instance(synthesizer.instance)
        token_text = (
            f"{_format_compact_token_count(tokens)} tokens" if tokens is not None else ""
        )
        return "QA panel synthesizing..." + token_text

    lane_runs = [
        run
        for run in runs
        if run.agent_kind.startswith(_QA_PANEL_AGENT_KIND_PREFIX)
        and run.agent_kind != _QA_PANEL_SYNTHESIZER_AGENT_KIND
        and _system_agent_run_iteration(run) == workflow.iteration
    ]
    total = len(lane_runs)
    completed = sum(1 for run in lane_runs if run.status == SystemAgentRun.STATUS_COMPLETED)
    token_total = 0
    has_tokens = False
    for run in lane_runs:
        tokens = codex_events.latest_goal_tokens_for_instance(run.instance)
        if tokens is None:
            continue
        has_tokens = True
        token_total += tokens
    progress = f"{completed}/{total} lanes complete" if total else "starting"
    token_text = (
        f", {_format_compact_token_count(token_total)} tokens" if has_tokens else ""
    )
    return f"QA panel working...{progress}{token_text}"


def _system_agent_run_iteration(run: SystemAgentRun) -> int:
    value = run.input.get("iteration") if isinstance(run.input, dict) else 0
    return value if isinstance(value, int) and value >= 0 else 0


def _format_compact_token_count(value: int) -> str:
    value = max(0, value)
    for index, (scale, suffix) in enumerate(_COMPACT_TOKEN_UNITS):
        if value < scale:
            continue
        amount = _format_compact_token_amount(value, scale)
        if amount == "1000" and index > 0:
            next_scale, next_suffix = _COMPACT_TOKEN_UNITS[index - 1]
            return _format_compact_token_amount(value, next_scale) + next_suffix
        return amount + suffix
    return str(value)


def _format_compact_token_amount(value: int, scale: int) -> str:
    if value >= 10 * scale:
        return str((value + scale // 2) // scale)
    tenths = (value * 10 + scale // 2) // scale
    whole, fraction = divmod(tenths, 10)
    if fraction == 0:
        return str(whole)
    return f"{whole}.{fraction}"


def _running_system_workflow(
    session_id: str,
    workflow_id: int,
) -> SystemWorkflow | None:
    return SystemWorkflow.objects.filter(
        pk=workflow_id,
        main_thread_id=session_id,
        status=SystemWorkflow.STATUS_RUNNING,
    ).first()


def _running_system_agent_instance(workflow_id: int) -> CodexInstance | None:
    run = (
        SystemAgentRun.objects.filter(
            workflow_id=workflow_id,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        .select_related("instance")
        .order_by("-created_at")
        .first()
    )
    return run.instance if run is not None else None


def _workflow_for_instance(instance: CodexInstance) -> SystemWorkflow | None:
    if instance.workflow_id is None:
        return None
    return SystemWorkflow.objects.filter(pk=instance.workflow_id).first()


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
    return bool(instance.pid) and not codex_pool.worker_is_alive(instance)


def _current_status(instance_id: int) -> str:
    try:
        return CodexInstance.objects.values_list("status", flat=True).get(pk=instance_id)
    except CodexInstance.DoesNotExist:
        return "unknown"
