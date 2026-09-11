"""Continue registered PR watches between visible coding turns."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

from hitch.main.models import CodexInstance, SessionMetadata, SessionPullRequest
from hitch.main.runtime import codex_pool, rate_limit, server_lifecycle
from hitch.main.sessions import agent_tasks, lifecycle
from hitch.main.sessions.session_settings import _is_allowed_session_cwd
from hitch.main.workflows import pr_tracking, pr_watch

logger = logging.getLogger(__name__)
_OBSERVATION_SECONDS = 60
_SCHEDULER_SECONDS = 60
_scheduler = server_lifecycle.SchedulerHandle(
    thread_name="hitch-pr-watch",
    tick_interval_seconds=_SCHEDULER_SECONDS,
)


def start_pr_watch_scheduler() -> bool:
    """Called by the enabled maintenance owner; GitHub reads run separately."""
    return _scheduler.start(_scheduler_loop)


def _scheduler_loop() -> None:
    stop = threading.Event()
    while True:
        _scheduler.run_tick(poll_registered_prs)
        stop.wait(_SCHEDULER_SECONDS)


def poll_registered_prs() -> None:
    groups: dict[str, list[SessionPullRequest]] = {}
    for record in SessionPullRequest.objects.filter(state__watch_active=True):
        if _thread_is_busy(record.thread_id):
            continue
        url = pr_tracking.pr_handoff_for_record(record)["url"].lower()
        groups.setdefault(url, []).append(record)
    for url, records in groups.items():
        try:
            key = "pr-watch:" + hashlib.sha256(url.encode()).hexdigest()
            if not rate_limit.claim(key):
                continue
            observation = pr_watch.observe_pr(
                cwd=records[0].cwd,
                url=url,
                command_timeout_seconds=pr_watch._command_timeout_provider(
                    deadline=time.monotonic() + _OBSERVATION_SECONDS,
                    monotonic=time.monotonic,
                    cancel_requested=lambda: False,
                ),
            )
        except Exception:
            logger.exception("PR watch poll failed for %s; will retry", url)
            continue
        for record in records:
            try:
                _record_observation(record, observation)
            except Exception:
                logger.exception("PR watch delivery failed for %s; will retry", record.thread_id)


def _thread_is_busy(thread_id: str) -> bool:
    return CodexInstance.objects.filter(thread_id=thread_id, status__in=CodexInstance.ACTIVE_STATUSES).exists()


def _record_observation(record: SessionPullRequest, observation: dict[str, Any]) -> None:
    registration = pr_tracking.registration_for_record(record)
    with lifecycle.hold(record.thread_id, blocking=False) as acquired:
        if not acquired:
            return
        record.refresh_from_db()
        if (
            not pr_tracking._registration_owns_record(record, registration)
            or record.state.get(pr_tracking.WATCH_ACTIVE_STATE_KEY) is not True
        ):
            return
        if _thread_is_busy(record.thread_id):
            return
        result = pr_watch._watch_result(
            observation,
            previous_feedback_fingerprint=pr_tracking._previous_feedback_fingerprint(record),
            previous_event_fingerprint=str(record.state.get(pr_tracking.WATCH_DELIVERED_STATE_KEY, "")),
        )
        snapshot = result or pr_watch._result_from_observation("pending", observation)
        pr_tracking.record_pr_watch_result(registration, snapshot, delivered=False)
        if result is None or result["status"] not in {"attention", "action_required"}:
            return
        if pr_watch.event_fingerprint(result) == record.state.get(pr_tracking.WATCH_DELIVERED_STATE_KEY):
            return
        if SessionMetadata.objects.filter(thread_id=record.thread_id, codex_archived=True).exists():
            return
        previous = (
            CodexInstance.objects.filter(
                thread_id=record.thread_id,
                purpose=CodexInstance.PURPOSE_USER,
            )
            .order_by("-pk")
            .first()
        )
        if previous is None:
            return
        _resume_watch(record, previous)


def _resume_watch(record: SessionPullRequest, previous: CodexInstance) -> None:
    if not _is_allowed_session_cwd(record.cwd):
        logger.warning("PR watch cannot resume %s: checkout is no longer allowed", record.thread_id)
        return
    task = agent_tasks.watch_pr_task(pr_tracking.pr_handoff_for_record(record)["url"])
    kwargs: dict[str, Any] = {
        key: getattr(previous, key)
        for key in (
            "model",
            "reasoning_effort",
            "sandbox_policy",
            "approval_mode",
            "web_search_mode",
            "enable_memories",
            "developer_instructions",
            "hitch_extra_instructions",
        )
    }
    codex_pool.spawn_turn(
        thread_id=record.thread_id,
        cwd=record.cwd,
        prompt=task.prompt,
        agent_kind=task.agent_kind,
        **kwargs,
    )
