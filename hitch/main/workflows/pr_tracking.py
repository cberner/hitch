"""Durable UI state for agent-invoked pull-request watches."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from django.db import transaction
from django.utils import timezone

from hitch.main.git_support import resolved_path
from hitch.main.models import (
    CodexInstance,
    Project,
    SessionMetadata,
    SessionPullRequest,
)
from hitch.main.repos import (
    AutoPullError,
    AutoPullResult,
    pull_default_branch_from_origin,
    repo_root,
    same_repo_or_worktree,
)
from hitch.main.runtime import rate_limit
from hitch.main.sessions.agent_tasks import (
    PR_AGENT_KINDS,
    PR_PUBLISH_AGENT_KIND,
    PR_WATCH_AGENT_KIND,
)
from hitch.main.workflows import pr_watch
from hitch.main.workflows.gh_cli import (
    _GH_CLI_TIMEOUT_SECONDS,
    _GH_PR_VIEW_FIELDS,
    _gh_pr_view_payload,
    _GhPrOpenError,
)
from hitch.main.workflows.pr_handoff import (
    _compact_pr_handoff,
    _merge_pr_handoff_dicts,
    _pr_handoff_head_changed,
    _pr_handoff_identity_changed,
    _pr_handoff_is_terminal,
)
from hitch.main.workflows.pr_stage_refresh_state import (
    _PR_STAGE_REFRESH_MIN_SECONDS,
    _pr_handoff_selector,
    _pr_stage_rate_limit_key,
    _pr_stage_refresh_globally_due,
    _should_refresh_pr_snapshot_for_stage,
)

logger = logging.getLogger(__name__)

PR_HANDOFF_STATE_KEY = "pr_handoff"
PR_GATES_STATE_KEY = "pr_gates"
AUTO_PULL_RESULT_STATE_KEY = "auto_pull_result"
_HITCH_HANDOFF_STATE_KEY = "hitch_pr_handoff"
_STAGE_REFRESH_STATE_KEY = "pr_stage_refresh"
_WATCH_OWNER_INSTANCE_STATE_KEY = SessionPullRequest.WATCH_OWNER_INSTANCE_STATE_KEY
_WATCH_OWNER_MESSAGE_STATE_KEY = "watch_owner_message_index"
_SUPERSEDED_BY_INSTANCE_STATE_KEY = (
    SessionPullRequest.SUPERSEDED_BY_INSTANCE_STATE_KEY
)
_SUPERSEDED_AT_STATE_KEY = "superseded_at"
_PR_STAGE_REFRESH_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class PrWatchRegistration:
    record_id: int
    owner_instance_id: int
    owner_message_index: int | None


@dataclass(frozen=True)
class OrdinaryPrWatchPreflight:
    record_id: int | None
    record_updated_at: datetime | None
    record_state: dict[str, Any] | None
    requires_checkout_validation: bool


@dataclass(frozen=True)
class _PrTarget:
    cwd: str


def ordinary_pr_watch_preflight(
    *, thread_id: str, requested_pr: dict[str, Any]
) -> OrdinaryPrWatchPreflight:
    record = SessionPullRequest.objects.filter(thread_id=thread_id).first()
    current_pr = pr_handoff_for_record(record)
    already_current = bool(
        record is not None
        and record.is_current
        and current_pr
        and pr_watch.pr_identity_matches(current_pr, requested_pr)
    )
    return OrdinaryPrWatchPreflight(
        record_id=record.pk if record is not None else None,
        record_updated_at=record.updated_at if record is not None else None,
        record_state=dict(record.state) if record is not None else None,
        requires_checkout_validation=not already_current,
    )


def begin_pr_watch_invocation(
    *,
    thread_id: str,
    cwd: str,
    instance_id: int,
    user_message_index: int | None,
    agent_kind: str,
    requested_pr: dict[str, Any],
    ordinary_preflight: OrdinaryPrWatchPreflight | None = None,
) -> tuple[PrWatchRegistration | None, str]:
    """Register an invocation-owned PR before entering the polling loop."""
    ordinary_turn = not agent_kind
    if not ordinary_turn and agent_kind not in PR_AGENT_KINDS:
        return None, ""
    with transaction.atomic():
        record, _created = SessionPullRequest.objects.select_for_update().get_or_create(
            thread_id=thread_id,
            defaults={"cwd": cwd},
        )
        if _record_has_newer_instance(
            record, instance_id
        ) or _newer_user_instance_exists(thread_id, instance_id):
            raise pr_watch.PrWatchError(
                "a newer session turn already owns this pull request"
            )
        current_pr = pr_handoff_for_record(record)
        registration_agent_kind = agent_kind
        if ordinary_turn:
            if not _ordinary_preflight_matches_record(
                ordinary_preflight, record, created=_created
            ):
                raise pr_watch.PrWatchError(
                    "session pull request changed before watch registration; retry"
                )
            assert ordinary_preflight is not None
            if ordinary_preflight.requires_checkout_validation:
                registration_agent_kind = PR_PUBLISH_AGENT_KIND
            elif (
                record.is_current
                and current_pr
                and pr_watch.pr_identity_matches(current_pr, requested_pr)
            ):
                registration_agent_kind = PR_WATCH_AGENT_KIND
            else:
                raise pr_watch.PrWatchError(
                    "session pull request changed before watch registration; retry"
                )
        if registration_agent_kind != PR_PUBLISH_AGENT_KIND and current_pr:
            _validate_pr_identity(current_pr, requested_pr)
        previous_fingerprint = _previous_feedback_fingerprint(record)
        identity_changed = bool(current_pr) and not pr_watch.pr_identity_matches(
            current_pr, requested_pr
        )
        state = dict(record.state)
        state.pop(_SUPERSEDED_BY_INSTANCE_STATE_KEY, None)
        state.pop(_SUPERSEDED_AT_STATE_KEY, None)
        if identity_changed:
            previous_fingerprint = ""
            state.pop(pr_watch.PR_WATCH_RESULT_STATE_KEY, None)
            state.pop(pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY, None)
            state.pop(PR_GATES_STATE_KEY, None)
            state.pop(AUTO_PULL_RESULT_STATE_KEY, None)
            state[PR_HANDOFF_STATE_KEY] = requested_pr
        else:
            state[PR_HANDOFF_STATE_KEY] = _merge_pr_handoff_dicts(
                current_pr, requested_pr
            )
        state[_HITCH_HANDOFF_STATE_KEY] = _handoff_marker(
            state[PR_HANDOFF_STATE_KEY]
        )
        state[_WATCH_OWNER_INSTANCE_STATE_KEY] = instance_id
        state[_WATCH_OWNER_MESSAGE_STATE_KEY] = user_message_index
        record.cwd = cwd
        record.state = state
        record.save(update_fields=["cwd", "state", "updated_at"])
        return (
            PrWatchRegistration(
                record_id=record.pk,
                owner_instance_id=instance_id,
                owner_message_index=user_message_index,
            ),
            previous_fingerprint,
        )


def _ordinary_preflight_matches_record(
    preflight: OrdinaryPrWatchPreflight | None,
    record: SessionPullRequest,
    *,
    created: bool,
) -> bool:
    if preflight is None:
        return False
    if preflight.record_id is None:
        return created
    return bool(
        not created
        and record.pk == preflight.record_id
        and record.updated_at == preflight.record_updated_at
        and record.state == preflight.record_state
    )


def _record_has_newer_instance(
    record: SessionPullRequest, instance_id: int
) -> bool:
    for key in (
        _WATCH_OWNER_INSTANCE_STATE_KEY,
        _SUPERSEDED_BY_INSTANCE_STATE_KEY,
    ):
        value = record.state.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > instance_id
        ):
            return True
    return False


def _newer_user_instance_exists(thread_id: str, instance_id: int) -> bool:
    return CodexInstance.objects.filter(
        thread_id=thread_id,
        purpose=CodexInstance.PURPOSE_USER,
        pk__gt=instance_id,
    ).exists()


def record_pr_watch_result(
    registration: PrWatchRegistration | None,
    result: dict[str, Any],
) -> None:
    if registration is None:
        return
    merged = False
    with transaction.atomic():
        record = (
            SessionPullRequest.objects.select_for_update()
            .filter(pk=registration.record_id)
            .first()
        )
        if record is None or not _registration_owns_record(record, registration):
            return
        current_pr = pr_handoff_for_record(record)
        observed_pr = _compact_pr_handoff(result.get("pr"))
        _validate_pr_identity(current_pr, observed_pr)
        state = {
            **record.state,
            pr_watch.PR_WATCH_RESULT_STATE_KEY: result,
            pr_watch.PR_WATCH_RESULT_TURN_INDEX_STATE_KEY: (
                registration.owner_message_index
            ),
            PR_GATES_STATE_KEY: (
                result.get("gates") if isinstance(result.get("gates"), list) else []
            ),
        }
        if observed_pr:
            state[PR_HANDOFF_STATE_KEY] = _merge_pr_handoff_dicts(
                current_pr, observed_pr
            )
        record.state = state
        record.save(update_fields=["state", "updated_at"])
        merged = result.get("status") == "terminal" and _pr_handoff_is_merged(
            pr_handoff_for_record(record)
        )
    if merged:
        _maybe_auto_pull_default_repo_after_pr_merge(registration)


def _registration_owns_record(
    record: SessionPullRequest, registration: PrWatchRegistration
) -> bool:
    return bool(
        record.is_current
        and record.state.get(_WATCH_OWNER_INSTANCE_STATE_KEY)
        == registration.owner_instance_id
        and record.state.get(_WATCH_OWNER_MESSAGE_STATE_KEY)
        == registration.owner_message_index
    )


def _validate_pr_identity(
    expected: dict[str, Any], observed: dict[str, Any]
) -> None:
    if expected and observed and not pr_watch.pr_identity_matches(expected, observed):
        raise pr_watch.PrWatchError(
            "url must identify the pull request registered for this session"
        )


def _previous_feedback_fingerprint(record: SessionPullRequest) -> str:
    previous = record.state.get(pr_watch.PR_WATCH_RESULT_STATE_KEY)
    if not isinstance(previous, dict):
        return ""
    value = previous.get("feedback_fingerprint")
    return value if isinstance(value, str) else ""


def record_for_thread(thread_id: str) -> SessionPullRequest | None:
    record = stored_record_for_thread(thread_id)
    return record if record_is_current(record) else None


def stored_record_for_thread(thread_id: str) -> SessionPullRequest | None:
    return SessionPullRequest.objects.filter(thread_id=thread_id).first()


def record_is_current(record: SessionPullRequest | None) -> bool:
    return record is not None and record.is_current


def watch_registered_by_instance(
    record: SessionPullRequest | None, instance_id: int
) -> bool:
    if not record_is_current(record):
        return False
    assert record is not None
    return bool(
        record.state.get(_WATCH_OWNER_INSTANCE_STATE_KEY) == instance_id
    )


def records_by_thread_id(thread_ids: list[str]) -> dict[str, SessionPullRequest]:
    return {
        record.thread_id: record
        for record in SessionPullRequest.objects.filter(thread_id__in=thread_ids)
    }


def pr_handoff_for_record(
    record: SessionPullRequest | None,
) -> dict[str, Any]:
    if record is None:
        return {}
    return _compact_pr_handoff(record.state.get(PR_HANDOFF_STATE_KEY))


def pr_watch_progress(state: dict[str, Any] | None) -> list[dict[str, str]]:
    raw_gates = state.get(PR_GATES_STATE_KEY) if isinstance(state, dict) else []
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


def pr_handoff_stage_refresh_due(record: SessionPullRequest | None) -> bool:
    if not record_is_current(record):
        return False
    assert record is not None
    handoff = pr_handoff_for_record(record)
    if not _should_refresh_record(record, handoff, force=False):
        return False
    return _pr_stage_refresh_globally_due(handoff)


def refresh_unarchived_session_pr_stages(*, limit: int | None = None) -> int:
    active_thread_ids = SessionMetadata.objects.filter(
        codex_archived=False,
        codex_updated_at__isnull=False,
    ).values_list("thread_id", flat=True)
    records = SessionPullRequest.objects.filter(
        thread_id__in=active_thread_ids
    ).order_by("updated_at", "pk")
    refreshed = 0
    for record in records:
        if limit is not None and refreshed >= limit:
            break
        if not pr_handoff_stage_refresh_due(record):
            continue
        if not _claim_pr_stage_refresh(record):
            continue
        refreshed_pr_handoff_for_stage(record, force=True)
        refreshed += 1
    return refreshed


def _claim_pr_stage_refresh(record: SessionPullRequest) -> bool:
    now = timezone.now()
    state = {
        **record.state,
        _STAGE_REFRESH_STATE_KEY: {"attempted_at": int(now.timestamp())},
    }
    updated = SessionPullRequest.objects.filter(
        pk=record.pk,
        updated_at=record.updated_at,
    ).update(state=state, updated_at=now)
    if updated != 1:
        return False
    record.state = state
    record.updated_at = now
    return True


def refreshed_pr_handoff_for_stage(
    record: SessionPullRequest | None, *, force: bool = False
) -> dict[str, Any]:
    if not record_is_current(record):
        return {}
    assert record is not None
    handoff = pr_handoff_for_record(record)
    if not _should_refresh_record(record, handoff, force=force):
        return handoff
    selector = _pr_handoff_selector(handoff)
    if not selector:
        return handoff
    rate_limit_key = _pr_stage_rate_limit_key(handoff)
    if not force and rate_limit_key and not rate_limit.claim(rate_limit_key):
        return handoff
    refresh_claimed, current = _begin_pr_stage_observation(record.pk, handoff)
    if not refresh_claimed:
        return current
    handoff = current
    selector = _pr_handoff_selector(handoff)
    if not selector:
        return handoff
    try:
        observed = _gh_pr_view(
            record,
            selector=selector,
            source_tool="gh_pr_stage_refresh",
            timeout_seconds=_PR_STAGE_REFRESH_TIMEOUT_SECONDS,
        )
    except _GhPrOpenError:
        logger.exception("failed to refresh PR stage for session %s", record.thread_id)
        return _current_pr_handoff(record.pk)
    if observed is None:
        return _current_pr_handoff(record.pk)
    return _record_pr_stage_observation(record.pk, handoff, observed)


def _begin_pr_stage_observation(
    record_id: int, expected: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Persist refresh backoff without overwriting a concurrent watch call."""
    with transaction.atomic():
        record = (
            SessionPullRequest.objects.select_for_update()
            .filter(pk=record_id)
            .first()
        )
        if record is None:
            return False, {}
        current = pr_handoff_for_record(record)
        if (
            not current
            or not pr_watch.pr_identity_matches(expected, current)
            or not _should_refresh_record(record, current, force=True)
        ):
            return False, current
        _mark_pr_stage_refresh_attempt(record)
        record.save(update_fields=["state", "updated_at"])
        return True, current


def _record_pr_stage_observation(
    record_id: int,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Merge a refresh only while the observed PR is still registered."""
    with transaction.atomic():
        record = (
            SessionPullRequest.objects.select_for_update()
            .filter(pk=record_id)
            .first()
        )
        if record is None:
            return {}
        current = pr_handoff_for_record(record)
        if (
            not record_is_current(record)
            or not current
            or not pr_watch.pr_identity_matches(expected, current)
            or not pr_watch.pr_identity_matches(current, observed)
        ):
            return current if record_is_current(record) else {}
        _merge_pr_handoff(record, observed)
        record.save(update_fields=["state", "updated_at"])
        return pr_handoff_for_record(record)


def _current_pr_handoff(record_id: int) -> dict[str, Any]:
    record = SessionPullRequest.objects.filter(pk=record_id).first()
    return pr_handoff_for_record(record) if record_is_current(record) else {}


def pr_snapshot_stage_refresh_due(
    *,
    cwd: str,
    snapshot: Mapping[str, Any] | None,
    attempted_at: datetime | None,
    force: bool = False,
) -> bool:
    handoff = _compact_pr_handoff(snapshot)
    if not _should_refresh_pr_snapshot_for_stage(
        cwd,
        handoff,
        attempted_at=attempted_at,
        force=force,
    ):
        return False
    return force or _pr_stage_refresh_globally_due(handoff)


def refreshed_pr_snapshot_for_stage(
    *,
    cwd: str,
    snapshot: Mapping[str, Any] | None,
    force: bool = False,
) -> dict[str, Any]:
    handoff = _compact_pr_handoff(snapshot)
    if not _should_refresh_pr_snapshot_for_stage(
        cwd,
        handoff,
        attempted_at=None,
        force=force,
    ):
        return handoff
    selector = _pr_handoff_selector(handoff)
    if not selector:
        return handoff
    rate_limit_key = _pr_stage_rate_limit_key(handoff)
    if not force and rate_limit_key and not rate_limit.claim(rate_limit_key):
        return handoff
    target = _PrTarget(cwd=cwd)
    try:
        observed = _gh_pr_view(
            target,
            selector=selector,
            source_tool="gh_pr_stage_refresh",
            timeout_seconds=_PR_STAGE_REFRESH_TIMEOUT_SECONDS,
        )
    except _GhPrOpenError:
        logger.exception("failed to refresh PR stage for %s", selector)
        return handoff
    if observed is None or _pr_handoff_identity_changed(handoff, observed):
        return handoff
    return _merge_pr_handoff_dicts(handoff, observed)


def _should_refresh_record(
    record: SessionPullRequest, handoff: dict[str, Any], *, force: bool
) -> bool:
    if not record_is_current(record) or _pr_handoff_is_terminal(handoff):
        return False
    if _handoff_marker(record.state.get(_HITCH_HANDOFF_STATE_KEY)) != _handoff_marker(
        handoff
    ):
        return False
    if not Path(record.cwd).is_dir():
        return False
    if force:
        return True
    last_attempted_at = _pr_stage_refresh_attempted_at(record)
    if last_attempted_at <= 0:
        return True
    return int(timezone.now().timestamp()) - last_attempted_at >= (
        _PR_STAGE_REFRESH_MIN_SECONDS
    )


def _mark_pr_stage_refresh_attempt(record: SessionPullRequest) -> None:
    record.state = {
        **record.state,
        _STAGE_REFRESH_STATE_KEY: {"attempted_at": int(timezone.now().timestamp())},
    }


def _pr_stage_refresh_attempted_at(record: SessionPullRequest) -> int:
    value = record.state.get(_STAGE_REFRESH_STATE_KEY)
    if not isinstance(value, dict):
        return 0
    attempted_at = value.get("attempted_at")
    return (
        attempted_at
        if isinstance(attempted_at, int) and not isinstance(attempted_at, bool)
        else 0
    )


def supersede_pr_after_turn(instance: CodexInstance) -> None:
    """Retire stale PR UI state after unrelated visible session activity."""
    if (
        instance.purpose != CodexInstance.PURPOSE_USER
        or instance.workflow_id is not None
        or instance.agent_kind == PR_WATCH_AGENT_KIND
        or not isinstance(instance.pk, int)
    ):
        return
    instance_id = instance.pk
    with transaction.atomic():
        record = (
            SessionPullRequest.objects.select_for_update()
            .filter(thread_id=instance.thread_id)
            .first()
        )
        if record is None:
            return
        owner_id = record.state.get(_WATCH_OWNER_INSTANCE_STATE_KEY)
        if owner_id == instance_id:
            return
        if (
            isinstance(owner_id, int)
            and not isinstance(owner_id, bool)
            and owner_id > instance_id
        ):
            return
        superseded_by = record.state.get(_SUPERSEDED_BY_INSTANCE_STATE_KEY)
        if (
            isinstance(superseded_by, int)
            and not isinstance(superseded_by, bool)
            and superseded_by >= instance_id
        ):
            return
        ended_at = getattr(instance, "ended_at", None) or timezone.now()
        record.state = {
            **record.state,
            _SUPERSEDED_BY_INSTANCE_STATE_KEY: instance_id,
            _SUPERSEDED_AT_STATE_KEY: int(ended_at.timestamp()),
        }
        record.save(update_fields=["state", "updated_at"])


def _merge_pr_handoff(record: SessionPullRequest, update: dict[str, Any]) -> None:
    current = pr_handoff_for_record(record)
    compact = _compact_pr_handoff(update)
    reset_gates = _pr_handoff_identity_changed(
        current, compact
    ) or _pr_handoff_head_changed(current, compact)
    record.state = {
        **record.state,
        PR_HANDOFF_STATE_KEY: _merge_pr_handoff_dicts(current, compact),
    }
    if reset_gates:
        record.state.pop(PR_GATES_STATE_KEY, None)


def _handoff_marker(value: Any) -> dict[str, Any]:
    handoff = _compact_pr_handoff(value)
    marker = {
        key: handoff[key]
        for key in ("url", "repository_full_name", "pr_number")
        if key in handoff
    }
    if "url" in marker or (
        "repository_full_name" in marker and "pr_number" in marker
    ):
        return marker
    return {}


def _gh_pr_view(
    target: Any,
    *,
    selector: str | None = None,
    source_tool: str,
    timeout_seconds: int = _GH_CLI_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    payload = _gh_pr_view_payload(
        target,
        selector=selector,
        fields=_GH_PR_VIEW_FIELDS,
        optional=selector is None,
        timeout_seconds=timeout_seconds,
    )
    if payload is None:
        return None
    return pr_watch.pr_handoff_from_gh_view(payload, source_tool=source_tool)


def _pr_handoff_is_merged(handoff: dict[str, Any]) -> bool:
    state = handoff.get("state")
    merged_at = handoff.get("merged_at")
    return (
        handoff.get("merged") is True
        or (isinstance(merged_at, str) and bool(merged_at.strip()))
        or (isinstance(state, str) and state.lower() == "merged")
    )


def _maybe_auto_pull_default_repo_after_pr_merge(
    registration: PrWatchRegistration,
) -> None:
    record = SessionPullRequest.objects.filter(pk=registration.record_id).first()
    if (
        record is None
        or not _registration_owns_record(record, registration)
        or not _pr_handoff_is_merged(pr_handoff_for_record(record))
    ):
        return
    metadata = (
        SessionMetadata.objects.select_related("project")
        .filter(thread_id=record.thread_id)
        .first()
    )
    project = metadata.project if metadata is not None else None
    if project is None or not project.auto_pull_enabled:
        return
    with transaction.atomic():
        locked = SessionPullRequest.objects.select_for_update().get(
            pk=registration.record_id
        )
        if (
            not _registration_owns_record(locked, registration)
            or not _pr_handoff_is_merged(pr_handoff_for_record(locked))
        ):
            return
        if locked.state.get(AUTO_PULL_RESULT_STATE_KEY):
            return
        skip_reason = _auto_pull_skip_reason(locked, project)
        if skip_reason:
            _record_auto_pull_result_locked(
                locked, {"status": "skipped", "reason": skip_reason}
            )
            return
        _record_auto_pull_result_locked(locked, {"status": "running"})
    try:
        result = pull_default_branch_from_origin(project.repo_path)
    except AutoPullError as exc:
        logger.warning(
            "auto-pull failed for project %s after PR watch: %s", project.pk, exc
        )
        _record_auto_pull_result(
            registration, {"status": "failed", "error": str(exc)}
        )
        return
    except Exception as exc:
        logger.exception(
            "unexpected auto-pull failure for project %s after PR watch", project.pk
        )
        _record_auto_pull_result(
            registration, {"status": "failed", "error": str(exc)}
        )
        return
    _record_auto_pull_result(registration, _auto_pull_result_dict(result))


def _auto_pull_skip_reason(record: SessionPullRequest, project: Project) -> str:
    cwd = record.cwd.strip()
    if not cwd:
        return "session checkout is unavailable"
    if _same_checkout(cwd, project.repo_path):
        return "default checkout is the active session checkout"
    if not same_repo_or_worktree(cwd, project.repo_path, project.git_common_dir):
        return "project repository does not match session checkout"
    return ""


def _same_checkout(cwd: str, repo_path: str) -> bool:
    cwd_root = repo_root(cwd)
    cwd_path = cwd_root if cwd_root is not None else Path(cwd).expanduser()
    return resolved_path(cwd_path) == resolved_path(Path(repo_path).expanduser())


def _auto_pull_result_dict(result: AutoPullResult) -> dict[str, object]:
    return {
        "status": "pulled" if result.changed else "up_to_date",
        "branch": result.branch,
        "before_sha": result.before_sha,
        "after_sha": result.after_sha,
        "changed": result.changed,
    }


def _record_auto_pull_result(
    registration: PrWatchRegistration, result: dict[str, object]
) -> None:
    try:
        with transaction.atomic():
            record = SessionPullRequest.objects.select_for_update().get(
                pk=registration.record_id
            )
            if not _registration_owns_record(record, registration):
                return
            _record_auto_pull_result_locked(record, result)
    except Exception:
        logger.exception(
            "failed to record auto-pull result for PR state %s",
            registration.record_id,
        )


def _record_auto_pull_result_locked(
    record: SessionPullRequest, result: dict[str, object]
) -> None:
    record.state = {**record.state, AUTO_PULL_RESULT_STATE_KEY: result}
    record.save(update_fields=["state", "updated_at"])
