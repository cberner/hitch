"""Durable UI state for agent-invoked pull-request watches."""

from __future__ import annotations

import logging
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
from hitch.main.sessions.agent_tasks import (
    PR_AGENT_KINDS,
    PR_PUBLISH_AGENT_KIND,
    PR_WATCH_AGENT_KIND,
)
from hitch.main.workflows import pr_watch
from hitch.main.workflows.pr_handoff import (
    _compact_pr_handoff,
    _merge_pr_handoff_dicts,
)

logger = logging.getLogger(__name__)

PR_HANDOFF_STATE_KEY = "pr_handoff"
PR_GATES_STATE_KEY = "pr_gates"
AUTO_PULL_RESULT_STATE_KEY = "auto_pull_result"
_WATCH_OWNER_INSTANCE_STATE_KEY = SessionPullRequest.WATCH_OWNER_INSTANCE_STATE_KEY
_WATCH_OWNER_MESSAGE_STATE_KEY = "watch_owner_message_index"
_SUPERSEDED_BY_INSTANCE_STATE_KEY = (
    SessionPullRequest.SUPERSEDED_BY_INSTANCE_STATE_KEY
)
_SUPERSEDED_AT_STATE_KEY = "superseded_at"


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
