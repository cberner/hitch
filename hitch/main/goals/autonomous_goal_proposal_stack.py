"""Proposal and accepted-session guards shared by autonomous goals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

from django.utils import timezone

from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    ProposedSession,
    SessionPullRequest,
    SystemWorkflow,
)
from hitch.main.runtime.sdk_values import is_nonbool_int
from hitch.main.sequences import unique_nonempty

AUTONOMOUS_GOAL_TOOL_PROTOCOL_METADATA_KEY = "autonomous_goal_tool_protocol"
AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_METADATA_KEY = "approved_snapshot_sha"
AUTONOMOUS_GOAL_APPROVED_SNAPSHOT_REF_METADATA_KEY = "approved_snapshot_ref"
AUTONOMOUS_GOAL_ACCEPTED_SNAPSHOT_METADATA_KEY = "accepted_snapshot_sha"
_AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY = "stacked_diff_continuation_stopped_reason"
_PR_HANDOFF_STATE_KEY = "pr_handoff"


def _proposal_outcome_metadata(
    proposal: ProposedSession, updates: dict[str, object] | None = None
) -> dict[str, object]:
    metadata = dict(proposal.outcome_metadata) if isinstance(proposal.outcome_metadata, dict) else {}
    for key, value in (updates or {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


def _autonomous_goal_pending_proposal_blocks_start(
    autonomous_goal: AutonomousGoal,
) -> bool:
    return autonomous_goal.proposed_sessions.filter(
        inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).exists()


def _autonomous_goal_pending_proposal_blocking_ids(
    autonomous_goals: Iterable[AutonomousGoal],
) -> set[int]:
    goal_ids = [goal.pk for goal in autonomous_goals if goal.pk is not None]
    return {
        goal_id
        for goal_id in ProposedSession.objects.filter(
            autonomous_goal_id__in=goal_ids,
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        ).values_list("autonomous_goal_id", flat=True)
        if isinstance(goal_id, int)
    }


def _proposal_metadata_non_negative_int(metadata: Mapping[str, object], key: str) -> int | None:
    value = metadata.get(key)
    return value if is_nonbool_int(value) and value >= 0 else None


def _autonomous_goal_unresolved_failure_notice_exists(
    autonomous_goal: AutonomousGoal,
) -> bool:
    return autonomous_goal.proposed_sessions.filter(
        inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        outcome_status=ProposedSession.OUTCOME_UNSET,
        outcome_metadata__automation_status="failed",
    ).exists()


def _autonomous_goal_accepted_session_blocks_start(
    autonomous_goal: AutonomousGoal,
) -> bool:
    return bool(_autonomous_goal_accepted_session_blocking_ids([autonomous_goal]))


def _autonomous_goal_accepted_session_blocking_ids(
    autonomous_goals: Iterable[AutonomousGoal],
) -> set[int]:
    goal_ids = [goal.pk for goal in autonomous_goals if goal.pk is not None]
    if not goal_ids:
        return set()
    blocking_ids: set[int] = set()
    claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
    claim_lookup = f"outcome_metadata__{claim_key}__isnull"
    now = timezone.now()
    claimed_proposals = ProposedSession.objects.filter(
        autonomous_goal_id__in=goal_ids,
        outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        accepted_session__isnull=True,
        **{claim_lookup: False},
    ).values_list("autonomous_goal_id", "outcome_metadata")
    for goal_id, metadata in claimed_proposals:
        if isinstance(goal_id, int) and ProposedSession.accepted_session_start_claim_is_active(metadata, now=now):
            blocking_ids.add(goal_id)

    accepted_rows = tuple(
        ProposedSession.objects.filter(
            autonomous_goal_id__in=goal_ids,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=False,
            accepted_session__codex_archived=False,
        ).values_list(
            "autonomous_goal_id",
            "accepted_session__thread_id",
            "accepted_session__codex_updated_at",
        )
    )
    thread_ids = [
        thread_id for _goal_id, thread_id, _updated_at in accepted_rows if isinstance(thread_id, str) and thread_id
    ]
    live_thread_ids = _accepted_session_live_thread_ids(thread_ids)
    done_prs = _accepted_session_done_prs_by_thread_id(thread_ids)
    for goal_id, thread_id, codex_updated_at in accepted_rows:
        if not isinstance(goal_id, int):
            continue
        if isinstance(thread_id, str) and thread_id in live_thread_ids:
            blocking_ids.add(goal_id)
            continue
        registered_pr = done_prs.get(thread_id) if isinstance(thread_id, str) else None
        if registered_pr is not None and _accepted_session_pr_is_current(registered_pr, codex_updated_at):
            continue
        blocking_ids.add(goal_id)
    return blocking_ids


def _accepted_session_live_thread_ids(thread_ids: Iterable[str]) -> set[str]:
    ids = unique_nonempty(thread_ids)
    if not ids:
        return set()
    live_ids = set(
        CodexInstance.objects.filter(
            thread_id__in=ids,
            status__in=CodexInstance.ACTIVE_STATUSES,
        ).values_list("thread_id", flat=True)
    )
    # Compatibility for sessions promoted before accepted proposals began
    # starting fresh visible threads.
    live_ids.update(
        SystemWorkflow.objects.filter(
            main_thread_id__in=ids,
            status=SystemWorkflow.STATUS_RUNNING,
        ).values_list("main_thread_id", flat=True)
    )
    return {thread_id for thread_id in live_ids if isinstance(thread_id, str)}


def _accepted_session_done_prs_by_thread_id(
    thread_ids: Iterable[str],
) -> dict[str, SessionPullRequest]:
    ids = unique_nonempty(thread_ids)
    if not ids:
        return {}
    done_prs: dict[str, SessionPullRequest] = {}
    for registered_pr in SessionPullRequest.objects.filter(thread_id__in=ids):
        if not registered_pr.is_current:
            continue
        state = registered_pr.state if isinstance(registered_pr.state, Mapping) else {}
        if _pr_snapshot_done_stage_key(state.get(_PR_HANDOFF_STATE_KEY)) is not None:
            done_prs[registered_pr.thread_id] = registered_pr
    return done_prs


def _accepted_session_pr_is_current(registered_pr: SessionPullRequest, codex_updated_at: object) -> bool:
    state = registered_pr.state if isinstance(registered_pr.state, Mapping) else {}
    watch_owner_id = state.get(SessionPullRequest.WATCH_OWNER_INSTANCE_STATE_KEY)
    if is_nonbool_int(watch_owner_id) and watch_owner_id > 0:
        return registered_pr.is_current
    if registered_pr.updated_at is None or not isinstance(codex_updated_at, datetime):
        return True
    return registered_pr.updated_at >= codex_updated_at


def _pr_snapshot_done_stage_key(snapshot: object) -> str | None:
    if not isinstance(snapshot, Mapping):
        return None
    state = _string(snapshot.get("state")).lower()
    if snapshot.get("merged") is True or _string(snapshot.get("merged_at")) or state == "merged":
        return "done_merged"
    if state == "closed":
        return "done_closed"
    return None


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""
