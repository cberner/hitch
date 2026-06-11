"""Autonomous-goal proposal-stack and continuation validation helpers.

Pure query/metadata helpers that decide whether an autonomous goal has a
pending proposal blocking its start, whether a single pending proposal is a
valid stacked-diff continuation, and whether in-flight automation exists.
Leaf module: imports nothing from ``system_agents`` to avoid an import cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from django.db import models
from django.utils import timezone

from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    ProposedSession,
    SystemWorkflow,
)

AUTONOMOUS_GOAL_AUTONOMY_ACCEPTED_BY = "autonomous_goal_autonomy"
LEGACY_AUTONOMOUS_GOAL_AUTONOMY_ACCEPTED_BY = "standing_order_autonomy"
_AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY = (
    "stacked_diff_continuation_stopped_reason"
)
_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKENS_USED_METADATA_KEY = (
    "proposal_budget_tokens_used"
)


@dataclass(frozen=True)
class _AutonomousGoalProposalStackMetadata:
    depth: int
    iteration: int


@dataclass(frozen=True)
class _AutonomousGoalPendingProposalState:
    blocking_goal_ids: set[int]
    continuable_stack_goal_ids: set[int]


def _proposal_outcome_metadata(
    proposal: ProposedSession, updates: dict[str, object]
) -> dict[str, object]:
    metadata = (
        dict(proposal.outcome_metadata)
        if isinstance(proposal.outcome_metadata, dict)
        else {}
    )
    for key, value in updates.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


def _autonomous_goal_pending_proposal_blocks_start(
    autonomous_goal: AutonomousGoal,
) -> bool:
    return bool(_autonomous_goal_pending_proposal_blocking_ids([autonomous_goal]))


def _autonomous_goal_pending_proposal_blocking_ids(
    autonomous_goals: Iterable[AutonomousGoal],
) -> set[int]:
    return _autonomous_goal_pending_proposal_state(
        autonomous_goals
    ).blocking_goal_ids


def _autonomous_goal_pending_proposal_state(
    autonomous_goals: Iterable[AutonomousGoal],
) -> _AutonomousGoalPendingProposalState:
    goals_by_id = {goal.pk: goal for goal in autonomous_goals}
    if not goals_by_id:
        return _AutonomousGoalPendingProposalState(
            blocking_goal_ids=set(),
            continuable_stack_goal_ids=set(),
        )
    pending_by_goal_id: dict[int, list[ProposedSession]] = {
        goal_id: [] for goal_id in goals_by_id
    }
    pending_proposal_rows = (
        ProposedSession.objects.select_related("candidate_session", "source_workflow")
        .filter(
            autonomous_goal_id__in=list(goals_by_id),
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        .order_by("autonomous_goal_id", "-created_at", "-id")
    )
    for proposal in pending_proposal_rows:
        if proposal.autonomous_goal_id is not None:
            pending_by_goal_id[proposal.autonomous_goal_id].append(proposal)
    blocking_goal_ids: set[int] = set()
    continuable_stack_goal_ids: set[int] = set()
    for goal_id, pending_proposals in pending_by_goal_id.items():
        if not pending_proposals:
            continue
        if (
            _autonomous_goal_stack_continuation_proposal_from_pending(
                pending_proposals, goals_by_id[goal_id]
            )
            is None
        ):
            blocking_goal_ids.add(goal_id)
        else:
            continuable_stack_goal_ids.add(goal_id)
    return _AutonomousGoalPendingProposalState(
        blocking_goal_ids=blocking_goal_ids,
        continuable_stack_goal_ids=continuable_stack_goal_ids,
    )


def _autonomous_goal_stack_continuation_proposal(
    autonomous_goal: AutonomousGoal,
) -> ProposedSession | None:
    return _autonomous_goal_stack_continuation_proposal_from_pending(
        _autonomous_goal_pending_proposals(autonomous_goal), autonomous_goal
    )


def _autonomous_goal_pending_proposals(
    autonomous_goal: AutonomousGoal,
) -> list[ProposedSession]:
    return list(
        autonomous_goal.proposed_sessions.select_related(
            "candidate_session", "source_workflow"
        )
        .filter(
            inbox_kind=ProposedSession.INBOX_KIND_PROPOSAL,
            outcome_status=ProposedSession.OUTCOME_UNSET,
        )
        .order_by("-created_at", "-id")
    )


def _autonomous_goal_stack_continuation_proposal_from_pending(
    pending_proposals: list[ProposedSession], autonomous_goal: AutonomousGoal
) -> ProposedSession | None:
    if len(pending_proposals) != 1:
        return None
    proposal = pending_proposals[0]
    return (
        proposal
        if _autonomous_goal_proposal_allows_stack_continuation(
            proposal, autonomous_goal
        )
        else None
    )


def _claim_autonomous_goal_stack_continuation_proposal(
    proposal: ProposedSession,
) -> ProposedSession | None:
    outcome_metadata = {
        **_proposal_outcome_metadata(proposal, {}),
        "stacked_diff_hidden_until_complete": False,
    }
    applied = ProposedSession.objects.filter(
        pk=proposal.pk,
        outcome_status=ProposedSession.OUTCOME_UNSET,
        accepted_session__isnull=True,
    ).update(
        outcome_metadata=outcome_metadata,
        updated_at=timezone.now(),
    )
    if not applied:
        return None
    proposal.outcome_metadata = outcome_metadata
    return proposal


def _autonomous_goal_proposal_allows_stack_continuation(
    proposal: ProposedSession, autonomous_goal: AutonomousGoal
) -> bool:
    if not autonomous_goal.auto_proposal_enabled:
        return False
    if proposal.outcome_status != ProposedSession.OUTCOME_UNSET:
        return False
    if proposal.inbox_kind != ProposedSession.INBOX_KIND_PROPOSAL:
        return False
    if proposal.candidate_session is None:
        return False
    candidate_cwd = proposal.candidate_session.cwd.strip()
    if not candidate_cwd or candidate_cwd == autonomous_goal.project.repo_path:
        return False
    metadata = _proposal_outcome_metadata(proposal, {})
    if metadata.get(_AUTONOMOUS_GOAL_STACKED_CONTINUATION_STOP_REASON_METADATA_KEY):
        return False
    if not _autonomous_goal_proposal_budget_allows_stack_continuation(
        proposal, autonomous_goal, metadata=metadata
    ):
        return False
    return (
        _autonomous_goal_proposal_stack_continuation_metadata(
            proposal, autonomous_goal
        )
        is not None
    )


def _autonomous_goal_proposal_budget_allows_stack_continuation(
    proposal: ProposedSession,
    autonomous_goal: AutonomousGoal,
    *,
    metadata: Mapping[str, object] | None = None,
) -> bool:
    budget = autonomous_goal.proposal_budget or 0
    if budget <= 0:
        return True
    if metadata is None:
        metadata = _proposal_outcome_metadata(proposal, {})
    tokens_used = _proposal_metadata_non_negative_int(
        metadata, _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKENS_USED_METADATA_KEY
    )
    return tokens_used is None or tokens_used < budget


def _proposal_metadata_non_negative_int(
    metadata: Mapping[str, object], key: str
) -> int | None:
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _autonomous_goal_proposal_stack_continuation_metadata(
    proposal: ProposedSession, autonomous_goal: AutonomousGoal
) -> _AutonomousGoalProposalStackMetadata | None:
    metadata = _proposal_outcome_metadata(proposal, {})
    depth_value = metadata.get("stacked_diff_depth")
    iteration_value = metadata.get("stacked_diff_iteration")
    if not _valid_autonomous_goal_stack_metadata_int(
        depth_value
    ) or not _valid_autonomous_goal_stack_metadata_int(iteration_value):
        return None
    depth = min(
        cast(int, depth_value),
        autonomous_goal.effective_stacked_diff_depth,
        AutonomousGoal.STACKED_DIFF_DEPTH_MAX,
    )
    if depth <= AutonomousGoal.STACKED_DIFF_DEPTH_MIN:
        return None
    iteration = cast(int, iteration_value)
    if iteration < 1 or iteration >= depth:
        return None
    return _AutonomousGoalProposalStackMetadata(depth=depth, iteration=iteration)


def _valid_autonomous_goal_stack_metadata_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _autonomous_goal_proposal_stack_iteration(proposal: ProposedSession) -> int:
    metadata = (
        proposal.outcome_metadata if isinstance(proposal.outcome_metadata, dict) else {}
    )
    value = metadata.get("stacked_diff_iteration")
    return max(value, 1) if isinstance(value, int) else 1


def _autonomous_goal_unresolved_failure_notice_exists(
    autonomous_goal: AutonomousGoal,
) -> bool:
    return autonomous_goal.proposed_sessions.filter(
        inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        outcome_status=ProposedSession.OUTCOME_UNSET,
        outcome_metadata__automation_status="failed",
    ).exists()


def _autonomous_goal_start_claim_exists(autonomous_goal: AutonomousGoal) -> bool:
    claim_key = ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY
    claim_lookup = f"outcome_metadata__{claim_key}__isnull"
    claimed_metadatas = (
        ProposedSession.objects.filter(
            project=autonomous_goal.project,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=True,
            **{claim_lookup: False},
        )
        .filter(_autonomous_goal_in_flight_proposal_criteria())
        .values_list("outcome_metadata", flat=True)
    )
    now = timezone.now()
    return any(
        ProposedSession.accepted_session_start_claim_is_active(metadata, now=now)
        for metadata in claimed_metadatas
    )


def _autonomous_goal_in_flight_proposal_criteria() -> models.Q:
    return (
        models.Q(outcome_metadata__accepted_by=AUTONOMOUS_GOAL_AUTONOMY_ACCEPTED_BY)
        | models.Q(
            outcome_metadata__accepted_by=LEGACY_AUTONOMOUS_GOAL_AUTONOMY_ACCEPTED_BY
        )
        | models.Q(
            autonomous_goal__isnull=False,
            outcome_metadata__accepted_by="user",
        )
        | (
            models.Q(autonomous_goal__isnull=False)
            & (
                models.Q(outcome_metadata__auto_pr_enabled=True)
                | models.Q(outcome_metadata__auto_qa_enabled=True)
            )
        )
    )


def _autonomous_goal_in_flight_automation_exists(autonomous_goal: AutonomousGoal) -> bool:
    if _autonomous_goal_start_claim_exists(autonomous_goal):
        return True
    accepted_thread_ids = (
        ProposedSession.objects.filter(
            project=autonomous_goal.project,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=False,
        )
        .filter(_autonomous_goal_in_flight_proposal_criteria())
        .exclude(accepted_session__thread_id="")
        .values_list("accepted_session__thread_id", flat=True)
    )
    if CodexInstance.objects.filter(
        thread_id__in=accepted_thread_ids,
        status__in=CodexInstance.ACTIVE_STATUSES,
    ).exists():
        return True
    return SystemWorkflow.objects.filter(
        kind=SystemWorkflow.KIND_PR_QA,
        main_thread_id__in=accepted_thread_ids,
        status=SystemWorkflow.STATUS_RUNNING,
    ).exists()
