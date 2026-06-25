"""Autonomous-goal pages and proposal-outcome endpoints."""
from datetime import datetime
from typing import Any

from django.db import transaction
from django.db.models import Q
from django.http import (
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from hitch.main.goals.autonomous_goal_form import (
    _attach_autonomous_goal_display_state,
    _validated_autonomous_goal_values,
)
from hitch.main.goals.autonomous_goal_proposal_stack import (
    _autonomous_goal_accepted_session_blocks_start,
    _proposal_outcome_metadata,
)
from hitch.main.goals.autonomous_goal_run_display import (
    _attach_autonomous_goal_run_state,
    _autonomous_goal_workflow_for_log,
)
from hitch.main.models import (
    AutonomousGoal,
    Project,
    ProposedSession,
)
from hitch.main.runtime import reconciliation
from hitch.main.sessions.project_visibility import (
    _filter_proposed_sessions_by_project_visibility,
)
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_settings import (
    _active_project_from_request,
    _cached_models_and_settings,
    _selected_project_for_settings,
    _session_project_visibility_for_settings,
    _stored_settings,
)
from hitch.main.sessions.settings_cookies import (
    _WEB_SEARCH_MODE_OPTIONS,
    _apply_cookie_updates,
)
from hitch.main.views import common
from hitch.main.workflows import autonomous_goals as goal_workflows
from hitch.main.workflows import system_agents
from hitch.main.worktrees import (
    WorktreeCleanupError,
)

_AUTONOMOUS_GOAL_TITLE_MAX_LEN = 200

@require_http_methods(["GET"])
def autonomous_goals(request: HttpRequest) -> HttpResponse:
    reconciliation.reconcile_dead_if_due()
    models_data, resolved_settings = _cached_models_and_settings(request)
    current_settings = resolved_settings.values
    cookie_updates = resolved_settings.cookie_updates
    projects = list(Project.objects.all())
    current_project = _selected_project_for_settings(current_settings, projects)
    goals = (
        list(
            AutonomousGoal.objects.filter(
                project=current_project,
                deleted_at__isnull=True,
            ).select_related("project")
        )
        if current_project is not None
        else []
    )
    local_branch_choices = (
        common.local_branch_names(current_project.repo_path)
        if current_project is not None
        else []
    )
    _attach_autonomous_goal_run_state(goals)
    _attach_autonomous_goal_display_state(goals)
    settings_context = common._settings_context(current_settings, models_data)
    response = render(
        request,
        "autonomous_goals.html",
        {
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "current_project": current_project,
            "autonomous_goals": goals,
            "autonomous_goal_create_url": reverse("create_autonomous_goal"),
            "autonomous_goal_run_all_url": reverse("run_autonomous_goals"),
            "autonomous_goal_run_busy": request.GET.get("ag_run_busy") == "1",
            "ambition_choices": AutonomousGoal.AMBITION_CHOICES,
            "default_ambition": AutonomousGoal.AMBITION_INCREMENTAL,
            "autonomy_choices": AutonomousGoal.AUTONOMY_CHOICES,
            "default_autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            "default_auto_qa": False,
            "auto_qa_supported_autonomies": tuple(AutonomousGoal.AUTO_QA_AUTONOMIES),
            "auto_qa_required_autonomies": tuple(
                AutonomousGoal.AUTO_QA_REQUIRED_AUTONOMIES
            ),
            "default_auto_proposal": False,
            "stacked_diff_supported_autonomies": tuple(
                AutonomousGoal.STACKED_DIFF_AUTONOMIES
            ),
            "default_stacked_diff_depth": AutonomousGoal.STACKED_DIFF_DEPTH_MIN,
            "stacked_diff_depth_min": AutonomousGoal.STACKED_DIFF_DEPTH_MIN,
            "stacked_diff_depth_max": AutonomousGoal.STACKED_DIFF_DEPTH_MAX,
            "default_proposal_budget": "",
            "confidence_choices": AutonomousGoal.CONFIDENCE_CHOICES,
            "default_confidence": AutonomousGoal.CONFIDENCE_HIGH,
            "web_search_mode_choices": _WEB_SEARCH_MODE_OPTIONS,
            "default_web_search_mode": AutonomousGoal.WEB_SEARCH_DEFAULT,
            "local_branch_choices": local_branch_choices,
            "title_max_len": _AUTONOMOUS_GOAL_TITLE_MAX_LEN,
            **settings_context,
        },
    )
    _apply_cookie_updates(response, cookie_updates)
    return response

@require_http_methods(["POST"])
def create_autonomous_goal(request: HttpRequest) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    values, error = _validated_autonomous_goal_values(
        request,
        local_branches=common.local_branch_names(project.repo_path),
    )
    if error is not None:
        return HttpResponseBadRequest(error)
    assert values is not None
    AutonomousGoal.objects.create(
        project=project,
        title=values.title,
        goal=values.goal,
        ambition=values.ambition,
        autonomy=values.autonomy,
        auto_qa_enabled=values.auto_qa_enabled,
        auto_proposal_enabled=values.auto_proposal_enabled,
        stacked_diff_depth=values.stacked_diff_depth,
        proposal_budget=values.proposal_budget,
        confidence_threshold=values.confidence_threshold,
        web_search_mode=values.web_search_mode,
        auto_merge_to_local_branch=values.auto_merge_to_local_branch,
        auto_merge_branch=values.auto_merge_branch,
    )
    return redirect("autonomous_goals")

@require_http_methods(["POST"])
def edit_autonomous_goal(request: HttpRequest, autonomous_goal_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    autonomous_goal = AutonomousGoal.objects.filter(
        pk=autonomous_goal_id,
        project=project,
        deleted_at__isnull=True,
    ).first()
    if autonomous_goal is None:
        raise Http404("autonomous goal not found")
    values, error = _validated_autonomous_goal_values(
        request,
        autonomy_default=autonomous_goal.autonomy,
        auto_qa_default=autonomous_goal.auto_qa_enabled,
        web_search_default=autonomous_goal.web_search_mode,
        auto_proposal_default=autonomous_goal.auto_proposal_enabled,
        stacked_diff_depth_default=autonomous_goal.stacked_diff_depth,
        proposal_budget_default=autonomous_goal.proposal_budget,
        local_branches=common.local_branch_names(project.repo_path),
    )
    if error is not None:
        return HttpResponseBadRequest(error)
    assert values is not None

    updates: list[str] = []
    for field in (
        "title",
        "goal",
        "ambition",
        "autonomy",
        "auto_qa_enabled",
        "auto_proposal_enabled",
        "stacked_diff_depth",
        "proposal_budget",
        "confidence_threshold",
        "web_search_mode",
        "auto_merge_to_local_branch",
        "auto_merge_branch",
    ):
        value = getattr(values, field)
        if getattr(autonomous_goal, field) != value:
            setattr(autonomous_goal, field, value)
            updates.append(field)
    if updates:
        if autonomous_goal.auto_proposal_last_no_proposal_sha:
            autonomous_goal.auto_proposal_last_no_proposal_sha = ""
            updates.append("auto_proposal_last_no_proposal_sha")
        autonomous_goal.save(update_fields=[*updates, "updated_at"])
    return redirect("autonomous_goals")

@require_http_methods(["POST"])
def delete_autonomous_goal(
    request: HttpRequest, autonomous_goal_id: int
) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    stop_error = system_agents.AUTONOMOUS_GOAL_DELETED_ERROR
    with transaction.atomic():
        autonomous_goal = (
            AutonomousGoal.objects.select_for_update()
            .filter(
                pk=autonomous_goal_id,
                project=project,
                deleted_at__isnull=True,
            )
            .first()
        )
        if autonomous_goal is None:
            raise Http404("autonomous goal not found")
        if not goal_workflows.stop_running_autonomous_goal_workflow(
            autonomous_goal.pk, stop_error
        ):
            return HttpResponseBadRequest("autonomous goal run could not be stopped")
        deleted_at = timezone.now()
        cleanup_proposals = _dismiss_unresolved_autonomous_goal_proposals(
            autonomous_goal,
            reason=stop_error,
            now=deleted_at,
        )
        autonomous_goal.deleted_at = deleted_at
        autonomous_goal.auto_proposal_enabled = False
        autonomous_goal.save(
            update_fields=["deleted_at", "auto_proposal_enabled", "updated_at"]
        )
    for proposal in cleanup_proposals:
        _cleanup_proposed_session_candidate_worktree(proposal)
    return redirect("autonomous_goals")

def _dismiss_unresolved_autonomous_goal_proposals(
    autonomous_goal: AutonomousGoal, *, reason: str, now: datetime
) -> list[ProposedSession]:
    proposals = list(
        ProposedSession.objects.select_for_update()
        .select_related("candidate_session")
        .filter(
            autonomous_goal=autonomous_goal,
        )
        .filter(_autonomous_goal_cleanup_proposal_filter())
    )
    if proposals:
        for proposal in proposals:
            metadata = (
                dict(proposal.outcome_metadata)
                if isinstance(proposal.outcome_metadata, dict)
                else {}
            )
            metadata["stacked_diff_hidden_until_complete"] = False
            proposal.outcome_status = ProposedSession.OUTCOME_DISMISSED
            proposal.outcome_notes = reason
            proposal.outcome_metadata = metadata
            proposal.updated_at = now
        ProposedSession.objects.bulk_update(
            proposals,
            ["outcome_status", "outcome_notes", "outcome_metadata", "updated_at"],
        )
    return proposals

def _autonomous_goal_cleanup_proposal_filter() -> Q:
    return Q(outcome_status=ProposedSession.OUTCOME_UNSET) | Q(
        outcome_status=ProposedSession.OUTCOME_DISMISSED,
        outcome_metadata__stacked_diff_hidden_until_complete=True,
    )

@require_http_methods(["POST"])
def run_autonomous_goal(request: HttpRequest, autonomous_goal_id: int) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    autonomous_goal = AutonomousGoal.objects.filter(
        pk=autonomous_goal_id,
        project=project,
        deleted_at__isnull=True,
    ).first()
    if autonomous_goal is None:
        raise Http404("autonomous goal not found")
    if _autonomous_goal_accepted_session_blocks_start(autonomous_goal):
        return redirect("autonomous_goals")
    workflow = goal_workflows.start_autonomous_goal_workflow_if_queue_idle(
        autonomous_goal=autonomous_goal,
        use_worktrees=True,
    )
    if workflow is None:
        return _redirect_autonomous_goals_run_busy()
    return redirect("autonomous_goals")

@require_http_methods(["POST"])
def run_autonomous_goals(request: HttpRequest) -> HttpResponse:
    project = _active_project_from_request(request)
    if project is None:
        return HttpResponseBadRequest("active project is required")
    runnable_goals = []
    for autonomous_goal in AutonomousGoal.objects.filter(
        project=project,
        deleted_at__isnull=True,
    ):
        if _autonomous_goal_accepted_session_blocks_start(autonomous_goal):
            continue
        runnable_goals.append(autonomous_goal)
    result = goal_workflows.start_autonomous_goal_workflows_or_queue(
        autonomous_goals=runnable_goals,
        use_worktrees=True,
    )
    if (
        result.started_workflow is None
        and result.queued_count == 0
        and goal_workflows.autonomous_goal_queue_busy()
    ):
        return _redirect_autonomous_goals_run_busy()
    return redirect("autonomous_goals")


def _redirect_autonomous_goals_run_busy() -> HttpResponse:
    return redirect(f"{reverse('autonomous_goals')}?ag_run_busy=1")


@require_http_methods(["GET"])
def autonomous_goal_run_log(request: HttpRequest, workflow_id: int) -> HttpResponse:
    workflow = _autonomous_goal_workflow_for_log(request, workflow_id)
    run = workflow.agent_runs.exclude(thread_id="").order_by("-created_at").first()
    if run is None:
        raise Http404("autonomous goal run log not found")
    return common._render_session_detail(
        request,
        run.thread_id,
        read_only=True,
        display_title="Autonomous goal run log",
    )

@require_http_methods(["POST"])
def update_proposed_session_outcome(
    request: HttpRequest, proposed_session_id: int
) -> HttpResponse:
    if proposed_session_id < 1 or proposed_session_id > common._MAX_BIGAUTOFIELD:
        return HttpResponseBadRequest("proposed session is required")
    current_settings = _stored_settings(request)
    project_visibility = _session_project_visibility_for_settings(
        current_settings, list(Project.objects.all())
    )
    proposed_session_query = _filter_proposed_sessions_by_project_visibility(
        ProposedSession.objects.select_related(
            "project",
            "autonomous_goal__project",
            "candidate_session",
        ).filter(pk=proposed_session_id),
        project_visibility,
    )
    proposed_session = proposed_session_query.first()
    if proposed_session is None:
        return HttpResponseBadRequest("proposed session is required")
    outcome_status = request.POST.get("outcome_status", "")
    # OUTCOME_UNSET is the inbox's pending state, not a decision the endpoint can
    # apply; accepting it as a target would let a request re-open a resolved item.
    valid_statuses = {
        choice[0] for choice in ProposedSession.OUTCOME_CHOICES
    } - {ProposedSession.OUTCOME_UNSET}
    if outcome_status not in valid_statuses:
        return HttpResponseBadRequest("outcome status is invalid")
    outcome_notes = request.POST.get(
        "reason", request.POST.get("outcome_notes", "")
    ).strip()
    if (
        proposed_session.inbox_kind == ProposedSession.INBOX_KIND_PROPOSAL
        and outcome_status == ProposedSession.OUTCOME_REJECTED
        and not outcome_notes
    ):
        return HttpResponseBadRequest("reason is required")
    if (
        proposed_session.inbox_kind == ProposedSession.INBOX_KIND_NOTICE
        and outcome_status != ProposedSession.OUTCOME_DISMISSED
    ):
        return HttpResponseBadRequest("outcome status is invalid")
    update_values: dict[str, Any] = {
        "outcome_status": outcome_status,
        "outcome_notes": outcome_notes,
        # update() bypasses save(), so the auto_now updated_at must be set here.
        "updated_at": timezone.now(),
    }
    outcome_metadata = _proposal_outcome_metadata(
        proposed_session,
        {"resolved_by": "user"},
    )
    if outcome_status == ProposedSession.OUTCOME_ACCEPTED:
        update_values["accepted_session"] = proposed_session.candidate_session
        outcome_metadata = _proposal_outcome_metadata(
            proposed_session,
            {
                **outcome_metadata,
                "accepted_by": "user",
                "accepted_session_id": (
                    proposed_session.candidate_session_id
                    if proposed_session.candidate_session_id is not None
                    else None
                ),
                "accepted_thread_id": (
                    proposed_session.candidate_session.thread_id
                    if proposed_session.candidate_session is not None
                    else ""
                ),
            },
        )
    update_values["outcome_metadata"] = outcome_metadata
    # Inbox decisions are one-way: only an undecided item may be resolved
    # (UNSET -> accepted/rejected/dismissed), matching the OUTCOME_UNSET filter
    # the new-session entry points already apply. Enforce it with a single
    # conditional UPDATE gated on the row still being OUTCOME_UNSET rather than a
    # read-then-write: two near-simultaneous requests (e.g. a stale-tab reject
    # racing an accept) could both read OUTCOME_UNSET before either commits, and
    # the loser would clobber the accepted outcome and re-hide the live session.
    # The atomic WHERE clause serializes the decision -- exactly one request
    # matches a row; the loser updates nothing and bails before any side effects.
    # (A row lock would also work, but only on backends that honor
    # select_for_update; a conditional UPDATE is correct on every backend.)
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).update(**update_values)
    if not applied:
        return HttpResponseBadRequest("proposed session has already been resolved")
    # Mirror the committed values onto the instance for the cleanup side effect.
    for field, value in update_values.items():
        setattr(proposed_session, field, value)
    stack_continuation_stopped = common._stop_autonomous_goal_stack_after_proposal_resolution(
        proposed_session
    )
    if (
        outcome_status == ProposedSession.OUTCOME_ACCEPTED
        and proposed_session.candidate_session is not None
    ):
        common._rename_codex_thread_from_proposal(
            proposed_session=proposed_session,
            session_metadata=proposed_session.candidate_session,
            settings=_stored_settings(request),
        )
    if outcome_status in {
        ProposedSession.OUTCOME_DISMISSED,
        ProposedSession.OUTCOME_REJECTED,
    } and stack_continuation_stopped:
        _cleanup_proposed_session_candidate_worktree(proposed_session)
    return redirect("inbox")

def _cleanup_proposed_session_candidate_worktree(
    proposed_session: ProposedSession,
) -> None:
    if proposed_session.accepted_session_id is not None:
        return
    candidate = proposed_session.candidate_session
    if candidate is None or not candidate.cwd:
        return
    try:
        common.cleanup_managed_worktree_path(candidate.cwd)
    except WorktreeCleanupError:
        common.logger.exception(
            "failed to clean up candidate worktree for proposed session %s",
            proposed_session.pk,
        )
