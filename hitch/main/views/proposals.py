"""Resolve proposed-session Inbox items."""

from typing import Any

from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from hitch.main.models import Project, ProposedSession
from hitch.main.proposals.proposed_sessions import _proposal_outcome_metadata
from hitch.main.sessions.project_visibility import _filter_proposed_sessions_by_project_visibility
from hitch.main.sessions.session_settings import _session_project_visibility_for_settings, _stored_settings
from hitch.main.views import common


@require_http_methods(["POST"])
def update_proposed_session_outcome(request: HttpRequest, proposed_session_id: int) -> HttpResponse:
    if proposed_session_id < 1 or proposed_session_id > common._MAX_BIGAUTOFIELD:
        return HttpResponseBadRequest("proposed session is required")
    current_settings = _stored_settings(request)
    project_visibility = _session_project_visibility_for_settings(current_settings, list(Project.objects.all()))
    proposed_session_query = _filter_proposed_sessions_by_project_visibility(
        ProposedSession.objects.select_related(
            "project",
        ).filter(pk=proposed_session_id),
        project_visibility,
    )
    proposed_session = proposed_session_query.first()
    if proposed_session is None:
        return HttpResponseBadRequest("proposed session is required")
    outcome_status = request.POST.get("outcome_status", "")
    # OUTCOME_UNSET is the inbox's pending state, not a decision the endpoint can
    # apply; accepting it as a target would let a request re-open a resolved item.
    valid_statuses = {choice[0] for choice in ProposedSession.OUTCOME_CHOICES} - {ProposedSession.OUTCOME_UNSET}
    if outcome_status not in valid_statuses:
        return HttpResponseBadRequest("outcome status is invalid")
    outcome_notes = request.POST.get("reason", request.POST.get("outcome_notes", "")).strip()
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
    if outcome_status == ProposedSession.OUTCOME_ACCEPTED:
        return HttpResponseBadRequest("proposal must be started before acceptance")
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
    update_values["outcome_metadata"] = outcome_metadata
    # A conditional update makes concurrent Inbox decisions one-way on every
    # database backend; a stale tab cannot overwrite an accepted proposal.
    applied = ProposedSession.objects.filter(
        pk=proposed_session.pk,
        outcome_status=ProposedSession.OUTCOME_UNSET,
    ).update(**update_values)
    if not applied:
        return HttpResponseBadRequest("proposed session has already been resolved")
    return redirect("inbox")
