"""HTTP endpoints and helpers for resolving session approvals.

Owns the POST endpoints that record a user's decision on a pending
command/file approval, answer a user-input request, and stop the
in-progress turn for a session, plus the small parsing/validation
helpers those endpoints share.
"""

from __future__ import annotations

import json
from typing import Any

from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from hitch.main.models import ApprovalRequest, CodexInstance, UserInputRequest
from hitch.main.runtime import codex_pool
from hitch.main.sessions.settings_cookies import _MAX_BIGAUTOFIELD
from hitch.main.workflows import system_agents


def _parse_instance_id(raw: str) -> tuple[int | None, str | None]:
    try:
        instance_id = int(raw)
    except ValueError:
        return None, "invalid instance id"
    # Cross-check against the column type up front so a tampered value past
    # the BigAutoField range can't leak a backend-specific OverflowError or
    # DataError out as a 500 from ``objects.get``.
    if instance_id < 1 or instance_id > _MAX_BIGAUTOFIELD:
        return None, "instance id out of range"
    return instance_id, None


# String decisions the approval endpoint accepts. Some approval requests also
# offer structured decisions, such as acceptWithExecpolicyAmendment; those are
# validated against the original app-server payload before being stored.
_VALID_APPROVAL_DECISIONS = frozenset(
    {
        ApprovalRequest.DECISION_ACCEPT,
        ApprovalRequest.DECISION_DECLINE,
        ApprovalRequest.DECISION_CANCEL,
    }
)


def _posted_approval_decision(
    request: HttpRequest, approval: ApprovalRequest
) -> tuple[str | None, Any, str | None]:
    raw_payload = request.POST.get("decision_payload", "").strip()
    if raw_payload:
        try:
            decision_payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None, None, "invalid decision"
        if not _valid_structured_approval_decision(decision_payload):
            return None, None, "invalid decision"
        if not _approval_offered_decision(approval, decision_payload):
            return None, None, "invalid decision"
        return ApprovalRequest.DECISION_ACCEPT, decision_payload, None

    raw_decision = request.POST.get("decision", "").strip()
    decision = ApprovalRequest.normalize_decision(raw_decision)
    if decision not in _VALID_APPROVAL_DECISIONS:
        return None, None, "invalid decision"
    return decision, None, None


def _valid_structured_approval_decision(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) != {"acceptWithExecpolicyAmendment"}:
        return False
    body = value["acceptWithExecpolicyAmendment"]
    if not isinstance(body, dict):
        return False
    amendment = body.get("execpolicy_amendment")
    return (
        isinstance(amendment, list)
        and bool(amendment)
        and all(isinstance(part, str) and part for part in amendment)
    )


def _approval_offered_decision(approval: ApprovalRequest, decision: Any) -> bool:
    available = approval.params.get("availableDecisions")
    if not isinstance(available, list):
        return False
    return any(option == decision for option in available)


@require_http_methods(["POST"])
def resolve_approval(request: HttpRequest, approval_id: int) -> HttpResponse:
    """Record the user's decision on a pending command/file approval.

    The worker's polling loop wakes on the row update and answers the
    SDK's JSON-RPC request with the recorded wire decision. The response is
    intentionally minimal (200 with the recorded status string) so the
    browser-side fetch can surface success without parsing JSON.

    Returns 409 if the approval has already been resolved — racing two
    clicks shouldn't silently overwrite an earlier choice that the worker
    has already returned to codex.
    """
    try:
        approval = ApprovalRequest.objects.get(pk=approval_id)
    except ApprovalRequest.DoesNotExist:
        return HttpResponse("approval not found", status=404)
    if approval.decision:
        return HttpResponse("approval already resolved", status=409)
    decision, decision_payload, error = _posted_approval_decision(request, approval)
    if error is not None or decision is None:
        return HttpResponseBadRequest(error or "invalid decision")
    # Filter on ``decision=""`` so two concurrent POSTs can't both succeed
    # in flipping the row away from pending.
    updated = ApprovalRequest.objects.filter(pk=approval_id, decision="").update(
        decision=decision,
        decision_payload=decision_payload,
        decided_at=timezone.now(),
    )
    if not updated:
        return HttpResponse("approval already resolved", status=409)
    return HttpResponse(decision, content_type="text/plain")


@require_http_methods(["POST"])
def resolve_input_request(request: HttpRequest, input_id: int) -> HttpResponse:
    raw_answers = request.POST.get("answers", "").strip()
    try:
        parsed = json.loads(raw_answers) if raw_answers else {}
    except json.JSONDecodeError:
        return HttpResponseBadRequest("invalid answers")
    if not isinstance(parsed, dict):
        return HttpResponseBadRequest("invalid answers")
    answers: dict[str, Any] = {}
    for key, value in parsed.items():
        key = key.strip()
        if isinstance(value, str):
            value = value.strip()
        if key:
            answers[key] = value
    response: dict[str, Any] = {"answers": answers}
    try:
        input_request = UserInputRequest.objects.get(pk=input_id)
    except UserInputRequest.DoesNotExist:
        return HttpResponse("input request not found", status=404)
    if input_request.response is not None:
        return HttpResponse("input request already resolved", status=409)
    updated = UserInputRequest.objects.filter(pk=input_id, response__isnull=True).update(
        response=response,
        responded_at=timezone.now(),
    )
    if not updated:
        return HttpResponse("input request already resolved", status=409)
    return HttpResponse(json.dumps(response), content_type="application/json")


@require_http_methods(["POST"])
def stop_session(request: HttpRequest, session_id: str) -> HttpResponse:
    """Interrupt the in-progress turn for ``session_id``.

    The Stop button posts the active worker's id (as ``instance``) so a
    stale tab can't abort an unrelated worker or workflow. A workflow-owned
    instance stops its matching workflow as a unit. When the form value is
    missing (older cached page, direct POST) we fall back to the current work.

    No-ops cleanly when no worker is active so a double-click after the
    turn already finished still lands on the session page rather than 404.
    """
    raw = request.POST.get("instance", "").strip()
    if raw:
        instance_id, error = _parse_instance_id(raw)
        if error is not None or instance_id is None:
            return HttpResponseBadRequest(error or "invalid instance id")
        workflow_id = (
            CodexInstance.objects.filter(pk=instance_id, thread_id=session_id)
            .values_list("workflow_id", flat=True)
            .first()
        )
        if workflow_id is None or not system_agents.stop_active_workflow(
            session_id, expected_workflow_id=workflow_id
        ):
            codex_pool.interrupt_instance(
                instance_id, expected_thread_id=session_id
            )
    else:
        if not system_agents.stop_active_workflow(session_id):
            codex_pool.interrupt_active(session_id)
    return redirect("session", session_id=session_id)
