"""User controls for Codex's native session goal."""

import sqlite3
from typing import Any, cast

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from openai_codex.errors import CodexError

from hitch.main.models import SessionMetadata
from hitch.main.runtime import app_server_pool, codex_pool, reconciliation, session_goals
from hitch.main.sessions import lifecycle
from hitch.main.sessions.hitch_instructions import hitch_instructions_for_turn
from hitch.main.sessions.session_settings import (
    _effective_approval_mode_for_session,
    _effective_sandbox_policy_for_cwd,
    _stored_settings,
)
from hitch.main.views import common
from hitch.main.views.session_actions import _read_thread_cwd


@require_POST
def update_goal(request: HttpRequest, session_id: str) -> HttpResponse:
    action = request.POST.get("action", "")
    change: dict[str, Any] = {"action": action}
    if action not in {"start", "budget", "pause", "resume", "clear"}:
        return JsonResponse({"error": "Unknown goal action."}, status=400)
    if action == "start":
        objective = request.POST.get("objective", "").strip()
        if not objective or len(objective) > 4000:
            return JsonResponse({"error": "Enter a goal of up to 4,000 characters."}, status=400)
        change["objective"] = objective
    if action in {"start", "budget"}:
        raw_budget = request.POST.get("token_budget", "").strip()
        if raw_budget and (not raw_budget.isascii() or not raw_budget.isdecimal() or len(raw_budget) > 16):
            return JsonResponse({"error": "Enter a positive whole number for the token budget."}, status=400)
        budget = int(raw_budget) if raw_budget else None
        if budget is not None and not 1 <= budget <= 2**53 - 1:
            return JsonResponse({"error": "The token budget must be between 1 and 9,007,199,254,740,991."}, status=400)
        change["tokenBudget"] = budget
    try:
        with lifecycle.hold(session_id):
            metadata = SessionMetadata.objects.filter(thread_id=session_id).first()
            if metadata and metadata.is_hidden_system_session:
                return JsonResponse({"error": "This session is read-only."}, status=409)
            if metadata and metadata.codex_archived:
                return JsonResponse({"error": "Unarchive this session before changing its goal."}, status=409)
            cwd = metadata.cwd if metadata and metadata.cwd else _read_thread_cwd(request, session_id)
            if not cwd or not common._is_allowed_session_cwd(cwd):
                return JsonResponse(
                    {"error": "Session is unavailable or outside the allowed repositories."}, status=400,
                )
            reconciliation.reconcile_dead_for_thread(session_id)
            active = codex_pool.latest_active_for_thread(session_id)
            goal = session_goals.current_goal(session_id)
            if action == "start" and goal:
                raise ValueError("Clear the existing goal before setting a new one.")
            if action != "start" and not goal:
                raise ValueError("This session no longer has a goal. Refresh the page.")
            if action != "clear" and goal and goal.get("status") == "complete":
                raise ValueError("This goal is complete. Clear it before setting a new goal.")
            if (
                action == "resume" and goal and goal.get("tokenBudget")
                and goal.get("tokensUsed", 0) >= goal["tokenBudget"]
            ):
                raise ValueError("Increase or remove the token budget before resuming.")
            if active:
                if action in {"start", "resume"}:
                    raise ValueError("Wait for the current turn to finish before starting a goal.")
                goal = session_goals.request_live_change(active, change)
                return JsonResponse({"goal": goal})
            home = session_goals.prepare_home(session_id)
            settings = _stored_settings(request)
            lease = session_goals.acquire_home(session_id)
            try:
                with app_server_pool.open_codex(lambda: common.Codex(
                    config=codex_pool.app_server_config(sqlite_home=home),
                )) as codex:
                    if action == "start":
                        # Persist paused first: the detached worker owns activation.
                        result = cast(dict[str, Any], codex._client._request_raw("thread/goal/set", {
                            "threadId": session_id, "objective": change["objective"],
                            "tokenBudget": change["tokenBudget"], "status": "paused",
                        }))
                        goal = result["goal"]
                    elif action != "resume":
                        goal = session_goals.apply_goal(codex._client, session_id, change)
                session_goals.compact_cleared_home(session_id)
            finally:
                lease.release()
            if action in {"start", "resume"}:
                previous = codex_pool.latest_for_thread(session_id)
                codex_pool.spawn_turn(
                    thread_id=session_id, cwd=cwd, prompt="",
                    resume_goal=True,
                    model=(metadata.model if metadata and metadata.model else previous.model if previous else None),
                    reasoning_effort=(
                        metadata.reasoning_effort if metadata and metadata.model
                        else previous.reasoning_effort if previous else None
                    ),
                    sandbox_policy=_effective_sandbox_policy_for_cwd(settings, cwd) or None,
                    approval_mode=_effective_approval_mode_for_session(settings, session_id, metadata),
                    enable_memories=settings.enable_memories,
                    web_search_mode=settings.web_search_mode or None,
                    hitch_extra_instructions=hitch_instructions_for_turn(settings.hitch_extra_instructions),
                )
                return JsonResponse({"goal": goal, "started": True})
            return JsonResponse({"goal": goal})
    except (ValueError, CodexError, OSError, sqlite3.Error) as exc:
        return JsonResponse({"error": str(exc)}, status=409)
