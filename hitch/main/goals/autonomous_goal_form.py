"""Autonomous-goal form validation and budget-display formatting.

Leaf helpers extracted from ``views.py``: they validate POSTed
autonomous-goal form fields and format proposal-budget values for
display. Depend only on models and stdlib — never on ``views``.
"""

from __future__ import annotations

from decimal import Decimal, DecimalException
from typing import NamedTuple

from django.http import HttpRequest

from hitch.main.codex_pool import _VALID_WEB_SEARCH_MODES
from hitch.main.models import AutonomousGoal
from hitch.main.settings_cookies import _MAX_BIGAUTOFIELD
from hitch.main.workflows.agent_io import _AUTONOMOUS_GOAL_TITLE_MAX_LEN

_MAX_BIGAUTOFIELD_DECIMAL = Decimal(_MAX_BIGAUTOFIELD)
_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_UNIT = 1_000_000
_MAX_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_MILLIONS = _MAX_BIGAUTOFIELD_DECIMAL / Decimal(
    _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_UNIT
)


class AutonomousGoalValues(NamedTuple):
    title: str
    goal: str
    ambition: str
    autonomy: str
    auto_qa_enabled: bool
    auto_proposal_enabled: bool
    stacked_diff_depth: int
    proposal_budget: int | None
    confidence_threshold: str
    web_search_mode: str
    auto_merge_to_local_branch: bool
    auto_merge_branch: str


def _validated_autonomous_goal_title(raw_title: str) -> tuple[str, str | None]:
    title = raw_title.strip()
    if not title:
        return "", "title is required"
    if len(title) > _AUTONOMOUS_GOAL_TITLE_MAX_LEN:
        return "", "title is too long"
    return title, None


def _validated_autonomous_goal_values(
    request: HttpRequest,
    *,
    autonomy_default: str = AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
    auto_qa_default: bool = False,
    web_search_default: str = AutonomousGoal.WEB_SEARCH_DEFAULT,
    auto_proposal_default: bool = False,
    stacked_diff_depth_default: int = AutonomousGoal.STACKED_DIFF_DEPTH_MIN,
    proposal_budget_default: int | None = None,
    local_branches: list[str] | None = None,
) -> tuple[AutonomousGoalValues | None, str | None]:
    title, error = _validated_autonomous_goal_title(request.POST.get("title", ""))
    if error is not None:
        return None, error
    goal = request.POST.get("goal", "").strip()
    if not goal:
        return None, "goal is required"
    ambition = request.POST.get("ambition", "").strip()
    valid_ambitions = {value for value, _label in AutonomousGoal.AMBITION_CHOICES}
    if ambition not in valid_ambitions:
        return None, "ambition is invalid"
    autonomy = (
        request.POST.get("autonomy", autonomy_default).strip()
        or autonomy_default
    )
    valid_autonomies = {value for value, _label in AutonomousGoal.AUTONOMY_CHOICES}
    if autonomy not in valid_autonomies:
        return None, "autonomy is invalid"
    required_auto_qa = AutonomousGoal.auto_qa_required_for_autonomy(autonomy)
    supported_auto_qa = AutonomousGoal.auto_qa_supported_for_autonomy(autonomy)
    auto_qa_values = [value.strip() for value in request.POST.getlist("auto_qa")]
    if any(value not in {"", "false", "true"} for value in auto_qa_values):
        return None, "auto-QA setting is invalid"
    if required_auto_qa:
        auto_qa_enabled = False
    elif auto_qa_values:
        auto_qa_enabled = auto_qa_values[-1] == "true" and supported_auto_qa
    else:
        auto_qa_enabled = auto_qa_default and supported_auto_qa
    auto_proposal_enabled, auto_proposal_error = _posted_autonomous_goal_bool(
        request.POST.get("auto_proposal"),
        default=auto_proposal_default,
        setting_name="auto-proposal",
    )
    if auto_proposal_error is not None:
        return None, auto_proposal_error
    stacked_diff_depth, stacked_diff_depth_error = (
        _posted_autonomous_goal_stacked_diff_depth(
            request.POST.get("stacked_diff_depth"),
            default=stacked_diff_depth_default,
            autonomy=autonomy,
        )
    )
    if stacked_diff_depth_error is not None:
        return None, stacked_diff_depth_error
    proposal_budget, proposal_budget_error = _posted_autonomous_goal_proposal_budget(
        request.POST.get("proposal_budget"),
        default=proposal_budget_default,
    )
    if proposal_budget_error is not None:
        return None, proposal_budget_error
    threshold = request.POST.get("confidence_threshold", "").strip()
    valid_thresholds = {value for value, _label in AutonomousGoal.CONFIDENCE_CHOICES}
    if threshold not in valid_thresholds:
        return None, "confidence threshold is invalid"
    web_search_mode = (
        request.POST["web_search_mode"].strip()
        if "web_search_mode" in request.POST
        else web_search_default
    )
    if web_search_mode not in {"", *_VALID_WEB_SEARCH_MODES}:
        return None, "web search setting is invalid"
    auto_merge = request.POST.get("auto_merge_to_local_branch", "").strip()
    if auto_merge not in {"", "false", "true"}:
        return None, "auto merge setting is invalid"
    auto_merge_to_local_branch = auto_merge == "true"
    auto_merge_branch = request.POST.get("auto_merge_branch", "").strip()
    valid_local_branches = set(local_branches or [])
    if auto_merge_to_local_branch:
        if not auto_qa_enabled:
            return None, "auto merge requires auto-QA"
        if not auto_merge_branch:
            return None, "auto merge branch is required"
        if auto_merge_branch not in valid_local_branches:
            return None, "auto merge branch is invalid"
    else:
        auto_merge_branch = ""
    return AutonomousGoalValues(
        title=title,
        goal=goal,
        ambition=ambition,
        autonomy=autonomy,
        auto_qa_enabled=auto_qa_enabled,
        auto_proposal_enabled=auto_proposal_enabled,
        stacked_diff_depth=stacked_diff_depth,
        proposal_budget=proposal_budget,
        confidence_threshold=threshold,
        web_search_mode=web_search_mode,
        auto_merge_to_local_branch=auto_merge_to_local_branch,
        auto_merge_branch=auto_merge_branch,
    ), None


def _posted_autonomous_goal_stacked_diff_depth(
    raw: str | None, *, default: int, autonomy: str
) -> tuple[int, str | None]:
    supported = AutonomousGoal.stacked_diff_supported_for_autonomy(autonomy)
    if raw is None or not raw.strip():
        return (default if supported else AutonomousGoal.STACKED_DIFF_DEPTH_MIN), None
    try:
        depth = int(raw.strip())
    except ValueError:
        return 0, "stacked diff depth is invalid"
    if (
        depth < AutonomousGoal.STACKED_DIFF_DEPTH_MIN
        or depth > AutonomousGoal.STACKED_DIFF_DEPTH_MAX
    ):
        return 0, "stacked diff depth is invalid"
    if not supported and depth != AutonomousGoal.STACKED_DIFF_DEPTH_MIN:
        return 0, "stacked diff depth requires draft patch or draft PR"
    return (depth if supported else AutonomousGoal.STACKED_DIFF_DEPTH_MIN), None


def _posted_autonomous_goal_proposal_budget(
    raw: str | None, *, default: int | None
) -> tuple[int | None, str | None]:
    if raw is None:
        return default, None
    raw = raw.strip()
    if not raw:
        return None, None
    try:
        budget_millions = Decimal(raw)
    except DecimalException:
        return None, "proposal budget is invalid"
    if (
        not budget_millions.is_finite()
        or budget_millions <= 0
        or budget_millions > _MAX_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_MILLIONS
    ):
        return None, "proposal budget is invalid"
    try:
        budget_decimal = budget_millions * _AUTONOMOUS_GOAL_PROPOSAL_BUDGET_UNIT
    except DecimalException:
        return None, "proposal budget is invalid"
    if budget_decimal > _MAX_BIGAUTOFIELD_DECIMAL:
        return None, "proposal budget is invalid"
    if budget_decimal != budget_decimal.to_integral_value():
        return None, "proposal budget is invalid"
    budget = int(budget_decimal)
    if budget < 1 or budget > _MAX_BIGAUTOFIELD:
        return None, "proposal budget is invalid"
    return budget, None


def _posted_autonomous_goal_bool(
    raw: str | None, *, default: bool, setting_name: str
) -> tuple[bool, str | None]:
    if raw is None:
        return default, None
    value = raw.strip().lower()
    if value in {"", "false"}:
        return False, None
    if value == "true":
        return True, None
    return False, f"{setting_name} is invalid"


def _attach_autonomous_goal_display_state(goals: list[AutonomousGoal]) -> None:
    for goal in goals:
        goal.proposal_budget_form_value = _autonomous_goal_budget_millions_value(  # type: ignore[attr-defined]
            goal.proposal_budget
        )
        goal.proposal_budget_display = _autonomous_goal_budget_display(  # type: ignore[attr-defined]
            goal.proposal_budget
        )


def _autonomous_goal_budget_millions_value(budget: int | None) -> str:
    if budget is None:
        return ""
    return _trim_decimal_text(
        format(Decimal(budget) / Decimal(_AUTONOMOUS_GOAL_PROPOSAL_BUDGET_UNIT), "f")
    )


def _autonomous_goal_budget_display(budget: int | None) -> str:
    if budget is None:
        return ""
    return f"{_autonomous_goal_budget_millions_value(budget)}M tokens"


def _trim_decimal_text(value: str) -> str:
    return value.rstrip("0").rstrip(".") if "." in value else value
