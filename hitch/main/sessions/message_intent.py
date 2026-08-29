"""Parsing of posted-message intent and PR/QA activation slash commands."""

from __future__ import annotations

from typing import NamedTuple

from django.http import HttpRequest

from hitch.main.sessions.pr_prompts import (
    PR_SLASH_DISPLAY_PROMPT,
    is_pr_creation_prompt,
)
from hitch.main.sessions.session_settings import _QA_SLASH_PROMPT

_PLAN_SLASH_COMMAND = "/plan"
_PR_SLASH_COMMAND = "/pr"
_PR_NOW_SLASH_COMMAND = "/pr-now"
_FIX_PR_SLASH_COMMAND = "/fix-pr"
_QA_SLASH_COMMAND = "/qa"


class _MessageIntent(NamedTuple):
    prompt: str
    plan_mode: bool
    allow_pending_plan_default: bool
    explicit_plan_mode: bool
    pr_activation: bool = False
    pr_now_activation: bool = False
    fix_pr_activation: bool = False
    qa_activation: bool = False


def _message_intent(request: HttpRequest) -> _MessageIntent:
    prompt = request.POST.get("prompt", "").strip()
    plan_mode = request.POST.get("plan_mode", "").strip().lower() == "true"
    default_plan_mode_raw = request.POST.get("default_plan_mode")
    default_plan_mode = (
        default_plan_mode_raw.strip().lower() == "true"
        if default_plan_mode_raw is not None
        else False
    )
    default_plan_mode_posted = default_plan_mode_raw is not None
    plan_mode_changed = (
        plan_mode != default_plan_mode if default_plan_mode_posted else plan_mode
    )
    explicit_plan_mode = (
        request.POST.get("plan_mode_explicit", "").strip().lower() == "true"
        or plan_mode_changed
    )
    parts = prompt.split(maxsplit=1)
    if not parts:
        return _MessageIntent(prompt, plan_mode, True, explicit_plan_mode)
    command = parts[0].lower()
    if command == _PLAN_SLASH_COMMAND:
        return _MessageIntent(
            parts[1].strip() if len(parts) > 1 else "",
            True,
            True,
            True,
        )
    if command == _PR_SLASH_COMMAND:
        return _MessageIntent(
            PR_SLASH_DISPLAY_PROMPT, False, False, False, pr_activation=True
        )
    if command == _PR_NOW_SLASH_COMMAND:
        return _MessageIntent(
            PR_SLASH_DISPLAY_PROMPT, False, False, False, pr_now_activation=True
        )
    if command == _FIX_PR_SLASH_COMMAND:
        return _MessageIntent(
            _FIX_PR_SLASH_COMMAND, False, False, False, fix_pr_activation=True
        )
    if command == _QA_SLASH_COMMAND:
        return _MessageIntent(
            _QA_SLASH_PROMPT, False, False, False, qa_activation=True
        )
    if not plan_mode and is_pr_creation_prompt(prompt):
        return _MessageIntent(prompt, False, False, False, pr_activation=True)
    if not plan_mode and prompt == _QA_SLASH_PROMPT:
        return _MessageIntent(
            _QA_SLASH_PROMPT, False, False, False, qa_activation=True
        )
    return _MessageIntent(
        prompt,
        plan_mode,
        True,
        explicit_plan_mode,
        pr_activation=is_pr_creation_prompt(prompt),
        qa_activation=prompt == _QA_SLASH_PROMPT,
    )
