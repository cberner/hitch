"""Editable Hitch guidance, supplied separately from submitted user messages."""

from hitch.main.sessions.pr_prompts import PR_SLASH_PROMPT
from hitch.main.sessions.review_prompts import optional_review_prompt

DEFAULT_HITCH_EXTRA_INSTRUCTIONS = (
    "Follow the Hitch workflow selected for this turn. In Plan mode, plan only; "
    "do not perform automatic review or publish a pull request. When the "
    "workflow is None, respond to the user's request without starting an "
    "automatic review or pull-request workflow.\n\n"
    "When Auto-QA or Auto-PR is selected, after completing the user's requested "
    "implementation, continue in the same turn:\n\n"
    f"{optional_review_prompt(prepare_pull_request=False)}\n\n"
    "When Auto-PR is selected, also complete the following:\n\n"
    f"{PR_SLASH_PROMPT}"
)


def hitch_instructions_for_turn(
    override: str | None,
    *,
    auto_pr_enabled: bool = False,
    auto_qa_enabled: bool = False,
    plan_mode: bool = False,
    pr_title: str = "",
) -> str:
    instructions = DEFAULT_HITCH_EXTRA_INSTRUCTIONS if override is None else override
    if not instructions.strip():
        return ""
    if plan_mode:
        workflow = "Plan"
    elif auto_pr_enabled:
        workflow = "Auto-PR"
    elif auto_qa_enabled:
        workflow = "Auto-QA"
    else:
        workflow = "None"
    context = f"Hitch workflow for this turn: {workflow}."
    if auto_pr_enabled and not plan_mode and (title := " ".join(pr_title.split())):
        context += f"\nRequested pull request title: {title}"
    return f"{context}\n\n{instructions}"


def combined_developer_instructions(personal: str, hitch: str | None) -> str:
    return "\n\n".join(part for part in (personal, hitch) if part)
