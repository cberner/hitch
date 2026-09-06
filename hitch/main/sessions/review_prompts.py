"""Review guidance that leaves delegation to the coding agent."""

from __future__ import annotations

from hitch.main.sessions.pr_prompts import PR_SLASH_PROMPT

QA_SLASH_DISPLAY_PROMPT = (
    "Ask the coding agent to inspect the changes and optionally use a reviewer subagent"
)


def optional_review_prompt(
    *,
    prepare_pull_request: bool,
) -> str:
    prompt = (
        "Inspect the complete current change set relative to an appropriate "
        "merge base, including committed, staged, unstaged, and untracked "
        "changes plus relevant surrounding code and tests. Improve the changes "
        "as needed. "
        "Delegate review to Codex subagents as you see fit, and assess their "
        "findings yourself. Fix valid issues and run the relevant tests before "
        "finishing."
    )
    if prepare_pull_request:
        prompt = f"{prompt}\n\n{PR_SLASH_PROMPT}"
    return prompt
