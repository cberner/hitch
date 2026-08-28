"""Prompts that recommend Codex's optional native reviewer subagent."""

from __future__ import annotations

from hitch.main.sessions.pr_prompts import PR_SLASH_PROMPT


def optional_review_prompt(
    *,
    prepare_pull_request: bool,
    auto_merge_branch: str = "",
) -> str:
    prompt = (
        "Inspect the complete current changes and improve them as needed. You "
        "have a native, read-only Codex reviewer subagent named "
        "`hitch_reviewer`. Using it is recommended, but not required; use your "
        "judgment. If useful, invoke `spawn_agent` with "
        "`agent_type=\"hitch_reviewer\"` and `fork_turns=\"none\"`, ask it to "
        "review the complete current changes, and assess its findings yourself. "
        "Fix every valid issue and request another independent pass only if it "
        "would be useful. Run the relevant tests before finishing."
    )
    if auto_merge_branch:
        prompt += (
            " The completed changes will be merged into local branch "
            f"`{auto_merge_branch}`, so include that target in any review focus."
        )
    if prepare_pull_request:
        prompt = f"{prompt}\n\n{PR_SLASH_PROMPT}"
    return prompt
