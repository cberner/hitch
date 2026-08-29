"""Shared recognition of prompts that belong to Hitch's PR workflow."""

PR_SLASH_DISPLAY_PROMPT = (
    "Rebase on the default branch, clean it up, and then open a PR"
)
PR_SLASH_PROMPT = (
    "Rebase on the default branch, polish it, get it ready, "
    "and commit the final changes. "
    "Do not push the branch or open a PR; Hitch will push and open it "
    "after this turn completes."
)

_PR_PROMPT_ALIASES = frozenset(
    {
        "/pr",
        PR_SLASH_DISPLAY_PROMPT,
        "Rebase on the repository's default branch, clean it up, and then open a PR",
        "Rebase on master, clean it up, and then open a PR",
        "Polish it, get it ready, and open or update the PR.",
        PR_SLASH_PROMPT,
        "Rebase on the repository's default branch, polish it, get it ready, and "
        "commit the final changes. Do not push the branch or open a PR; Hitch will "
        "push and open it after this turn completes.",
        "Rebase on master, polish it, get it ready, and commit the final changes. Do "
        "not push the branch or open a PR; Hitch will push and open it after this turn "
        "completes.",
        "Polish it, get it ready, and commit the final changes. Do not push the branch "
        "or open a PR; Hitch will push and open it after this turn completes.",
        "Polish it, get it ready, commit the final changes, and push the branch. Do "
        "not open a PR; Hitch will open it after this turn completes.",
    }
)
_PR_WORKFLOW_PROMPT_PREFIXES = (
    "Hitch review workflow could not complete.",
    "Hitch PR workflow could not complete.",
    "Hitch PR monitor found follow-up work on the active PR.",
)


def is_pr_creation_prompt(text: str) -> bool:
    return text.strip() in _PR_PROMPT_ALIASES


def is_pr_workflow_notice(text: str) -> bool:
    return text.strip().startswith(_PR_WORKFLOW_PROMPT_PREFIXES)
