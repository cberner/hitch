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
        "Do a thorough review of the diff. Rebase on master, clean it up, and then "
        "open a PR",
        "Do a thorough review of the diff. Rebase on master, clean it up, and then "
        "open a PR. After opening it, poll the PR every 2 minutes until you have CI "
        "status and at least one review signal: code review comments, a thumbs up "
        "emoji on the PR, or an explicit review approval. On each poll, check whether "
        "the PR has merge conflicts. Address CI failures, review comments, merge "
        "conflicts, and any other blocking issues; push fixes and keep looping until "
        "CI, review, and mergeability are all clean. Stop and report back if any "
        "single polling iteration has no results after 30 minutes.",
    }
)
_PR_WORKFLOW_PROMPT_PREFIXES = (
    "Hitch QA agent could not complete the PR workflow.",
    "Hitch review workflow could not complete.",
    "Hitch PR workflow could not complete.",
    "Hitch PR monitor found follow-up work on the active PR.",
)


def is_pr_creation_prompt(text: str) -> bool:
    return text.strip() in _PR_PROMPT_ALIASES


def is_pr_workflow_notice(text: str) -> bool:
    return text.strip().startswith(_PR_WORKFLOW_PROMPT_PREFIXES)
