"""Prompts and legacy aliases for agent-owned PR turns."""

PR_SLASH_DISPLAY_PROMPT = (
    "Rebase on the default branch, clean it up, and then open a PR"
)
_LEGACY_PR_SLASH_PROMPT = (
    "Rebase on the default branch, polish it, get it ready, "
    "and commit the final changes. "
    "Do not push the branch or open a PR; Hitch will push and open it "
    "after this turn completes."
)
_LEGACY_HITCH_PUBLISHED_PR_PROMPTS = frozenset(
    {
        _LEGACY_PR_SLASH_PROMPT,
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
PR_SLASH_PROMPT = (
    "Rebase on the default branch, polish it, get it ready, "
    "run the relevant tests, and commit the final changes. Use Codex's built-in "
    "PR publishing tool to push the branch and open or update the pull request "
    "with a clear title and description. Then call `hitch.watch_pr` with the "
    "full PR URL; that registers the PR with Hitch and polls its checks and "
    "reviews. Assess any feedback, fix valid issues, test, commit, and publish "
    "follow-up changes, then call `hitch.watch_pr` again. If there are changes "
    "to publish, do not finish without calling `hitch.watch_pr`."
)

_PR_PROMPT_ALIASES = frozenset(
    {
        "/pr",
        PR_SLASH_DISPLAY_PROMPT,
        "Rebase on the repository's default branch, clean it up, and then open a PR",
        "Rebase on master, clean it up, and then open a PR",
        "Polish it, get it ready, and open or update the PR.",
        PR_SLASH_PROMPT,
        *_LEGACY_HITCH_PUBLISHED_PR_PROMPTS,
    }
)
_LEGACY_PR_WORKFLOW_NOTICE_PREFIXES = (
    "Hitch review workflow could not complete.",
    "Hitch PR workflow could not complete.",
    "Hitch PR monitor found follow-up work on the active PR.",
)


def is_pr_creation_prompt(text: str) -> bool:
    return text.strip() in _PR_PROMPT_ALIASES


def is_legacy_hitch_published_pr_prompt(text: str) -> bool:
    """Return whether a pre-watch PR turn expected Hitch to publish for it."""
    return any(prompt in text for prompt in _LEGACY_HITCH_PUBLISHED_PR_PROMPTS)


def is_pr_workflow_notice(text: str) -> bool:
    """Recognize old wrapper narration while parsing pre-upgrade rollouts."""
    return text.strip().startswith(_LEGACY_PR_WORKFLOW_NOTICE_PREFIXES)
