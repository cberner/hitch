"""Prompts and command aliases for agent-owned PR turns."""

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
_LEGACY_SINGLE_WATCH_PR_PROMPT = (
    "Rebase on the default branch, polish it, get it ready, "
    "run the relevant tests, and commit the final changes. Use Codex's built-in "
    "PR publishing tool to push the branch and open or update the pull request "
    "with a clear title and description. Then call `hitch.watch_pr` with the "
    "full PR URL; that registers the PR with Hitch and polls its checks and "
    "reviews. Assess any feedback, fix valid issues, test, commit, and publish "
    "follow-up changes, then call `hitch.watch_pr` again. If there are changes "
    "to publish, do not finish without calling `hitch.watch_pr`."
)
_LEGACY_BOUNDED_WATCH_INSTRUCTIONS = (
    "A watch invocation ends when it returns; registration does not keep a "
    "background watcher running. After `attention` or `action_required`, assess "
    "the feedback, address valid issues, and call `hitch.watch_pr` again even "
    "when no code changes are needed. Continue until `ready` or `terminal`; "
    "report `timed_out`, a tool failure, or a blocker you cannot resolve clearly. "
    "Do not finish merely because CI is green while review is still pending."
)
PR_WATCH_FOLLOW_UP_INSTRUCTIONS = (
    "Registration keeps watching after an invocation returns, including after "
    "`ready` or `timed_out`. Assess `attention` or `action_required` feedback, "
    "address valid issues, and call `hitch.watch_pr` again even when no code "
    "changes are needed. Hitch resumes the visible session for new feedback "
    "while it is idle. Watching stops only when the PR is merged or closed, "
    "or you explicitly call `hitch.unwatch_pr` with its URL. Report unresolved "
    "blockers or tool failures clearly; use `unwatch_pr` if you decide to stop."
)
PR_SLASH_PROMPT = (
    f"{_LEGACY_SINGLE_WATCH_PR_PROMPT} {PR_WATCH_FOLLOW_UP_INSTRUCTIONS}"
)

_PR_PROMPT_ALIASES = frozenset(
    {
        "/pr",
        PR_SLASH_DISPLAY_PROMPT,
        "Rebase on the repository's default branch, clean it up, and then open a PR",
        "Rebase on master, clean it up, and then open a PR",
        "Polish it, get it ready, and open or update the PR.",
        PR_SLASH_PROMPT,
        _LEGACY_SINGLE_WATCH_PR_PROMPT,
        f"{_LEGACY_SINGLE_WATCH_PR_PROMPT} {_LEGACY_BOUNDED_WATCH_INSTRUCTIONS}",
        *_LEGACY_HITCH_PUBLISHED_PR_PROMPTS,
    }
)


def is_pr_creation_prompt(text: str) -> bool:
    return text.strip() in _PR_PROMPT_ALIASES
