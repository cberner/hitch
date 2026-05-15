# Agent instructions for hitch

This file tells coding agents how to work productively in this repository.

## Before completing a task

**Always run `just test` and confirm it passes before telling the user you are done.**
This target runs the `pre` recipe first, which executes `uv sync --all-groups`,
`uv run ruff check .`, and `uv run mypy .`, and then runs the Django test suite.
If any of those fail, fix the underlying issue — do not bypass checks.

## Style guide
- Comments should be brief and focus on important invariants, architectural details, or other
  long-term relevant information. They should not contain minor implementation details of the current
  commit.

## Tests
When adding new features, add tests — but aim for high code coverage and important integration
tests without adding too many lines of new test code. Expanding a logically related existing test
is often a good way to achieve coverage without bloating the suite.

## Git commits
1) git commits should use your human's name and email address for authorship. Add "Assisted-by:" and
   your agent name at the end of the commit message. In the same style as the
   [Linux Kernel's coding assistant guidelines](https://github.com/torvalds/linux/blob/master/Documentation/process/coding-assistants.rst).
2) Make one commit per feature / bug fix when opening a PR. Multiple commits or "fixup" commits are
   should not be merged to master.
