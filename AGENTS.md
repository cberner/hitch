# Agent instructions for hitch

This file tells coding agents how to work productively in this repository.

## Specs
- Before working on a feature, check `docs/specs/` for a relevant spec.
- Treat specs as authoritative for behavior, UX, and acceptance criteria.
- If a user's request contradicts a spec, flag the contradiction to the user before implementing.
- If code changes intentionally change behavior covered by a spec, update the spec in the same change or explicitly tell the user it needs a follow-up update.

## Before completing codebase-changing work

**Run `just test` and confirm it passes after making any change that can affect the codebase
in the current working directory.**
This target builds the Podman test image, runs the `test-pre` recipe first, and
then runs the Django test suite inside a container with isolated Hitch and
Codex state. The image build performs `uv sync --all-groups --locked`;
`test-pre` runs `ruff check .` and `mypy .` in the container.
If any of those fail, fix the underlying issue — do not bypass checks.

## Style guide
- Comments should be brief and focus on important invariants, architectural details, or other
  long-term relevant information. They should not contain minor implementation details of the current
  commit.

## Tests
When adding new features, add tests — but aim for high code coverage and important integration
tests without adding too many lines of new test code. 90% coverage is a good target for new
features; it does not have to be 100%. Expanding a logically related existing test is often a good
way to achieve coverage without bloating the suite.

## Git commits
1) git commits should use your human's name and email address for authorship. Add "Assisted-by:" and
   your agent name at the end of the commit message. In the same style as the
   [Linux Kernel's coding assistant guidelines](https://github.com/torvalds/linux/blob/master/Documentation/process/coding-assistants.rst).
2) Make one commit per feature / bug fix when opening a PR. Multiple commits or "fixup" commits are
   should not be merged to master.
