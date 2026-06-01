import subprocess
from typing import override
from unittest.mock import patch

from django.test import SimpleTestCase

from hitch.main import open_pr


def _completed(
    args: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


class _FakeRunner:
    """Dispatches ``subprocess.run`` calls to canned results keyed by argv."""

    def __init__(self, responses: dict[tuple[str, ...], subprocess.CompletedProcess[str]]):
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(list(command))
        for key, result in self.responses.items():
            if tuple(command[: len(key)]) == key:
                return result
        return _completed(command, returncode=0, stdout="")


def _branch_and_head() -> dict[tuple[str, ...], subprocess.CompletedProcess[str]]:
    return {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): _completed(
            [], stdout="feature\n"
        ),
        ("git", "rev-parse", "HEAD"): _completed([], stdout="abc123\n"),
        ("git", "push"): _completed([], returncode=0),
    }


class OpenPullRequestTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        # ``subprocess`` is a shared module, so patching open_pr's reference to
        # ``subprocess.run`` also intercepts repos' default-branch lookup. Stub
        # that lookup so the fake runner only ever answers open_pr's own calls.
        patcher = patch(
            "hitch.main.open_pr.repos.symbolic_default_branch_name",
            return_value=None,
        )
        self.symbolic_default_branch_name = patcher.start()
        self.addCleanup(patcher.stop)

    def test_creates_pr_when_none_exists(self) -> None:
        responses = _branch_and_head()
        responses[("gh", "pr", "view")] = _completed([], returncode=1)
        responses[("gh", "pr", "create")] = _completed(
            [], stdout="https://github.com/cberner/hitch/pull/42\n"
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner):
            opened = open_pr.open_pull_request("/repo")

        self.assertEqual(opened.url, "https://github.com/cberner/hitch/pull/42")
        self.assertEqual(opened.repository_full_name, "cberner/hitch")
        self.assertEqual(opened.pr_number, 42)
        self.assertEqual(opened.head_sha, "abc123")
        self.assertEqual(
            opened.as_handoff(),
            {
                "url": "https://github.com/cberner/hitch/pull/42",
                "repository_full_name": "cberner/hitch",
                "pr_number": 42,
                "state": "open",
                "head_sha": "abc123",
                "latest_commit_sha": "abc123",
            },
        )

    def test_reuses_existing_open_pr_without_creating(self) -> None:
        responses = _branch_and_head()
        responses[("gh", "pr", "view")] = _completed(
            [],
            stdout=(
                '{"url": "https://github.com/cberner/hitch/pull/7", '
                '"state": "OPEN"}'
            ),
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner):
            opened = open_pr.open_pull_request("/repo")

        self.assertEqual(opened.pr_number, 7)
        self.assertFalse(
            any(call[:3] == ["gh", "pr", "create"] for call in runner.calls)
        )

    def test_creates_new_pr_when_only_a_closed_pr_exists(self) -> None:
        # gh pr view can resolve a historical closed PR; it must not be reused.
        responses = _branch_and_head()
        responses[("gh", "pr", "view")] = _completed(
            [],
            stdout=(
                '{"url": "https://github.com/cberner/hitch/pull/7", '
                '"state": "CLOSED"}'
            ),
        )
        responses[("gh", "pr", "create")] = _completed(
            [], stdout="https://github.com/cberner/hitch/pull/8\n"
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner):
            opened = open_pr.open_pull_request("/repo")

        self.assertEqual(opened.pr_number, 8)
        self.assertTrue(
            any(call[:3] == ["gh", "pr", "create"] for call in runner.calls)
        )

    def test_raises_when_push_fails(self) -> None:
        responses = _branch_and_head()
        responses[("git", "push")] = _completed(
            [], returncode=1, stderr="remote rejected"
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner), self.assertRaises(
            open_pr.OpenPrError
        ):
            open_pr.open_pull_request("/repo")

    def test_raises_when_on_default_branch(self) -> None:
        self.symbolic_default_branch_name.return_value = "main"
        responses = _branch_and_head()
        responses[("git", "rev-parse", "--abbrev-ref", "HEAD")] = _completed(
            [], stdout="main\n"
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner), self.assertRaises(
            open_pr.OpenPrError
        ):
            open_pr.open_pull_request("/repo")

        self.assertFalse(any(call[:2] == ["git", "push"] for call in runner.calls))

    def test_fails_closed_on_common_default_when_unresolved(self) -> None:
        # repos cannot infer a default branch, but ``main`` must still be refused.
        self.symbolic_default_branch_name.return_value = None
        responses = _branch_and_head()
        responses[("git", "rev-parse", "--abbrev-ref", "HEAD")] = _completed(
            [], stdout="master\n"
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner), self.assertRaises(
            open_pr.OpenPrError
        ):
            open_pr.open_pull_request("/repo")

        self.assertFalse(any(call[:2] == ["git", "push"] for call in runner.calls))

    def test_allows_integration_branch_when_default_resolved(self) -> None:
        # ``develop`` is a common-name candidate, but with a resolved ``main``
        # default it should be allowed to open a PR into it.
        self.symbolic_default_branch_name.return_value = "main"
        responses = _branch_and_head()
        responses[("git", "rev-parse", "--abbrev-ref", "HEAD")] = _completed(
            [], stdout="develop\n"
        )
        responses[("gh", "pr", "view")] = _completed([], returncode=1)
        responses[("gh", "pr", "create")] = _completed(
            [], stdout="https://github.com/cberner/hitch/pull/55\n"
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner):
            opened = open_pr.open_pull_request("/repo")

        self.assertEqual(opened.pr_number, 55)
        self.assertTrue(
            any(
                call[:2] == ["git", "push"] and "--force-with-lease" in call
                for call in runner.calls
            )
        )

    def test_raises_when_worktree_dirty(self) -> None:
        responses = _branch_and_head()
        responses[("git", "-c", "core.fsmonitor=false", "status", "--porcelain")] = (
            _completed([], stdout=" M hitch/main/open_pr.py\n")
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner), self.assertRaises(
            open_pr.OpenPrError
        ):
            open_pr.open_pull_request("/repo")

        self.assertFalse(any(call[:2] == ["git", "push"] for call in runner.calls))

    def test_raises_on_detached_head(self) -> None:
        responses = _branch_and_head()
        responses[("git", "rev-parse", "--abbrev-ref", "HEAD")] = _completed(
            [], stdout="HEAD\n"
        )
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner), self.assertRaises(
            open_pr.OpenPrError
        ):
            open_pr.open_pull_request("/repo")

    def test_raises_when_create_reports_no_url(self) -> None:
        responses = _branch_and_head()
        responses[("gh", "pr", "view")] = _completed([], returncode=1)
        responses[("gh", "pr", "create")] = _completed([], stdout="done\n")
        runner = _FakeRunner(responses)

        with patch("hitch.main.open_pr.subprocess.run", runner), self.assertRaises(
            open_pr.OpenPrError
        ):
            open_pr.open_pull_request("/repo")
