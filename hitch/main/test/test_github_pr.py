"""Tests for the ``gh``/``git``-backed PR helper."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from hitch.main import github_pr


def _completed(
    args: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


_PR_VIEW_JSON = {
    "number": 42,
    "url": "https://github.com/cberner/hitch/pull/42",
    "state": "OPEN",
    "isDraft": False,
    "merged": False,
    "mergedAt": None,
    "mergeCommit": None,
    "mergeable": "MERGEABLE",
    "title": "Add feature",
    "baseRefName": "master",
    "headRefName": "feature",
    "headRefOid": "abc123",
    "createdAt": "2026-05-01T00:00:00Z",
    "updatedAt": "2026-05-02T00:00:00Z",
    "closedAt": None,
    "reviewDecision": "CHANGES_REQUESTED",
    "reviews": [{"state": "CHANGES_REQUESTED"}],
    "latestReviews": [{"state": "CHANGES_REQUESTED"}],
    "reviewThreads": [
        {"isResolved": False, "isOutdated": False, "path": "a.py", "line": 3, "id": "t1"},
        {"isResolved": True, "isOutdated": False, "path": "b.py", "id": "t2"},
    ],
    "comments": [
        {"body": "please   fix this", "url": "https://x/1", "createdAt": "2026-05-02T00:00:00Z"}
    ],
    "reactionGroups": [{"content": "THUMBS_UP", "users": {"totalCount": 2}}],
    "statusCheckRollup": [
        {"__typename": "CheckRun", "name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"},
        {"__typename": "CheckRun", "name": "lint", "status": "IN_PROGRESS"},
        {"__typename": "StatusContext", "context": "ci/ext", "state": "SUCCESS"},
    ],
}


class GithubPrHelperTests(SimpleTestCase):
    @patch("hitch.main.github_pr.subprocess.run")
    def test_current_branch_returns_named_branch(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(["git"], stdout="feature\n")

        self.assertEqual(github_pr.current_branch("/repo"), "feature")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_current_branch_rejects_detached_head(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(["git"], stdout="HEAD\n")

        with self.assertRaises(github_pr.GithubCliError):
            github_pr.current_branch("/repo")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_run_raises_on_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(["git"], returncode=1, stderr="boom")

        with self.assertRaises(github_pr.GithubCliError) as ctx:
            github_pr.push_branch("/repo", "feature")
        self.assertIn("boom", str(ctx.exception))

    @patch("hitch.main.github_pr.subprocess.run")
    def test_push_branch_refuses_protected_branches(self, mock_run: MagicMock) -> None:
        # Default branch resolves to master; force-pushing it must be refused.
        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/master\n")
            raise AssertionError(f"should not run {argv}")

        mock_run.side_effect = fake_run

        for branch in ("master", "main"):
            with self.assertRaises(github_pr.GithubCliError):
                github_pr.push_branch("/repo", branch)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_push_branch_refuses_custom_default_branch(
        self, mock_run: MagicMock
    ) -> None:
        pushed: list[list[str]] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/trunk\n")
            pushed.append(argv)
            return _completed(argv)

        mock_run.side_effect = fake_run

        with self.assertRaises(github_pr.GithubCliError):
            github_pr.push_branch("/repo", "trunk")
        self.assertEqual(pushed, [])

    @patch("hitch.main.github_pr.subprocess.run")
    def test_mark_ready_invokes_gh_pr_ready(self, mock_run: MagicMock) -> None:
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return _completed(argv)

        mock_run.side_effect = fake_run

        github_pr.mark_ready("/repo", pr_number=42)

        self.assertEqual(calls, [["gh", "pr", "ready", "42"]])


    @patch("hitch.main.github_pr.subprocess.run")
    def test_run_wraps_os_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("gh missing")

        with self.assertRaises(github_pr.GithubCliError):
            github_pr.push_branch("/repo", "feature")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_enterprise_pr_threads_pass_hostname(self, mock_run: MagicMock) -> None:
        # A GitHub Enterprise PR URL must route the GraphQL thread fetch to that
        # host via --hostname, not the default github.com.
        enterprise = {
            **_PR_VIEW_JSON,
            "url": "https://ghe.example.com/cberner/hitch/pull/42",
        }
        graphql_argv: list[str] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["gh", "api", "graphql"]:
                graphql_argv.extend(argv)
                return _completed(
                    argv,
                    stdout=json.dumps(
                        {
                            "data": {
                                "repository": {
                                    "pullRequest": {
                                        "reviewThreads": {
                                            "nodes": [],
                                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                                        }
                                    }
                                }
                            }
                        }
                    ),
                )
            return _completed(argv, stdout=json.dumps(enterprise))

        mock_run.side_effect = fake_run

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["repository_full_name"], "cberner/hitch")
        self.assertIn("--hostname", graphql_argv)
        self.assertIn("ghe.example.com", graphql_argv)
        # A plain github.com PR must NOT pass --hostname.
        self.assertEqual(github_pr._host_from_url("https://github.com/o/r/pull/1"), "")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_outdated_unresolved_thread_still_counts(
        self, mock_run: MagicMock
    ) -> None:
        # An outdated-but-unresolved thread still blocks merge under required
        # conversation resolution, so it must be counted as unresolved.
        threads_json = json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": [
                                    {
                                        "id": "t1",
                                        "isResolved": False,
                                        "isOutdated": True,
                                        "path": "a.py",
                                        "line": 3,
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }
            }
        )

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, stdout=threads_json)
            return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))

        mock_run.side_effect = fake_run

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["unresolved_thread_count"], 1)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_maps_review_ci_and_threads(
        self, mock_run: MagicMock
    ) -> None:
        threads_json = json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": [
                                    {"isResolved": False, "isOutdated": False, "path": "a.py", "line": 3, "id": "t1"},
                                    {"isResolved": True, "isOutdated": False, "path": "b.py", "id": "t2"},
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }
            }
        )

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, stdout=threads_json)
            return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))

        mock_run.side_effect = fake_run

        snapshot = github_pr.fetch_pr_snapshot("/repo", branch="feature")

        assert snapshot is not None
        self.assertEqual(snapshot["url"], "https://github.com/cberner/hitch/pull/42")
        self.assertEqual(snapshot["repository_full_name"], "cberner/hitch")
        self.assertEqual(snapshot["pr_number"], 42)
        self.assertEqual(snapshot["state"], "open")
        self.assertIs(snapshot["merged"], False)
        self.assertIs(snapshot["draft"], False)
        self.assertIs(snapshot["mergeable"], True)
        self.assertEqual(snapshot["head_sha"], "abc123")
        self.assertEqual(snapshot["latest_commit_sha"], "abc123")
        # An explicit changes-requested decision wins over the thumbs-up reaction.
        self.assertEqual(snapshot["review_signal"], "changes_requested")
        self.assertEqual(snapshot["reaction_count"], 2)
        self.assertEqual(snapshot["unresolved_thread_count"], 1)
        self.assertEqual(snapshot["review_thread_count"], 2)
        self.assertEqual(snapshot["unresolved_threads"], [{"path": "a.py", "line": 3, "id": "t1"}])
        self.assertEqual(snapshot["comment_count"], 1)
        self.assertEqual(snapshot["latest_comments"][0]["body"], "please fix this")
        self.assertEqual(snapshot["ci_status"], "failure")
        self.assertEqual(snapshot["failing_jobs"], [{"name": "tests"}])
        self.assertEqual(snapshot["pending_jobs"], [{"name": "lint"}])

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_thumbs_up_and_clean_ci(
        self, mock_run: MagicMock
    ) -> None:
        data = {
            **_PR_VIEW_JSON,
            "reviewDecision": "",
            "reviews": [],
            "reviewThreads": [],
            "statusCheckRollup": [
                {"__typename": "CheckRun", "name": "tests", "status": "COMPLETED", "conclusion": "SUCCESS"},
            ],
        }
        mock_run.return_value = _completed(["gh"], stdout=json.dumps(data))

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["review_signal"], "thumbs_up")
        self.assertEqual(snapshot["ci_status"], "success")
        self.assertEqual(snapshot["failing_jobs"], [])
        self.assertEqual(snapshot["pending_jobs"], [])

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_unknown_check_status_is_pending(
        self, mock_run: MagicMock
    ) -> None:
        # A check with neither a usable status nor state must not be counted as
        # a passed check; treat it as still pending.
        data = {
            **_PR_VIEW_JSON,
            "reviewDecision": "APPROVED",
            "statusCheckRollup": [{"__typename": "CheckRun", "name": "mystery"}],
        }
        mock_run.return_value = _completed(["gh"], stdout=json.dumps(data))

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "pending")
        self.assertEqual(snapshot["pending_jobs"], [{"name": "mystery"}])
        self.assertEqual(snapshot["failing_jobs"], [])

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_no_checks_is_success(self, mock_run: MagicMock) -> None:
        data = {**_PR_VIEW_JSON, "statusCheckRollup": []}
        mock_run.return_value = _completed(["gh"], stdout=json.dumps(data))

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "success")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_thread_fetch_failure_leaves_unresolved_count_unknown(
        self, mock_run: MagicMock
    ) -> None:
        # A failed GraphQL thread fetch must not record zero unresolved threads,
        # or an approved PR could pass the review gate without observing threads.
        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, returncode=1, stderr="rate limited")
            return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))

        mock_run.side_effect = fake_run

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertNotIn("unresolved_thread_count", snapshot)
        self.assertNotIn("review_thread_count", snapshot)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_inline_review_comment_text_surfaced_to_feedback(
        self, mock_run: MagicMock
    ) -> None:
        threads_json = json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": [
                                    {
                                        "id": "t1",
                                        "isResolved": False,
                                        "isOutdated": False,
                                        "path": "a.py",
                                        "line": 3,
                                        "comments": {
                                            "nodes": [
                                                {"body": "please rename this", "url": "https://x/c1"}
                                            ]
                                        },
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }
            }
        )

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, stdout=threads_json)
            return _completed(argv, stdout=json.dumps({**_PR_VIEW_JSON, "comments": []}))

        mock_run.side_effect = fake_run

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["unresolved_thread_count"], 1)
        bodies = [c.get("body", "") for c in snapshot["latest_comments"]]
        self.assertTrue(any("please rename this" in b for b in bodies))
        self.assertTrue(any("a.py" in b for b in bodies))

    @patch("hitch.main.github_pr.subprocess.run")
    def test_changes_requested_review_body_surfaced(self, mock_run: MagicMock) -> None:
        # A changes-requested review with only a main body (no inline thread)
        # must still surface its text so the fix agent knows what to change.
        data = {
            **_PR_VIEW_JSON,
            "reviewDecision": "CHANGES_REQUESTED",
            "reviews": [
                {"state": "CHANGES_REQUESTED", "body": "Please add tests", "url": "https://x/r1"}
            ],
            "comments": [],
        }
        mock_run.return_value = _completed(["gh"], stdout=json.dumps(data))

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["review_signal"], "changes_requested")
        bodies = [c.get("body", "") for c in snapshot["latest_comments"]]
        self.assertTrue(any("Please add tests" in b for b in bodies))

    @patch("hitch.main.github_pr.subprocess.run")
    def test_open_or_update_pr_tolerates_existing_pr_on_create(
        self, mock_run: MagicMock
    ) -> None:
        # A closed PR for the head is not found by the open-only lookup, so
        # create is attempted and gh reports it already exists; fall through to
        # reading the PR back rather than failing.
        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/main\n")
            if argv[:2] == ["git", "push"]:
                return _completed(argv)
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, stdout="{}")
            if argv[:3] == ["gh", "pr", "list"]:
                return _completed(argv, stdout="[]")
            if argv[:3] == ["gh", "pr", "create"]:
                return _completed(
                    argv,
                    returncode=1,
                    stderr="a pull request for branch feature already exists",
                )
            if argv[:3] == ["gh", "pr", "view"]:
                return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))
            raise AssertionError(f"unexpected argv {argv}")

        mock_run.side_effect = fake_run

        snapshot = github_pr.open_or_update_pr("/repo", branch="feature")

        self.assertEqual(snapshot["pr_number"], 42)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_open_or_update_pr_raises_on_real_create_error(
        self, mock_run: MagicMock
    ) -> None:
        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/main\n")
            if argv[:2] == ["git", "push"]:
                return _completed(argv)
            if argv[:3] == ["gh", "pr", "list"]:
                return _completed(argv, stdout="[]")
            if argv[:3] == ["gh", "pr", "create"]:
                return _completed(argv, returncode=1, stderr="validation failed")
            if argv[:3] == ["gh", "pr", "view"]:
                return _completed(argv, returncode=1, stderr="no pull requests found")
            raise AssertionError(f"unexpected argv {argv}")

        mock_run.side_effect = fake_run

        with self.assertRaises(github_pr.GithubCliError):
            github_pr.open_or_update_pr("/repo", branch="feature")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_review_threads_paginate_across_pages(self, mock_run: MagicMock) -> None:
        def thread_page(node_id: str, *, resolved: bool, has_next: bool, cursor: str | None) -> str:
            return json.dumps(
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "nodes": [
                                        {
                                            "id": node_id,
                                            "isResolved": resolved,
                                            "isOutdated": False,
                                            "path": "a.py",
                                            "line": 1,
                                        }
                                    ],
                                    "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                                }
                            }
                        }
                    }
                }
            )

        calls = {"n": 0}

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["gh", "api", "graphql"]:
                calls["n"] += 1
                if any(a.startswith("after=") for a in argv):
                    # Second page: an unresolved thread beyond the first 100.
                    return _completed(argv, stdout=thread_page("t2", resolved=False, has_next=False, cursor=None))
                return _completed(argv, stdout=thread_page("t1", resolved=True, has_next=True, cursor="CUR"))
            return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))

        mock_run.side_effect = fake_run

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(calls["n"], 2)
        self.assertEqual(snapshot["review_thread_count"], 2)
        # The unresolved thread on the second page must be counted.
        self.assertEqual(snapshot["unresolved_thread_count"], 1)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_review_threads_incomplete_pages_stay_unknown(
        self, mock_run: MagicMock
    ) -> None:
        # A page set that never terminates (always hasNextPage) must be treated
        # as unobserved rather than authoritative.
        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(
                    argv,
                    stdout=json.dumps(
                        {
                            "data": {
                                "repository": {
                                    "pullRequest": {
                                        "reviewThreads": {
                                            "nodes": [],
                                            "pageInfo": {"hasNextPage": True, "endCursor": "CUR"},
                                        }
                                    }
                                }
                            }
                        }
                    ),
                )
            return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))

        mock_run.side_effect = fake_run

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertNotIn("unresolved_thread_count", snapshot)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_derives_merged_from_state(
        self, mock_run: MagicMock
    ) -> None:
        data = {**_PR_VIEW_JSON, "state": "MERGED", "mergedAt": "2026-05-03T00:00:00Z"}
        data.pop("merged", None)
        mock_run.return_value = _completed(["gh"], stdout=json.dumps(data))

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertIs(snapshot["merged"], True)
        self.assertEqual(snapshot["state"], "merged")
        # ``merged`` must not be requested from gh (unsupported --json field).
        self.assertNotIn("merged", github_pr._PR_VIEW_FIELDS.split(","))
        self.assertNotIn("reviewThreads", github_pr._PR_VIEW_FIELDS.split(","))

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_returns_none_when_no_pr(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _completed(
            ["gh"], returncode=1, stderr="no pull requests found for branch feature"
        )

        self.assertIsNone(github_pr.fetch_pr_snapshot("/repo", branch="feature"))

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_raises_on_real_error(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(["gh"], returncode=1, stderr="network down")

        with self.assertRaises(github_pr.GithubCliError):
            github_pr.fetch_pr_snapshot("/repo", branch="feature")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_open_or_update_pr_creates_when_missing(self, mock_run: MagicMock) -> None:
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/main\n")
            if argv[:2] == ["git", "push"]:
                return _completed(argv)
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, stdout="{}")
            if argv[:3] == ["gh", "pr", "list"]:
                return _completed(argv, stdout="[]")
            if argv[:3] == ["gh", "pr", "create"]:
                return _completed(argv, stdout="created")
            if argv[:3] == ["gh", "pr", "view"]:
                return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))
            raise AssertionError(f"unexpected argv {argv}")

        mock_run.side_effect = fake_run

        snapshot = github_pr.open_or_update_pr("/repo", branch="feature")

        self.assertEqual(snapshot["pr_number"], 42)
        commands = [argv[:3] for argv in calls]
        push_calls = [argv for argv in calls if argv[:2] == ["git", "push"]]
        self.assertTrue(push_calls)
        self.assertIn("--force-with-lease", push_calls[0])
        self.assertIn(["gh", "pr", "create"], commands)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_open_or_update_pr_passes_title_base_draft(
        self, mock_run: MagicMock
    ) -> None:
        create_argv: list[str] = []

        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/main\n")
            if argv[:2] == ["git", "push"]:
                return _completed(argv)
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, stdout="{}")
            if argv[:3] == ["gh", "pr", "list"]:
                return _completed(argv, returncode=1, stderr="boom")
            if argv[:3] == ["gh", "pr", "create"]:
                create_argv.extend(argv)
                return _completed(argv)
            if argv[:3] == ["gh", "pr", "view"]:
                return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))
            raise AssertionError(f"unexpected argv {argv}")

        mock_run.side_effect = fake_run

        github_pr.open_or_update_pr(
            "/repo", branch="feature", base="main", title="Feat", body="body", draft=True
        )

        self.assertIn("--base", create_argv)
        self.assertIn("--title", create_argv)
        self.assertIn("--draft", create_argv)
        self.assertNotIn("--fill", create_argv)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_open_or_update_pr_raises_when_snapshot_unreadable(
        self, mock_run: MagicMock
    ) -> None:
        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/main\n")
            if argv[:2] == ["git", "push"]:
                return _completed(argv)
            if argv[:3] == ["gh", "pr", "list"]:
                return _completed(argv, stdout="[]")
            if argv[:3] == ["gh", "pr", "create"]:
                return _completed(argv)
            if argv[:3] == ["gh", "pr", "view"]:
                return _completed(argv, returncode=1, stderr="no pull requests found")
            raise AssertionError(f"unexpected argv {argv}")

        mock_run.side_effect = fake_run

        with self.assertRaises(github_pr.GithubCliError):
            github_pr.open_or_update_pr("/repo", branch="feature")

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_maps_conflicting_merged_and_status_context(
        self, mock_run: MagicMock
    ) -> None:
        data = {
            **_PR_VIEW_JSON,
            "state": "MERGED",
            "merged": True,
            "mergedAt": "2026-05-03T00:00:00Z",
            "mergeCommit": {"oid": "deadbeef"},
            "mergeable": "CONFLICTING",
            "reviewDecision": "APPROVED",
            "reactionGroups": [],
            "comments": {"nodes": [{"body": "hi"}]},
            "reviewThreads": {"nodes": []},
            "statusCheckRollup": [
                {"__typename": "StatusContext", "context": "ci/ext", "state": "FAILURE"},
            ],
        }
        mock_run.return_value = _completed(["gh"], stdout=json.dumps(data))

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["state"], "merged")
        self.assertIs(snapshot["merged"], True)
        self.assertEqual(snapshot["merged_at"], "2026-05-03T00:00:00Z")
        self.assertEqual(snapshot["merge_commit_sha"], "deadbeef")
        self.assertIs(snapshot["mergeable"], False)
        self.assertEqual(snapshot["review_signal"], "approved")
        self.assertEqual(snapshot["ci_status"], "failure")
        self.assertEqual(snapshot["failing_jobs"], [{"name": "ci/ext"}])
        # ``comments`` arrived in {"nodes": [...]} GraphQL shape.
        self.assertEqual(snapshot["comment_count"], 1)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_commented_signal_and_pending_status_context(
        self, mock_run: MagicMock
    ) -> None:
        data = {
            **_PR_VIEW_JSON,
            "reviewDecision": "REVIEW_REQUIRED",
            "reviews": [{"state": "COMMENTED"}],
            "reactionGroups": [],
            "statusCheckRollup": [
                {"__typename": "StatusContext", "context": "ci/ext", "state": "PENDING"},
            ],
        }
        mock_run.return_value = _completed(["gh"], stdout=json.dumps(data))

        snapshot = github_pr.fetch_pr_snapshot("/repo", pr_number=42)

        assert snapshot is not None
        self.assertEqual(snapshot["review_signal"], "commented")
        self.assertEqual(snapshot["ci_status"], "pending")
        self.assertEqual(snapshot["pending_jobs"], [{"name": "ci/ext"}])

    @patch("hitch.main.github_pr.subprocess.run")
    def test_fetch_pr_snapshot_raises_on_bad_json(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(["gh"], stdout="not json")

        with self.assertRaises(github_pr.GithubCliError):
            github_pr.fetch_pr_snapshot("/repo", pr_number=42)

    @patch("hitch.main.github_pr.subprocess.run")
    def test_open_or_update_pr_skips_create_when_pr_exists(
        self, mock_run: MagicMock
    ) -> None:
        def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if argv[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return _completed(argv, stdout="origin/main\n")
            if argv[:2] == ["git", "push"]:
                return _completed(argv)
            if argv[:3] == ["gh", "api", "graphql"]:
                return _completed(argv, stdout="{}")
            if argv[:3] == ["gh", "pr", "list"]:
                return _completed(argv, stdout=json.dumps([{"number": 42}]))
            if argv[:3] == ["gh", "pr", "view"]:
                return _completed(argv, stdout=json.dumps(_PR_VIEW_JSON))
            if argv[:3] == ["gh", "pr", "create"]:
                raise AssertionError("should not create when a PR already exists")
            raise AssertionError(f"unexpected argv {argv}")

        mock_run.side_effect = fake_run

        snapshot = github_pr.open_or_update_pr("/repo", branch="feature")

        self.assertEqual(snapshot["pr_number"], 42)
