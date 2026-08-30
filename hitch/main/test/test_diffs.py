import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from hitch.main import diffs as diffs_module
from hitch.main.diffs import build_worktree_diff
from hitch.main.test.support import _git


class WorktreeDiffTests(SimpleTestCase):
    def test_session_preview_bounds_raw_diff_characters(self) -> None:
        raw_diff = "x" * (diffs_module._MAX_DIFF_PREVIEW_CHARS + 1)
        with (
            patch("hitch.main.diffs._worktree_diff_text", return_value=raw_diff),
            patch("hitch.main.diffs._parse_unified_diff") as parse_diff,
        ):
            build_worktree_diff("/repo")

        parse_diff.assert_called_once_with(
            raw_diff[: diffs_module._MAX_DIFF_PREVIEW_CHARS],
            truncated=True,
        )

    def test_session_preview_bounds_rendered_diff_lines(self) -> None:
        changed_line_count = diffs_module._MAX_DIFF_PREVIEW_LINES + 50
        changed_lines = "\n".join(
            f"+changed line {index}" for index in range(changed_line_count)
        )
        raw_diff = (
            "diff --git a/large.txt b/large.txt\n"
            "--- a/large.txt\n"
            "+++ b/large.txt\n"
            f"@@ -0,0 +1,{changed_line_count} @@\n"
            f"{changed_lines}\n"
        )

        with patch("hitch.main.diffs._worktree_diff_text", return_value=raw_diff):
            diff = build_worktree_diff("/repo")

        self.assertTrue(diff.truncated)
        self.assertLessEqual(
            sum(len(file.lines) for file in diff.files),
            diffs_module._MAX_DIFF_PREVIEW_LINES - 1,
        )
        rendered = "\n".join(
            line.html for file in diff.files for line in file.lines
        )
        self.assertIn("changed line 0", rendered)
        self.assertNotIn(f"changed line {changed_line_count - 1}", rendered)

    def test_builds_highlighted_diff_for_tracked_and_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")

            tracked.write_text("def answer():\n    return 2\n")
            (repo / "new_file.py").write_text("def created():\n    return 3\n")

            diff = build_worktree_diff(str(repo))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.file_count, 2)
        self.assertEqual(diff.additions, 3)
        self.assertEqual(diff.deletions, 1)

        files = {file.path: file for file in diff.files}
        self.assertEqual(files["example.py"].status, "Modified")
        self.assertEqual(files["new_file.py"].status, "Added")
        rendered = "\n".join(line.html for file in diff.files for line in file.lines)
        self.assertIn('<span class="k">return</span>', rendered)

    def test_builds_diff_for_committed_branch_changes_against_origin_master(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            worktree = root / "feature-worktree"
            subprocess.run(["git", "init", "--initial-branch=master", str(repo)], check=True, capture_output=True)
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
            _git(repo, "worktree", "add", "-b", "feature", str(worktree), "master")

            branch_file = worktree / "example.py"
            branch_file.write_text("def answer():\n    return 2\n")
            _git(worktree, "commit", "-am", "feature change")

            diff = build_worktree_diff(str(worktree))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.file_count, 1)
        self.assertEqual(diff.files[0].path, "example.py")
        self.assertEqual(diff.additions, 1)
        self.assertEqual(diff.deletions, 1)

    def test_branch_diff_origin_head_beats_closer_non_default_ref(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            worktree = root / "feature-worktree"
            subprocess.run(["git", "init", "--initial-branch=main", str(repo)], check=True, capture_output=True)
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
            _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
            _git(repo, "worktree", "add", "-b", "feature", str(worktree), "main")

            branch_file = worktree / "example.py"
            branch_file.write_text("def answer():\n    return 2\n")
            _git(worktree, "commit", "-am", "feature change")
            _git(worktree, "update-ref", "refs/remotes/origin/master", "HEAD")

            diff = build_worktree_diff(str(worktree))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.file_count, 1)
        self.assertEqual(diff.files[0].path, "example.py")

    def test_merge_base_execution_error_falls_back_to_ref(self) -> None:
        # A merge-base failure that is not "no common ancestor" (e.g. a timeout
        # or lock, surfaced as None) must not be treated as disjoint history.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(
                ["git", "init", "--initial-branch=master", str(repo)],
                check=True, capture_output=True,
            )
            (repo / "base.py").write_text("def base():\n    return 1\n")
            (repo / "shared.py").write_text("def shared():\n    return 0\n")
            _git(repo, "add", "base.py", "shared.py")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
            (repo / "base.py").write_text("def base():\n    return 2\n")
            _git(repo, "commit", "-am", "local change")

            real_git_output = diffs_module._git_output

            def fake_git_output(
                repo_arg: Path,
                args: list[str],
                *,
                allow_statuses: set[int] | None = None,
            ) -> str | None:
                if args[:1] == ["merge-base"]:
                    return None  # simulate a timeout / lock, not a clean status 1
                return real_git_output(repo_arg, args, allow_statuses=allow_statuses)

            with patch.object(
                diffs_module, "_git_output", side_effect=fake_git_output
            ):
                diff = build_worktree_diff(str(repo))

        # Falls back to diffing against origin/master, so only the changed file
        # shows -- not the whole tree as additions (the empty-tree path).
        paths = {file.path for file in diff.files}
        self.assertIn("base.py", paths)
        self.assertNotIn("shared.py", paths)

    def test_unborn_repo_still_shows_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "new_file.py").write_text("def created():\n    return 3\n")

            diff = build_worktree_diff(str(repo))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.files[0].path, "new_file.py")
        self.assertEqual(diff.files[0].status, "Added")

    def test_git_output_returns_none_on_spawn_failure(self) -> None:
        with patch(
            "hitch.main.git_support.subprocess.run", side_effect=OSError("no git")
        ):
            self.assertIsNone(diffs_module._git_output(Path("/repo"), ["status"]))
