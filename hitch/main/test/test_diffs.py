import os
import subprocess
import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from hitch.main.diffs import build_worktree_diff


def _git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Hitch Tests",
        "GIT_AUTHOR_EMAIL": "hitch@example.com",
        "GIT_COMMITTER_NAME": "Hitch Tests",
        "GIT_COMMITTER_EMAIL": "hitch@example.com",
    }
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)


class WorktreeDiffTests(SimpleTestCase):
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

    def test_branch_diff_uses_merge_base_when_origin_master_has_advanced(self) -> None:
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

            (repo / "upstream.py").write_text("def upstream():\n    return 10\n")
            _git(repo, "add", "upstream.py")
            _git(repo, "commit", "-m", "upstream change")
            _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")

            branch_file = worktree / "example.py"
            branch_file.write_text("def answer():\n    return 2\n")
            _git(worktree, "commit", "-am", "feature change")

            diff = build_worktree_diff(str(worktree))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.file_count, 1)
        self.assertEqual(diff.files[0].path, "example.py")
        self.assertEqual(diff.additions, 1)
        self.assertEqual(diff.deletions, 1)

    def test_branch_diff_prefers_origin_head_over_stale_origin_main(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            worktree = root / "feature-worktree"
            subprocess.run(["git", "init", "--initial-branch=master", str(repo)], check=True, capture_output=True)
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")

            (repo / "upstream.py").write_text("def upstream():\n    return 10\n")
            _git(repo, "add", "upstream.py")
            _git(repo, "commit", "-m", "upstream change")
            _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
            _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/master")
            _git(repo, "worktree", "add", "-b", "feature", str(worktree), "master")

            branch_file = worktree / "example.py"
            branch_file.write_text("def answer():\n    return 2\n")
            _git(worktree, "commit", "-am", "feature change")

            diff = build_worktree_diff(str(worktree))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.file_count, 1)
        self.assertEqual(diff.files[0].path, "example.py")

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

    def test_branch_diff_prefers_closest_remote_base_without_origin_head(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            worktree = root / "feature-worktree"
            subprocess.run(["git", "init", "--initial-branch=master", str(repo)], check=True, capture_output=True)
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")

            (repo / "upstream.py").write_text("def upstream():\n    return 10\n")
            _git(repo, "add", "upstream.py")
            _git(repo, "commit", "-m", "upstream change")
            _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
            _git(repo, "worktree", "add", "-b", "feature", str(worktree), "master")

            branch_file = worktree / "example.py"
            branch_file.write_text("def answer():\n    return 2\n")
            _git(worktree, "commit", "-am", "feature change")

            diff = build_worktree_diff(str(worktree))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.file_count, 1)
        self.assertEqual(diff.files[0].path, "example.py")

    def test_branch_diff_falls_back_to_origin_master_without_merge_base(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", "--initial-branch=feature", str(repo)], check=True, capture_output=True)
            (repo / "feature.py").write_text("def feature():\n    return 1\n")
            _git(repo, "add", "feature.py")
            _git(repo, "commit", "-m", "feature")

            _git(repo, "checkout", "--orphan", "remote-master")
            _git(repo, "rm", "-rf", ".")
            (repo / "remote.py").write_text("def remote():\n    return 2\n")
            _git(repo, "add", "remote.py")
            _git(repo, "commit", "-m", "remote master")
            _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
            _git(repo, "checkout", "feature")

            diff = build_worktree_diff(str(repo))

        self.assertTrue(diff.has_changes)
        self.assertIn("feature.py", {file.path for file in diff.files})

    def test_origin_master_diff_does_not_require_local_master_branch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            worktree = root / "feature-worktree"
            subprocess.run(["git", "init", "--initial-branch=main", str(repo)], check=True, capture_output=True)
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
            _git(repo, "worktree", "add", "-b", "feature", str(worktree), "main")

            branch_file = worktree / "example.py"
            branch_file.write_text("def answer():\n    return 2\n")
            _git(worktree, "commit", "-am", "feature change")

            diff = build_worktree_diff(str(worktree))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.files[0].path, "example.py")

    def test_builds_diff_for_committed_branch_changes_against_origin_main(self) -> None:
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
            _git(repo, "worktree", "add", "-b", "feature", str(worktree), "main")

            branch_file = worktree / "example.py"
            branch_file.write_text("def answer():\n    return 2\n")
            _git(worktree, "commit", "-am", "feature change")

            diff = build_worktree_diff(str(worktree))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.file_count, 1)
        self.assertEqual(diff.files[0].path, "example.py")
        self.assertEqual(diff.additions, 1)
        self.assertEqual(diff.deletions, 1)

    def test_committed_branch_changes_fall_back_to_head_without_origin_master(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", "--initial-branch=master", str(repo)], check=True, capture_output=True)
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "checkout", "-b", "feature")
            tracked.write_text("def answer():\n    return 2\n")
            _git(repo, "commit", "-am", "feature change")

            diff = build_worktree_diff(str(repo))

        self.assertFalse(diff.has_changes)

    def test_hunk_lines_that_look_like_file_headers_stay_in_hunk(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            tracked = repo / "operators.txt"
            tracked.write_text("-- i;\nkeep\n")
            _git(repo, "add", "operators.txt")
            _git(repo, "commit", "-m", "initial")

            tracked.write_text("++ i;\nkeep\n")

            diff = build_worktree_diff(str(repo))

        self.assertEqual(diff.additions, 1)
        self.assertEqual(diff.deletions, 1)
        self.assertEqual(diff.files[0].path, "operators.txt")
        rendered = "\n".join(line.html for file in diff.files for line in file.lines)
        self.assertIn("++ i;", rendered)
        self.assertIn("-- i;", rendered)

    def test_unborn_repo_still_shows_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "new_file.py").write_text("def created():\n    return 3\n")

            diff = build_worktree_diff(str(repo))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.files[0].path, "new_file.py")
        self.assertEqual(diff.files[0].status, "Added")

    def test_quoted_binary_diff_header_preserves_path_with_space_b_slash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            path = repo / "x b" / "y.bin"
            path.parent.mkdir()
            path.write_bytes(b"\0old")
            _git(repo, "add", "x b/y.bin")
            _git(repo, "commit", "-m", "initial")

            path.write_bytes(b"\0new")

            diff = build_worktree_diff(str(repo))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.files[0].path, "x b/y.bin")

    def test_untracked_symlink_does_not_render_target_contents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            outside = Path(raw) / "outside.txt"
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            outside.write_text("do not leak")
            (repo / "link.txt").symlink_to(outside)

            diff = build_worktree_diff(str(repo))

        rendered = "\n".join(line.html for file in diff.files for line in file.lines)
        self.assertIn("Symlink not shown", rendered)
        self.assertNotIn("do not leak", rendered)

    def test_non_utf8_untracked_path_does_not_crash_diff_build(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "visible.txt").write_text("shown\n")
            bad_path = os.path.join(os.fsencode(repo), b"bad-\xff.txt")
            fd = os.open(bad_path, os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                os.write(fd, b"hidden\n")
            finally:
                os.close(fd)

            diff = build_worktree_diff(str(repo))

        self.assertIn("visible.txt", {file.path for file in diff.files})

    def test_mnemonic_prefix_config_does_not_leak_into_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            _git(repo, "config", "diff.mnemonicPrefix", "true")
            tracked = repo / "example.py"
            tracked.write_text("def answer():\n    return 1\n")
            _git(repo, "add", "example.py")
            _git(repo, "commit", "-m", "initial")

            tracked.write_text("def answer():\n    return 2\n")

            diff = build_worktree_diff(str(repo))

        self.assertEqual(diff.files[0].path, "example.py")

    def test_quoted_git_path_decodes_octal_utf8_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            filename = "\N{MICRO SIGN}.txt"
            tracked = repo / filename
            tracked.write_text("old\n")
            _git(repo, "add", filename)
            _git(repo, "commit", "-m", "initial")

            tracked.write_text("new\n")

            diff = build_worktree_diff(str(repo))

        self.assertEqual(diff.files[0].path, filename)

    def test_rename_headers_decode_quoted_git_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            old_name = "\N{MICRO SIGN} old.txt"
            new_name = "\N{MICRO SIGN} new.txt"
            old_path = repo / old_name
            old_path.write_text("same\n")
            _git(repo, "add", old_name)
            _git(repo, "commit", "-m", "initial")

            old_path.rename(repo / new_name)
            _git(repo, "add", "-A")

            diff = build_worktree_diff(str(repo))

        self.assertEqual(diff.files[0].status, "Renamed")
        self.assertEqual(diff.files[0].old_path, old_name)
        self.assertEqual(diff.files[0].path, new_name)
