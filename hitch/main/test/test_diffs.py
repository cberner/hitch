import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase
from pygments.lexers import PythonLexer

from hitch.main import diffs as diffs_module
from hitch.main.diffs import build_worktree_diff, build_worktree_diff_text
from hitch.main.test.support import _git


class WorktreeDiffTests(SimpleTestCase):
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

    def test_highlight_reuses_cached_lexer_class_for_repeated_filename(self) -> None:
        diffs_module._lexer_class_for_filename.cache_clear()
        try:
            with patch(
                "hitch.main.diffs.find_lexer_class_for_filename",
                return_value=PythonLexer,
            ) as find_lexer_class:
                diffs_module._highlight_code("example.py", "first = 1")
                diffs_module._highlight_code("example.py", "second = 2")

            self.assertEqual(find_lexer_class.call_count, 1)
        finally:
            diffs_module._lexer_class_for_filename.cache_clear()

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

    def test_disjoint_history_shows_branch_content_without_unrelated_deletions(
        self,
    ) -> None:
        # origin/master exists but shares no history with HEAD (orphan branch /
        # re-pointed origin). Diffing against it would render its whole tree as
        # spurious deletions, so the branch content is diffed against the empty
        # tree: feature.py shows as added and remote.py never appears.
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
        paths = {file.path for file in diff.files}
        self.assertIn("feature.py", paths)
        self.assertNotIn("remote.py", paths)

    def test_disjoint_history_uses_repo_object_format_empty_tree(self) -> None:
        # The empty-tree base must match the repo's object format: the sha1
        # empty-tree id is not a valid object in a sha256 repo, which would make
        # the diff silently fall back to HEAD and drop the branch content.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            init = subprocess.run(
                [
                    "git", "init", "--object-format=sha256",
                    "--initial-branch=feature", str(repo),
                ],
                capture_output=True,
            )
            if init.returncode != 0:
                self.skipTest("git lacks sha256 object-format support")
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
        paths = {file.path for file in diff.files}
        self.assertIn("feature.py", paths)
        self.assertNotIn("remote.py", paths)

    def test_shallow_clone_diffs_against_base_ref_not_empty_tree(self) -> None:
        # A shallow clone can omit the shared ancestor, so merge-base reports no
        # common ancestor even though HEAD and origin/main are related. The diff
        # must use the base ref directly (showing only the real change), not the
        # empty tree (which would render every file as an addition).
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            origin = root / "origin"
            subprocess.run(
                ["git", "init", "--initial-branch=main", str(origin)],
                check=True, capture_output=True,
            )
            (origin / "base.py").write_text("def base():\n    return 1\n")
            (origin / "shared.py").write_text("def shared():\n    return 0\n")
            _git(origin, "add", "base.py", "shared.py")
            _git(origin, "commit", "-m", "ancestor")
            _git(origin, "checkout", "-b", "feature")
            (origin / "base.py").write_text("def base():\n    return 2\n")
            _git(origin, "commit", "-am", "feature change")
            _git(origin, "checkout", "main")
            (origin / "base.py").write_text("def base():\n    return 3\n")
            _git(origin, "commit", "-am", "main change")

            clone = root / "clone"
            _git(
                root, "clone", "--depth=1", "--no-single-branch",
                "--branch", "feature", f"file://{origin}", str(clone),
            )
            # The clone is shallow and lacks the ancestor shared with origin/main.
            self.assertEqual(
                _git(clone, "rev-parse", "--is-shallow-repository"), "true"
            )

            diff = build_worktree_diff(str(clone))

        paths = {file.path for file in diff.files}
        self.assertIn("base.py", paths)
        self.assertNotIn("shared.py", paths)

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

    def test_rename_preserves_a_prefixed_directory_segment(self) -> None:
        # git's ``rename from``/``rename to`` headers carry the actual file
        # paths -- unlike ``--- a/...``/``+++ b/...``, they are never tagged
        # with ``--src-prefix``/``--dst-prefix``. Files that legitimately live
        # in a directory whose name starts with ``a/`` or ``b/`` therefore look
        # like prefixed paths and must not have their leading directory
        # stripped.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "a").mkdir()
            old_name = "a/old.txt"
            new_name = "a/new.txt"
            (repo / old_name).write_text("same\n")
            _git(repo, "add", old_name)
            _git(repo, "commit", "-m", "initial")

            (repo / old_name).rename(repo / new_name)
            _git(repo, "add", "-A")

            diff = build_worktree_diff(str(repo))

        self.assertEqual(diff.files[0].status, "Renamed")
        self.assertEqual(diff.files[0].old_path, old_name)
        self.assertEqual(diff.files[0].path, new_name)

    def test_tracked_and_untracked_changes_do_not_phantom_context_line(self) -> None:
        # ``build_worktree_diff`` joins ``git diff`` output -- which always
        # ends with a newline -- to the synthetic untracked-file diff with
        # ``"\n".join``. ``splitlines()`` therefore yields a blank string at
        # the boundary, and previously the parser fell through to the
        # context branch and appended a phantom blank line with bogus line
        # numbers past EOF to the last tracked file. This regresses to a
        # visible "line 2" pseudo-context in a single-line file, plus a
        # misleading additions/deletions tail in the rendered diff.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            tracked = repo / "tracked.txt"
            tracked.write_text("line1\n")
            _git(repo, "add", "tracked.txt")
            _git(repo, "commit", "-m", "initial")

            tracked.write_text("line1 modified\n")
            (repo / "untracked.txt").write_text("brand new\n")

            diff = build_worktree_diff(str(repo))

        paths = [file.path for file in diff.files]
        self.assertIn("tracked.txt", paths)
        self.assertIn("untracked.txt", paths)
        tracked_file = next(file for file in diff.files if file.path == "tracked.txt")
        context_line_numbers = [
            (line.old_lineno, line.new_lineno)
            for line in tracked_file.lines
            if line.kind == "context"
        ]
        self.assertEqual(context_line_numbers, [])

    def test_form_feed_in_content_does_not_split_diff_lines(self) -> None:
        # git frames diff output on ``\n`` only, treating form feeds (common in
        # Python/Emacs/C source as page-break markers), vertical tabs, and the
        # Unicode line/paragraph separators as ordinary line content.
        # ``str.splitlines`` instead breaks on all of them, which used to tear a
        # single diff line in two: the model added exactly one line, but the
        # tail after the form feed -- here beginning with ``-`` -- was reparsed
        # as a phantom deletion, inventing a bogus ``deletions`` total and
        # shifting the old-side line numbers of every following context line.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            tracked = repo / "page.txt"
            tracked.write_text("one\ntwo\nthree\n")
            _git(repo, "add", "page.txt")
            _git(repo, "commit", "-m", "initial")

            tracked.write_text("one\ninserted\x0c-dashed\ntwo\nthree\n")

            diff = build_worktree_diff(str(repo))

        self.assertEqual(diff.additions, 1)
        self.assertEqual(diff.deletions, 0)
        page = next(file for file in diff.files if file.path == "page.txt")
        added = [line for line in page.lines if line.kind == "add"]
        self.assertEqual(len(added), 1)
        self.assertIn("inserted", added[0].html)
        self.assertIn("dashed", added[0].html)
        self.assertEqual([line for line in page.lines if line.kind == "remove"], [])
        # The unchanged trailing lines keep their true old-side line numbers
        # rather than being pushed down by the phantom deletion.
        context = [
            (line.old_lineno, line.new_lineno)
            for line in page.lines
            if line.kind == "context"
        ]
        self.assertEqual(context, [(1, 1), (2, 3), (3, 4)])

    def test_form_feed_in_untracked_file_keeps_one_line_per_newline(self) -> None:
        # The same ``splitlines`` over-split inflated the synthetic untracked
        # file diff: a file with N ``\n``-delimited lines but an embedded form
        # feed rendered N+1 ``+`` rows and a wrong ``@@ -0,0 +1,N @@`` count.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "new.txt").write_text("header\x0cbody\ntail\n")

            diff_text = build_worktree_diff_text(str(repo))
            diff = build_worktree_diff(str(repo))

        self.assertIn("@@ -0,0 +1,2 @@", diff_text)
        new_file = next(file for file in diff.files if file.path == "new.txt")
        added = [line for line in new_file.lines if line.kind == "add"]
        self.assertEqual(len(added), 2)
        self.assertIn("header", added[0].html)
        self.assertIn("body", added[0].html)
        self.assertIn("tail", added[1].html)

    def test_untracked_file_without_trailing_newline_marks_eof(self) -> None:
        # The synthetic untracked-file diff is rendered into the session view
        # and -- when auto-merge is disabled -- forwarded verbatim to the QA
        # reviewer prompt via ``build_worktree_diff_text``. Without the
        # ``\ No newline at end of file`` marker the diff is indistinguishable
        # from a file that does end with ``\n``, so the reviewer cannot tell
        # whether the new file is missing the trailing newline that lint tools
        # commonly flag.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "missing.txt").write_bytes(b"line1\nline2")
            (repo / "single.txt").write_bytes(b"only")
            (repo / "complete.txt").write_bytes(b"line1\nline2\n")

            diff_text = build_worktree_diff_text(str(repo))
            diff = build_worktree_diff(str(repo))

        missing_section, _, after = diff_text.partition("+++ b/missing.txt\n")
        missing_body, _, _ = after.partition("diff --git ")
        self.assertIn("\\ No newline at end of file", missing_body)
        single_section, _, after = diff_text.partition("+++ b/single.txt\n")
        single_body, _, _ = after.partition("diff --git ")
        self.assertIn("\\ No newline at end of file", single_body)
        complete_section, _, after = diff_text.partition("+++ b/complete.txt\n")
        complete_body, _, _ = after.partition("diff --git ")
        self.assertNotIn("\\ No newline at end of file", complete_body)

        files = {file.path: file for file in diff.files}
        missing_meta = [line.html for line in files["missing.txt"].lines if line.kind == "meta"]
        self.assertTrue(
            any("No newline at end of file" in html for html in missing_meta),
            msg=f"expected EOF marker in meta lines, got {missing_meta!r}",
        )
        complete_meta = [line.html for line in files["complete.txt"].lines if line.kind == "meta"]
        self.assertFalse(
            any("No newline at end of file" in html for html in complete_meta),
            msg=f"unexpected EOF marker in meta lines, got {complete_meta!r}",
        )

    def test_git_output_returns_none_on_spawn_failure(self) -> None:
        with patch(
            "hitch.main.git_support.subprocess.run", side_effect=OSError("no git")
        ):
            self.assertIsNone(diffs_module._git_output(Path("/repo"), ["status"]))
