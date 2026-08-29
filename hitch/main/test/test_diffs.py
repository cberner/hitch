import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from django.test import SimpleTestCase

from hitch.main import diffs as diffs_module
from hitch.main.diffs import (
    IncompleteDiffError,
    build_worktree_diff,
    build_worktree_diff_text,
)
from hitch.main.git_support import GitCommandError
from hitch.main.git_support import run_git as git_run_git
from hitch.main.test.support import _git


def _init_repo_with_submodule(
    root: Path, *, submodule_path: str = "vendor/lib"
) -> Path:
    repo = root / "repo"
    submodule_source = root / "submodule-source"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "init", str(submodule_source)], check=True, capture_output=True
    )
    (submodule_source / "README.md").write_text("nested\n")
    _git(submodule_source, "add", "README.md")
    _git(submodule_source, "commit", "-m", "initial")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule_source),
        submodule_path,
    )
    _git(repo, "commit", "-m", "add submodule")
    return repo


class WorktreeDiffTests(SimpleTestCase):
    def test_reviewer_diff_rejects_unavailable_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing"
            nongit = Path(raw) / "nongit"
            nongit.mkdir()

            for cwd in (None, str(missing), str(nongit)):
                with self.subTest(cwd=cwd), self.assertRaises(IncompleteDiffError):
                    build_worktree_diff_text(cwd)

    def test_reviewer_diff_rejects_omitted_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            for index in range(diffs_module._MAX_UNTRACKED_FILES + 1):
                (repo / f"new-{index}.txt").write_text(f"change {index}\n")

            preview = build_worktree_diff(str(repo))
            with self.assertRaisesRegex(IncompleteDiffError, "untracked files"):
                build_worktree_diff_text(str(repo))

        self.assertFalse(preview.truncated)
        self.assertIn("untracked files omitted", preview.files[-1].lines[-1].html)

    def test_reviewer_diff_rejects_truncated_untracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "large.txt").write_bytes(
                b"x" * (diffs_module._MAX_UNTRACKED_FILE_BYTES + 1)
            )

            preview = build_worktree_diff(str(repo))
            with self.assertRaisesRegex(IncompleteDiffError, "large.txt"):
                build_worktree_diff_text(str(repo))

        self.assertIn("File preview truncated", preview.files[0].lines[-1].html)

    def test_reviewer_diff_rejects_tracked_change_above_handoff_limit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            tracked = repo / "large.txt"
            tracked.write_text("initial\n")
            _git(repo, "add", tracked.name)
            _git(repo, "commit", "-m", "initial")
            tracked.write_text("x" * (diffs_module.REVIEWER_DIFF_MAX_BYTES + 1))

            with self.assertRaisesRegex(
                IncompleteDiffError, "reviewer handoff limit"
            ):
                build_worktree_diff_text(str(repo))

    def test_reviewer_diff_handoff_limit_counts_utf8_bytes(self) -> None:
        with (
            patch.object(diffs_module, "REVIEWER_DIFF_MAX_BYTES", 10),
            patch(
                "hitch.main.diffs._strict_worktree_diff_text",
                return_value="☃" * 4,
            ),
            self.assertRaisesRegex(IncompleteDiffError, "10-byte reviewer"),
        ):
            build_worktree_diff_text("/repo")

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
            review_diff = build_worktree_diff_text(str(repo))

        self.assertTrue(diff.has_changes)
        paths = {file.path for file in diff.files}
        self.assertIn("feature.py", paths)
        self.assertNotIn("remote.py", paths)
        self.assertIn("feature.py", review_diff)
        self.assertNotIn("remote.py", review_diff)

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
            with self.assertRaisesRegex(IncompleteDiffError, "shallow repository"):
                build_worktree_diff_text(str(clone))

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

    def test_reviewer_diff_does_not_treat_head_lookup_failure_as_unborn(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "tracked.txt").write_text("initial\n")
            _git(repo, "add", "tracked.txt")
            _git(repo, "commit", "-m", "initial")

            def fail_head_lookup(
                cwd: str | Path, args: list[str], **kwargs: Any
            ) -> subprocess.CompletedProcess[bytes]:
                if args == ["rev-parse", "--verify", "--quiet", "HEAD"]:
                    raise GitCommandError("timed out")
                return git_run_git(cwd, args, **kwargs)

            with patch.object(
                diffs_module, "run_git", side_effect=fail_head_lookup
            ), self.assertRaisesRegex(IncompleteDiffError, "git could not run"):
                build_worktree_diff_text(str(repo))

    def test_unborn_repo_still_shows_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "new_file.py").write_text("def created():\n    return 3\n")

            diff = build_worktree_diff(str(repo))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.files[0].path, "new_file.py")
        self.assertEqual(diff.files[0].status, "Added")

    def test_reviewer_diff_includes_staged_file_on_unborn_branch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "staged.txt").write_text("must be reviewed\n")
            _git(repo, "add", "staged.txt")

            diff_text = build_worktree_diff_text(str(repo))

        self.assertIn("diff --git a/staged.txt b/staged.txt", diff_text)
        self.assertIn("+must be reviewed", diff_text)

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
            with self.assertRaisesRegex(IncompleteDiffError, "tracked binary"):
                build_worktree_diff_text(str(repo))

        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.files[0].path, "x b/y.bin")

    def test_reviewer_diff_rejects_non_utf8_tracked_text(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            path = repo / "latin1.txt"
            path.write_bytes(b"old-\xe9\n")
            _git(repo, "add", "latin1.txt")
            _git(repo, "commit", "-m", "initial")
            path.write_bytes(b"new-\xea\n")

            with self.assertRaisesRegex(IncompleteDiffError, "tracked non-UTF-8"):
                build_worktree_diff_text(str(repo))

    def test_reviewer_diff_rejects_nul_forced_to_text(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / ".gitattributes").write_text("*.dat diff\n")
            path = repo / "forced.dat"
            path.write_bytes(b"old\0value\n")
            _git(repo, "add", ".gitattributes", "forced.dat")
            _git(repo, "commit", "-m", "initial")
            path.write_bytes(b"new\0value\n")

            with self.assertRaisesRegex(IncompleteDiffError, "NUL-bearing"):
                build_worktree_diff_text(str(repo))

    def test_reviewer_diff_disables_textconv_filters(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            converter = repo / "normalize.sh"
            converter.write_text("#!/bin/sh\nprintf 'normalized\\n'\n")
            converter.chmod(0o755)
            (repo / ".gitattributes").write_text("*.dat diff=normalize\n")
            data = repo / "value.dat"
            data.write_text("old value\n")
            _git(repo, "config", "diff.normalize.textconv", str(converter))
            _git(repo, "add", ".gitattributes", "normalize.sh", "value.dat")
            _git(repo, "commit", "-m", "initial")
            data.write_text("new value\n")

            diff_text = build_worktree_diff_text(str(repo))

        self.assertIn("-old value", diff_text)
        self.assertIn("+new value", diff_text)

    def test_reviewer_diff_rejects_dirty_submodule_despite_ignore_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = _init_repo_with_submodule(root)
            _git(repo, "config", "diff.ignoreSubmodules", "all")
            _git(repo, "config", "diff.submodule", "log")
            (repo / "vendor/lib/README.md").write_text("dirty nested change\n")

            with self.assertRaisesRegex(IncompleteDiffError, "dirty submodule"):
                build_worktree_diff_text(str(repo))

    def test_reviewer_diff_rejects_renamed_dirty_submodule_with_ambiguous_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = _init_repo_with_submodule(root, submodule_path="x b/y")
            _git(repo, "mv", "x b/y", "new")
            (repo / "new/README.md").write_text("dirty nested change\n")

            with self.assertRaisesRegex(IncompleteDiffError, "dirty submodule"):
                build_worktree_diff_text(str(repo))

    def test_reviewer_diff_allows_gitlink_replaced_by_marker_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = _init_repo_with_submodule(root)
            shutil.rmtree(repo / "vendor/lib")
            (repo / "vendor/lib").write_text("Subproject commit abc-dirty\n")
            _git(repo, "add", "-A")

            diff_text = build_worktree_diff_text(str(repo))

        self.assertIn("deleted file mode 160000", diff_text)
        self.assertIn("new file mode 100644", diff_text)
        self.assertIn("+Subproject commit abc-dirty", diff_text)

    def test_untracked_symlink_does_not_render_target_contents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            outside = Path(raw) / "outside.txt"
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            outside.write_text("do not leak")
            (repo / "link.txt").symlink_to(outside)

            diff = build_worktree_diff(str(repo))
            with self.assertRaisesRegex(IncompleteDiffError, "symbolic link"):
                build_worktree_diff_text(str(repo))

        rendered = "\n".join(line.html for file in diff.files for line in file.lines)
        self.assertIn("Symlink not shown", rendered)
        self.assertNotIn("do not leak", rendered)

    def test_non_utf8_untracked_path_does_not_crash_diff_build(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "visible.txt").write_text("shown\n")
            (repo / "bad-\N{REPLACEMENT CHARACTER}.txt").write_text("decoy\n")
            bad_path = os.path.join(os.fsencode(repo), b"bad-\xff.txt")
            fd = os.open(bad_path, os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                os.write(fd, b"hidden\n")
            finally:
                os.close(fd)

            diff = build_worktree_diff(str(repo))
            with self.assertRaisesRegex(IncompleteDiffError, "non-UTF-8 path"):
                build_worktree_diff_text(str(repo))

        self.assertIn("visible.txt", {file.path for file in diff.files})

    def test_reviewer_diff_rejects_tabbed_untracked_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "before\tafter.txt").write_text("cannot be represented safely\n")

            with self.assertRaisesRegex(IncompleteDiffError, "untracked path"):
                build_worktree_diff_text(str(repo))

    def test_reviewer_diff_rejects_untracked_binary_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "asset.bin").write_bytes(b"\0not reviewable")

            preview = build_worktree_diff(str(repo))
            with self.assertRaisesRegex(IncompleteDiffError, "untracked binary"):
                build_worktree_diff_text(str(repo))

        self.assertIn("Binary file not shown", preview.files[0].lines[-1].html)

    def test_reviewer_diff_uses_default_when_core_file_mode_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            _git(repo, "config", "--unset", "core.fileMode")
            script = repo / "run.sh"
            script.write_text("#!/bin/sh\nexit 0\n")
            script.chmod(0o755)

            diff_text = build_worktree_diff_text(str(repo))

        self.assertIn("new file mode 100755", diff_text)

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

    def test_untracked_file_without_trailing_newline_marks_eof(self) -> None:
        # The synthetic untracked-file diff is rendered into the session view
        # and exposed verbatim through ``build_worktree_diff_text``. Without the
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
