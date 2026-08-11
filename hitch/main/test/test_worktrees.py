import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import cast, override
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from hitch.main import worktrees
from hitch.main.test.support import _git, _init_repo
from hitch.main.worktrees import (
    WorktreeCreationError,
    cleanup_managed_worktree_path,
    cleanup_worktree,
    create_worktree_for_session,
    discover_managed_worktrees,
    is_managed_worktree_path,
    snapshot_worktree_to_commit,
)


def _init_unborn_repo(repo: Path) -> None:
    subprocess.run(
        ["git", "init", "--initial-branch=master", str(repo)],
        check=True,
        capture_output=True,
    )


class ManagedWorktreeTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        invalidation_patcher = patch(
            "hitch.main.runtime.disk_cleanup.invalidate_hitch_home_disk_usage"
        )
        self.mock_invalidate_disk_usage = invalidation_patcher.start()
        self.addCleanup(invalidation_patcher.stop)

    def test_snapshot_worktree_to_commit_includes_dirty_and_untracked_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "source"
            _init_repo(repo)
            head = _git(repo, "rev-parse", "HEAD")
            (repo / "README.md").write_text("changed\n")
            (repo / "staged.txt").write_text("staged\n")
            _git(repo, "add", "staged.txt")
            (repo / "untracked.txt").write_text("untracked\n")
            status_before = _git(repo, "status", "--short")

            with patch.dict(
                os.environ,
                {
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": str(Path(raw) / "home"),
                    "XDG_CONFIG_HOME": str(Path(raw) / "xdg"),
                },
                clear=True,
            ):
                snapshot = snapshot_worktree_to_commit(repo)

            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head)
            self.assertEqual(_git(repo, "rev-parse", f"{snapshot}^"), head)
            self.assertEqual(_git(repo, "show", f"{snapshot}:README.md"), "changed")
            self.assertEqual(_git(repo, "show", f"{snapshot}:staged.txt"), "staged")
            self.assertEqual(
                _git(repo, "show", f"{snapshot}:untracked.txt"), "untracked"
            )
            self.assertEqual(_git(repo, "status", "--short"), status_before)

    def test_snapshot_worktree_to_commit_rejects_non_repo_source(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            self.assertRaisesRegex(
                WorktreeCreationError, "source cwd is not a git repository"
            ),
        ):
            snapshot_worktree_to_commit(raw)

    def test_snapshot_worktree_to_commit_handles_unborn_head(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "source"
            _init_unborn_repo(repo)
            (repo / "README.md").write_text("hello\n")

            with patch.dict(
                os.environ,
                {
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": str(Path(raw) / "home"),
                    "XDG_CONFIG_HOME": str(Path(raw) / "xdg"),
                },
                clear=True,
            ):
                snapshot = snapshot_worktree_to_commit(repo)

            self.assertEqual(_git(repo, "show", f"{snapshot}:README.md"), "hello")
            self.assertEqual(
                _git(repo, "rev-list", "--parents", "-n", "1", snapshot), snapshot
            )
            self.assertEqual(_git(repo, "status", "--short"), "?? README.md")

    def test_snapshot_worktree_to_commit_reports_tree_write_failure(self) -> None:
        git_results = iter(["", "", "", None])
        with (
            patch("hitch.main.worktrees._repo_root", return_value=Path("/repo")),
            patch(
                "hitch.main.worktrees._git",
                side_effect=lambda *args, **kwargs: next(git_results),
            ),
            self.assertRaisesRegex(
                WorktreeCreationError, "failed to write worktree snapshot tree"
            ),
        ):
            snapshot_worktree_to_commit("/repo")

    def test_snapshot_worktree_to_commit_reports_commit_failure(self) -> None:
        git_results = iter(["", "", "", "tree-sha\n", None])
        with (
            patch("hitch.main.worktrees._repo_root", return_value=Path("/repo")),
            patch(
                "hitch.main.worktrees._git",
                side_effect=lambda *args, **kwargs: next(git_results),
            ),
            self.assertRaisesRegex(
                WorktreeCreationError, "failed to create worktree snapshot commit"
            ),
        ):
            snapshot_worktree_to_commit("/repo")

    def test_creates_branch_and_worktree_under_settings_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source repo"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                worktree = managed_worktree.path

                self.assertTrue(worktree.is_dir())
                self.assertTrue((worktree / ".git").exists())
                self.assertEqual(worktree.parent.parent, managed)
                self.assertEqual(discover_managed_worktrees(), [worktree])
                self.assertTrue(is_managed_worktree_path(worktree))
                self.assertFalse(is_managed_worktree_path(repo))
                self.assertEqual(managed_worktree.source_repo, repo)

            self.assertEqual(_git(repo, "branch", "--show-current"), "master")
            branch = _git(worktree, "branch", "--show-current")
            self.assertEqual(branch, managed_worktree.branch)
            self.assertRegex(branch, r"^hitch/source-repo/\d{14}-[0-9a-f]{8}$")
            self.assertEqual((worktree / "README.md").read_text(), "hello\n")

    def test_creates_worktree_from_base_ref_with_hooks_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)
            _git(repo, "checkout", "-b", "release")
            (repo / "README.md").write_text("release\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "release")
            _git(repo, "checkout", "master")
            hook = repo / ".git" / "hooks" / "post-checkout"
            hook.write_text("#!/bin/sh\ntouch ../hook-ran\nexit 1\n")
            hook.chmod(0o755)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(
                    str(repo),
                    base_ref="refs/heads/release",
                    disable_hooks=True,
                )

            self.assertEqual(
                (managed_worktree.path / "README.md").read_text(), "release\n"
            )
            self.assertFalse((repo.parent / "hook-ran").exists())
            self.assertEqual(_git(repo, "branch", "--show-current"), "master")

    def test_creates_orphan_worktree_for_unborn_head(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "empty"
            managed = root / "managed"
            _init_unborn_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))

                self.assertTrue(managed_worktree.path.is_dir())
                self.assertTrue((managed_worktree.path / ".git").exists())
                self.assertEqual(
                    _git(managed_worktree.path, "branch", "--show-current"),
                    managed_worktree.branch,
                )
                cleanup_worktree(managed_worktree)

            self.assertFalse(managed_worktree.path.exists())

    def test_sanitizes_repo_name_for_branch_segment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "foo..bar.lock"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))

            self.assertEqual(managed_worktree.path.parent.name, "foo-bar")
            self.assertRegex(
                managed_worktree.branch, r"^hitch/foo-bar/\d{14}-[0-9a-f]{8}$"
            )
            self.assertEqual(
                _git(managed_worktree.path, "branch", "--show-current"),
                managed_worktree.branch,
            )

    def test_discovers_only_direct_managed_worktree_roots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                decoy = managed_worktree.path.parent / "not-a-worktree"
                decoy.mkdir()
                nested_repo = managed_worktree.path / "vendor" / "nested"
                _init_repo(nested_repo)

                self.assertEqual(discover_managed_worktrees(), [managed_worktree.path])

    def test_discovery_helpers_tolerate_unreadable_paths(self) -> None:
        unreadable_root = MagicMock(spec=Path)
        unreadable_root.iterdir.side_effect = OSError("no access")

        self.assertEqual(list(worktrees._child_dirs(cast(Path, unreadable_root))), [])

        unreadable_child = MagicMock(spec=Path)
        unreadable_child.is_dir.side_effect = OSError("no access")
        readable_child = MagicMock(spec=Path)
        readable_child.is_dir.return_value = True
        mixed_root = MagicMock(spec=Path)
        mixed_root.iterdir.return_value = [unreadable_child, readable_child]

        self.assertEqual(
            list(worktrees._child_dirs(cast(Path, mixed_root))), [readable_child]
        )

    def test_resolved_path_falls_back_when_resolution_fails(self) -> None:
        path = MagicMock(spec=Path)
        path.resolve.side_effect = OSError("no access")

        self.assertEqual(worktrees._resolved_path(cast(Path, path)), path)

    def test_reports_managed_dir_creation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with (
                override_settings(HITCH_WORKTREES_DIR=managed),
                patch("hitch.main.worktrees.Path.mkdir", side_effect=OSError("nope")),
                self.assertRaisesRegex(WorktreeCreationError, "nope"),
            ):
                create_worktree_for_session(str(repo))

    def test_cleanup_removes_worktree_and_branch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                cleanup_worktree(managed_worktree)

            self.assertFalse(managed_worktree.path.exists())
            self.assertEqual(
                _git(repo, "branch", "--list", managed_worktree.branch), ""
            )
            self.mock_invalidate_disk_usage.assert_called_once_with()

    def test_cleanup_managed_worktree_path_removes_worktree_and_branch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                cleaned = cleanup_managed_worktree_path(str(managed_worktree.path))

            self.assertTrue(cleaned)
            self.assertFalse(managed_worktree.path.exists())
            self.assertEqual(
                _git(repo, "branch", "--list", managed_worktree.branch), ""
            )
            self.mock_invalidate_disk_usage.assert_called_once_with()

    def test_cleanup_managed_worktree_path_preserves_checked_out_user_branch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)
            _git(repo, "branch", "release")

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                _git(managed_worktree.path, "checkout", "release")
                cleaned = cleanup_managed_worktree_path(str(managed_worktree.path))

            self.assertTrue(cleaned)
            self.assertFalse(managed_worktree.path.exists())
            self.assertEqual(
                _git(repo, "branch", "--list", managed_worktree.branch), ""
            )
            self.assertIn("release", _git(repo, "branch", "--list", "release"))

    def test_cleanup_managed_worktree_path_ignores_unmanaged_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            unmanaged = root / "unmanaged"
            managed = root / "managed"
            _init_repo(unmanaged)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                cleaned = cleanup_managed_worktree_path(str(unmanaged))

            self.assertFalse(cleaned)
            self.assertTrue(unmanaged.exists())
            self.mock_invalidate_disk_usage.assert_not_called()

    def test_cleanup_is_idempotent_after_worktree_already_removed(self) -> None:
        # A failed ``git worktree remove`` (here: the path is already gone)
        # falls back to reaping the directory and pruning the admin entry, so
        # repeated cleanup converges instead of raising forever.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                cleanup_worktree(managed_worktree)
                cleanup_worktree(managed_worktree)
            self.assertFalse(managed_worktree.path.exists())
            self.assertEqual(
                _git(repo, "branch", "--list", managed_worktree.branch), ""
            )
            self.mock_invalidate_disk_usage.assert_called_once_with()

    def test_cleanup_does_not_invalidate_when_directory_remains(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            worktree_path = root / "managed" / "repo" / "worktree"
            worktree_path.mkdir(parents=True)
            managed_worktree = worktrees.ManagedWorktree(
                path=worktree_path,
                branch="hitch/repo/worktree",
                source_repo=root / "source",
            )
            with (
                patch.object(
                    worktrees,
                    "_git",
                    side_effect=[worktrees.WorktreeCleanupError("remove failed"), None],
                ),
                patch("hitch.main.worktrees.shutil.rmtree"),
                patch.object(worktrees, "_branch_exists", return_value=False),
            ):
                cleanup_worktree(managed_worktree)

            self.assertTrue(worktree_path.exists())
            self.mock_invalidate_disk_usage.assert_not_called()

    def test_cleanup_managed_worktree_path_reaps_directory_without_gitlink(
        self,
    ) -> None:
        # A worktree half-deleted out-of-band loses its .git link, so the
        # source repo can't be located; cleanup must still free the directory
        # instead of returning False forever.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                (managed_worktree.path / ".git").unlink()

                self.assertTrue(
                    cleanup_managed_worktree_path(str(managed_worktree.path))
                )
            self.assertFalse(managed_worktree.path.exists())
            self.mock_invalidate_disk_usage.assert_called_once_with()

    def test_cleanup_managed_worktree_path_reaps_when_source_repo_gone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                shutil.rmtree(repo)

                self.assertTrue(
                    cleanup_managed_worktree_path(str(managed_worktree.path))
                )
            self.assertFalse(managed_worktree.path.exists())
            self.mock_invalidate_disk_usage.assert_called_once_with()

    def test_cleanup_managed_worktree_path_never_reaps_non_leaf_directories(
        self,
    ) -> None:
        # The gitlink-missing fallback must only delete a branch-addressable
        # <base>/<slug>/<suffix> leaf: a stale cwd naming the managed base or
        # a slug directory must not wipe the live worktrees beneath it.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                slug_dir = managed_worktree.path.parent

                self.assertFalse(cleanup_managed_worktree_path(str(managed)))
                self.assertFalse(cleanup_managed_worktree_path(str(slug_dir)))
            self.assertTrue(managed_worktree.path.is_dir())
            self.mock_invalidate_disk_usage.assert_not_called()

    def test_create_worktree_cleans_up_after_add_failure(self) -> None:
        # ``worktree add -b`` creates the branch and admin entry before the
        # checkout; a failure mid-add (e.g. a timeout SIGKILLing git) must not
        # strand them, or every retry leaks another branch.
        def _fail_after_partial_add(
            repo: Path, args: list[str], *, disable_hooks: bool
        ) -> None:
            branch = args[args.index("-b") + 1]
            path = Path(args[args.index("-b") + 2])
            _git(repo, "branch", branch)
            path.mkdir(parents=True, exist_ok=True)
            (path / "partial.txt").write_text("partial checkout\n")
            raise WorktreeCreationError("simulated timeout")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with (
                override_settings(HITCH_WORKTREES_DIR=managed),
                patch.object(worktrees, "_add_worktree", _fail_after_partial_add),
                self.assertRaisesRegex(WorktreeCreationError, "simulated timeout"),
            ):
                create_worktree_for_session(str(repo))

            self.assertEqual(_git(repo, "branch", "--list", "hitch/*"), "")
            leftovers = [
                child
                for repo_dir in (managed.iterdir() if managed.is_dir() else [])
                for child in repo_dir.iterdir()
            ]
            self.assertEqual(leftovers, [])

    def test_git_wrapper_reports_spawn_failure(self) -> None:
        with (
            patch("hitch.main.git_support.subprocess.run", side_effect=OSError("no git")),
            self.assertRaisesRegex(WorktreeCreationError, "no git"),
        ):
            worktrees._git(Path("/repo"), ["status"])

        with patch("hitch.main.git_support.subprocess.run", side_effect=OSError("no git")):
            self.assertIsNone(
                worktrees._git(Path("/repo"), ["status"], raise_on_error=False)
            )

    def test_discovers_no_worktrees_when_base_dir_is_missing(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(HITCH_WORKTREES_DIR=Path(raw) / "managed"),
        ):
            self.assertEqual(discover_managed_worktrees(), [])

    def test_rejects_non_repo_source(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(HITCH_WORKTREES_DIR=Path(raw) / "managed"),
            self.assertRaisesRegex(
                WorktreeCreationError, "source cwd is not a git repository"
            ),
        ):
            create_worktree_for_session(raw)
