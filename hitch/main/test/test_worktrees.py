import os
import subprocess
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from hitch.main import worktrees
from hitch.main.worktrees import (
    WorktreeCleanupError,
    WorktreeCreationError,
    cleanup_managed_worktree_path,
    cleanup_worktree,
    create_worktree_for_session,
    discover_managed_worktrees,
)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Hitch Tests",
        "GIT_AUTHOR_EMAIL": "hitch@example.com",
        "GIT_COMMITTER_NAME": "Hitch Tests",
        "GIT_COMMITTER_EMAIL": "hitch@example.com",
    }
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    subprocess.run(
        ["git", "init", "--initial-branch=master", str(repo)],
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")


def _init_unborn_repo(repo: Path) -> None:
    subprocess.run(
        ["git", "init", "--initial-branch=master", str(repo)],
        check=True,
        capture_output=True,
    )


class ManagedWorktreeTests(SimpleTestCase):
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

    def test_cleanup_reports_git_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "source"
            managed = root / "managed"
            _init_repo(repo)

            with override_settings(HITCH_WORKTREES_DIR=managed):
                managed_worktree = create_worktree_for_session(str(repo))
                cleanup_worktree(managed_worktree)
                with self.assertRaises(WorktreeCleanupError):
                    cleanup_worktree(managed_worktree)

    def test_git_wrapper_reports_spawn_failure(self) -> None:
        with (
            patch("hitch.main.worktrees.subprocess.run", side_effect=OSError("no git")),
            self.assertRaisesRegex(WorktreeCreationError, "no git"),
        ):
            worktrees._git(Path("/repo"), ["status"])

        with patch("hitch.main.worktrees.subprocess.run", side_effect=OSError("no git")):
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
