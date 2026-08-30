import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase

from hitch.main import repos
from hitch.main.git_support import GitCommandError
from hitch.main.repos import (
    AutoPullError,
    default_branch_commit_hash,
    discover_repos,
    git_common_dir,
    pull_default_branch_from_origin,
    repo_root,
    same_repo_or_worktree,
)
from hitch.main.test.support import _git, _init_repo


class HermeticGitEnvTests(TestCase):
    def test_repo_lookups_ignore_inherited_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            _init_repo(repo)
            with patch.dict(os.environ, {"GIT_DIR": str(Path(raw) / "nowhere")}):
                self.assertEqual(repo_root(repo), repo.resolve())


class DiscoverReposTests(TestCase):
    def _mk_repo(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".git").mkdir()

    def test_finds_repo_at_depth_one(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            self._mk_repo(home / "proj")
            self.assertEqual(discover_repos(home), [home / "proj"])

    def test_returns_empty_when_home_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            missing = Path(raw_home) / "does-not-exist"
            self.assertEqual(discover_repos(missing), [])

    def test_git_common_dir_matches_linked_worktree_to_source_repo(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repo = root / "repo"
            worktree = root / "feature-worktree"
            repo.mkdir()
            _git(repo, "init")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("hello\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "worktree", "add", "-b", "feature", str(worktree), "HEAD")
            (repo / "subdir").mkdir()
            (worktree / "subdir").mkdir()

            self.assertEqual(git_common_dir(repo), git_common_dir(worktree))
            self.assertEqual(repo_root(repo / "subdir"), repo)
            self.assertEqual(repo_root(worktree / "subdir"), worktree)
            self.assertIsNone(repo_root(root / "missing"))
            self.assertTrue(same_repo_or_worktree(worktree, repo, str(git_common_dir(repo))))
            self.assertTrue(same_repo_or_worktree(repo, repo, str(git_common_dir(repo))))

    def test_default_branch_commit_hash_prefers_origin_head(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=master")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("master\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "master")
            _git(repo, "checkout", "-b", "main")
            (repo / "README.md").write_text("main\n")
            _git(repo, "commit", "-am", "main")
            main_sha = _git(repo, "rev-parse", "HEAD")
            _git(repo, "update-ref", "refs/remotes/origin/main", main_sha)
            _git(
                repo,
                "symbolic-ref",
                "refs/remotes/origin/HEAD",
                "refs/remotes/origin/main",
            )
            _git(repo, "checkout", "master")

            self.assertEqual(default_branch_commit_hash(repo), main_sha)

    def test_default_branch_commit_hash_does_not_guess_between_main_and_master(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=master")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("master\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "master")
            _git(repo, "checkout", "-b", "main")
            (repo / "README.md").write_text("main\n")
            _git(repo, "commit", "-am", "main")

            self.assertIsNone(default_branch_commit_hash(repo))

    def test_pull_default_branch_from_origin_rejects_non_default_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            origin = root / "origin.git"
            repo = root / "repo"
            _init_repo(source, initial_branch="main", configure_user=True)
            _git(source, "clone", "--bare", str(source), str(origin))
            _git(source, "clone", str(origin), str(repo))
            _git(repo, "checkout", "-b", "feature")

            with self.assertRaisesRegex(AutoPullError, "not default branch main"):
                pull_default_branch_from_origin(repo)

    def test_pull_default_branch_from_origin_requires_origin_head(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            origin = root / "origin.git"
            repo = root / "repo"
            _init_repo(source, initial_branch="main", configure_user=True)
            _git(source, "checkout", "-b", "feature")
            (source / "README.md").write_text("feature\n")
            _git(source, "commit", "-am", "feature")
            _git(source, "checkout", "main")
            _git(source, "clone", "--bare", str(source), str(origin))
            _git(source, "clone", str(origin), str(repo))
            _git(repo, "checkout", "-b", "feature", "origin/feature")
            _git(repo, "branch", "-D", "main")
            _git(repo, "update-ref", "-d", "refs/remotes/origin/HEAD")

            with self.assertRaisesRegex(
                AutoPullError, "project default branch is unavailable"
            ):
                pull_default_branch_from_origin(repo)

    def test_pull_default_branch_from_origin_scopes_fetch_options(self) -> None:
        with (
            patch.object(repos, "_repo_root", return_value=Path("/repo")),
            patch.object(repos, "_origin_default_branch_name", return_value="main"),
            patch.object(repos, "_current_branch_name", return_value="main"),
            patch.object(repos, "_worktree_is_clean", return_value=True),
            patch.object(
                repos,
                "_commit_hash_for_ref",
                side_effect=["abc123", "def456"],
            ),
            patch.object(
                repos,
                "_run_git_for_auto_pull",
                return_value=SimpleNamespace(returncode=0),
            ) as mock_run,
        ):
            pull_default_branch_from_origin("/repo")

        mock_run.assert_called_once()
        self.assertEqual(
            mock_run.call_args.args[1],
            [
                "pull",
                "--ff-only",
                "--no-recurse-submodules",
                "--no-tags",
                "origin",
                "main",
            ],
        )

    def test_pull_default_branch_from_origin_reports_up_to_date_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            origin = root / "origin.git"
            repo = root / "repo"
            _init_repo(source, initial_branch="main", configure_user=True)
            _git(source, "clone", "--bare", str(source), str(origin))
            _git(source, "clone", str(origin), str(repo))
            head_sha = _git(repo, "rev-parse", "HEAD")

            result = pull_default_branch_from_origin(repo)

            self.assertEqual(result.branch, "main")
            self.assertFalse(result.changed)
            self.assertEqual(result.before_sha, head_sha)
            self.assertEqual(result.after_sha, head_sha)

    def test_pull_default_branch_from_origin_allows_ahead_only_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            origin = root / "origin.git"
            repo = root / "repo"
            _init_repo(source, initial_branch="main", configure_user=True)
            _git(source, "clone", "--bare", str(source), str(origin))
            _git(source, "clone", str(origin), str(repo))
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("local\n")
            _git(repo, "commit", "-am", "local")
            head_sha = _git(repo, "rev-parse", "HEAD")

            result = pull_default_branch_from_origin(repo)

            self.assertEqual(result.branch, "main")
            self.assertFalse(result.changed)
            self.assertEqual(result.before_sha, head_sha)
            self.assertEqual(result.after_sha, head_sha)
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), head_sha)

    def test_pull_default_branch_from_origin_rejects_dirty_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            origin = root / "origin.git"
            repo = root / "repo"
            _init_repo(source, initial_branch="main", configure_user=True)
            _git(source, "clone", "--bare", str(source), str(origin))
            _git(source, "clone", str(origin), str(repo))
            (repo / "README.md").write_text("local edit\n")

            with self.assertRaisesRegex(AutoPullError, "uncommitted changes"):
                pull_default_branch_from_origin(repo)

    def test_pull_default_branch_from_origin_reports_pull_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            origin = root / "origin.git"
            repo = root / "repo"
            writer = root / "writer"
            _init_repo(source, initial_branch="main", configure_user=True)
            _git(source, "clone", "--bare", str(source), str(origin))
            _git(source, "clone", str(origin), str(repo))
            _git(source, "clone", str(origin), str(writer))
            (writer / "README.md").write_text("remote\n")
            _git(writer, "commit", "-am", "remote")
            _git(writer, "push", "origin", "main")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("local\n")
            _git(repo, "commit", "-am", "local")

            with self.assertRaises(AutoPullError):
                pull_default_branch_from_origin(repo)

    def test_auto_pull_wraps_git_spawn_failures(self) -> None:
        with (
            patch.object(
                repos,
                "run_git",
                side_effect=GitCommandError("git missing"),
            ),
            self.assertRaisesRegex(AutoPullError, "git missing"),
        ):
            repos._run_git_for_auto_pull(Path("/repo"), ["pull"])

    def test_empty_git_failure_message_is_generic(self) -> None:
        self.assertEqual(
            repos._git_failure_message(SimpleNamespace(stderr=b"", stdout=b"")),
            "git pull failed",
        )

    def test_pull_default_branch_from_origin_rejects_missing_repo(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw_root,
            self.assertRaisesRegex(AutoPullError, "repository is unavailable"),
        ):
            missing = Path(raw_root) / "missing"
            pull_default_branch_from_origin(missing)

    def test_git_output_returns_none_on_spawn_failure(self) -> None:
        with patch(
            "hitch.main.git_support.subprocess.run", side_effect=OSError("no git")
        ):
            self.assertIsNone(repos._git_output(Path("/repo"), ["status"]))
