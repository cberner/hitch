import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase

from hitch.main import repos
from hitch.main.repos import (
    default_branch_checkout_commit_hash,
    default_branch_commit_hash,
    discover_repos,
    git_common_dir,
    same_repo_or_worktree,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


class DiscoverReposTests(TestCase):
    def _mk_repo(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".git").mkdir()

    def test_finds_repo_at_depth_one(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            self._mk_repo(home / "proj")
            self.assertEqual(discover_repos(home), [home / "proj"])

    def test_finds_repo_at_depth_two(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            self._mk_repo(home / "code" / "proj")
            self.assertEqual(discover_repos(home), [home / "code" / "proj"])

    def test_does_not_recurse_into_repo(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            self._mk_repo(home / "outer")
            # A nested repo inside another repo must not be reported.
            self._mk_repo(home / "outer" / "vendored")
            self.assertEqual(discover_repos(home), [home / "outer"])

    def test_skips_beyond_max_depth(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            self._mk_repo(home / "a" / "b" / "deep")
            self.assertEqual(discover_repos(home), [])

    def test_returns_sorted_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            self._mk_repo(home / "zeta")
            self._mk_repo(home / "alpha")
            self._mk_repo(home / "code" / "beta")
            result = discover_repos(home)
            self.assertEqual(
                result,
                sorted([home / "alpha", home / "code" / "beta", home / "zeta"]),
            )

    def test_handles_git_file_not_directory(self) -> None:
        # Submodules and worktrees store .git as a file rather than a directory.
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            repo = home / "worktree-style"
            repo.mkdir()
            (repo / ".git").write_text("gitdir: /elsewhere\n")
            self.assertEqual(discover_repos(home), [repo])

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

            self.assertEqual(git_common_dir(repo), git_common_dir(worktree))
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
            self.assertIsNone(default_branch_checkout_commit_hash(repo))
            _git(repo, "checkout", "main")
            self.assertEqual(default_branch_checkout_commit_hash(repo), main_sha)

    def test_default_branch_commit_hash_falls_back_to_local_main(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("hello\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "initial")
            main_sha = _git(repo, "rev-parse", "HEAD")

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

    def test_default_branch_commit_hash_does_not_guess_between_origin_main_and_master(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=stable")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("stable\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "stable")
            master_sha = _git(repo, "rev-parse", "HEAD")
            _git(repo, "update-ref", "refs/remotes/origin/master", master_sha)
            (repo / "README.md").write_text("main\n")
            _git(repo, "commit", "-am", "main")
            main_sha = _git(repo, "rev-parse", "HEAD")
            _git(repo, "update-ref", "refs/remotes/origin/main", main_sha)

            self.assertIsNone(default_branch_commit_hash(repo))
            self.assertIsNone(default_branch_checkout_commit_hash(repo))

    def test_default_branch_checkout_rejects_ambiguous_local_named_branches(
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
            _git(repo, "checkout", "master")

            self.assertIsNone(default_branch_checkout_commit_hash(repo))

    def test_default_branch_checkout_rejects_feature_branch_at_same_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("hello\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "initial")
            _git(repo, "checkout", "-b", "feature")

            self.assertIsNone(default_branch_checkout_commit_hash(repo))

    def test_default_branch_checkout_commit_hash_requires_clean_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("hello\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "initial")
            main_sha = _git(repo, "rev-parse", "HEAD")

            self.assertEqual(default_branch_checkout_commit_hash(repo), main_sha)

            (repo / "README.md").write_text("local change\n")

            self.assertIsNone(default_branch_checkout_commit_hash(repo))

    def test_default_branch_checkout_commit_hash_disables_fsmonitor(self) -> None:
        calls: list[list[str]] = []

        def fake_git_output(_cwd: Path, args: list[str]) -> str | None:
            calls.append(args)
            if args == ["rev-parse", "--show-toplevel"]:
                return "/repo\n"
            if args == ["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"]:
                return "refs/remotes/origin/main\n"
            if args == [
                "rev-parse",
                "--verify",
                "--quiet",
                "refs/remotes/origin/main^{commit}",
            ]:
                return f"{'a' * 40}\n"
            if args == ["symbolic-ref", "--quiet", "--short", "HEAD"]:
                return "main\n"
            if args == ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"]:
                return f"{'a' * 40}\n"
            if args == ["-c", "core.fsmonitor=false", "status", "--porcelain"]:
                return ""
            return None

        with patch.object(repos, "_git_output", side_effect=fake_git_output):
            self.assertEqual(
                default_branch_checkout_commit_hash(Path("/repo")),
                "a" * 40,
            )

        self.assertIn(
            ["-c", "core.fsmonitor=false", "status", "--porcelain"],
            calls,
        )

    def test_default_branch_commit_hash_falls_back_to_head_for_custom_branch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            repo = Path(raw_root) / "repo"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=stable")
            _git(repo, "config", "user.email", "dev@example.com")
            _git(repo, "config", "user.name", "Dev")
            (repo / "README.md").write_text("hello\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "initial")
            head_sha = _git(repo, "rev-parse", "HEAD")

            self.assertEqual(default_branch_commit_hash(repo), head_sha)
            self.assertEqual(default_branch_checkout_commit_hash(repo), head_sha)
