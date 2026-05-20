import subprocess
import tempfile
from pathlib import Path

from django.test import TestCase

from hitch.main.repos import discover_repos, git_common_dir, same_repo_or_worktree


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
