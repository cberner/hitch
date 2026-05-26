import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from hitch.main.diffs import build_worktree_diff_text
from hitch.main.local_merges import (
    LocalBranchMergeError,
    LocalBranchMergeResult,
    _auto_merge_source_base_ref,
    _source_worktree_tree,
    build_auto_merge_review_patch,
    local_branch_names,
    merge_worktree_diff_to_branch,
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
        ["git", "init", "--initial-branch=main", str(repo)],
        check=True,
        capture_output=True,
    )
    _git(repo, "config", "user.name", "Hitch Tests")
    _git(repo, "config", "user.email", "hitch@example.com")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")


def _merge_reviewed_patch(source_cwd: Path, branch: str) -> LocalBranchMergeResult:
    review = build_auto_merge_review_patch(source_cwd, branch)
    return merge_worktree_diff_to_branch(
        source_cwd, branch, review.patch, review.target_sha
    )


def _add_submodule(repo: Path, submodule_src: Path) -> None:
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule_src),
        "vendor/lib",
    )
    _git(repo, "commit", "-m", "add submodule")


class LocalMergeTests(SimpleTestCase):
    def test_local_branch_names_returns_local_branches(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            _init_repo(repo)
            _git(repo, "branch", "release")

            self.assertEqual(local_branch_names(repo), ["main", "release"])

    def test_merge_worktree_diff_to_checked_out_branch_commits_patch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")

            result = _merge_reviewed_patch(session, "main")

            self.assertTrue(result.changed)
            self.assertEqual(result.branch, "main")
            self.assertEqual(_git(repo, "rev-parse", "main"), result.commit_sha)
            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\napproved"
            )

    def test_merge_worktree_diff_rejects_dirty_source_as_target_branch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            _init_repo(repo)
            (repo / "README.md").write_text("hello\napproved\n")

            with self.assertRaisesRegex(
                LocalBranchMergeError,
                "target branch is checked out in the source worktree",
            ):
                _merge_reviewed_patch(repo, "main")

            self.assertEqual(_git(repo, "status", "--porcelain"), "M README.md")

    def test_merge_worktree_diff_carries_untracked_files_and_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").unlink()
            (session / "notes.txt").write_text("new session notes\n")

            result = _merge_reviewed_patch(session, "main")

            self.assertTrue(result.changed)
            with self.assertRaises(subprocess.CalledProcessError):
                _git(repo, "show", "main:README.md")
            self.assertEqual(
                _git(repo, "show", "main:notes.txt"), "new session notes"
            )

    def test_merge_worktree_diff_allows_file_to_directory_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            (repo / "docs").write_text("old docs\n")
            _git(repo, "add", "docs")
            _git(repo, "commit", "-m", "add docs file")
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "docs").unlink()
            (session / "docs").mkdir()
            (session / "docs" / "intro.txt").write_text("new docs\n")

            result = _merge_reviewed_patch(session, "main")

            self.assertTrue(result.changed)
            self.assertEqual(_git(repo, "cat-file", "-t", "main:docs"), "tree")
            self.assertEqual(
                _git(repo, "show", "main:docs/intro.txt"), "new docs"
            )

    def test_merge_worktree_diff_disables_target_worktree_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")
            post_checkout = repo / ".git" / "hooks" / "post-checkout"
            post_checkout.write_text("#!/bin/sh\ntouch ../hook-ran\nexit 1\n")
            post_checkout.chmod(0o755)
            post_merge = repo / ".git" / "hooks" / "post-merge"
            post_merge.write_text("#!/bin/sh\ntouch ../hook-ran\nexit 1\n")
            post_merge.chmod(0o755)

            result = _merge_reviewed_patch(session, "main")

            self.assertTrue(result.changed)
            self.assertFalse((repo.parent / "hook-ran").exists())
            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\napproved"
            )

    def test_merge_worktree_diff_updates_unchecked_out_branch_ref(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            initial_main = _git(repo, "rev-parse", "main")
            _git(repo, "branch", "release")
            _git(repo, "worktree", "add", "-b", "session", str(session), "release")
            (session / "README.md").write_text("hello\nrelease approved\n")
            reference_hook = repo / ".git" / "hooks" / "reference-transaction"
            reference_hook.write_text("#!/bin/sh\ntouch ../hook-ran\nexit 1\n")
            reference_hook.chmod(0o755)

            result = _merge_reviewed_patch(session, "release")

            self.assertTrue(result.changed)
            self.assertEqual(result.target_worktree, "")
            self.assertEqual(_git(repo, "rev-parse", "release"), result.commit_sha)
            self.assertEqual(_git(repo, "rev-parse", "main"), initial_main)
            self.assertEqual(
                _git(repo, "show", "release:README.md"), "hello\nrelease approved"
            )
            self.assertEqual(_git(repo, "show", "main:README.md"), "hello")
            self.assertFalse((repo.parent / "hook-ran").exists())

    def test_auto_merge_preserves_sparse_checkout_entries(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            (repo / "src").mkdir()
            (repo / "src" / "keep.txt").write_text("keep\n")
            (repo / "docs").mkdir()
            (repo / "docs" / "excluded.txt").write_text("docs\n")
            _git(repo, "add", "src/keep.txt", "docs/excluded.txt")
            _git(repo, "commit", "-m", "add sparse paths")
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            _git(session, "sparse-checkout", "init", "--cone")
            _git(session, "sparse-checkout", "set", "src")
            self.assertFalse((session / "docs" / "excluded.txt").exists())
            (session / "src" / "keep.txt").write_text("keep changed\n")

            review = build_auto_merge_review_patch(session, "main")
            result = merge_worktree_diff_to_branch(
                session, "main", review.patch, review.target_sha
            )

            self.assertTrue(result.changed)
            self.assertNotIn("docs/excluded.txt", review.patch)
            self.assertEqual(_git(repo, "show", "main:src/keep.txt"), "keep changed")
            self.assertEqual(_git(repo, "show", "main:docs/excluded.txt"), "docs")

    def test_merge_rejects_dirty_checked_out_target_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            release = root / "release"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "branch", "release")
            initial_release = _git(repo, "rev-parse", "release")
            _git(repo, "worktree", "add", str(release), "release")
            _git(repo, "worktree", "add", "-b", "session", str(session), "release")
            (session / "README.md").write_text("hello\napproved\n")
            review = build_auto_merge_review_patch(session, "release")
            (release / "README.md").write_text("dirty local edit\n")

            with self.assertRaisesRegex(LocalBranchMergeError, "uncommitted changes"):
                merge_worktree_diff_to_branch(
                    session,
                    "release",
                    review.patch,
                    review.target_sha,
                )

            self.assertEqual(_git(repo, "rev-parse", "release"), initial_release)
            self.assertEqual((release / "README.md").read_text(), "dirty local edit\n")

    def test_auto_merge_patch_is_relative_to_diverged_target_branch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            initial_main = _git(repo, "rev-parse", "main")
            _git(repo, "checkout", "-b", "release")
            (repo / "README.md").write_text("release base\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "release base")
            _git(repo, "checkout", "main")
            _git(repo, "worktree", "add", "-b", "session", str(session), "release")
            (session / "README.md").write_text("release base\napproved\n")

            result = _merge_reviewed_patch(session, "release")

            self.assertTrue(result.changed)
            self.assertEqual(_git(repo, "rev-parse", "main"), initial_main)
            self.assertEqual(
                _git(repo, "show", "release:README.md"), "release base\napproved"
            )
            self.assertEqual(_git(repo, "show", "main:README.md"), "hello")

    def test_follow_up_review_does_not_replay_merged_patch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")

            first = _merge_reviewed_patch(session, "main")
            second_review = build_auto_merge_review_patch(session, "main")
            second = merge_worktree_diff_to_branch(
                session,
                "main",
                second_review.patch,
                second_review.target_sha,
            )

            self.assertTrue(first.changed)
            self.assertEqual(second_review.patch, "")
            self.assertFalse(second.changed)
            self.assertEqual(second.commit_sha, first.commit_sha)

    def test_follow_up_review_can_revert_previously_merged_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")
            first = _merge_reviewed_patch(session, "main")
            (session / "README.md").write_text("hello\n")

            review = build_auto_merge_review_patch(session, "main")
            second = merge_worktree_diff_to_branch(
                session,
                "main",
                review.patch,
                review.target_sha,
            )

            self.assertTrue(first.changed)
            self.assertTrue(second.changed)
            self.assertIn("-approved", review.patch)
            self.assertEqual(_git(repo, "show", "main:README.md"), "hello")

    def test_follow_up_review_rebuilds_source_base_after_target_rewind(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "branch", "release")
            initial_release = _git(repo, "rev-parse", "release")
            _git(repo, "worktree", "add", "-b", "session", str(session), "release")
            (session / "README.md").write_text("hello\napproved\n")
            first = _merge_reviewed_patch(session, "release")
            _git(
                repo,
                "update-ref",
                "refs/heads/release",
                initial_release,
                first.commit_sha,
            )

            review = build_auto_merge_review_patch(session, "release")
            second = merge_worktree_diff_to_branch(
                session,
                "release",
                review.patch,
                review.target_sha,
            )

            self.assertTrue(first.changed)
            self.assertTrue(second.changed)
            self.assertIn("+approved", review.patch)
            self.assertEqual(
                _git(repo, "show", "release:README.md"), "hello\napproved"
            )

    def test_follow_up_review_uses_legacy_source_base_without_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")
            first = _merge_reviewed_patch(session, "main")
            source_tree = _source_worktree_tree(session)
            legacy_source_base = _git(
                session,
                "commit-tree",
                source_tree,
                "-m",
                "legacy source base without parent",
            )
            _git(
                session,
                "update-ref",
                _auto_merge_source_base_ref(session, "refs/heads/main"),
                legacy_source_base,
            )
            (session / "notes.txt").write_text("follow-up\n")

            review = build_auto_merge_review_patch(session, "main")
            result = merge_worktree_diff_to_branch(
                session,
                "main",
                review.patch,
                review.target_sha,
            )

            self.assertTrue(first.changed)
            self.assertTrue(result.changed)
            self.assertNotIn("README.md", review.patch)
            self.assertIn("notes.txt", review.patch)
            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\napproved"
            )
            self.assertEqual(_git(repo, "show", "main:notes.txt"), "follow-up")

    def test_auto_merge_patch_keeps_target_commits_added_before_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (repo / "later.txt").write_text("target-only\n")
            _git(repo, "add", "later.txt")
            _git(repo, "commit", "-m", "advance target")
            (session / "README.md").write_text("hello\napproved\n")

            review = build_auto_merge_review_patch(session, "main")
            result = merge_worktree_diff_to_branch(
                session,
                "main",
                review.patch,
                review.target_sha,
            )

            self.assertNotIn("later.txt", review.patch)
            self.assertTrue(result.changed)
            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\napproved"
            )
            self.assertEqual(_git(repo, "show", "main:later.txt"), "target-only")

    def test_follow_up_patch_after_target_advanced_contains_only_new_delta(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (repo / "later.txt").write_text("target-only\n")
            _git(repo, "add", "later.txt")
            _git(repo, "commit", "-m", "advance target")
            (session / "README.md").write_text("hello\napproved\n")
            first = _merge_reviewed_patch(session, "main")
            (session / "notes.txt").write_text("new follow-up\n")

            review = build_auto_merge_review_patch(session, "main")
            second = merge_worktree_diff_to_branch(
                session,
                "main",
                review.patch,
                review.target_sha,
            )

            self.assertTrue(first.changed)
            self.assertTrue(second.changed)
            self.assertNotIn("README.md", review.patch)
            self.assertIn("notes.txt", review.patch)
            self.assertEqual(_git(repo, "show", "main:later.txt"), "target-only")
            self.assertEqual(
                _git(repo, "show", "main:notes.txt"), "new follow-up"
            )

    def test_merge_rejects_target_branch_moved_after_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")
            review = build_auto_merge_review_patch(session, "main")
            (repo / "README.md").write_text("hello\ntarget moved\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "target moved")

            with self.assertRaisesRegex(LocalBranchMergeError, "target branch changed"):
                merge_worktree_diff_to_branch(
                    session,
                    "main",
                    review.patch,
                    review.target_sha,
                )

            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\ntarget moved"
            )

    def test_merge_worktree_diff_rejects_incomplete_reviewed_diff(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "asset.bin").write_bytes(b"\0not reviewable")
            reviewed_patch = build_worktree_diff_text(str(session))

            with self.assertRaisesRegex(
                LocalBranchMergeError, "reviewed diff is incomplete"
            ):
                merge_worktree_diff_to_branch(
                    session,
                    "main",
                    reviewed_patch,
                    _git(repo, "rev-parse", "main"),
                )

            with self.assertRaises(subprocess.CalledProcessError):
                _git(repo, "show", "main:asset.bin")

    def test_reviewed_patch_allows_marker_text_in_content(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            marker_text = (
                "hello\n"
                "Binary files are not always binary\n"
                "Binary file not shown\n"
                "File preview truncated\n"
                "Symlink not shown\n"
                "4 untracked files omitted from diff preview\n"
                "GIT binary patch\n"
                "new file mode 120000\n"
                "Subproject commit docs\n"
            )
            (session / "README.md").write_text(marker_text)

            result = _merge_reviewed_patch(session, "main")

            self.assertTrue(result.changed)
            self.assertEqual(_git(repo, "show", "main:README.md"), marker_text.strip())

    def test_auto_merge_review_patch_rejects_binary_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "asset.bin").write_bytes(b"\0not reviewable")

            with self.assertRaisesRegex(
                LocalBranchMergeError, "binary or symlink changes"
            ):
                build_auto_merge_review_patch(session, "main")

    def test_auto_merge_review_patch_rejects_modified_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            (repo / "link").symlink_to("old-target")
            _git(repo, "add", "link")
            _git(repo, "commit", "-m", "add symlink")
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "link").unlink()
            (session / "link").symlink_to("new-target")

            with self.assertRaisesRegex(
                LocalBranchMergeError, "binary or symlink changes"
            ):
                build_auto_merge_review_patch(session, "main")

    def test_auto_merge_review_patch_preserves_unchanged_symlink_to_directory(
        self,
    ) -> None:
        # A tracked symlink whose target is a directory must round-trip
        # through the source-worktree tree builder unchanged. Otherwise
        # ``_source_worktree_tree`` silently drops it, the resulting
        # ``base_sha vs source_tree_sha`` diff shows the symlink as deleted,
        # and ``_validate_reviewable_patch`` rejects the whole patch even
        # though the user only touched an unrelated regular file.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            (repo / "shared").mkdir()
            (repo / "shared" / "data.txt").write_text("payload\n")
            _git(repo, "add", "shared/data.txt")
            (repo / "vendor").symlink_to("shared")
            _git(repo, "add", "vendor")
            _git(repo, "commit", "-m", "add symlink to directory")
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")

            result = _merge_reviewed_patch(session, "main")

            self.assertTrue(result.changed)
            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\napproved"
            )
            # The symlink must still be a symlink on the merged branch, not
            # a deletion or a regular file.
            self.assertIn("120000", _git(repo, "ls-tree", "main", "vendor"))

    def test_auto_merge_review_patch_rejects_oversized_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "large.txt").write_text("x\n" * 300_000)

            with self.assertRaisesRegex(LocalBranchMergeError, "too large to review"):
                build_auto_merge_review_patch(session, "main")

    def test_auto_merge_review_patch_does_not_run_clean_filters(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            (repo / "filtered.txt").write_text("base\n")
            _git(repo, "add", "filtered.txt")
            _git(repo, "commit", "-m", "filtered file")
            (repo / ".gitattributes").write_text("*.txt filter=explode\n")
            _git(repo, "add", ".gitattributes")
            _git(repo, "commit", "-m", "attributes")
            _git(
                repo,
                "config",
                "filter.explode.clean",
                "touch ../filter-ran && exit 1",
            )
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "filtered.txt").write_text("base\napproved\n")

            review = build_auto_merge_review_patch(session, "main")

            self.assertIn("+approved", review.patch)
            self.assertFalse((repo.parent / "filter-ran").exists())

    def test_auto_merge_review_patch_preserves_unchanged_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            submodule_src = root / "submodule-src"
            session = root / "session"
            _init_repo(repo)
            _init_repo(submodule_src)
            _add_submodule(repo, submodule_src)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            (session / "README.md").write_text("hello\napproved\n")

            result = _merge_reviewed_patch(session, "main")

            self.assertTrue(result.changed)
            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\napproved"
            )
            self.assertIn("160000", _git(repo, "ls-tree", "main", "vendor/lib"))

    def test_auto_merge_review_patch_rejects_removed_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            submodule_src = root / "submodule-src"
            session = root / "session"
            _init_repo(repo)
            _init_repo(submodule_src)
            _add_submodule(repo, submodule_src)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            shutil.rmtree(session / "vendor" / "lib")

            with self.assertRaisesRegex(LocalBranchMergeError, "submodule changes"):
                build_auto_merge_review_patch(session, "main")

    def test_auto_merge_review_patch_preserves_sparse_submodule_entry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            submodule_src = root / "submodule-src"
            session = root / "session"
            _init_repo(repo)
            _init_repo(submodule_src)
            _add_submodule(repo, submodule_src)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            _git(session, "update-index", "--skip-worktree", "vendor/lib")
            shutil.rmtree(session / "vendor" / "lib")
            (session / "README.md").write_text("hello\napproved\n")

            review = build_auto_merge_review_patch(session, "main")
            result = merge_worktree_diff_to_branch(
                session,
                "main",
                review.patch,
                review.target_sha,
            )

            self.assertTrue(result.changed)
            self.assertNotIn("vendor/lib", review.patch)
            self.assertEqual(
                _git(repo, "show", "main:README.md"), "hello\napproved"
            )

    def test_auto_merge_review_patch_rejects_file_replacing_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            submodule_src = root / "submodule-src"
            session = root / "session"
            _init_repo(repo)
            _init_repo(submodule_src)
            _add_submodule(repo, submodule_src)
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            shutil.rmtree(session / "vendor" / "lib")
            (session / "vendor" / "lib").write_text("not a submodule\n")

            with self.assertRaisesRegex(LocalBranchMergeError, "submodule changes"):
                build_auto_merge_review_patch(session, "main")

    def test_auto_merge_review_patch_rejects_submodule_gitlink_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            submodule_src = root / "submodule-src"
            session = root / "session"
            _init_repo(repo)
            _init_repo(submodule_src)
            _add_submodule(repo, submodule_src)
            (submodule_src / "README.md").write_text("hello from submodule v2\n")
            _git(submodule_src, "add", "README.md")
            _git(submodule_src, "commit", "-m", "submodule v2")
            new_sha = _git(submodule_src, "rev-parse", "HEAD")
            _git(repo, "worktree", "add", "-b", "session", str(session), "HEAD")
            _git(
                session,
                "update-index",
                "--cacheinfo",
                f"160000,{new_sha},vendor/lib",
            )

            with self.assertRaisesRegex(LocalBranchMergeError, "submodule changes"):
                build_auto_merge_review_patch(session, "main")

    def test_auto_merge_review_patch_rejects_unresolved_merge_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            session = root / "session"
            _init_repo(repo)
            # ``other`` carries a conflicting edit so a later merge into the
            # session worktree leaves README.md unmerged in the index.
            _git(repo, "checkout", "-b", "other")
            (repo / "README.md").write_text("other side\n")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "other side")
            _git(repo, "checkout", "main")
            _git(repo, "worktree", "add", "-b", "session", str(session), "main")
            (session / "README.md").write_text("session side\n")
            _git(session, "add", "README.md")
            _git(session, "commit", "-m", "session side")
            merge = subprocess.run(
                ["git", "-C", str(session), "merge", "other"],
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "Hitch Tests",
                    "GIT_AUTHOR_EMAIL": "hitch@example.com",
                    "GIT_COMMITTER_NAME": "Hitch Tests",
                    "GIT_COMMITTER_EMAIL": "hitch@example.com",
                },
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(merge.returncode, 0)
            self.assertIn("<<<<<<<", (session / "README.md").read_text())
            staged = subprocess.run(
                ["git", "-C", str(session), "ls-files", "-s", "--", "README.md"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(len(staged.stdout.splitlines()), 3)

            with self.assertRaisesRegex(
                LocalBranchMergeError, "unresolved merge conflict"
            ):
                build_auto_merge_review_patch(session, "main")

            # main must not pick up the conflict markers via the bug path.
            self.assertEqual(_git(repo, "show", "main:README.md"), "hello")
