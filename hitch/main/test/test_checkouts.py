"""Checkout identity, lease ordering, and retention contracts."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from hitch.main import checkouts, worktrees
from hitch.main.models import CodexInstance, ProposedSession, SessionMetadata, SessionPullRequest
from hitch.main.sessions import checkout_protection


class CheckoutLeaseTests(SimpleTestCase):
    def test_aliases_share_identity_and_nested_leases(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(HITCH_WORKTREES_DIR=Path(raw) / "managed"),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            root = Path(raw)
            checkout = root / "managed" / "repo" / "checkout"
            checkout.mkdir(parents=True)
            descendant = checkout / "src"
            descendant.mkdir()
            alias = root / "alias"
            alias.symlink_to(checkout, target_is_directory=True)
            self.assertEqual(checkouts.identity(alias), checkouts.identity(checkout))
            self.assertEqual(worktrees._managed_branch_for_path(alias), "hitch/repo/checkout")
            self.assertIsNone(checkouts.identity(root))
            self.assertIsNone(checkouts.identity(""))
            self.assertIsNotNone(checkouts.identity(checkout / "missing"))
            self.assertEqual(checkouts.managed_key(descendant), str(checkout))
            self.assertFalse(worktrees.cleanup_managed_worktree_path(str(descendant)))
            with self.assertRaises(worktrees.WorktreeCleanupError):
                worktrees._managed_branch_for_path(descendant)
            with self.assertRaises(FileNotFoundError), checkouts.hold(str(checkout / "missing"), require_exists=True):
                self.fail("an existing checkout root must not hide a missing cwd")

            def claim_alias() -> bool:
                with checkouts.hold(str(alias), blocking=False) as acquired:
                    return acquired

            with (
                checkouts.hold(str(descendant), require_exists=True),
                checkouts.hold_many([str(alias), str(checkout), str(descendant), ""]),
                checkouts.hold(str(alias), require_exists=True),
            ):
                self.assertFalse(executor.submit(claim_alias).result(timeout=5))
            self.assertTrue(executor.submit(claim_alias).result(timeout=5))

    def test_group_contention_releases_partial_leases(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            override_settings(HITCH_WORKTREES_DIR=Path(raw)),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            first, second = Path(raw) / "a", Path(raw) / "b"
            first.mkdir()
            second.mkdir()

            def claim_group() -> bool:
                with checkouts.hold_many([str(second), str(first)], blocking=False) as acquired:
                    return acquired

            def claim_first() -> bool:
                with checkouts.hold(str(first), blocking=False) as acquired:
                    return acquired

            with checkouts.hold(str(second)):
                self.assertFalse(executor.submit(claim_group).result(timeout=5))
                self.assertTrue(executor.submit(claim_first).result(timeout=5))
            self.assertTrue(executor.submit(claim_group).result(timeout=5))

    def test_unresolvable_paths_are_not_removed_or_started(self) -> None:
        with patch.object(Path, "resolve", side_effect=OSError("cannot resolve")):
            self.assertIsNone(checkouts.managed_key("/managed/repo/checkout"))
            self.assertFalse(worktrees.cleanup_managed_worktree_path("/managed/repo/checkout"))
            with self.assertRaisesRegex(OSError, "cannot resolve"), checkouts.hold("/managed/repo/checkout"):
                self.fail("startup must not continue without its checkout lease")


class CheckoutRetentionTests(TestCase):
    def test_snapshot_and_leased_recheck_agree_on_retention(self) -> None:
        now = timezone.now()
        cases = (
            ("visible", False, False, "", None, False),
            ("recent", True, False, "", now - timedelta(minutes=30), False),
            ("old", True, False, "", now - timedelta(hours=2), True),
            ("merged", True, False, "done_merged", now, True),
            ("closed", True, False, "done_closed", now, True),
            ("system", False, True, "", None, True),
            ("promoted", False, True, "", None, False),
        )
        with tempfile.TemporaryDirectory() as raw, override_settings(HITCH_WORKTREES_DIR=Path(raw)):
            for name, archived, hidden, stage, archived_at, removable in cases:
                with self.subTest(name=name):
                    cwd = str(Path(raw) / name)
                    Path(cwd).mkdir()
                    metadata = SessionMetadata.objects.create(
                        thread_id=name, cwd=cwd, codex_archived=archived, is_hidden_system_session=hidden,
                        derived_stage=stage, codex_archived_at=archived_at,
                    )
                    if name == "promoted":
                        ProposedSession.objects.create(
                            outcome_status=ProposedSession.OUTCOME_ACCEPTED, accepted_session=metadata,
                        )
                    snapshot = checkout_protection.snapshot(now=now)
                    self.assertEqual(not snapshot.protects(metadata), removable)
                    with checkout_protection.removal_lease(cwd, aliases=(), changed_since=now) as acquired:
                        self.assertEqual(acquired, removable)

    def test_known_aliases_remain_protected_without_recent_updates(self) -> None:
        with tempfile.TemporaryDirectory() as raw, override_settings(HITCH_WORKTREES_DIR=Path(raw)):
            checkout, alias = Path(raw) / "checkout", Path(raw) / "alias"
            checkout.mkdir()
            alias.symlink_to(checkout, target_is_directory=True)
            metadata = SessionMetadata.objects.create(thread_id="alias", cwd=str(alias))
            changed_since = timezone.now() + timedelta(hours=1)
            for protection in ("visible", "worker", "watch", "terminal", "proposal"):
                with self.subTest(protection=protection):
                    metadata.codex_archived = protection != "visible"
                    metadata.codex_archived_at = timezone.now() - timedelta(hours=2)
                    metadata.save()
                    if protection == "worker":
                        CodexInstance.objects.create(
                            thread_id="worker", cwd=str(alias), pid=0, status=CodexInstance.STATUS_STARTING,
                        )
                    elif protection in {"watch", "terminal"}:
                        key = "watch_active" if protection == "watch" else "watch_terminal_pending"
                        record = SessionPullRequest.objects.create(
                            thread_id=protection, cwd=str(alias), state={key: True},
                        )
                    elif protection == "proposal":
                        ProposedSession.objects.create(source_session=metadata)
                    with checkout_protection.removal_lease(
                        str(checkout), aliases=(str(alias),), changed_since=changed_since,
                    ) as acquired:
                        self.assertFalse(acquired)
                    if protection == "worker":
                        CodexInstance.objects.all().delete()
                    elif protection in {"watch", "terminal"}:
                        record.delete()
