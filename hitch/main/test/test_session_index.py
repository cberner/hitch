from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone
from openai_codex import Codex

from hitch.main.models import SessionIndexSyncState, SessionMetadata
from hitch.main.sessions import lifecycle, session_index


def _thread(thread_id: str, *, updated_at: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        id=thread_id,
        name=thread_id,
        preview="",
        cwd="/repo",
        path="",
        updated_at=updated_at,
    )


class SessionIndexRefreshTests(TestCase):
    def test_indexing_protects_both_stored_and_observed_checkouts(self) -> None:
        with tempfile.TemporaryDirectory() as raw, override_settings(
            HITCH_WORKTREES_DIR=Path(raw),
        ), ThreadPoolExecutor(max_workers=1) as executor:
            incoming, old = Path(raw) / "incoming", Path(raw) / "old"
            incoming.mkdir()
            old.mkdir()
            upsert = session_index._upsert_thread_locked

            def competing_claim(cwd: str) -> bool:
                with lifecycle.hold_worktree(cwd, blocking=False) as acquired:
                    return acquired

            def register(thread: Any, **kwargs: Any) -> SessionMetadata:
                paths = [str(incoming), str(old)] if thread.id == "moved" else [str(incoming)]
                for path in paths:
                    self.assertFalse(executor.submit(competing_claim, path).result(timeout=5))
                return upsert(thread, **kwargs)

            SessionMetadata.objects.create(thread_id="moved", cwd=str(old))
            for thread_id in ("unseen", "moved"):
                with self.subTest(thread_id=thread_id):
                    thread = _thread(thread_id)
                    thread.cwd = str(incoming)
                    with patch.object(session_index, "_upsert_thread_locked", side_effect=register):
                        result = session_index.upsert_thread(thread, projects=[])
                    self.assertIsNotNone(result)
                    self.assertEqual(SessionMetadata.objects.get(thread_id=thread_id).cwd, str(incoming))
                    for path in (incoming, old):
                        self.assertTrue(executor.submit(competing_claim, str(path)).result(timeout=5))

    def test_stale_active_observation_does_not_undo_archive(self) -> None:
        SessionMetadata.objects.create(
            thread_id="archive-race",
            cwd="/repo",
            codex_archived=False,
        )
        observed_at = timezone.now()
        session_index.update_cached_archived("archive-race", archived=True)

        session_index.upsert_thread(
            _thread("archive-race"),
            projects=[],
            observed_at=observed_at,
        )

        metadata = SessionMetadata.objects.get(thread_id="archive-race")
        self.assertTrue(metadata.codex_archived)

    def test_upsert_thread_does_not_regress_worker_bump(self) -> None:
        # A worker turn on an isolated sqlite_home bumped the cached row; the web
        # home still reports the pre-turn (older) updated_at. A DB-only refresh
        # must not drag the session's recency back below the worker bump.
        bumped = datetime(2026, 6, 6, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="bumped-thread",
            cwd="/repo",
            codex_created_at=datetime(2026, 1, 1, tzinfo=UTC),
            codex_updated_at=bumped,
            codex_last_synced_at=bumped,
        )

        session_index.upsert_thread(_thread("bumped-thread", updated_at=1), projects=[])

        metadata = SessionMetadata.objects.get(thread_id="bumped-thread")
        self.assertEqual(metadata.codex_updated_at, bumped)

        # A genuinely newer web timestamp still advances it.
        newer = int(datetime(2027, 1, 1, tzinfo=UTC).timestamp())
        session_index.upsert_thread(
            _thread("bumped-thread", updated_at=newer), projects=[]
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_updated_at, datetime(2027, 1, 1, tzinfo=UTC))

    def test_upsert_thread_preserves_existing_hidden_system_flag(self) -> None:
        SessionMetadata.objects.create(
            thread_id="system-thread",
            cwd="/repo",
            codex_created_at=datetime(2026, 1, 1, tzinfo=UTC),
            codex_updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            codex_last_synced_at=datetime(2026, 1, 1, tzinfo=UTC),
            is_hidden_system_session=True,
        )

        session_index.upsert_thread(_thread("system-thread", updated_at=1), projects=[])

        metadata = SessionMetadata.objects.get(thread_id="system-thread")
        self.assertTrue(metadata.is_hidden_system_session)

    def test_capped_refresh_keeps_never_complete_source_incomplete(self) -> None:
        thread_list = MagicMock(
            return_value=SimpleNamespace(
                data=[_thread("newest-thread")],
                next_cursor="page-2",
            )
        )
        codex = cast(Codex, SimpleNamespace(thread_list=thread_list))

        result = session_index.refresh_from_codex(
            codex,
            projects=[],
            include_active=True,
            max_pages=1,
        )

        self.assertEqual(result.active_next_cursor, "page-2")
        state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        )
        self.assertFalse(state.is_complete)
        self.assertEqual(state.next_cursor, "page-2")
        self.assertTrue(session_index.has_pending_pages(archived=False))
