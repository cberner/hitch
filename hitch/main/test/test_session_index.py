from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from django.test import TestCase
from django.utils import timezone
from openai_codex import Codex, CodexError

from hitch.main.models import SessionIndexSyncState, SessionMetadata
from hitch.main.sessions import session_index


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

    def test_active_window_failure_keeps_cursor_and_reports_failed(self) -> None:
        # A failed window must not advance: it reports failed and hands back the
        # start cursor so the scheduler retries the same spot next tick.
        thread_list = MagicMock(side_effect=CodexError("thread list down"))
        codex = cast(Codex, SimpleNamespace(thread_list=thread_list))

        result = session_index.refresh_active_window(
            codex, projects=[], start_cursor="page-4", max_pages=5
        )

        self.assertTrue(result.failed)
        self.assertFalse(result.complete)
        self.assertEqual(result.synced, 0)
        self.assertEqual(result.next_cursor, "page-4")
        # A failed window leaves the request-path sync state untouched.
        self.assertFalse(
            SessionIndexSyncState.objects.filter(
                source=SessionIndexSyncState.SOURCE_ACTIVE
            ).exists()
        )

    def test_active_window_self_referential_cursor_resets_to_front(self) -> None:
        # A thread_list response that hands back the same cursor it was called
        # with must not pin the scheduler on that page forever: the window is
        # seeded with start_cursor so the duplicate is caught on the first page,
        # and the next cursor resets to the front for a clean pass next tick.
        thread_list = MagicMock(
            return_value=SimpleNamespace(
                data=[_thread("stuck-thread")],
                next_cursor="page-2",
            )
        )
        codex = cast(Codex, SimpleNamespace(thread_list=thread_list))

        result = session_index.refresh_active_window(
            codex, projects=[], start_cursor="page-2", max_pages=5
        )

        self.assertEqual(result.synced, 1)
        self.assertFalse(result.complete)
        self.assertFalse(result.failed)
        # Reset to the front rather than resuming from the self-referential page.
        self.assertEqual(result.next_cursor, "")
        # Detected on the first page (seeded guard), so it never re-fetched the
        # same stuck page within the window.
        self.assertEqual(thread_list.call_count, 1)

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
