from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from django.test import TestCase
from openai_codex import AppServerError, Codex

from hitch.main import session_index
from hitch.main.models import SessionIndexSyncState, SessionMetadata


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
    def test_local_session_stores_and_preserves_codex_path(self) -> None:
        metadata = session_index.upsert_local_session(
            thread_id="local-thread",
            cwd="/repo",
            name="Initial title",
            codex_path="/root/.codex/sessions/rollout-local-thread.jsonl",
        )

        self.assertEqual(
            metadata.codex_path,
            "/root/.codex/sessions/rollout-local-thread.jsonl",
        )

        metadata = session_index.upsert_local_session(
            thread_id="local-thread",
            cwd="/repo",
            name="Updated title",
        )

        self.assertEqual(
            metadata.codex_path,
            "/root/.codex/sessions/rollout-local-thread.jsonl",
        )

    def test_active_window_resumes_from_cursor_without_marking_synced(self) -> None:
        # A mid-list window (more pages remain) must page from start_cursor and
        # must NOT advance the request-path SessionIndexSyncState cursor.
        thread_list = MagicMock(
            return_value=SimpleNamespace(
                data=[_thread("mid-thread")],
                next_cursor="page-3",
            )
        )
        codex = cast(Codex, SimpleNamespace(thread_list=thread_list))

        result = session_index.refresh_active_window(
            codex, projects=[], start_cursor="page-2", max_pages=1
        )

        self.assertEqual(result.synced, 1)
        self.assertEqual(result.next_cursor, "page-3")
        self.assertFalse(result.complete)
        self.assertFalse(result.failed)
        # Resumed from the supplied cursor rather than the front.
        self.assertEqual(thread_list.call_args.kwargs["cursor"], "page-2")
        # Background windows leave the request-path sync state untouched.
        self.assertFalse(
            SessionIndexSyncState.objects.filter(
                source=SessionIndexSyncState.SOURCE_ACTIVE
            ).exists()
        )

    def test_active_window_failure_keeps_cursor_and_reports_failed(self) -> None:
        # A failed window must not advance: it reports failed and hands back the
        # start cursor so the scheduler retries the same spot next tick.
        thread_list = MagicMock(side_effect=AppServerError("thread list down"))
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

    def test_active_window_completion_bumps_freshness_and_clears_cursor(self) -> None:
        thread_list = MagicMock(
            return_value=SimpleNamespace(
                data=[_thread("last-thread")],
                next_cursor="",
            )
        )
        codex = cast(Codex, SimpleNamespace(thread_list=thread_list))

        result = session_index.refresh_active_window(
            codex, projects=[], start_cursor="page-9", max_pages=5
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.next_cursor, "")
        # A completed pass bumps the freshness signal so an idle dashboard still
        # skips its own refresh.
        state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        )
        self.assertTrue(state.is_complete)

    def test_capped_refresh_preserves_previously_complete_source(self) -> None:
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=datetime.now(UTC),
            is_complete=True,
        )
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

        self.assertEqual(result.synced, 1)
        self.assertFalse(result.failed)
        self.assertEqual(result.active_next_cursor, "page-2")
        state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        )
        self.assertTrue(state.is_complete)
        self.assertEqual(state.next_cursor, "page-2")
        self.assertTrue(session_index.has_pending_pages(archived=False))
        self.assertTrue(
            SessionMetadata.objects.filter(thread_id="newest-thread").exists()
        )

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

    def test_complete_refresh_clears_incomplete_cursor(self) -> None:
        SessionIndexSyncState.objects.create(
            source=SessionIndexSyncState.SOURCE_ACTIVE,
            last_synced_at=datetime.now(UTC),
            is_complete=False,
            next_cursor="page-2",
        )
        thread_list = MagicMock(
            return_value=SimpleNamespace(data=[_thread("final-thread")])
        )
        codex = cast(Codex, SimpleNamespace(thread_list=thread_list))

        session_index.refresh_from_codex(
            codex,
            projects=[],
            include_active=True,
            max_pages=1,
        )

        state = SessionIndexSyncState.objects.get(
            source=SessionIndexSyncState.SOURCE_ACTIVE
        )
        self.assertTrue(state.is_complete)
        self.assertEqual(state.next_cursor, "")
        self.assertFalse(session_index.has_pending_pages(archived=False))
