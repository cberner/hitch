from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from django.test import TestCase
from openai_codex import Codex

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
