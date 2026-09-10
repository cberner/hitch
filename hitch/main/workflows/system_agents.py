"""Identify historical Hitch system sessions for display and usage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from hitch.main.models import (
    CodexInstance,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
)
from hitch.main.sessions import session_index


def legacy_promoted_system_thread_ids() -> set[str]:
    """Return accepted threads that must remain visible, including legacy promotions."""
    return set(
        ProposedSession.objects.filter(
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session__isnull=False,
        ).values_list("accepted_session__thread_id", flat=True)
    )


def hidden_thread_ids() -> set[str]:
    hidden_ids = set(SystemAgentRun.objects.exclude(thread_id="").values_list("thread_id", flat=True).distinct())
    hidden_ids.update(
        CodexInstance.objects.filter(purpose=CodexInstance.PURPOSE_SYSTEM_AGENT)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    hidden_ids.update(
        SessionMetadata.objects.filter(is_hidden_system_session=True)
        .exclude(thread_id="")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    return hidden_ids - legacy_promoted_system_thread_ids()


def hidden_thread_ids_from_threads(threads: Iterable[Any]) -> set[str]:
    hidden_ids = {
        thread_id
        for thread in threads
        if isinstance(thread_id := getattr(thread, "id", None), str) and hitch_system_agent_thread(thread)
    }
    return hidden_ids - legacy_promoted_system_thread_ids()


def hitch_system_agent_thread(thread: Any) -> bool:
    return session_index.hidden_system_session_from_metadata(
        name=_thread_metadata_value(getattr(thread, "name", None)).strip(),
        preview=_thread_metadata_value(getattr(thread, "preview", None)).strip(),
        thread_source=_thread_metadata_value(getattr(thread, "thread_source", None)),
    )


def _thread_metadata_value(value: Any) -> str:
    root = getattr(value, "root", value)
    raw = getattr(root, "value", root)
    return raw if isinstance(raw, str) else ""
