"""Ownership rules for classifying hidden system sessions."""

from __future__ import annotations

from typing import Any

from django.db.models import QuerySet

from hitch.main.legacy_agent_records import without_legacy_redacted_agent_records


def system_session_owner_rows(rows: QuerySet[Any]) -> QuerySet[Any]:
    """Return agent rows whose thread is owned by a hidden system session.

    Legacy demo turns ran on the user's visible thread. Their rows remain as
    transcript-redaction evidence, but they never transfer ownership of that
    thread to the system-session lifecycle.
    """
    return without_legacy_redacted_agent_records(rows)
