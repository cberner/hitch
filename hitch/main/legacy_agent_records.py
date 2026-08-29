"""Visibility rules for records retained from removed agent features."""

from typing import Any

from django.db.models import QuerySet

LEGACY_REDACTED_AGENT_KINDS = ("demo",)


def is_legacy_redacted_agent_record(record: Any | None) -> bool:
    return record is not None and getattr(record, "agent_kind", "") in LEGACY_REDACTED_AGENT_KINDS


def without_legacy_redacted_agent_records(
    rows: QuerySet[Any],
) -> QuerySet[Any]:
    return rows.exclude(agent_kind__in=LEGACY_REDACTED_AGENT_KINDS)


def only_legacy_redacted_agent_records(rows: QuerySet[Any]) -> QuerySet[Any]:
    return rows.filter(agent_kind__in=LEGACY_REDACTED_AGENT_KINDS)
