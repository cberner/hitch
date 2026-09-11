"""Durable model and effort overrides for subsequent session turns."""

from hitch.main.models import SessionMetadata


def session_model_override(thread_id: str) -> tuple[str, str] | None:
    return (
        SessionMetadata.objects.filter(thread_id=thread_id)
        .exclude(model="")
        .values_list("model", "reasoning_effort")
        .first()
    )
