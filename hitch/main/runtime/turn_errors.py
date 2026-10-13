"""Classify terminal errors without losing an unresolved capacity warning."""

from dataclasses import dataclass
from typing import Any

from openai_codex.generated.v2_all import ErrorNotification, Turn, TurnStatus
from openai_codex.models import Notification


def is_model_connection_failure(code: Any, message: str) -> bool:
    if isinstance(code, dict):
        code = next(iter(code), "")
    return code in (
        "httpConnectionFailed", "responseStreamDisconnected", "responseStreamConnectionFailed",
        "responseTooManyFailedAttempts",
    ) or (
        code in (None, "other")
        and message.strip().lower().startswith("stream disconnected before completion:")
    )


@dataclass
class CapacityFailureTracker:
    thread_id: str
    turn_id: str
    details: str = ""

    def observe(self, event: Notification) -> None:
        payload = event.payload
        if (
            getattr(payload, "thread_id", None) != self.thread_id
            or getattr(payload, "turn_id", None) != self.turn_id
        ):
            return
        if isinstance(payload, ErrorNotification):
            error = payload.error
            code = error.codex_error_info.model_dump(mode="json", by_alias=True) if error.codex_error_info else None
            details = error.additional_details or error.message
            if payload.will_retry and (
                code == "serverOverloaded"
                or (
                    is_model_connection_failure(code, error.message)
                    and "currently experiencing high demand" in details.lower()
                )
            ):
                self.details = details
            return
        # Usage updates and output from existing tools do not prove model recovery.
        item = getattr(payload, "item", None)
        item_type = getattr(getattr(item, "root", item), "type", None)
        if event.method in (
            "item/agentMessage/delta", "item/reasoning/summaryTextDelta",
            "item/reasoning/textDelta", "item/plan/delta",
        ) or (
            event.method == "item/started" and item_type in (
                "agentMessage", "reasoning", "plan", "commandExecution", "fileChange",
                "webSearch", "mcpToolCall", "dynamicToolCall", "collabAgentToolCall",
                "imageView", "imageGeneration",
            )
        ) or (event.method == "item/completed" and item_type in ("agentMessage", "reasoning", "plan")):
            self.details = ""

    def is_capacity_failure(self, turn: Turn | None) -> bool:
        if turn is None or turn.id != self.turn_id or turn.status != TurnStatus.failed or turn.error is None:
            return False
        error = turn.error
        code = error.codex_error_info.model_dump(mode="json", by_alias=True) if error.codex_error_info else None
        return code == "serverOverloaded" or (
            bool(self.details) and is_model_connection_failure(code, error.message)
        )
