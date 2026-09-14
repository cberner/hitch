"""Own the idle-to-starting transition for an existing Codex thread."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Required, TypedDict, Unpack

from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.runtime import codex_pool
from hitch.main.sessions import lifecycle
from hitch.main.sessions.execution_settings import PreviousTurnApproval, RequestApproval, resolve_approval


class TurnOptions(TypedDict, total=False):
    cwd: Required[str]
    prompt: Required[str]
    input_image_paths: list[str] | None
    model: str | None
    stored_model: str | None
    reasoning_effort: str | None
    stored_reasoning_effort: str | None
    sandbox_policy: str | None
    web_search_mode: str | None
    enable_memories: bool
    collaboration_mode: str | None
    plan_mode: bool
    developer_instructions: str | None
    hitch_extra_instructions: str | None
    new_thread: bool
    purpose: str
    agent_kind: str
    user_message_index: int | None


class TurnStartup:
    """One startup attempt, valid only inside its session lease."""

    def __init__(self, thread_id: str) -> None:
        self._thread_id = thread_id
        self._available = False

    def spawn(
        self, *, approval: RequestApproval | PreviousTurnApproval, **kwargs: Unpack[TurnOptions],
    ) -> CodexInstance:
        if not self._available:
            raise RuntimeError("turn startup claim is no longer available")
        self._available = False
        metadata = SessionMetadata.objects.filter(thread_id=self._thread_id).only(
            "approval_mode", "approval_snapshot_mode", "approval_snapshot_instance_id",
        ).first()
        return codex_pool._spawn_turn(
            thread_id=self._thread_id, approval_mode=resolve_approval(metadata, approval).mode, **kwargs,
        )


@contextmanager
def claim(thread_id: str, *, blocking: bool = True) -> Iterator[TurnStartup | None]:
    """Yield a startup claim, or None if the lease or an active worker is busy.

    Callers prepare the turn and perform acceptance/rollback bookkeeping inside
    the claim. Steering is a separate operation and must happen before claiming.
    """
    with lifecycle.hold(thread_id, blocking=blocking) as acquired:
        if not acquired or codex_pool.latest_active_for_thread(thread_id) is not None:
            yield None
            return
        startup = TurnStartup(thread_id)
        startup._available = True
        try:
            yield startup
        finally:
            startup._available = False
