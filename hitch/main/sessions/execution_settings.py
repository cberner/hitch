"""Approval precedence for request-driven turns and inherited background turns."""

from dataclasses import dataclass
from typing import Literal

from hitch.main.models import CodexInstance, SessionMetadata
from hitch.main.sessions.settings_cookies import _DEFAULT_APPROVAL_MODE, _VALID_APPROVAL_MODES


@dataclass(frozen=True)
class RequestApproval:
    mode: str


@dataclass(frozen=True)
class PreviousTurnApproval:
    instance_id: int
    mode: str


@dataclass(frozen=True)
class ResolvedApproval:
    mode: str
    source: Literal["session_override", "request", "session_snapshot", "previous_turn", "default"]


def approval_override(metadata: SessionMetadata | None) -> str:
    mode = metadata.approval_mode if metadata is not None else ""
    return mode if mode in _VALID_APPROVAL_MODES else ""


def resolve_approval(
    metadata: SessionMetadata | None,
    defaults: RequestApproval | PreviousTurnApproval,
) -> ResolvedApproval:
    """Explicit overrides win; only inherited turns consume reset snapshots."""
    override = approval_override(metadata)
    if override:
        return ResolvedApproval(override, "session_override")
    if (
        isinstance(defaults, PreviousTurnApproval)
        and metadata is not None
        and metadata.approval_snapshot_instance_id == defaults.instance_id
        and metadata.approval_snapshot_mode in _VALID_APPROVAL_MODES
    ):
        return ResolvedApproval(metadata.approval_snapshot_mode, "session_snapshot")
    if defaults.mode not in _VALID_APPROVAL_MODES:
        return ResolvedApproval(_DEFAULT_APPROVAL_MODE, "default")
    return ResolvedApproval(
        defaults.mode, "request" if isinstance(defaults, RequestApproval) else "previous_turn",
    )


def save_session_approval(
    thread_id: str, *, cwd: str, override: str, defaults: RequestApproval,
) -> ResolvedApproval:
    """Save an override and its reset snapshot under the caller's session lease."""
    if override and override not in _VALID_APPROVAL_MODES:
        raise ValueError("invalid approval mode")
    resolved = resolve_approval(SessionMetadata(approval_mode=override), defaults)
    owner_id = CodexInstance.objects.filter(
        thread_id=thread_id, purpose=CodexInstance.PURPOSE_USER,
    ).order_by("-pk").values_list("pk", flat=True).first()
    SessionMetadata.objects.update_or_create(
        thread_id=thread_id,
        defaults={
            "cwd": cwd,
            "approval_mode": override,
            "approval_snapshot_mode": resolved.mode,
            "approval_snapshot_instance_id": owner_id,
        },
    )
    return resolved
