"""Shared PR visibility and stage-cache policy for session list and detail."""

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from hitch.main.models import CodexInstance, SessionPullRequest
from hitch.main.runtime import rollout
from hitch.main.sessions import agent_tasks, session_stage
from hitch.main.workflows import pr_tracking

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrDisplayState:
    record: SessionPullRequest | None
    snapshot: dict[str, Any]

    @property
    def has_registered_pr(self) -> bool:
        return bool(self.snapshot)


def pr_display_state(
    record: SessionPullRequest | None, active_instance: CodexInstance | None,
) -> PrDisplayState:
    if not pr_tracking.record_is_current(record):
        record = None
    if (
        active_instance is not None
        and active_instance.agent_kind == agent_tasks.PR_PUBLISH_AGENT_KIND
        and not pr_tracking.watch_registered_by_instance(record, active_instance.pk)
    ):
        record = None
    snapshot = pr_tracking.pr_handoff_for_record(record)
    return PrDisplayState(record if snapshot else None, snapshot)


@dataclass(frozen=True)
class StageCache:
    key: str
    source_mtime_ns: int

    def usable_stage(self, current_mtime_ns: int | None) -> session_stage.SessionStage | None:
        if current_mtime_ns is None or self.source_mtime_ns != current_mtime_ns:
            return None
        stage = session_stage.stage_for_key(self.key)
        # PR and live-input stages depend on state outside the rollout's mtime.
        return stage if stage in (
            session_stage.NEW, session_stage.PLAN, session_stage.IMPLEMENTATION, session_stage.QA,
        ) else None


@dataclass(frozen=True)
class StageHistory:
    entries: Iterable[Mapping[str, Any]]
    complete: bool


@dataclass(frozen=True)
class StageResolution:
    stage: session_stage.SessionStage
    should_persist: bool


def resolve_stage(
    *,
    entries: Iterable[Mapping[str, Any]],
    history_complete: bool,
    pr: PrDisplayState,
    active_instance: CodexInstance | None = None,
    awaiting_user_input: bool = False,
    cache: StageCache | None = None,
    source_mtime_ns: int | None = None,
    leading_user_text: str | None = None,
    history_loader: Callable[[], StageHistory] | None = None,
) -> StageResolution:
    """Resolve authoritative state before touching a cache or transcript."""
    if active_instance is not None or awaiting_user_input or pr.has_registered_pr:
        stage = session_stage.derive_stage(
            active_instance=active_instance, awaiting_user_input=awaiting_user_input,
            pr_snapshot=pr.snapshot,
        )
        # Persist PR stages for durable classification, but never reuse them
        # without a current registration. Worker/input stages are transient.
        return StageResolution(stage, history_complete and active_instance is None and not awaiting_user_input)
    if cache is not None and (cached := cache.usable_stage(source_mtime_ns)) is not None:
        return StageResolution(cached, False)
    if history_loader is not None:
        history = history_loader()
        entries, history_complete = history.entries, history.complete
    if not history_complete and leading_user_text is not None:
        entries = chain(({"kind": "user", "text": leading_user_text},), entries)
    return StageResolution(session_stage.derive_stage(entries=entries), history_complete)


def read_stage_history(
    rollout_path: Path | None, *, has_activity: bool,
) -> StageHistory:
    stage_data = None
    if rollout_path is not None:
        try:
            stage_data = rollout.session_stage_data(rollout_path)
        except Exception:
            logger.exception("failed to parse rollout %s for session stage", rollout_path)
    entries = stage_data.entries if stage_data is not None and stage_data.entries else (
        ({"kind": "user"},) if has_activity else ()
    )
    return StageHistory(entries, complete=stage_data is not None)
