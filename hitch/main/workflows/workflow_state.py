"""Generic accessors over a SystemWorkflow's persisted ``state`` dict.

System workflows persist loosely typed key/value state. This module owns the
dependency-free typed readers over that dict (string/int/bool/dict coercions),
the SessionMetadata lookup from a state-stored session id, and the confidence
threshold comparison those accessors feed.
"""

from __future__ import annotations

from typing import Any

from hitch.main.models import AutonomousGoal, SessionMetadata, SystemWorkflow
from hitch.main.workflows import engine

_CONFIDENCE_RANK = {
    AutonomousGoal.CONFIDENCE_MEDIUM: 1,
    AutonomousGoal.CONFIDENCE_HIGH: 2,
    AutonomousGoal.CONFIDENCE_VERY_HIGH: 3,
}


def _checked_state_key(workflow: SystemWorkflow, key: str) -> str:
    """Refuse to read a state key the workflow's kind does not declare.

    Catches a typo'd key (or a read wired to the wrong workflow object) at
    the call site instead of silently returning the type's default.
    """
    declared = engine.declared_state_keys(workflow.kind)
    if declared is not None and key not in declared:
        raise KeyError(
            f"undeclared state key {key!r} for workflow kind {workflow.kind!r}; "
            "declare it on the kind's WorkflowHandler.state_keys"
        )
    return key


def _state_dict(workflow: SystemWorkflow, key: str) -> dict[str, Any]:
    value = workflow.state.get(_checked_state_key(workflow, key))
    return dict(value) if isinstance(value, dict) else {}


def _state_string(workflow: SystemWorkflow, key: str) -> str:
    value = workflow.state.get(_checked_state_key(workflow, key))
    return value if isinstance(value, str) else ""


def _state_int(workflow: SystemWorkflow, key: str) -> int:
    value = workflow.state.get(_checked_state_key(workflow, key))
    return value if isinstance(value, int) and value >= 0 else 0


def _state_bool(workflow: SystemWorkflow, key: str) -> bool:
    return workflow.state.get(_checked_state_key(workflow, key)) is True


def _confidence_meets_threshold(confidence: str, threshold: str) -> bool:
    return _CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK.get(threshold, 0)


def _session_metadata_from_state(
    workflow: SystemWorkflow, key: str
) -> SessionMetadata | None:
    session_id = _state_int(workflow, key)
    if session_id < 1:
        return None
    return SessionMetadata.objects.filter(pk=session_id).first()
