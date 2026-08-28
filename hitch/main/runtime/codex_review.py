"""Configuration for Hitch's native Codex reviewer subagent."""

from __future__ import annotations

import json
from pathlib import Path

REVIEWER_AGENT_NAME = "hitch_reviewer"
REVIEWER_AGENT_DESCRIPTION = (
    "Review the repository's complete current changes as an independent, "
    "read-only reviewer and return prioritized actionable findings."
)

_REVIEWER_AGENT_CONFIG = Path(__file__).with_name("hitch_reviewer.toml").resolve()


def reviewer_config_overrides() -> tuple[str, ...]:
    """Register the reviewer role on one visible coding worker's app-server."""
    prefix = f"agents.{REVIEWER_AGENT_NAME}"
    return (
        "features.multi_agent=true",
        f"{prefix}.description={json.dumps(REVIEWER_AGENT_DESCRIPTION)}",
        f"{prefix}.config_file={json.dumps(str(_REVIEWER_AGENT_CONFIG))}",
    )
