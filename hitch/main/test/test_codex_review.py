from __future__ import annotations

import json
import tomllib
from pathlib import Path

from django.test import SimpleTestCase

from hitch.main.runtime.codex_review import (
    REVIEWER_AGENT_DESCRIPTION,
    REVIEWER_AGENT_NAME,
    reviewer_config_overrides,
)


class ReviewerAgentConfigTests(SimpleTestCase):
    def test_registers_native_read_only_reviewer(self) -> None:
        feature, description, config_file = reviewer_config_overrides()

        self.assertEqual(feature, "features.multi_agent=true")
        self.assertEqual(
            description,
            f"agents.{REVIEWER_AGENT_NAME}.description="
            f"{json.dumps(REVIEWER_AGENT_DESCRIPTION)}",
        )
        config_path = Path(json.loads(config_file.split("=", 1)[1]))
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["sandbox_mode"], "read-only")
        self.assertEqual(config["approval_policy"], "never")
        self.assertNotIn("model", config)
        self.assertIn("Do not modify files", config["developer_instructions"])
