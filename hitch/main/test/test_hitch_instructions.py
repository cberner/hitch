import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from hitch.main.runtime import instruction_baseline
from hitch.main.sessions.hitch_instructions import (
    DEFAULT_HITCH_EXTRA_INSTRUCTIONS,
    hitch_instructions_for_turn,
)


class HitchInstructionsTests(SimpleTestCase):
    def test_imported_baseline_uses_typed_metadata_and_stops_at_unknown_history(self) -> None:
        message = {"type": "response_item", "payload": {
            "type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "Keep my instructions"}],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["generic.developer_instructions"]},
        }}
        skills = {"type": "response_item", "payload": {
            "type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "Unrelated skills update"}],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["host_skills.instructions"]},
        }}
        self.assertIsNone(instruction_baseline.recorded_developer_instructions(None))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rollout.jsonl"
            self.assertIsNone(instruction_baseline.recorded_developer_instructions(path))
            for suffix, expected in (
                ([skills], "Keep my instructions"),
                ([{"type": "compacted", "payload": {}}], None),
                ([{"type": "event_msg", "payload": {"type": "thread_rolled_back"}}], None),
                ([{"type": "response_item", "payload": {"type": "message", "role": "developer", "content": []}}], None),
            ):
                path.write_text("\n".join(json.dumps(event) for event in [message, *suffix]))
                self.assertEqual(instruction_baseline.recorded_developer_instructions(path), expected)
            path.write_text(json.dumps(message))
            with patch.object(instruction_baseline, "_MAX_BASELINE_BYTES", 10):
                self.assertIsNone(instruction_baseline.recorded_developer_instructions(path))
            path.write_bytes(b"invalid json\xff")
            self.assertIsNone(instruction_baseline.recorded_developer_instructions(path))

    def test_defaults_follow_selected_workflow(self) -> None:
        for auto_pr, auto_qa, plan, workflow in (
            (False, False, False, "None"),
            (False, True, False, "Auto-QA"),
            (True, True, False, "Auto-PR"),
            (True, True, True, "Plan"),
        ):
            with self.subTest(workflow=workflow):
                text = hitch_instructions_for_turn(
                    None, auto_pr_enabled=auto_pr, auto_qa_enabled=auto_qa,
                    plan_mode=plan, pr_title=" Proposed\nchange ",
                )
                self.assertTrue(text.startswith(f"Hitch workflow for this turn: {workflow}."))
                self.assertTrue(text.endswith(DEFAULT_HITCH_EXTRA_INSTRUCTIONS))
                self.assertEqual("Requested pull request title: Proposed change" in text, auto_pr and not plan)

    def test_override_replaces_defaults_and_empty_disables(self) -> None:
        custom = "Do my review. Preserve {literal braces}."
        self.assertEqual(
            hitch_instructions_for_turn(custom, auto_pr_enabled=True),
            f"Hitch workflow for this turn: Auto-PR.\n\n{custom}",
        )
        self.assertEqual(hitch_instructions_for_turn("", auto_pr_enabled=True), "")
        self.assertEqual(hitch_instructions_for_turn(" \n", auto_qa_enabled=True), "")
