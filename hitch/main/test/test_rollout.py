import json
import tempfile
from pathlib import Path
from typing import Any, override

from django.test import TestCase

from hitch.main import rollout


def _line(line_type: str, payload: dict[str, Any], *, timestamp: str = "2025-01-05T12:00:00Z") -> str:
    return json.dumps({"timestamp": timestamp, "type": line_type, "payload": payload})


def _write_rollout(lines: list[str]) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="rollout-",
        suffix=".jsonl",
        mode="w",
        delete=False,
    ) as tmp:
        tmp.write("\n".join(lines))
        if lines:
            tmp.write("\n")
        return Path(tmp.name)


class IterEntriesTests(TestCase):
    @override
    def tearDown(self) -> None:
        for path in getattr(self, "_paths", []):
            path.unlink(missing_ok=True)

    def _make(self, lines: list[str]) -> Path:
        path = _write_rollout(lines)
        self._paths = [*getattr(self, "_paths", []), path]
        return path

    def test_function_call_for_shell_yields_command_execution(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "cargo build --release"}),
                        "call_id": "call-1",
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["kind"], "tool_call")
        self.assertEqual(entry["type"], "commandExecution")
        self.assertEqual(entry["label"], "Command")
        self.assertEqual(entry["detail"], "cargo build --release")

    def test_shell_command_legacy_argument_name(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "shell_command",
                        "arguments": json.dumps({"command": "ls -la"}),
                        "call_id": "call-2",
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["detail"], "ls -la")

    def test_function_call_for_non_shell_tool_is_ignored(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "request_user_input",
                        "arguments": "{}",
                        "call_id": "call-3",
                    },
                ),
            ]
        )
        self.assertEqual(list(rollout.iter_entries(path)), [])

    def test_failed_exec_command_surfaces_failed_status(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "false"}),
                        "call_id": "call-4",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-4",
                        "output": json.dumps({"exit_code": 1, "output": ""}),
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["status"], "failed")

    def test_in_progress_command_without_output(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "sleep 100"}),
                        "call_id": "call-5",
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["status"], "inProgress")

    def test_local_shell_call_is_rendered(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "local_shell_call",
                        "call_id": "ls-1",
                        "status": "completed",
                        "action": {"type": "exec", "command": ["ls", "-la"]},
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["type"], "commandExecution")
        self.assertEqual(entries[0]["detail"], "ls -la")
        self.assertIsNone(entries[0]["status"])

    def test_user_and_agent_messages(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "do the thing"}),
                _line("event_msg", {"type": "agent_message", "message": "okay"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["kind"], "user")
        self.assertEqual(entries[0]["text"], "do the thing")
        self.assertEqual(entries[1]["kind"], "agent")
        self.assertEqual(entries[1]["text"], "okay")

    def test_patch_apply_end_to_file_change(self) -> None:
        path = self._make(
            [
                _line(
                    "event_msg",
                    {
                        "type": "patch_apply_end",
                        "call_id": "patch-1",
                        "status": "completed",
                        "changes": {"hitch/main/views.py": {}},
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["type"], "fileChange")
        self.assertEqual(entries[0]["detail"], "hitch/main/views.py")
        self.assertIsNone(entries[0]["status"])

    def test_patch_apply_end_failed_status(self) -> None:
        path = self._make(
            [
                _line(
                    "event_msg",
                    {
                        "type": "patch_apply_end",
                        "call_id": "patch-2",
                        "status": "failed",
                        "changes": {"a.py": {}, "b.py": {}},
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["status"], "failed")
        self.assertEqual(entries[0]["detail"], "a.py (+1 more)")

    def test_web_search_and_context_compaction(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "web_search_end", "query": "how to django"}),
                _line("event_msg", {"type": "context_compacted"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["type"], "webSearch")
        self.assertEqual(entries[0]["detail"], "how to django")
        self.assertEqual(entries[1]["type"], "contextCompaction")

    def test_entries_preserve_rollout_file_order(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "build it"}),
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "cargo build"}),
                        "call_id": "c1",
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "built"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        kinds = [e.get("kind") for e in entries]
        self.assertEqual(kinds, ["user", "tool_call", "agent"])

    def test_timestamps_are_unix_seconds(self) -> None:
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "hi"},
                    timestamp="2026-04-14T23:05:00Z",
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        # 2026-04-14T23:05:00Z == 1776207900 unix seconds
        self.assertEqual(entries[0]["timestamp"], 1776207900)

    def test_malformed_lines_are_skipped(self) -> None:
        path = self._make(
            [
                "not valid json",
                _line("event_msg", {"type": "user_message", "message": "hi"}),
                "",
                "{unclosed",
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "hi")

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(list(rollout.iter_entries(Path("/nonexistent/rollout.jsonl"))), [])

    def test_response_item_message_does_not_duplicate_agent_text(self) -> None:
        # Both an event_msg::agent_message and a response_item::message
        # (role=assistant) are typically present for the same agent turn. Only
        # the event-derived entry should be emitted, mirroring the SDK.
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "okay"}],
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "okay"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "agent")

    def test_empty_agent_message_is_dropped(self) -> None:
        path = self._make([_line("event_msg", {"type": "agent_message", "message": ""})])
        self.assertEqual(list(rollout.iter_entries(path)), [])

    def test_agent_message_phase_is_preserved(self) -> None:
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "agent_message", "message": "interim", "phase": "commentary"},
                ),
                _line(
                    "event_msg",
                    {"type": "agent_message", "message": "answer", "phase": "final_answer"},
                ),
                _line("event_msg", {"type": "agent_message", "message": "no phase"}),
                _line(
                    "event_msg",
                    {"type": "agent_message", "message": "nonstring", "phase": 7},
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["phase"], "commentary")
        self.assertEqual(entries[1]["phase"], "final_answer")
        self.assertIsNone(entries[2]["phase"])
        # Non-string phases are coerced to None so the collapser doesn't crash
        # when codex emits something unexpected.
        self.assertIsNone(entries[3]["phase"])

    def test_mcp_tool_call_end(self) -> None:
        # Codex serialises `result: Result<CallToolResult, String>` as
        # `{"Ok": {...}}` or `{"Err": "..."}`. An Ok with no error renders
        # without a badge; Err and Ok-but-is_error both surface as "failed".
        path = self._make(
            [
                _line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "call_id": "m1",
                        "invocation": {"server": "github", "tool": "create_pr"},
                        "result": {"Ok": {"content": [], "is_error": False}},
                    },
                ),
                _line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "call_id": "m2",
                        "invocation": {"server": "github", "tool": "create_pr"},
                        "result": {"Err": "connection refused"},
                    },
                ),
                _line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "call_id": "m3",
                        "invocation": {"server": "github", "tool": "create_pr"},
                        "result": {"Ok": {"content": [], "is_error": True}},
                    },
                ),
                # Missing/unrecognised result shape leaves the badge unset
                # rather than guessing.
                _line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "call_id": "m4",
                        "invocation": {"server": "github", "tool": "create_pr"},
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["type"], "mcpToolCall")
        self.assertEqual(entries[0]["detail"], "github / create_pr")
        self.assertIsNone(entries[0]["status"])
        self.assertEqual(entries[1]["status"], "failed")
        self.assertEqual(entries[2]["status"], "failed")
        self.assertIsNone(entries[3]["status"])

    def test_user_message_with_images(self) -> None:
        path = self._make(
            [
                _line(
                    "event_msg",
                    {
                        "type": "user_message",
                        "message": "look at this",
                        "images": ["https://example.com/a.png", "https://example.com/b.png"],
                        "local_images": ["/tmp/c.png"],
                    },
                ),
                # Image-only prompt: message is empty but attachments still
                # need to surface so the user entry doesn't render blank.
                _line(
                    "event_msg",
                    {
                        "type": "user_message",
                        "message": "",
                        "images": ["https://example.com/d.png"],
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(
            entries[0]["text"],
            "look at this\n[image]\n[image]\n[image: /tmp/c.png]",
        )
        self.assertEqual(entries[1]["text"], "[image]")

    def test_agent_reasoning(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "agent_reasoning", "text": "first line\nrest"}),
                _line("event_msg", {"type": "agent_reasoning", "text": ""}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "reasoning")
        # Long reasoning collapses to its first line so the row stays compact.
        self.assertEqual(entries[0]["detail"], "first line")

    def test_review_mode_transitions(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "entered_review_mode"}),
                _line("event_msg", {"type": "exited_review_mode"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual([e["type"] for e in entries], ["enteredReviewMode", "exitedReviewMode"])

    def test_unknown_event_types_are_skipped(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "thread_goal_updated", "goal": {}}),
                _line("event_msg", {"type": "user_message", "message": "hi"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "user")

    def test_non_response_item_lines_are_skipped(self) -> None:
        # session_meta / turn_context / compacted markers have no UI entry.
        path = self._make(
            [
                _line("session_meta", {"id": "abc", "cwd": "."}),
                _line("turn_context", {"turn_id": "t1"}),
                _line("compacted", {"message": "summary"}),
                _line("event_msg", {"type": "user_message", "message": "hi"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)

    def test_function_call_arguments_edge_cases(self) -> None:
        # Non-string arguments fall through as empty; malformed JSON falls
        # back to the raw string; a JSON literal that isn't a dict also falls
        # back to the raw string.
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": None,
                        "call_id": "a1",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": "{not json",
                        "call_id": "a2",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": '"literal"',
                        "call_id": "a3",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": 42}),
                        "call_id": "a4",
                    },
                ),
            ]
        )
        details = [e["detail"] for e in rollout.iter_entries(path)]
        self.assertEqual(details, ["", "{not json", '"literal"', ""])

    def test_function_call_status_with_unparseable_output(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "ls"}),
                        "call_id": "s1",
                    },
                ),
                _line(
                    "response_item",
                    {"type": "function_call_output", "call_id": "s1", "output": "raw text"},
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "ls"}),
                        "call_id": "s2",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "s2",
                        "output": json.dumps({"exit_code": 0, "output": ""}),
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        # Plain-text output: no exit_code info, so no failure badge.
        self.assertIsNone(entries[0]["status"])
        # exit_code == 0: completed cleanly, no badge either.
        self.assertIsNone(entries[1]["status"])

    def test_local_shell_call_with_non_exec_action_is_skipped(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "local_shell_call",
                        "call_id": "ls",
                        "status": "completed",
                        "action": {"type": "kill"},
                    },
                ),
            ]
        )
        self.assertEqual(list(rollout.iter_entries(path)), [])

    def test_patch_apply_end_with_no_changes(self) -> None:
        path = self._make(
            [
                _line(
                    "event_msg",
                    {
                        "type": "patch_apply_end",
                        "call_id": "p",
                        "status": "completed",
                        "changes": {},
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(entries[0]["detail"], "")

    def test_timestamp_parsing_edge_cases(self) -> None:
        # Non-string, unparseable, and naive (no tz) timestamps. The naive
        # value is treated as UTC.
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "a"}, timestamp=""),
                _line("event_msg", {"type": "user_message", "message": "b"}, timestamp="garbage"),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "c"},
                    timestamp="2026-04-14T23:05:00",
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertIsNone(entries[0]["timestamp"])
        self.assertIsNone(entries[1]["timestamp"])
        self.assertEqual(entries[2]["timestamp"], 1776207900)

    def test_timestamp_missing_entirely_is_none(self) -> None:
        # Hand-craft a line without a timestamp key to exercise the non-string
        # branch in _iso_to_unix_seconds.
        path = self._make(
            [
                json.dumps(
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}}
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertIsNone(entries[0]["timestamp"])
