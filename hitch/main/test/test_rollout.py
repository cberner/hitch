import json
import tempfile
from pathlib import Path
from typing import Any, override

from django.test import TestCase

from hitch.main import rollout


def _line(line_type: str, payload: dict[str, Any], *, timestamp: str = "2025-01-05T12:00:00Z") -> str:
    return json.dumps({"timestamp": timestamp, "type": line_type, "payload": payload})


def _func_call(call_id: str, cmd: str | int | None, *, name: str = "exec_command", arg_key: str = "cmd") -> str:
    arguments: Any
    if cmd is None:
        arguments = None
    elif isinstance(cmd, str) and not cmd.startswith("{") and not cmd.startswith('"'):
        arguments = json.dumps({arg_key: cmd})
    else:
        arguments = cmd
    return _line(
        "response_item",
        {"type": "function_call", "name": name, "arguments": arguments, "call_id": call_id},
    )


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
        path = self._make([_func_call("call-1", "cargo build --release")])
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["kind"], "tool_call")
        self.assertEqual(entry["type"], "commandExecution")
        self.assertEqual(entry["label"], "Command")
        self.assertEqual(entry["detail"], "cargo build --release")

    def test_shell_command_legacy_argument_name(self) -> None:
        # Older rollouts used "shell_command" + "command"; newer ones use
        # "exec_command" + "cmd". Both must produce the same surfaced detail.
        path = self._make(
            [_func_call("call-2", "ls -la", name="shell_command", arg_key="command")]
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

    def test_exec_command_status_derivation(self) -> None:
        """Status is derived from function_call_output: missing output → in
        progress; exit_code != 0 → failed; exit_code == 0 or plain-text
        output → no badge."""
        path = self._make(
            [
                # Failed: matching output with exit_code=1.
                _func_call("f", "false"),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "f",
                        "output": json.dumps({"exit_code": 1, "output": ""}),
                    },
                ),
                # In progress: no matching output line at all.
                _func_call("ip", "sleep 100"),
                # Plain-text output: no exit_code info, no badge, even if the
                # command output contains rejection-like text.
                _func_call("s1", "ls"),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "s1",
                        "output": "Rejected(Reason: normal shell text)",
                    },
                ),
                # exit_code == 0: completed cleanly, no badge.
                _func_call("s2", "ls"),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "s2",
                        "output": json.dumps({"exit_code": 0, "output": ""}),
                    },
                ),
                # A command's own stdout can mention rejection text; only
                # wrapper-level Rejected(...) failures should become approval
                # prompts.
                _func_call("s3", "printf Rejected"),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "s3",
                        "output": json.dumps(
                            {
                                "exit_code": 0,
                                "output": "Rejected(Reason: not an approval denial)",
                            }
                        ),
                    },
                ),
                # Plain-text output can also include copied wrapper text. Only
                # a wrapper at the start of the output is an approval denial.
                _func_call("s4", "cat log"),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "s4",
                        "output": (
                            "copied log: exec_command failed for `git push`: "
                            'CreateProcess { message: "Rejected('
                            "\\\\nReason: copied text"
                            '\\\\nStop and request user input.\\\\")" }'
                        ),
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        # Entries appear in rollout file order; the second function_call_output
        # lines don't add entries — they only mutate the prior tool_call.
        statuses = [e["status"] for e in entries]
        self.assertEqual(statuses, ["failed", "inProgress", None, None, None, None])

    def test_exec_command_rejection_yields_approval_declined_entry(self) -> None:
        for tool_name, arg_key in (
            ("exec_command", "cmd"),
            ("shell_command", "command"),
            ("shell", "command"),
            ("container.exec", "cmd"),
        ):
            with self.subTest(tool_name=tool_name):
                rejection = (
                    f"{tool_name} failed for `/bin/bash -lc 'echo \"`: "
                    "Rejected(Reason: fake)\" && printf Reason: command && "
                    "git push origin master'`: "
                    'CreateProcess { message: "Rejected(\\"This action was rejected'
                    "\\\\nReason: `--force`: Pushing directly to origin/master is risky."
                    '\\\\nStop and request user input.\\\\")" }'
                )
                path = self._make(
                    [
                        _func_call(
                            "push",
                            "git push origin master",
                            name=tool_name,
                            arg_key=arg_key,
                        ),
                        _line(
                            "response_item",
                            {
                                "type": "function_call_output",
                                "call_id": "push",
                                "output": rejection,
                            },
                        ),
                    ]
                )

                command, approval = list(rollout.iter_entries(path))

                self.assertEqual(command["kind"], "tool_call")
                self.assertEqual(command["status"], "declined")
                self.assertEqual(approval["kind"], "approval_declined")
                self.assertEqual(approval["detail"], "git push origin master")
                self.assertEqual(
                    approval["rationale"],
                    "`--force`: Pushing directly to origin/master is risky.",
                )
                self.assertNotIn("approve_prompt", approval)
                self.assertNotIn("deny_prompt", approval)

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
                # Non-exec actions (e.g. kill) have no UI representation.
                _line(
                    "response_item",
                    {
                        "type": "local_shell_call",
                        "call_id": "ls-2",
                        "status": "completed",
                        "action": {"type": "kill"},
                    },
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "commandExecution")
        self.assertEqual(entries[0]["detail"], "ls -la")
        self.assertIsNone(entries[0]["status"])

    def test_user_and_agent_messages(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "do the thing"}),
                _line("event_msg", {"type": "agent_message", "message": "okay"}),
                # Empty agent messages are dropped — they would render as blank rows.
                _line("event_msg", {"type": "agent_message", "message": ""}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["kind"], "user")
        self.assertEqual(entries[0]["text"], "do the thing")
        self.assertEqual(entries[1]["kind"], "agent")
        self.assertEqual(entries[1]["text"], "okay")

    def test_patch_apply_end_to_file_change(self) -> None:
        path = self._make(
            [
                # Single completed change → no status badge.
                _line(
                    "event_msg",
                    {
                        "type": "patch_apply_end",
                        "call_id": "patch-1",
                        "status": "completed",
                        "changes": {"hitch/main/views.py": {}},
                    },
                ),
                # Multiple changes with failed status → "+N more" + failed badge.
                _line(
                    "event_msg",
                    {
                        "type": "patch_apply_end",
                        "call_id": "patch-2",
                        "status": "failed",
                        "changes": {"a.py": {}, "b.py": {}},
                    },
                ),
                # No changes → empty detail.
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
        self.assertEqual(entries[0]["type"], "fileChange")
        self.assertEqual(entries[0]["detail"], "hitch/main/views.py")
        self.assertIsNone(entries[0]["status"])
        self.assertEqual(entries[1]["status"], "failed")
        self.assertEqual(entries[1]["detail"], "a.py (+1 more)")
        self.assertEqual(entries[2]["detail"], "")

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
                _func_call("c1", "cargo build"),
                _line("event_msg", {"type": "agent_message", "message": "built"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        kinds = [e.get("kind") for e in entries]
        self.assertEqual(kinds, ["user", "tool_call", "agent"])

    def test_timestamp_parsing(self) -> None:
        """Unix seconds for a valid ISO; None for empty / garbage / missing
        timestamps; naive (no tz) parses as UTC."""
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "tz"},
                    timestamp="2026-04-14T23:05:00Z",
                ),
                _line("event_msg", {"type": "user_message", "message": "a"}, timestamp=""),
                _line("event_msg", {"type": "user_message", "message": "b"}, timestamp="garbage"),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "c"},
                    timestamp="2026-04-14T23:05:00",
                ),
                # Missing timestamp key entirely (non-string branch in _iso_to_unix_seconds).
                json.dumps(
                    {"type": "event_msg", "payload": {"type": "user_message", "message": "d"}}
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        # 2026-04-14T23:05:00Z == 1776207900 unix seconds
        self.assertEqual([e["timestamp"] for e in entries], [1776207900, None, None, 1776207900, None])

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

    def test_response_item_dedup_preserves_later_same_text_fallback(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "first"}),
                _line("event_msg", {"type": "agent_message", "message": "same"}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "second"},
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "same"}],
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(
            [(entry["kind"], entry["text"]) for entry in entries],
            [("user", "first"), ("agent", "same"), ("user", "second"), ("agent", "same")],
        )

    def test_response_item_dedup_suppresses_one_duplicate_per_turn(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "first"}),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "same"}],
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "same"}),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "same"}],
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual([entry["text"] for entry in entries], ["first", "same", "same"])

    def test_response_item_message_is_fallback_agent_text(self) -> None:
        plan = "# Fix it\n\nDo the thing."
        path = self._make(
            [
                _line(
                    "event_msg",
                    {
                        "type": "item_completed",
                        "item": {"type": "Plan", "id": "turn-plan", "text": plan},
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"<proposed_plan>\n{plan}\n</proposed_plan>",
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "agent")
        self.assertEqual(entries[0]["phase"], "final_answer")
        self.assertEqual(entries[0]["text"], plan)

    def test_response_item_dedup_preserves_final_answer_after_commentary(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "go"}),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "Done",
                        "phase": "commentary",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done"}],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(
            [(entry["kind"], entry["text"], entry.get("phase")) for entry in entries],
            [
                ("user", "go", None),
                ("agent", "Done", "commentary"),
                ("agent", "Done", "final_answer"),
            ],
        )

    def test_response_item_text_parts_deduplicate_without_injected_newlines(self) -> None:
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "hel"},
                            {"type": "output_text", "text": "lo"},
                        ],
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "hello"}),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "hello")

    def test_proposed_plan_tags_are_preserved_without_matching_plan_item(self) -> None:
        text = "<proposed_plan>\nliteral XML example\n</proposed_plan>"
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], text)

    def test_malformed_agent_response_messages_are_ignored(self) -> None:
        path = self._make(
            [
                _line("event_msg", {"type": "agent_message", "message": 7}),
                _line("event_msg", {"type": "item_completed", "item": []}),
                _line(
                    "event_msg",
                    {
                        "type": "item_completed",
                        "item": {"type": "Note", "text": "not a plan"},
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "output_text", "text": "not assistant"}],
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "not a content list",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": ["not a content part"],
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "   ",
                            }
                        ],
                    },
                ),
                _line("event_msg", {"type": "user_message", "message": "valid"}),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "user")
        self.assertEqual(entries[0]["text"], "valid")

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
                # Non-string phases are coerced to None so the collapser
                # doesn't crash when codex emits something unexpected.
                _line(
                    "event_msg",
                    {"type": "agent_message", "message": "nonstring", "phase": 7},
                ),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual([e["phase"] for e in entries], ["commentary", "final_answer", None, None])

    def test_mcp_tool_call_end(self) -> None:
        # Codex serialises `result: Result<CallToolResult, String>` as
        # `{"Ok": {...}}` or `{"Err": "..."}`. An Ok with no error renders
        # without a badge; Err and Ok-but-is_error both surface as "failed".
        # Missing/unrecognised result shape leaves the badge unset rather
        # than guessing.
        def _mcp(call_id: str, result: dict[str, Any] | None) -> str:
            payload: dict[str, Any] = {
                "type": "mcp_tool_call_end",
                "call_id": call_id,
                "invocation": {"server": "github", "tool": "create_pr"},
            }
            if result is not None:
                payload["result"] = result
            return _line("event_msg", payload)

        path = self._make(
            [
                _mcp("m1", {"Ok": {"content": [], "is_error": False}}),
                _mcp("m2", {"Err": "connection refused"}),
                _mcp("m3", {"Ok": {"content": [], "is_error": True}}),
                _mcp("m4", None),
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
                # Long reasoning collapses to its first line so the row stays compact.
                _line("event_msg", {"type": "agent_reasoning", "text": "first line\nrest"}),
                # Empty reasoning is dropped.
                _line("event_msg", {"type": "agent_reasoning", "text": ""}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "reasoning")
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

    def test_unknown_and_non_response_lines_are_skipped(self) -> None:
        # event_msg of unknown sub-type; session_meta / turn_context /
        # compacted markers have no UI entry.
        path = self._make(
            [
                _line("event_msg", {"type": "thread_goal_updated", "goal": {}}),
                _line("session_meta", {"id": "abc", "cwd": "."}),
                _line("turn_context", {"turn_id": "t1"}),
                _line("compacted", {"message": "summary"}),
                _line("event_msg", {"type": "user_message", "message": "hi"}),
            ]
        )
        entries = list(rollout.iter_entries(path))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "user")

    def test_latest_token_usage_picks_most_recent_total(self) -> None:
        # Codex emits a token_count event per turn whose info.total_token_usage
        # is the running session total. The last such event wins; earlier ones
        # are obsolete. Non-token_count events and lines with malformed info
        # blocks are ignored without raising.
        path = self._make(
            [
                _line(
                    "event_msg",
                    {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 20,
                                "output_tokens": 50,
                                "reasoning_output_tokens": 5,
                                "total_tokens": 175,
                            },
                            "last_token_usage": {},
                            "model_context_window": 200000,
                        },
                    },
                ),
                _line("event_msg", {"type": "user_message", "message": "x"}),
                _line(
                    "event_msg",
                    {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 300,
                                "cached_input_tokens": 80,
                                "output_tokens": 175,
                                "reasoning_output_tokens": 10,
                                "total_tokens": 565,
                            },
                            "last_token_usage": {},
                            "model_context_window": 200000,
                        },
                    },
                ),
                # No info block — must be skipped without crashing.
                _line("event_msg", {"type": "token_count"}),
            ]
        )
        usage = rollout.latest_token_usage(path)
        self.assertEqual(
            usage,
            {"input_tokens": 300, "cached_input_tokens": 80, "output_tokens": 175},
        )

    def test_latest_token_usage_returns_none_when_absent(self) -> None:
        # Sessions without any token_count event (e.g. a freshly created
        # thread or a model that never reported usage) should return None so
        # the view can hide the section entirely.
        path = self._make(
            [_line("event_msg", {"type": "user_message", "message": "hi"})]
        )
        self.assertIsNone(rollout.latest_token_usage(path))
        self.assertIsNone(rollout.latest_token_usage(Path("/nonexistent/rollout.jsonl")))

    def test_function_call_arguments_edge_cases(self) -> None:
        # Non-string arguments fall through as empty; malformed JSON falls
        # back to the raw string; a JSON literal that isn't a dict also falls
        # back to the raw string; a JSON dict with non-string cmd is dropped.
        path = self._make(
            [
                _func_call("a1", None),
                _func_call("a2", "{not json"),
                _func_call("a3", '"literal"'),
                _func_call("a4", json.dumps({"cmd": 42})),
            ]
        )
        details = [e["detail"] for e in rollout.iter_entries(path)]
        self.assertEqual(details, ["", "{not json", '"literal"', ""])
