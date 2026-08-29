import json
import tempfile
from pathlib import Path
from typing import Any, override

from django.test import TestCase

from hitch.main.runtime import rollout
from hitch.main.workflows import system_agents


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


class SessionHistoryPageTests(TestCase):
    def test_detects_persisted_namespaced_dynamic_tool(self) -> None:
        path = _write_rollout(
            [
                _line(
                    "session_meta",
                    {
                        "dynamic_tools": [
                            {
                                "type": "namespace",
                                "name": "hitch",
                                "tools": [
                                    {"type": "function", "name": "watch_pr"}
                                ],
                            }
                        ]
                    },
                )
            ]
        )
        self.addCleanup(path.unlink, missing_ok=True)

        self.assertTrue(
            rollout.has_dynamic_tool(
                path,
                namespace="hitch",
                name="watch_pr",
            )
        )
        self.assertFalse(
            rollout.has_dynamic_tool(
                path,
                namespace="hitch",
                name="missing",
            )
        )

    def test_compacts_one_oversized_message_record(self) -> None:
        path = _write_rollout(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Prompt"},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "x" * (70 * 1024),
                        "phase": "commentary",
                    },
                ),
            ]
        )
        self.addCleanup(path.unlink, missing_ok=True)

        page = rollout.session_history_page(path, message_target=2)

        self.assertIsNotNone(page)
        assert page is not None
        self.assertEqual(page.flat_entries[0]["text"], "Prompt")
        self.assertEqual(
            page.flat_entries[1]["text"],
            "[Oversized message omitted from paged history.]",
        )
        self.assertEqual(page.flat_entries[1]["phase"], "commentary")


    def test_retains_active_user_boundary_beyond_message_target(self) -> None:
        lines = [
            _line("event_msg", {"type": "user_message", "message": "Old prompt"}),
            _line("event_msg", {"type": "agent_message", "message": "Old answer"}),
            _line("event_msg", {"type": "user_message", "message": "Active prompt"}),
        ]
        lines.extend(
            _line(
                "event_msg",
                {"type": "agent_message", "message": f"Active answer {index}"},
            )
            for index in range(5)
        )
        path = _write_rollout(lines)
        self.addCleanup(path.unlink, missing_ok=True)

        page = rollout.session_history_page(
            path,
            message_target=2,
            active_user_identity=rollout.SessionHistoryUserIdentity(
                text="Active prompt",
                prompt="Active prompt",
                started_at=1736078400,
            ),
        )

        self.assertIsNotNone(page)
        assert page is not None
        self.assertEqual(page.flat_entries[0]["kind"], "user")
        self.assertTrue(page.flat_entries[0]["_hitch_active_user"])
        older = rollout.session_history_page(
            path,
            before_offset=page.start_offset,
            message_target=2,
        )
        self.assertIsNotNone(older)
        assert older is not None
        self.assertEqual(
            [entry["text"] for entry in older.flat_entries],
            ["Old prompt", "Old answer"],
        )

    def test_matches_delayed_oversized_active_prompt_but_not_older_repeat(
        self,
    ) -> None:
        prompt = "Repeated active prompt " * 4000
        path = _write_rollout(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": prompt},
                    timestamp="2025-01-05T11:59:59.900Z",
                ),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": prompt},
                    timestamp="2025-01-05T12:00:01Z",
                ),
            ]
        )
        self.addCleanup(path.unlink, missing_ok=True)

        page = rollout.session_history_page(
            path,
            message_target=2,
            active_user_identity=rollout.SessionHistoryUserIdentity(
                text=prompt,
                prompt=prompt,
                started_at=1736078400.5,
            ),
        )

        self.assertIsNotNone(page)
        assert page is not None
        self.assertEqual(
            [entry["_hitch_active_user"] for entry in page.flat_entries],
            [False, True],
        )


class LatestModelConfigTests(TestCase):
    def test_reads_latest_turn_settings_across_reverse_read_chunks(self) -> None:
        path = _write_rollout(
            [
                _line(
                    "event_msg",
                    {
                        "type": "thread_settings_applied",
                        "thread_settings": {
                            "model": "gpt-5.5",
                            "reasoning_effort": "high",
                        },
                    },
                ),
                _line(
                    "turn_context",
                    {"model": "gpt-5.6-sol", "effort": "xhigh"},
                ),
                _line("response_item", {"text": "x" * (70 * 1024)}),
                "incomplete trailing record",
            ]
        )
        self.addCleanup(path.unlink, missing_ok=True)

        expected = rollout.SessionModelConfig(
            model="gpt-5.6-sol",
            reasoning_effort="xhigh",
        )
        self.assertEqual(rollout.latest_model_config(path), expected)
        detail = rollout.session_detail_data(path)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail.latest_model_config, expected)

    def test_reads_nested_collaboration_mode_settings(self) -> None:
        path = _write_rollout(
            [
                _line(
                    "turn_context",
                    {"model": "gpt-5.5", "effort": "xhigh"},
                ),
                _line(
                    "turn_context",
                    {
                        "collaborationMode": {
                            "mode": "plan",
                            "settings": {
                                "model": "gpt-5.6-sol",
                                "reasoning_effort": "medium",
                            },
                        }
                    },
                ),
            ]
        )
        self.addCleanup(path.unlink, missing_ok=True)

        expected = rollout.SessionModelConfig(
            model="gpt-5.6-sol",
            reasoning_effort="medium",
        )
        self.assertEqual(rollout.latest_model_config(path), expected)
        detail = rollout.session_detail_data(path)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail.latest_model_config, expected)


class LatestPrUrlTests(TestCase):
    @override
    def tearDown(self) -> None:
        for path in getattr(self, "_paths", []):
            path.unlink(missing_ok=True)

    def _make(self, lines: list[str]) -> Path:
        path = _write_rollout(lines)
        self._paths = [*getattr(self, "_paths", []), path]
        return path

    def test_completed_pr_turn_without_url_clears_earlier_url(self) -> None:
        stale_url = "https://github.com/cberner/hitch/pull/93"
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "invocation": {
                            "server": "github",
                            "tool": "_create_pull_request",
                        },
                        "result": {"url": stale_url},
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "Opened."}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-empty",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-empty",
                        "output": json.dumps({"content": []}),
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "No PR."}),
            ]
        )

        self.assertIsNone(rollout.latest_pr_url(path))
        self.assertIsNone(rollout.latest_pr_snapshot(path))

    def test_pr_snapshot_reads_mcp_prefixed_response_item_tool_name(self) -> None:
        url = "https://github.com/cberner/hitch/pull/96"
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "mcp__codex_apps__github_fetch_pr",
                        "arguments": json.dumps(
                            {"repo_full_name": "cberner/hitch", "pr_number": 96}
                        ),
                        "call_id": "call-pr",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps(
                            {"url": url, "state": "closed", "merged": False}
                        ),
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Closed."}],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        self.assertEqual(
            rollout.latest_pr_snapshot(path),
            {
                "url": url,
                "state": "closed",
                "merged": False,
                "repository_full_name": "cberner/hitch",
                "pr_number": 96,
                "source_tool": "fetch_pr",
            },
        )

    def test_unrelated_non_pr_ci_check_does_not_keep_pr_snapshot(self) -> None:
        url = "https://github.com/cberner/hitch/pull/98"
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _func_call("call-pr", None, name="github_fetch_pr"),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps(
                            {"url": url, "state": "open", "head_sha": "abc123"}
                        ),
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Opened."}],
                        "phase": "final_answer",
                    },
                ),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _func_call(
                    "call-ci",
                    json.dumps(
                        {
                            "repo_full_name": "cberner/hitch",
                            "commit_sha": "unrelated",
                        }
                    ),
                    name="github_get_commit_combined_status",
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-ci",
                        "output": json.dumps(
                            {"statuses": [{"state": "success"}]}
                        ),
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Plan."}],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        self.assertIsNone(rollout.latest_pr_snapshot(path))

    def test_ignores_non_github_pr_mcp_output_with_pr_url(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "invocation": {"server": "linear", "tool": "create_issue"},
                        "result": {"url": url},
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "Done."}),
            ]
        )

        self.assertIsNone(rollout.latest_pr_url(path))


class IterEntriesTests(TestCase):
    @override
    def tearDown(self) -> None:
        for path in getattr(self, "_paths", []):
            path.unlink(missing_ok=True)

    def _make(self, lines: list[str]) -> Path:
        path = _write_rollout(lines)
        self._paths = [*getattr(self, "_paths", []), path]
        return path

    def test_shell_function_call_with_argv_array_renders_joined_command(self) -> None:
        # Codex's `shell` tool spec (see `core/src/tools/handlers/shell_spec.rs`
        # in codex-rs) and `container.exec` both carry the command as an
        # argv-style array of strings, mirroring the payload `local_shell_call`
        # uses. The function-call path is the route OSS models -- including the
        # `qwen2.5-coder:0.5b` model the README's `just run_qwen` recipe
        # configures -- take when they cannot emit `local_shell_call`. Without
        # joining the parts, every shell invocation that flows through this
        # path surfaces as a `Command:` row with no detail at all, so the user
        # cannot see what the agent actually ran.
        for tool_name, arg_key in (
            ("shell", "command"),
            ("container.exec", "cmd"),
        ):
            with self.subTest(tool_name=tool_name):
                arguments = json.dumps({arg_key: ["bash", "-lc", "ls -la"]})
                path = self._make(
                    [
                        _line(
                            "response_item",
                            {
                                "type": "function_call",
                                "name": tool_name,
                                "arguments": arguments,
                                "call_id": f"call-{tool_name}",
                            },
                        )
                    ]
                )
                entries = list(rollout.iter_entries(path))
                self.assertEqual(len(entries), 1)
                entry = entries[0]
                self.assertEqual(entry["kind"], "tool_call")
                self.assertEqual(entry["type"], "commandExecution")
                self.assertEqual(entry["detail"], "bash -lc ls -la")

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

    def test_deduped_response_item_memory_citation_attaches_to_event(self) -> None:
        raw_text = (
            "hello"
            "<oai-mem-citation>"
            "<citation_entries>\nMEMORY.md:1-2|note=[x]\n</citation_entries>"
            "</oai-mem-citation>"
            " world"
        )
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": raw_text}],
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "hello world"}),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "hello world")
        self.assertEqual(entries[0]["memory_citation"]["count"], 1)
        self.assertEqual(entries[0]["memory_citation"]["entries"][0]["note"], "x")

    def test_completed_plan_item_suppresses_matching_plan_mode_response(self) -> None:
        plan = "# Fix it\n\nDo the thing."
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
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

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)

    def test_completed_plan_item_suppresses_matching_plan_mode_event(self) -> None:
        plan = "# Fix it\n\nDo the thing."
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {
                        "type": "item_completed",
                        "item": {"type": "Plan", "id": "turn-plan", "text": plan},
                    },
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": f"<proposed_plan>\n{plan}\n</proposed_plan>",
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)

    def test_markdown_proposed_plan_after_plan_mode_session_renders_plan(self) -> None:
        initial_plan = "# Initial Plan\n\nStart here."
        plan = "# Revised Plan\n\nUse the replacement plan block."
        tagged_plan = f"<proposed_plan>\n{plan}\n</proposed_plan>"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    f"<proposed_plan>\n{initial_plan}\n"
                                    "</proposed_plan>"
                                ),
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Revise it"},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": tagged_plan,
                        "phase": "final_answer",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": tagged_plan}],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(
            [(entry["kind"], entry["text"]) for entry in entries],
            [
                ("user", "Plan it"),
                ("plan", initial_plan),
                ("user", "Revise it"),
                ("plan", plan),
            ],
        )

    def test_literal_proposed_plan_example_after_plan_mode_discussion_stays_agent(
        self,
    ) -> None:
        text = (
            "<proposed_plan>\n# Plan XML Example\n\n"
            "1. literal step\n2. still an example\n</proposed_plan>"
        )
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Discuss it"}),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "This can work."}],
                        "phase": "final_answer",
                    },
                ),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Show the tags."},
                ),
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

        self.assertEqual(entries[-1]["kind"], "agent")
        self.assertEqual(entries[-1]["text"], text)

    def test_plan_mode_stays_active_until_approval(self) -> None:
        text = "<proposed_plan>\n# Plan\n\nImplement it.\n</proposed_plan>"
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {"type": "task_started", "collaboration_mode_kind": "plan"},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "No plan this time.",
                        "phase": "final_answer",
                    },
                ),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Turn that into a plan."},
                ),
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], "# Plan\n\nImplement it.")

    def test_approval_turn_treats_proposed_plan_markup_as_agent_text(self) -> None:
        text = "<proposed_plan>\n# XML Example\n\nliteral example\n</proposed_plan>"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "<proposed_plan>\n# Plan\n\nDo it\n</proposed_plan>",
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Implement the plan."},
                ),
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
        plan_entries = [entry for entry in entries if entry["kind"] == "plan"]

        self.assertEqual(len(plan_entries), 1)
        self.assertEqual(entries[-1]["kind"], "agent")
        self.assertEqual(entries[-1]["text"], text)

    def test_default_mode_turn_exits_active_plan_mode_without_approval(self) -> None:
        text = "<proposed_plan>\n# Plan\n\nImplement it.\n</proposed_plan>"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "No proposed plan yet.",
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "answer directly"},
                ),
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

        self.assertEqual(entries[-1]["kind"], "agent")
        self.assertEqual(entries[-1]["text"], text)

    def test_plan_request_after_active_plan_mode_renders_pending_plan(self) -> None:
        text = "<proposed_plan>\n# Plan\n\nImplement it.\n</proposed_plan>"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "No proposed plan yet.",
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {
                        "type": "user_message",
                        "message": "Give me the plan and I'll approve it",
                    },
                ),
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], "# Plan\n\nImplement it.")

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
                    "event_msg",
                    {
                        "type": "item_completed",
                        "item": {"type": "Plan", "text": 7},
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
            "look at this\n[image]\n[image]\n[image]",
        )
        self.assertNotIn("/tmp/c.png", entries[0]["text"])
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

    def test_token_usage_history_returns_timestamped_totals(self) -> None:
        path = self._make(
            [
                "{not json",
                _line(
                    "event_msg",
                    {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 20,
                                "output_tokens": 50,
                                "total_tokens": 170,
                            }
                        },
                    },
                    timestamp="2025-01-05T12:00:00Z",
                ),
                _line("event_msg", {"type": "user_message", "message": "x"}),
                _line("event_msg", {"type": "token_count"}),
                _line(
                    "event_msg",
                    {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 300,
                                "cached_input_tokens": True,
                                "output_tokens": 175,
                                "total_tokens": "ignored",
                            }
                        },
                    },
                    timestamp="2025-01-06T12:00:00Z",
                ),
                _line(
                    "event_msg",
                    {
                        "type": "token_count",
                        "info": {"total_token_usage": {"input_tokens": 999}},
                    },
                    timestamp="",
                ),
            ]
        )

        self.assertEqual(
            rollout.token_usage_history(path),
            [
                {
                    "timestamp": 1736078400,
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 50,
                    "total_tokens": 170,
                },
                {
                    "timestamp": 1736164800,
                    "input_tokens": 300,
                    "cached_input_tokens": 0,
                    "output_tokens": 175,
                    "total_tokens": 0,
                },
            ],
        )

    def test_token_usage_history_returns_empty_when_unreadable(self) -> None:
        self.assertEqual(rollout.token_usage_history(Path("/nonexistent/rollout.jsonl")), [])

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


class ProposedPlanTextTests(TestCase):
    def test_missing_tags_return_none(self) -> None:
        self.assertIsNone(rollout.proposed_plan_text("just some agent reply"))
