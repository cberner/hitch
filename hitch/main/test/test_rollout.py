import json
import tempfile
from pathlib import Path
from typing import Any, override

from django.test import TestCase

from hitch.main import rollout, system_agents


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


class LatestPrUrlTests(TestCase):
    @override
    def tearDown(self) -> None:
        for path in getattr(self, "_paths", []):
            path.unlink(missing_ok=True)

    def _make(self, lines: list[str]) -> Path:
        path = _write_rollout(lines)
        self._paths = [*getattr(self, "_paths", []), path]
        return path

    def test_detects_pr_url_from_github_function_call_output(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
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
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": url}),
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
            ]
        )

        self.assertEqual(rollout.latest_pr_url(path), url)

    def test_detects_pr_url_when_function_call_output_follows_final_message(
        self,
    ) -> None:
        # OpenAI's Responses API can return a function_call and an assistant
        # final-answer message in a single output (the "issue the call and
        # narrate it" pattern). The Codex SDK records response items in
        # arrival order, so the function_call_output -- written when the tool
        # actually completes -- lands in the rollout AFTER the final-answer
        # message for the same turn. The session-page PR pill reads the URL
        # from this output, so dropping it leaves the rendered turn with no
        # link to the PR the user just opened.
        url = "https://github.com/cberner/hitch/pull/95"
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
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Opening the PR now."}
                        ],
                        "phase": "final_answer",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps({"url": url}),
                    },
                ),
            ]
        )

        self.assertEqual(rollout.latest_pr_url(path), url)

    def test_pr_snapshot_when_function_call_output_follows_final_message(
        self,
    ) -> None:
        # Same Responses-API shape as the URL test above, but checked against
        # ``latest_pr_snapshot``: the session stage and any closed/merged
        # state read this snapshot, so dropping the post-final
        # ``function_call_output`` leaves the page showing a PR link with no
        # identity in the snapshot -- the stage falls back to
        # ``IMPLEMENTATION`` instead of ``PR``/``DONE_*`` and the derived
        # stage cache stores the wrong value.
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
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Closed it."}],
                        "phase": "final_answer",
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
            ]
        )

        snapshot = rollout.latest_pr_snapshot(path)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["url"], url)
        self.assertEqual(snapshot["state"], "closed")
        self.assertIs(snapshot["merged"], False)

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

    def test_pr_snapshot_reads_ok_wrapped_mcp_result(self) -> None:
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
                        "invocation": {
                            "server": "github",
                            "tool": "_create_pull_request",
                        },
                        "result": {
                            "Ok": {"url": url, "state": "closed", "merged": False}
                        },
                    },
                ),
                _line("event_msg", {"type": "agent_message", "message": "Closed."}),
            ]
        )

        self.assertEqual(
            rollout.latest_pr_snapshot(path),
            {
                "url": url,
                "state": "closed",
                "merged": False,
                "repository_full_name": "cberner/hitch",
                "pr_number": 94,
                "source_tool": "create_pull_request",
            },
        )

    def test_pr_snapshot_reads_ok_wrapped_function_call_output_string(self) -> None:
        url = "https://github.com/cberner/hitch/pull/95"
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
                        "name": "github_create_pull_request",
                        "arguments": "{}",
                        "call_id": "call-pr",
                    },
                ),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-pr",
                        "output": json.dumps(
                            {"Ok": {"url": url, "state": "closed", "merged": False}}
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
                "pr_number": 95,
                "source_tool": "create_pull_request",
            },
        )

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

    def test_pr_snapshot_clears_after_later_non_pr_work_turn(self) -> None:
        url = "https://github.com/cberner/hitch/pull/97"
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
                        "name": "github_create_pull_request",
                        "arguments": "{}",
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
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
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

    def test_ignores_shell_output_with_pr_url(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "user_message", "message": system_agents.PR_SLASH_PROMPT},
                ),
                _func_call("call-gh-view", "gh pr view"),
                _line(
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": "call-gh-view",
                        "output": url,
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

    def test_invalid_utf8_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(
            prefix="rollout-",
            suffix=".jsonl",
            delete=False,
        ) as tmp:
            tmp.write(b"\xff\xfe\x00not json\n")
            path = Path(tmp.name)
        self._paths = [*getattr(self, "_paths", []), path]

        self.assertEqual(list(rollout.iter_entries(path)), [])

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

    def test_response_item_memory_citation_is_stripped_and_parsed(self) -> None:
        text = (
            "hello"
            "<oai-mem-citation>"
            "<citation_entries>\n"
            "MEMORY.md:1-2|note=[summary]\n"
            "</citation_entries>\n"
            "<rollout_ids>\n"
            "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4\n"
            "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4\n"
            "019cc2ea-1dff-7902-8d40-c8f6e5d83cc5\n"
            "</rollout_ids>"
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
                        "content": [{"type": "output_text", "text": text}],
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "hello world")
        self.assertNotIn("oai-mem-citation", entries[0]["text"])
        self.assertEqual(
            entries[0]["memory_citation"],
            {
                "count": 3,
                "entries": [
                    {
                        "path": "MEMORY.md",
                        "line_start": 1,
                        "line_end": 2,
                        "note": "summary",
                    }
                ],
                "thread_ids": [
                    "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4",
                    "019cc2ea-1dff-7902-8d40-c8f6e5d83cc5",
                ],
            },
        )

    def test_memory_citation_count_sums_entries_and_thread_ids(self) -> None:
        # The popover in _session_entry.html renders both ``entries`` and
        # ``thread_ids`` under a single "Memories used: N" summary, so the
        # count has to cover both kinds. A citation that mixes specific
        # file-line entries with prior-session ids previously rendered as
        # "Memories used: 2" while the popover expanded to five rows.
        text = (
            "answer"
            "<oai-mem-citation>"
            "<citation_entries>\n"
            "MEMORY.md:1-2|note=[a]\n"
            "MEMORY.md:5-9|note=[b]\n"
            "</citation_entries>\n"
            "<rollout_ids>\n"
            "thread-aaa\n"
            "thread-bbb\n"
            "thread-ccc\n"
            "</rollout_ids>"
            "</oai-mem-citation>"
        )
        path = self._make(
            [
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(len(entries), 1)
        citation = entries[0]["memory_citation"]
        self.assertEqual(len(citation["entries"]), 2)
        self.assertEqual(len(citation["thread_ids"]), 3)
        self.assertEqual(
            citation["count"],
            len(citation["entries"]) + len(citation["thread_ids"]),
        )

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

    def test_completed_plan_item_suppresses_matching_proposed_plan_message(self) -> None:
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
        self.assertEqual(entries[0]["kind"], "plan")
        self.assertEqual(entries[0]["text"], plan)

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

    def test_plan_mode_proposed_plan_without_completed_item_renders_plan(self) -> None:
        plan = "# Fallback Plan\n\nUse the final proposed plan block."
        path = self._make(
            [
                _line(
                    "turn_context",
                    {"collaboration_mode": {"mode": "plan"}},
                ),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
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
        self.assertEqual(entries[1]["kind"], "plan")
        self.assertEqual(entries[1]["text"], plan)

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

    def test_plan_like_proposed_plan_after_plan_mode_discussion_renders_plan(
        self,
    ) -> None:
        plan = (
            "**AG Proposal Rollout Flow**\n\n"
            "**Summary**\n"
            "- Draft the implementation in a hidden managed worktree.\n\n"
            "**Key Changes**\n"
            "- Link the inbox proposal to the hidden session.\n\n"
            "**Test Plan**\n"
            "- Run the focused view tests."
        )
        tagged_plan = f"<proposed_plan>\n{plan}\n</proposed_plan>"
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
                    {"type": "user_message", "message": "Turn that into the plan."},
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)
        self.assertFalse(any("<proposed_plan>" in entry.get("text", "") for entry in entries))

    def test_numbered_proposed_plan_after_plan_mode_discussion_renders_plan(
        self,
    ) -> None:
        plan = "1. Step one\n2. Step two"
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
                    {"type": "user_message", "message": "Turn that into steps."},
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)

    def test_simple_proposed_plan_after_plan_mode_discussion_renders_plan(
        self,
    ) -> None:
        plan = "# Plan\n\nImplement it."
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
                    {"type": "user_message", "message": "Turn that into a plan."},
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)

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

    def test_proposed_plan_with_tag_substring_heading_is_recognized_as_plan(
        self,
    ) -> None:
        # ``_looks_like_literal_plan_example`` uses the literal-example markers
        # ("example", "xml", "tag", "syntax", "literal") to reject literal
        # ``<proposed_plan>`` examples on the plan-mode followup turn. Matching
        # those markers as substrings (not as whole words) also rejects real
        # plans whose heading happens to contain "tag" inside an unrelated word
        # -- "Tagging strategy", "Stage rollout", "Vintage cleanup" -- so the
        # rollout view downgrades them to plain ``agent`` entries and the
        # auto-review gate (``_completed_turn_has_pending_proposed_plan``) lets
        # auto-PR/auto-QA proceed without surfacing the plan for approval.
        plan = (
            "# Tagging strategy\n\n"
            "1. Add tag CRUD endpoints\n"
            "2. Update docs"
        )
        text = f"<proposed_plan>\n{plan}\n</proposed_plan>"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Discuss it"}),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Sounds good."}],
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Plan the tagging work."},
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
        self.assertEqual(entries[-1]["text"], plan)
        self.assertTrue(rollout.entries_await_plan_approval(entries))

    def test_pending_plan_survives_trailing_commentary_in_flat_entries(self) -> None:
        # ``entries_await_plan_approval`` runs against BOTH the collapsed
        # session-view entries and the raw, un-collapsed rollout entries
        # (session-list stage in ``views`` and the auto-review gate
        # ``_completed_turn_has_pending_proposed_plan`` in ``system_agents``).
        # Codex commonly narrates after emitting the plan with a
        # ``commentary``-phase ``agent_message``. The collapse step folds that
        # narration into a skipped ``thinking`` entry, but the raw stream keeps
        # it as ``kind="agent"``. If that intermediate commentary were treated
        # as the turn's final reply, the gate would conclude the plan is no
        # longer pending and auto-PR/auto-QA would fire on a plan the user has
        # not approved yet.
        plan = "# Plan\n\n1. Do step one\n2. Do step two"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {"type": "item_completed", "item": {"type": "plan", "text": plan}},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": f"<proposed_plan>\n{plan}\n</proposed_plan>",
                        "phase": "final_answer",
                    },
                ),
                # Reading-tool narration plus a closing commentary remark, both
                # of which are intermediate and must not resolve the plan.
                _line(
                    "event_msg",
                    {
                        "type": "mcp_tool_call_end",
                        "invocation": {"server": "fs", "tool": "read"},
                        "result": {},
                    },
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "Let me know if you'd like any changes.",
                        "phase": "commentary",
                    },
                ),
            ]
        )

        flat_entries = list(rollout.iter_entries(path))

        # Sanity-check the shape this regression targets: a commentary agent
        # entry trails the plan in the raw stream.
        self.assertEqual(flat_entries[-1]["kind"], "agent")
        self.assertEqual(flat_entries[-1]["phase"], "commentary")
        self.assertTrue(any(entry["kind"] == "plan" for entry in flat_entries))

        pending = rollout.pending_plan_entry(flat_entries)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending["text"], plan)
        self.assertTrue(rollout.entries_await_plan_approval(flat_entries))

    def test_final_agent_reply_after_plan_resolves_pending_plan(self) -> None:
        # The mirror of the commentary case: a genuine final-answer agent reply
        # after the plan means the agent moved on, so the plan is no longer
        # pending. Guards against the commentary skip over-reaching.
        plan = "# Plan\n\n1. Do step one\n2. Do step two"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {"type": "item_completed", "item": {"type": "plan", "text": plan}},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": f"<proposed_plan>\n{plan}\n</proposed_plan>",
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Implement the plan."},
                ),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "Implemented and tests pass.",
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        flat_entries = list(rollout.iter_entries(path))

        self.assertFalse(rollout.entries_await_plan_approval(flat_entries))

    def test_final_proposed_plan_replaces_streamed_draft_plan(self) -> None:
        draft_plan = "# Draft Plan\n\nStill streaming."
        final_plan = "# Final Plan\n\nUse the canonical response item."
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": (
                            f"<proposed_plan>\n{draft_plan}\n</proposed_plan>"
                        ),
                        "phase": "final_answer",
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
                                "text": (
                                    f"<proposed_plan>\n{final_plan}\n</proposed_plan>"
                                ),
                            }
                        ],
                        "phase": "final_answer",
                    },
                ),
            ]
        )

        entries = list(rollout.iter_entries(path))

        self.assertEqual(
            [(entry["kind"], entry["text"]) for entry in entries],
            [("user", "Plan it"), ("plan", final_plan)],
        )

    def test_numbered_proposed_plan_after_pending_plan_renders_plan(self) -> None:
        initial_plan = "# Initial Plan\n\nStart here."
        plan = "1. Step one\n2. Step two"
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
                    {"type": "user_message", "message": "Use numbered steps"},
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)

    def test_task_started_after_user_marks_current_plan_turn(self) -> None:
        plan = "1. Step one\n2. Step two"
        path = self._make(
            [
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line(
                    "event_msg",
                    {"type": "task_started", "collaboration_mode_kind": "plan"},
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)

    def test_task_started_before_user_marks_next_plan_turn(self) -> None:
        plan = "1. Step one\n2. Step two"
        path = self._make(
            [
                _line(
                    "event_msg",
                    {"type": "task_started", "collaboration_mode_kind": "plan"},
                ),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
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

        self.assertEqual(entries[-1]["kind"], "plan")
        self.assertEqual(entries[-1]["text"], plan)

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

    def test_plan_mode_followup_fallback_expires_after_one_turn(self) -> None:
        text = "<proposed_plan>\n# Plan XML Example\n\nliteral example\n</proposed_plan>"
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
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line("event_msg", {"type": "user_message", "message": "Thanks"}),
                _line(
                    "response_item",
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Show the tags"},
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

    def test_response_item_after_pending_plan_resolves_plan_state(self) -> None:
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
                        "content": [{"type": "output_text", "text": "Implemented."}],
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Show the tags"},
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

    def test_non_markdown_proposed_plan_after_plan_mode_session_stays_agent_text(
        self,
    ) -> None:
        text = "<proposed_plan>\nliteral XML example\n</proposed_plan>"
        path = self._make(
            [
                _line("turn_context", {"collaboration_mode": {"mode": "plan"}}),
                _line("event_msg", {"type": "user_message", "message": "Plan it"}),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Show the tags"},
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

    def test_markdown_proposed_plan_after_plan_is_no_longer_pending_stays_agent_text(
        self,
    ) -> None:
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
                    "event_msg",
                    {
                        "type": "agent_message",
                        "message": "Implemented.",
                        "phase": "final_answer",
                    },
                ),
                _line("turn_context", {"collaboration_mode": {"mode": "default"}}),
                _line(
                    "event_msg",
                    {"type": "user_message", "message": "Show the tags"},
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
        # is the running session total and info.last_token_usage is the active
        # context size. The last such event wins; earlier ones are obsolete.
        # Non-token_count events and lines with malformed info blocks are
        # ignored without raising.
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
                            "last_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 20,
                                "output_tokens": 50,
                                "reasoning_output_tokens": 5,
                                "total_tokens": 175,
                            },
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
                            "last_token_usage": {
                                "input_tokens": 150,
                                "cached_input_tokens": 40,
                                "output_tokens": 100,
                                "reasoning_output_tokens": 5,
                                "total_tokens": 22038,
                            },
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
            {
                "input_tokens": 300,
                "cached_input_tokens": 80,
                "output_tokens": 175,
                "total_tokens": 565,
                "context_tokens": 22038,
                "model_context_window": 200000,
            },
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
