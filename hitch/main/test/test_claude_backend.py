"""Coverage for the Claude Code backend: event translation, option mapping,
the in-process propose_session tool, worker-command routing, and the
``spawn_new_session`` Claude path that mints a local session instead of calling
the Codex app-server.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from django.test import TestCase, override_settings

from hitch.main import claude_options, claude_translate, codex_pool, coding_agents
from hitch.main.models import CodexInstance


def _assistant(*blocks: Any, message_id: str = "m1") -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude", message_id=message_id)


class EventTranslatorTests(TestCase):
    def test_text_block_yields_started_and_completed_agent_message(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(_assistant(TextBlock(text="hello")))
        self.assertEqual(
            [method for method, _ in events],
            ["item/started", "item/completed"],
        )
        started, completed = (payload for _, payload in events)
        self.assertEqual(started["item"]["type"], "agentMessage")
        self.assertEqual(completed["item"]["text"], "hello")
        self.assertEqual(started["item"]["id"], completed["item"]["id"])

    def test_thinking_block_maps_to_reasoning_commentary(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(_assistant(ThinkingBlock(thinking="hmm", signature="s")))
        _, completed = events[1]
        self.assertEqual(completed["item"]["type"], "reasoning")
        self.assertEqual(completed["item"]["phase"], "commentary")
        self.assertEqual(completed["item"]["text"], "hmm")

    def test_bash_tool_opens_command_item_and_result_closes_it(self) -> None:
        translator = claude_translate.EventTranslator()
        opened = translator.translate(
            _assistant(ToolUseBlock(id="t1", name="Bash", input={"command": "ls -la"}))
        )
        self.assertEqual(opened[0][0], "item/started")
        item = opened[0][1]["item"]
        self.assertEqual(item["type"], "commandExecution")
        self.assertEqual(item["command"], "ls -la")

        closed = translator.translate(
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="done")])
        )
        self.assertEqual(closed[0][0], "item/completed")
        completed_item = closed[0][1]["item"]
        self.assertEqual(completed_item["status"], "completed")
        self.assertEqual(completed_item["result"], "done")

    def test_edit_tool_maps_to_file_change_with_path(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(
            _assistant(ToolUseBlock(id="e1", name="Edit", input={"file_path": "/repo/a.py"}))
        )
        item = events[0][1]["item"]
        self.assertEqual(item["type"], "fileChange")
        self.assertEqual(item["changes"], [{"path": "/repo/a.py"}])

    def test_error_tool_result_marks_item_failed(self) -> None:
        translator = claude_translate.EventTranslator()
        translator.translate(
            _assistant(ToolUseBlock(id="t1", name="Bash", input={"command": "x"}))
        )
        closed = translator.translate(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t1", content="boom", is_error=True)]
            )
        )
        self.assertEqual(closed[0][1]["item"]["status"], "failed")

    def test_mcp_tool_call_records_server_and_tool(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(
            _assistant(
                ToolUseBlock(id="g1", name="mcp__github__create_pr", input={"title": "x"})
            )
        )
        item = events[0][1]["item"]
        self.assertEqual(item["type"], "mcpToolCall")
        self.assertEqual(item["server"], "github")
        self.assertEqual(item["tool"], "create_pr")

    def test_todo_write_becomes_task_plan_update(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(
            _assistant(
                ToolUseBlock(
                    id="p1",
                    name="TodoWrite",
                    input={
                        "todos": [
                            {"content": "step one", "status": "in_progress"},
                            {"content": "step two", "status": "pending"},
                            {"content": "", "status": "completed"},
                        ]
                    },
                )
            )
        )
        self.assertEqual(len(events), 1)
        method, payload = events[0]
        self.assertEqual(method, "turn/plan/updated")
        self.assertEqual(
            payload["plan"],
            [
                {"step": "step one", "status": "inProgress"},
                {"step": "step two", "status": "pending"},
            ],
        )

    def test_exit_plan_mode_emits_plan_item(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(
            _assistant(ToolUseBlock(id="x1", name="ExitPlanMode", input={"plan": "# Plan"}))
        )
        self.assertEqual([m for m, _ in events], ["item/started", "item/completed"])
        self.assertEqual(events[1][1]["item"]["type"], "plan")
        self.assertEqual(events[1][1]["item"]["text"], "# Plan")

    def test_unknown_tool_result_is_ignored(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(
            UserMessage(content=[ToolResultBlock(tool_use_id="missing", content="x")])
        )
        self.assertEqual(events, [])

    def test_string_user_message_renders_as_user_item(self) -> None:
        translator = claude_translate.EventTranslator()
        events = translator.translate(UserMessage(content="hi there"))
        self.assertEqual(events[1][1]["item"]["type"], "userMessage")
        self.assertEqual(events[1][1]["item"]["text"], "hi there")


class ClaudeOptionsTests(TestCase):
    def test_plan_mode_takes_precedence(self) -> None:
        self.assertEqual(
            claude_options.resolve_permission_mode(
                plan_mode=True, sandbox_policy="dangerFullAccess", approval_mode="approve_all"
            ),
            "plan",
        )

    def test_approve_all_maps_to_bypass(self) -> None:
        self.assertEqual(
            claude_options.resolve_permission_mode(
                plan_mode=False, sandbox_policy=None, approval_mode="approve_all"
            ),
            "bypassPermissions",
        )

    def test_danger_full_access_keeps_approval_gate(self) -> None:
        # Full filesystem access must not silently bypass the approval gate.
        self.assertEqual(
            claude_options.resolve_permission_mode(
                plan_mode=False, sandbox_policy="dangerFullAccess", approval_mode="prompt_user"
            ),
            "default",
        )

    def test_web_disabled_blocks_webfetch_too(self) -> None:
        allowed, disallowed = claude_options.resolve_tool_lists(
            sandbox_policy=None, web_search_mode="disabled"
        )
        self.assertIn("WebFetch", disallowed)
        self.assertNotIn("WebFetch", allowed)

    def test_default_permission_mode(self) -> None:
        self.assertEqual(
            claude_options.resolve_permission_mode(
                plan_mode=False, sandbox_policy="workspaceWrite", approval_mode="prompt_user"
            ),
            "default",
        )

    def test_read_only_sandbox_disallows_write_tools_and_bash(self) -> None:
        allowed, disallowed = claude_options.resolve_tool_lists(
            sandbox_policy="readOnly", web_search_mode=None
        )
        self.assertIn("Read", allowed)
        for tool in claude_options.WRITE_TOOLS:
            self.assertIn(tool, disallowed)
        # A shell command can mutate the workspace, so read-only must block Bash.
        self.assertIn("Bash", disallowed)
        self.assertNotIn("Bash", allowed)

    def test_web_search_disabled_blocks_tool(self) -> None:
        allowed, disallowed = claude_options.resolve_tool_lists(
            sandbox_policy=None, web_search_mode="disabled"
        )
        self.assertIn("WebSearch", disallowed)
        self.assertNotIn("WebSearch", allowed)

    def test_map_effort_filters_unknown(self) -> None:
        self.assertEqual(claude_options.map_effort("high"), "high")
        self.assertIsNone(claude_options.map_effort("minimal"))
        self.assertIsNone(claude_options.map_effort(None))

    @patch("hitch.main.claude_options.claude_bin", return_value="/usr/bin/claude")
    def test_build_options_first_run_fixes_session_id(self, _bin: MagicMock) -> None:
        options = claude_options.build_options(
            cwd="/repo",
            model="claude-opus-4-8",
            plan_mode=True,
            session_id="abc123",
        )
        self.assertEqual(options.session_id, "abc123")
        self.assertIsNone(options.resume)
        self.assertEqual(options.permission_mode, "plan")
        self.assertEqual(options.cli_path, "/usr/bin/claude")

    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_build_options_resume_prefers_resume_over_session_id(self, _bin: MagicMock) -> None:
        options = claude_options.build_options(
            cwd="/repo",
            model="claude-opus-4-8",
            resume_session_id="sess-9",
            session_id="ignored",
            output_schema={"type": "object"},
            base_instructions="be terse",
        )
        self.assertEqual(options.resume, "sess-9")
        self.assertIsNone(options.session_id)
        self.assertEqual(options.output_format, {"type": "object"})
        self.assertIsInstance(options.system_prompt, dict)


    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_build_options_with_mcp_server_and_schema_and_hook(self, _bin: MagicMock) -> None:
        from hitch.main import claude_tools

        server = claude_tools.build_hitch_mcp_server(cwd="/repo", thread_id="t")
        options = claude_options.build_options(
            cwd="/repo",
            model="claude-opus-4-8",
            output_schema={"type": "object"},
            approval_mode="prompt_user",
            mcp_server=server,
            can_use_tool=lambda *a: None,
        )
        self.assertIn(claude_tools.PROPOSE_SESSION_TOOL_NAME, options.allowed_tools)
        assert isinstance(options.mcp_servers, dict)
        self.assertIn(claude_tools.HITCH_MCP_SERVER_NAME, options.mcp_servers)
        self.assertEqual(options.output_format, {"type": "object"})
        # can_use_tool requires the companion PreToolUse hook.
        self.assertIn("PreToolUse", options.hooks or {})


class ProposeSessionHandlerTests(TestCase):
    def test_relevant_files_must_be_a_list(self) -> None:
        from hitch.main import claude_tools

        result = claude_tools._run_propose_session(
            {"title": "t", "summary": "s", "prompt": "p", "relevant_files": "x"},
            "/repo",
            "thread",
        )
        self.assertTrue(result["is_error"])
        self.assertIn("relevant_files", result["content"][0]["text"])

    def test_unknown_cwd_project_is_reported(self) -> None:
        from hitch.main import claude_tools

        result = claude_tools._run_propose_session(
            {"title": "t", "summary": "s", "prompt": "p"}, "/no/such/repo", "thread"
        )
        self.assertTrue(result["is_error"])


class WorkerArgvRoutingTests(TestCase):
    def test_claude_backend_selects_claude_worker(self) -> None:
        argv = codex_pool._worker_argv(
            instance_id=7,
            model="claude-opus-4-8",
            plan_mode=True,
            enable_memories=True,
            collaboration_mode="default",
            backend=CodexInstance.BACKEND_CLAUDE,
        )
        self.assertIn("claude_worker", argv)
        self.assertNotIn("codex_worker", argv)
        self.assertIn("--plan-mode", argv)
        # Codex-only knobs must not be forwarded to the Claude worker.
        self.assertNotIn("--enable-memories", argv)
        self.assertNotIn("--collaboration-mode", argv)

    def test_codex_backend_default_command(self) -> None:
        argv = codex_pool._worker_argv(instance_id=3)
        self.assertIn("codex_worker", argv)


class SpawnClaudeSessionTests(TestCase):
    @patch("hitch.main.codex_pool._launch_worker_process")
    @patch("hitch.main.codex_pool.Codex")
    def test_claude_spawn_skips_codex_and_mints_local_session(
        self, mock_codex: MagicMock, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = codex_pool.WorkerLaunch(pid=1234)
        with (
            tempfile.TemporaryDirectory() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="do the thing",
                backend=CodexInstance.BACKEND_CLAUDE,
            )
        # The Codex app-server is never contacted for a Claude session.
        mock_codex.assert_not_called()
        self.assertEqual(instance.backend, CodexInstance.BACKEND_CLAUDE)
        self.assertTrue(instance.thread_id)
        self.assertEqual(instance.pid, 1234)
        # The worker is launched with the Claude backend forwarded.
        self.assertEqual(mock_launch.call_args.kwargs["backend"], CodexInstance.BACKEND_CLAUDE)

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_spawn_turn_inherits_backend_and_session_id(self, mock_launch: MagicMock) -> None:
        mock_launch.return_value = codex_pool.WorkerLaunch(pid=1)
        with (
            tempfile.TemporaryDirectory() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            CodexInstance.objects.create(
                thread_id="thread-c",
                cwd="/repo",
                prompt="first",
                events_path="x",
                pid=0,
                status=CodexInstance.STATUS_COMPLETED,
                backend=CodexInstance.BACKEND_CLAUDE,
                claude_session_id="sess-42",
            )
            instance = codex_pool.spawn_turn(
                thread_id="thread-c", cwd="/repo", prompt="next"
            )
        self.assertEqual(instance.backend, CodexInstance.BACKEND_CLAUDE)
        self.assertEqual(instance.claude_session_id, "sess-42")


class ApprovalRoutingTests(TestCase):
    def test_every_tool_maps_to_an_approval_method(self) -> None:
        from hitch.main.management.commands import claude_worker

        self.assertEqual(
            claude_worker._approval_method("Bash"),
            claude_worker._COMMAND_APPROVAL_METHOD,
        )
        self.assertEqual(
            claude_worker._approval_method("Edit"),
            claude_worker._FILE_APPROVAL_METHOD,
        )
        # An unknown / MCP tool must still be gated, not auto-allowed.
        self.assertEqual(
            claude_worker._approval_method("mcp__github__create_pr"),
            claude_worker._TOOL_APPROVAL_METHOD,
        )

    def test_generic_tool_approval_params_carry_tool_name(self) -> None:
        from hitch.main.management.commands import claude_worker

        params = claude_worker._approval_params(
            claude_worker._TOOL_APPROVAL_METHOD,
            "mcp__github__create_pr",
            {"title": "x"},
        )
        self.assertEqual(params["tool"], "mcp__github__create_pr")
        self.assertEqual(params["item"]["type"], "toolCall")


class SessionIndexInvalidationTests(TestCase):
    def test_codex_refresh_does_not_invalidate_claude_sessions(self) -> None:
        from django.utils import timezone

        from hitch.main import session_index
        from hitch.main.models import SessionMetadata

        now = timezone.now()
        # A Claude session: local metadata + a claude-backed instance.
        SessionMetadata.objects.create(
            thread_id="claude-thread",
            cwd="/repo",
            codex_updated_at=now,
            codex_archived=False,
        )
        CodexInstance.objects.create(
            thread_id="claude-thread",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=CodexInstance.BACKEND_CLAUDE,
        )
        # A Codex session absent from the refresh's seen set should be invalidated.
        SessionMetadata.objects.create(
            thread_id="codex-thread",
            cwd="/repo",
            codex_updated_at=now,
            codex_archived=False,
        )
        session_index._invalidate_absent_source_rows(
            archived=False, seen_thread_ids=set()
        )
        claude_row = SessionMetadata.objects.get(thread_id="claude-thread")
        codex_row = SessionMetadata.objects.get(thread_id="codex-thread")
        self.assertIsNotNone(claude_row.codex_updated_at)  # preserved
        self.assertIsNone(codex_row.codex_updated_at)  # invalidated


class ProviderBackendTests(TestCase):
    def test_backend_mapping(self) -> None:
        self.assertEqual(
            coding_agents.backend_for_provider(coding_agents.PROVIDER_CLAUDE),
            coding_agents.BACKEND_CLAUDE,
        )
        self.assertEqual(
            coding_agents.backend_for_provider(coding_agents.PROVIDER_CODEX),
            coding_agents.BACKEND_CODEX,
        )
        self.assertEqual(
            coding_agents.backend_for_provider("bogus"),
            coding_agents.BACKEND_CODEX,
        )


class ProposeSessionToolTests(TestCase):
    def test_build_server_exposes_propose_session(self) -> None:
        from hitch.main import claude_tools

        server = claude_tools.build_hitch_mcp_server(cwd="/repo", thread_id="t")
        self.assertEqual(server["name"], claude_tools.HITCH_MCP_SERVER_NAME)
        self.assertEqual(
            claude_tools.PROPOSE_SESSION_TOOL_NAME, "mcp__hitch__propose_session"
        )


def _result(subtype: str = "success", is_error: bool = False) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="sess-final",
    )


class SessionEntriesTests(TestCase):
    def _write(self, lines: list[dict[str, Any]]) -> str:
        import json
        import tempfile
        from pathlib import Path

        tmp_dir = tempfile.mkdtemp()
        path = Path(tmp_dir) / "events.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps(line) + "\n")
        return str(path)

    def test_entries_cover_user_agent_and_tool_items(self) -> None:
        from hitch.main import claude_session_entries

        def completed(item: dict[str, Any], recorded_at: int = 2_000_000) -> dict[str, Any]:
            return {
                "method": "item/completed",
                "payload": {"item": item},
                "recordedAt": recorded_at,
            }

        path = self._write(
            [
                completed({"id": "u", "type": "userMessage", "text": "hi"}),
                completed({"id": "a", "type": "agentMessage", "text": "hello"}),
                completed(
                    {
                        "id": "c",
                        "type": "commandExecution",
                        "command": "ls",
                        "status": "completed",
                    }
                ),
                completed(
                    {
                        "id": "f",
                        "type": "fileChange",
                        "changes": [{"path": "a.py"}, {"path": "b.py"}],
                        "status": "failed",
                    }
                ),
                {"method": "item/started", "payload": {"item": {"id": "x"}}},
            ]
        )
        entries = claude_session_entries.session_entries(path)
        kinds = [e["kind"] for e in entries]
        self.assertEqual(kinds, ["user", "agent", "tool_call", "tool_call"])
        self.assertEqual(entries[0]["timestamp"], 2)
        self.assertEqual(entries[2]["type"], "commandExecution")
        self.assertEqual(entries[2]["detail"], "ls")
        self.assertIsNone(entries[2]["status"])  # 'completed' is not surfaced
        self.assertEqual(entries[3]["detail"], "a.py, b.py")
        self.assertEqual(entries[3]["status"], "failed")

    def test_missing_file_returns_empty(self) -> None:
        from hitch.main import claude_session_entries

        self.assertEqual(claude_session_entries.session_entries("/no/such/file"), [])


class ImageBlockTests(TestCase):
    def test_builds_base64_blocks_and_skips_unknown(self) -> None:
        import tempfile
        from pathlib import Path

        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "a.png"
            png.write_bytes(b"\x89PNG\r\n")
            txt = Path(tmp) / "b.txt"
            txt.write_bytes(b"nope")
            blocks = claude_worker._image_content_blocks(
                [str(png), str(txt), "/missing.png", 5]
            )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[0]["source"]["media_type"], "image/png")
        self.assertTrue(blocks[0]["source"]["data"])


class AnthropicProxyTranslationTests(TestCase):
    """Locally-validatable translation logic for the integration-test proxy."""

    def test_flattens_system_and_message_blocks(self) -> None:
        from hitch.main.test import anthropic_ollama_proxy as proxy

        messages = proxy.anthropic_to_openai_messages(
            {
                "system": [{"type": "text", "text": "sys"}, {"type": "image"}],
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                ],
            }
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_sse_has_anthropic_event_sequence(self) -> None:
        from hitch.main.test import anthropic_ollama_proxy as proxy

        body = proxy.anthropic_sse("HELLO", "claude-x").decode()
        for event in (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ):
            self.assertIn(f"event: {event}", body)
        self.assertIn("HELLO", body)

    @patch("hitch.main.test.anthropic_ollama_proxy._call_ollama", return_value="HELLO")
    def test_proxy_server_returns_sse_for_messages(self, _ollama: MagicMock) -> None:
        import urllib.request

        from hitch.main.test.anthropic_ollama_proxy import AnthropicOllamaProxy

        with AnthropicOllamaProxy() as proxy:
            req = urllib.request.Request(
                f"{proxy.base_url}/v1/messages",
                data=b'{"model":"claude-x","messages":[{"role":"user","content":"hi"}]}',
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                body = resp.read().decode()
        self.assertIn("event: message_stop", body)
        self.assertIn("HELLO", body)


class SteerInputTests(TestCase):
    def test_steer_request_extracts_text_and_image_paths(self) -> None:
        import json

        from hitch.main.management.commands import claude_worker

        raw = json.dumps(
            {"op": "steer", "input": " go ", "inputImagePaths": ["/a.png", "", 3]}
        ).encode()
        text, paths = claude_worker._steer_request(raw)
        self.assertEqual(text, "go")
        self.assertEqual(paths, ["/a.png"])

    def test_steer_request_ignores_non_steer(self) -> None:
        import json

        from hitch.main.management.commands import claude_worker

        self.assertEqual(
            claude_worker._steer_request(json.dumps({"op": "other"}).encode()),
            ("", []),
        )

    def test_build_query_input_plain_text_without_images(self) -> None:
        from hitch.main.management.commands import claude_worker

        self.assertEqual(claude_worker._build_query_input("hello", []), "hello")

    def test_build_query_input_streams_when_images_present(self) -> None:
        import tempfile
        from collections.abc import AsyncIterator
        from pathlib import Path

        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "a.png"
            png.write_bytes(b"\x89PNG\r\n")
            result = claude_worker._build_query_input("hi", [str(png)])
        self.assertIsInstance(result, AsyncIterator)


class SessionEntriesCoverageTests(TestCase):
    """Cover the remaining item-type and parsing branches."""

    def _entry(self, item: dict[str, Any]) -> Any:
        import json

        from hitch.main import claude_session_entries

        line = json.dumps({"method": "item/completed", "payload": {"item": item}})
        return claude_session_entries._entry_from_event_line(line)

    def test_mcp_web_reasoning_plan_dynamic_items(self) -> None:
        self.assertEqual(
            self._entry(
                {"type": "mcpToolCall", "server": "github", "tool": "create_pr"}
            )["detail"],
            "github / create_pr",
        )
        self.assertEqual(
            self._entry({"type": "webSearch", "query": "django orm"})["detail"],
            "django orm",
        )
        self.assertEqual(
            self._entry({"type": "reasoning", "text": "first line\nsecond"})["detail"],
            "first line",
        )
        plan = self._entry({"type": "plan", "text": "# Plan"})
        self.assertEqual((plan["kind"], plan["text"]), ("plan", "# Plan"))
        self.assertEqual(
            self._entry({"type": "dynamicToolCall", "tool": "hitch.x"})["detail"],
            "hitch.x",
        )

    def test_empty_text_items_and_unknown_type_drop(self) -> None:
        self.assertIsNone(self._entry({"type": "agentMessage", "text": ""}))
        self.assertIsNone(self._entry({"type": "userMessage", "text": ""}))
        self.assertIsNone(self._entry({"type": "plan", "text": ""}))
        self.assertIsNone(self._entry({"type": "somethingElse"}))

    def test_malformed_and_non_item_lines_ignored(self) -> None:
        from hitch.main import claude_session_entries

        self.assertIsNone(claude_session_entries._entry_from_event_line("not json"))
        self.assertIsNone(claude_session_entries._entry_from_event_line(""))
        self.assertIsNone(
            claude_session_entries._entry_from_event_line('{"method": "item/started"}')
        )


class TranslateCoverageTests(TestCase):
    def test_websearch_and_mcp_and_user_text_blocks(self) -> None:
        translator = claude_translate.EventTranslator()
        ev = translator.translate(
            _assistant(ToolUseBlock(id="w", name="WebSearch", input={"query": "q"}))
        )
        self.assertEqual(ev[0][1]["item"]["type"], "webSearch")
        self.assertEqual(ev[0][1]["item"]["query"], "q")
        # tool result content as a list of text blocks is flattened
        translator.translate(
            _assistant(ToolUseBlock(id="m", name="mcp__hitch__propose_session", input={}))
        )
        closed = translator.translate(
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="m",
                        content=[{"type": "text", "text": "ok"}, {"type": "other"}],
                    )
                ]
            )
        )
        self.assertEqual(closed[0][1]["item"]["result"], "ok")

    def test_user_text_block_and_unknown_tool_result(self) -> None:
        translator = claude_translate.EventTranslator()
        ev = translator.translate(UserMessage(content=[TextBlock(text="hi")]))
        self.assertEqual(ev[1][1]["item"]["type"], "userMessage")
        self.assertEqual(
            translator.translate(
                UserMessage(content=[ToolResultBlock(tool_use_id="ghost", content="x")])
            ),
            [],
        )

    def test_generic_dynamic_tool_item(self) -> None:
        translator = claude_translate.EventTranslator()
        ev = translator.translate(
            _assistant(ToolUseBlock(id="t", name="Task", input={"a": 1}))
        )
        item = ev[0][1]["item"]
        self.assertEqual(item["type"], "dynamicToolCall")
        self.assertEqual(item["tool"], "Task")


class _FakeClient:
    """Stand-in for ClaudeSDKClient yielding a scripted message stream."""

    def __init__(self, messages: list[Any], *, options: Any = None) -> None:
        self._messages = messages
        self.options = options
        self.queries: list[str] = []
        self.interrupted = False

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def interrupt(self) -> None:
        self.interrupted = True

    async def receive_response(self) -> Any:
        for message in self._messages:
            yield message


class WorkerTurnTests(TestCase):
    def _run(self, messages: list[Any]) -> tuple[Any, str]:
        import asyncio

        from hitch.main.management.commands import claude_worker

        fake = _FakeClient(messages)

        def _factory(*, options: Any) -> _FakeClient:
            fake.options = options
            return fake

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            instance = CodexInstance(
                pk=1,
                thread_id="thread-x",
                cwd=tmp,
                prompt="please help",
                events_path=str(events_path),
                pid=0,
                status=CodexInstance.STATUS_RUNNING,
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            )
            with (
                open(events_path, "a", encoding="utf-8") as events_file,
                patch.object(claude_worker, "ClaudeSDKClient", _factory),
            ):
                runner = claude_worker._TurnRunner(
                    instance=instance,
                    events_file=events_file,
                    model="claude-opus-4-8",
                    reasoning_effort=None,
                    sandbox_policy=None,
                    approval_mode=None,
                    web_search_mode=None,
                    plan_mode=False,
                )
                asyncio.run(runner.run())
            return runner, events_path.read_text(encoding="utf-8")

    def test_stream_is_translated_and_session_captured(self) -> None:
        from claude_agent_sdk import SystemMessage

        messages = [
            SystemMessage(subtype="init", data={"session_id": "sess-final"}),
            _assistant(
                TextBlock(text="working on it"),
                ToolUseBlock(id="t1", name="Bash", input={"command": "ls"}),
            ),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")]),
            _result(),
        ]
        runner, written = self._run(messages)
        methods = [line for line in written.splitlines() if line.strip()]
        self.assertTrue(any('"item/started"' in line for line in methods))
        self.assertTrue(any('"item/completed"' in line for line in methods))
        self.assertTrue(any("commandExecution" in line for line in methods))
        self.assertEqual(runner.session_id, "sess-final")
        self.assertFalse(runner.failed)
        # The developer prompt is forwarded to the CLI.
        self.assertEqual(runner._client.queries, ["please help"])

    def test_error_result_marks_turn_failed(self) -> None:
        messages = [_assistant(TextBlock(text="oops")), _result(subtype="error", is_error=True)]
        runner, _written = self._run(messages)
        self.assertTrue(runner.failed)

    def test_stream_without_result_is_not_marked_completed(self) -> None:
        # A truncated/aborted stream (no ResultMessage) must not look successful.
        runner, _written = self._run([_assistant(TextBlock(text="partial"))])
        self.assertFalse(runner.saw_result)
        self.assertFalse(runner.failed)
