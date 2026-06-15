"""Coverage for the Claude Code backend: event translation, option mapping,
the in-process propose_session tool, worker-command routing, and the
``spawn_new_session`` Claude path that mints a local session instead of calling
the Codex app-server.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, cast, override
from unittest.mock import MagicMock, patch

from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from django.test import TestCase, TransactionTestCase, override_settings

from hitch.main import claude_options, coding_agents
from hitch.main.models import ApprovalRequest, CodexInstance, UserInputRequest
from hitch.main.runtime import claude_translate, codex_pool


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

    def test_empty_user_message_emits_nothing_without_suppression(self) -> None:
        # The worker now records image turns itself (the SDK echo can't represent
        # them), so the translator no longer invents an "[image]" marker for an
        # empty echo -- an unsuppressed empty user message yields no event.
        translator = claude_translate.EventTranslator()
        self.assertEqual(translator.translate(UserMessage(content=[])), [])

    def test_suppressed_user_prompt_echo_is_dropped(self) -> None:
        # When the worker has emitted a turn's userMessage (an image turn), the
        # matching SDK prompt echo -- empty (image-only) or text (mixed) -- is
        # dropped so the turn is not recorded twice.
        translator = claude_translate.EventTranslator()
        translator.suppress_next_user_prompt()
        self.assertEqual(translator.translate(UserMessage(content=[])), [])
        translator.suppress_next_user_prompt()
        self.assertEqual(
            translator.translate(UserMessage(content=[TextBlock(text="hi")])), []
        )
        # Suppression is one-shot: a later prompt echo records normally again.
        events = translator.translate(UserMessage(content="next"))
        self.assertEqual(events[1][1]["item"]["text"], "next")

    def test_suppression_does_not_swallow_tool_results(self) -> None:
        # A pending suppression must skip only a user *prompt* echo, never a
        # ToolResultBlock delivery (which also arrives as a UserMessage).
        translator = claude_translate.EventTranslator()
        translator.translate(
            _assistant(ToolUseBlock(id="t1", name="Bash", input={"command": "ls"}))
        )
        translator.suppress_next_user_prompt()
        events = translator.translate(
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="out")])
        )
        self.assertEqual(events[0][1]["item"]["status"], "completed")
        # The prompt echo that follows is still suppressed.
        self.assertEqual(translator.translate(UserMessage(content=[])), [])

    def test_user_message_events_helper_shape(self) -> None:
        events = claude_translate.user_message_events("uid:0", "text\n[image]")
        self.assertEqual([m for m, _ in events], ["item/started", "item/completed"])
        self.assertEqual(events[1][1]["item"]["type"], "userMessage")
        self.assertEqual(events[1][1]["item"]["text"], "text\n[image]")

    def test_cancel_pending_suppression_releases_echo(self) -> None:
        # A steer whose query is rejected releases its armed suppression, so the
        # next real user echo is not swallowed.
        translator = claude_translate.EventTranslator()
        translator.suppress_next_user_prompt()
        translator.cancel_pending_user_prompt_suppression()
        ev = translator.translate(UserMessage(content="hello"))
        self.assertEqual(ev[1][1]["item"]["text"], "hello")
        # Cancelling with nothing armed is a harmless no-op.
        translator.cancel_pending_user_prompt_suppression()


class ClaudeOptionsTests(TestCase):
    def test_plan_mode_takes_precedence(self) -> None:
        self.assertEqual(
            claude_options.resolve_permission_mode(
                plan_mode=True, sandbox_policy="dangerFullAccess", approval_mode="approve_all"
            ),
            "plan",
        )

    def test_approve_all_never_bypasses_permission_callback(self) -> None:
        # The callback must stay live even under full-access + approve_all so
        # AskUserQuestion can route to the input UI; "run everything" is preserved
        # by auto-approving inside the callback instead. So no config bypasses.
        for sandbox in ("dangerFullAccess", "workspaceWrite"):
            self.assertEqual(
                claude_options.resolve_permission_mode(
                    plan_mode=False,
                    sandbox_policy=sandbox,
                    approval_mode="approve_all",
                ),
                "default",
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
        # ``Monitor`` runs a background script under Bash rules; gate it too so
        # ``approve_all`` cannot run commands despite the read-only sandbox.
        self.assertIn("Monitor", disallowed)
        # ``BashOutput`` polls a background Bash run; block it alongside Bash.
        self.assertIn("BashOutput", disallowed)
        # ``PowerShell`` runs host commands natively (Windows / opt-in env var);
        # it must be blocked like Bash so read-only stays authoritative.
        self.assertIn("PowerShell", disallowed)

    def test_web_search_disabled_blocks_tool(self) -> None:
        allowed, disallowed = claude_options.resolve_tool_lists(
            sandbox_policy=None, web_search_mode="disabled"
        )
        self.assertIn("WebSearch", disallowed)
        self.assertNotIn("WebSearch", allowed)

    def test_cached_web_mode_blocks_live_web_access(self) -> None:
        # Claude has no cached-search mode, so "cached" must not grant live web
        # access: both WebSearch and (live-fetching) WebFetch are blocked.
        allowed, disallowed = claude_options.resolve_tool_lists(
            sandbox_policy=None, web_search_mode="cached"
        )
        self.assertIn("WebSearch", disallowed)
        self.assertIn("WebFetch", disallowed)
        self.assertNotIn("WebSearch", allowed)
        self.assertNotIn("WebFetch", allowed)

    def test_live_web_mode_allows_search(self) -> None:
        allowed, _disallowed = claude_options.resolve_tool_lists(
            sandbox_policy=None, web_search_mode="live"
        )
        self.assertIn("WebSearch", allowed)

    def test_default_web_mode_blocks_live_web(self) -> None:
        # The Codex-default (empty) and unset web settings must not grant live
        # web access for Claude; only an explicit "live" opt-in does.
        for mode in ("", None):
            allowed, disallowed = claude_options.resolve_tool_lists(
                sandbox_policy=None, web_search_mode=mode
            )
            self.assertIn("WebSearch", disallowed, mode)
            self.assertIn("WebFetch", disallowed, mode)
            self.assertNotIn("WebSearch", allowed, mode)
            self.assertNotIn("WebFetch", allowed, mode)

    def test_workspace_write_enables_bash_sandbox(self) -> None:
        # workspaceWrite must confine approved/auto-approved Bash to the repo;
        # the SDK does that via the sandbox setting, not the tool lists. It must
        # also close the unsandboxed-command escape so approve_all
        # (bypassPermissions) can't run Bash outside the sandbox unprompted.
        self.assertEqual(
            claude_options.resolve_sandbox_settings("workspaceWrite"),
            {
                "enabled": True,
                "allowUnsandboxedCommands": False,
                "autoAllowBashIfSandboxed": False,
            },
        )
        # Read-only blocks Bash outright, and dangerFullAccess is the opt-out,
        # so neither carries a bash sandbox.
        self.assertIsNone(claude_options.resolve_sandbox_settings("readOnly"))
        self.assertIsNone(claude_options.resolve_sandbox_settings("dangerFullAccess"))
        self.assertIsNone(claude_options.resolve_sandbox_settings(None))

    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_build_options_sets_sandbox_for_workspace_write(
        self, _bin: MagicMock
    ) -> None:
        options = claude_options.build_options(
            cwd="/repo", model="claude-opus-4-8", sandbox_policy="workspaceWrite"
        )
        self.assertEqual(
            options.sandbox,
            {
                "enabled": True,
                "allowUnsandboxedCommands": False,
                "autoAllowBashIfSandboxed": False,
            },
        )
        # The default (no sandbox policy) leaves the SDK at its own default.
        plain = claude_options.build_options(cwd="/repo", model="claude-opus-4-8")
        self.assertIsNone(plain.sandbox)

    def test_map_effort_filters_unknown(self) -> None:
        self.assertEqual(claude_options.map_effort("high"), "high")
        self.assertIsNone(claude_options.map_effort("minimal"))
        self.assertIsNone(claude_options.map_effort(None))

    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_filesystem_settings_gated_for_hidden_runs(self, _bin: MagicMock) -> None:
        # Visible user turns load repo/user .claude settings (CLAUDE.md, MCP);
        # hidden runs must not, so an untrusted repo's shell hooks can't run
        # outside the approval gate.
        visible = claude_options.build_options(cwd="/repo", model=None)
        self.assertEqual(visible.setting_sources, ["user", "project", "local"])
        hidden = claude_options.build_options(
            cwd="/repo", model=None, load_filesystem_settings=False
        )
        self.assertEqual(hidden.setting_sources, [])
        # Hidden runs must also ignore filesystem MCP (project .mcp.json / user /
        # plugin), so a project stdio MCP server can't launch before can_use_tool;
        # visible runs keep loading the user's own trusted project MCP.
        self.assertTrue(hidden.strict_mcp_config)
        self.assertFalse(visible.strict_mcp_config)

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
        # A bare schema must be wrapped so the SDK actually forwards it to the
        # CLI; otherwise it is silently dropped and no structured_output is set.
        self.assertEqual(
            options.output_format,
            {"type": "json_schema", "schema": {"type": "object"}},
        )
        self.assertIsInstance(options.system_prompt, dict)

    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_developer_instructions_ride_in_system_prompt(self, _bin: MagicMock) -> None:
        # Developer guidance must go through the system prompt, not the user
        # prompt, so it does not leak into the transcript or repeat on resume.
        options = claude_options.build_options(
            cwd="/repo",
            model="claude-opus-4-8",
            base_instructions="Base guidance.",
            developer_instructions="Use repo conventions.",
        )
        append = cast("dict[str, Any]", options.system_prompt)["append"]
        self.assertIn("Base guidance.", append)
        self.assertIn("Use repo conventions.", append)

    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_developer_instructions_alone_create_system_prompt(
        self, _bin: MagicMock
    ) -> None:
        options = claude_options.build_options(
            cwd="/repo",
            model="claude-opus-4-8",
            developer_instructions="Use repo conventions.",
        )
        system_prompt = cast("dict[str, Any]", options.system_prompt)
        self.assertIn("Use repo conventions.", system_prompt["append"])

    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_build_options_with_mcp_server_and_schema_and_hook(self, _bin: MagicMock) -> None:
        from hitch.main.runtime import claude_tools

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
        self.assertEqual(
            options.output_format,
            {"type": "json_schema", "schema": {"type": "object"}},
        )
        # can_use_tool requires the companion PreToolUse hook.
        self.assertIn("PreToolUse", options.hooks or {})

    @patch("hitch.main.claude_options.claude_bin", return_value=None)
    def test_build_options_passes_prewrapped_output_format(
        self, _bin: MagicMock
    ) -> None:
        # A caller that already wrapped the schema must not be double-wrapped.
        wrapped = {"type": "json_schema", "schema": {"type": "object"}}
        options = claude_options.build_options(
            cwd="/repo", model="claude-opus-4-8", output_schema=wrapped
        )
        self.assertEqual(options.output_format, wrapped)


class ProposeSessionHandlerTests(TestCase):
    def test_relevant_files_must_be_a_list(self) -> None:
        from hitch.main.runtime import claude_tools

        result = claude_tools._run_propose_session(
            {"title": "t", "summary": "s", "prompt": "p", "relevant_files": "x"},
            "/repo",
            "thread",
        )
        self.assertTrue(result["is_error"])
        self.assertIn("relevant_files", result["content"][0]["text"])

    def test_unknown_cwd_project_is_reported(self) -> None:
        from hitch.main.runtime import claude_tools

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
    def test_hidden_claude_reviewer_defaults_to_workspace_write(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = codex_pool.WorkerLaunch(pid=1)
        with (
            tempfile.TemporaryDirectory() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            # Hidden reviewer with the empty "Codex default" sandbox: resolved to
            # workspace-write so it can run tests like a Codex reviewer.
            hidden = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="qa",
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind="pr_qa",
                sandbox_policy=None,
            )
            # A visible user turn with the empty sandbox is left untouched (the
            # worker defaults those), so its persisted policy stays empty.
            visible = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="hi",
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_USER,
                sandbox_policy=None,
            )
        self.assertEqual(
            hidden.sandbox_policy, claude_options.SANDBOX_WORKSPACE_WRITE
        )
        self.assertEqual(visible.sandbox_policy, "")

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_hidden_claude_reviewer_keeps_explicit_sandbox(
        self, mock_launch: MagicMock
    ) -> None:
        mock_launch.return_value = codex_pool.WorkerLaunch(pid=1)
        with (
            tempfile.TemporaryDirectory() as events_dir,
            override_settings(CODEX_EVENTS_DIR=Path(events_dir)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd="/repo",
                prompt="qa",
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind="pr_qa",
                sandbox_policy=claude_options.SANDBOX_READ_ONLY,
            )
        self.assertEqual(instance.sandbox_policy, claude_options.SANDBOX_READ_ONLY)

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
        # ``Monitor`` runs a background script under Bash rules, so it follows the
        # command-execution path -- not the generic tool path.
        self.assertEqual(
            claude_worker._approval_method("Monitor"),
            claude_worker._COMMAND_APPROVAL_METHOD,
        )
        # ``BashOutput`` polls a background Bash run, so it shares Bash's
        # command-execution treatment (a hidden auto_review run must be able to
        # observe a background command to completion).
        self.assertEqual(
            claude_worker._approval_method("BashOutput"),
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

    def test_generic_tool_approval_params_carry_tool_name_and_arguments(self) -> None:
        from hitch.main.management.commands import claude_worker

        params = claude_worker._approval_params(
            claude_worker._TOOL_APPROVAL_METHOD,
            "mcp__github__create_pr",
            {"title": "x"},
        )
        self.assertEqual(params["tool"], "mcp__github__create_pr")
        self.assertEqual(params["item"]["type"], "toolCall")
        # The browser renders the item JSON, so the user must see the arguments
        # they are approving -- not just the tool name.
        self.assertEqual(params["item"]["arguments"], {"title": "x"})

    def test_file_approval_params_extract_path(self) -> None:
        from hitch.main.management.commands import claude_worker

        params = claude_worker._approval_params(
            claude_worker._FILE_APPROVAL_METHOD, "Edit", {"file_path": "/repo/a.py"}
        )
        self.assertEqual(params["item"]["type"], "fileChange")
        self.assertEqual(params["item"]["changes"], [{"path": "/repo/a.py"}])
        # Falls back to ``notebook_path`` when ``file_path``/``path`` are absent.
        nb = claude_worker._approval_params(
            claude_worker._FILE_APPROVAL_METHOD,
            "NotebookEdit",
            {"notebook_path": "/repo/n.ipynb"},
        )
        self.assertEqual(nb["item"]["changes"], [{"path": "/repo/n.ipynb"}])


class WorkerHelperEdgeCaseTests(TestCase):
    """Edge branches of the worker's pure helpers."""

    def test_ask_user_question_params_skips_malformed(self) -> None:
        from hitch.main.management.commands import claude_worker

        self.assertEqual(claude_worker._ask_user_question_params({}), [])
        self.assertEqual(
            claude_worker._ask_user_question_params({"questions": "nope"}), []
        )
        questions = claude_worker._ask_user_question_params(
            {
                "questions": [
                    "not-a-dict",
                    {"header": "No text"},  # missing question -> skipped
                    {
                        "question": "Pick?",
                        "header": "Pick",
                        "options": ["bad", {"description": "no label"}, {"label": "ok"}],
                    },
                ]
            }
        )
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["id"], "q2")
        self.assertEqual([o["label"] for o in questions[0]["options"]], ["ok"])

    def test_ask_user_answers_map_keys_by_question_and_header(self) -> None:
        from hitch.main.management.commands import claude_worker

        tool_input = {
            "questions": [
                {"question": "Which one?", "header": "Pick"},
                {"question": "Skipped?", "header": "Skip"},
            ]
        }
        answers = claude_worker._ask_user_answers_map(
            tool_input, {"q0": "alpha", "q1": ""}
        )
        # q0 is recorded under both its text and header; q1 (blank) is skipped.
        self.assertEqual(answers, {"Which one?": "alpha", "Pick": "alpha"})

    def test_image_content_blocks_non_list_is_empty(self) -> None:
        from hitch.main.management.commands import claude_worker

        self.assertEqual(claude_worker._image_content_blocks(None), [])
        self.assertEqual(claude_worker._image_content_blocks("x.png"), [])

    def test_decision_allows(self) -> None:
        from hitch.main.management.commands import claude_worker
        from hitch.main.models import ApprovalRequest

        self.assertTrue(claude_worker._decision_allows({"updated": 1}))
        self.assertTrue(
            claude_worker._decision_allows(ApprovalRequest.DECISION_ACCEPT)
        )
        self.assertFalse(claude_worker._decision_allows("decline"))

    def test_steer_request_handles_undecodable_line(self) -> None:
        from hitch.main.management.commands import claude_worker

        self.assertEqual(
            claude_worker._steer_request(b"\xff\xfe not json"), ("", [], "")
        )


class StopUnblocksApprovalWaitTests(TestCase):
    """A Stop click while ``can_use_tool`` waits on a browser approval must set
    the shared cancel flag ``_wait_for_decision`` polls, or the approval sits
    until its timeout instead of declining and interrupting the turn."""

    def test_sigterm_sets_shared_cancel_flag(self) -> None:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
        )
        runner = claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
            web_search_mode=None,
            plan_mode=False,
        )
        runner._client = MagicMock()
        with (
            patch.object(claude_worker, "request_cancel") as mock_cancel,
            patch(
                "hitch.main.management.commands.claude_worker.asyncio.create_task"
            ),
            patch.object(runner, "_interrupt", new=MagicMock()),
        ):
            runner._on_sigterm()
        mock_cancel.assert_called_once()
        self.assertTrue(runner._cancelled)


class AskUserQuestionTests(TestCase):
    """Claude's ``AskUserQuestion`` (plan-mode clarifications) is routed to the
    structured-input UI instead of a bare Run/Skip approval, and the selections
    are returned to the model."""

    def _runner(self) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode="prompt_user",
            web_search_mode=None,
            plan_mode=True,
        )

    def test_params_mapping_adds_ids_and_carries_options(self) -> None:
        from hitch.main.management.commands import claude_worker

        questions = claude_worker._ask_user_question_params(
            {
                "questions": [
                    {
                        "question": "Which library?",
                        "header": "Library",
                        "options": [
                            {"label": "requests", "description": "simple"},
                            {"label": "httpx", "description": "async"},
                        ],
                        "multiSelect": False,
                    }
                ]
            }
        )
        self.assertEqual(len(questions), 1)
        question = questions[0]
        self.assertEqual(question["id"], "q0")
        self.assertEqual(question["question"], "Which library?")
        self.assertEqual(question["header"], "Library")
        self.assertEqual([o["label"] for o in question["options"]], ["requests", "httpx"])
        self.assertTrue(question["requires_explicit_choice"])
        self.assertFalse(question["multi_select"])

    def test_params_mapping_carries_multi_select(self) -> None:
        from hitch.main.management.commands import claude_worker

        questions = claude_worker._ask_user_question_params(
            {
                "questions": [
                    {
                        "question": "Which features?",
                        "header": "Features",
                        "options": [{"label": "a"}, {"label": "b"}],
                        "multiSelect": True,
                    }
                ]
            }
        )
        self.assertTrue(questions[0]["multi_select"])

    def test_routes_to_input_request_and_returns_answers(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultAllow

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        tool_input = {
            "questions": [
                {
                    "question": "Which library?",
                    "header": "Library",
                    "options": [
                        {"label": "requests", "description": ""},
                        {"label": "httpx", "description": ""},
                    ],
                }
            ]
        }
        with (
            patch.object(
                claude_worker, "_create_pending_user_input", return_value=7
            ) as mock_create,
            patch.object(
                claude_worker,
                "_wait_for_user_input_response",
                return_value={"answers": {"q0": "httpx"}},
            ) as mock_wait,
        ):
            result = asyncio.run(
                runner._can_use_tool("AskUserQuestion", tool_input, None)
            )
        mock_create.assert_called_once()
        mock_wait.assert_called_once()
        # Selections ride back to the tool via ``answers`` in updated_input on an
        # allow -- keyed by both question text and header so the CLI can match.
        self.assertIsInstance(result, PermissionResultAllow)
        answers = result.updated_input["answers"]
        self.assertEqual(answers["Which library?"], "httpx")
        self.assertEqual(answers["Library"], "httpx")

    def test_unanswered_question_declines(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        tool_input = {
            "questions": [
                {
                    "question": "Which library?",
                    "header": "Library",
                    "options": [{"label": "requests", "description": ""}],
                }
            ]
        }
        with (
            patch.object(claude_worker, "_create_pending_user_input", return_value=7),
            patch.object(
                claude_worker,
                "_wait_for_user_input_response",
                return_value={"answers": {}},
            ),
        ):
            result = asyncio.run(
                runner._can_use_tool("AskUserQuestion", tool_input, None)
            )
        self.assertIsInstance(result, PermissionResultDeny)
        self.assertIn("did not answer", result.message)

    def test_routed_to_input_ui_even_under_deny_all(self) -> None:
        # A ``/plan`` clarification is not an "escalation": deny_all must still
        # surface the input UI rather than denying the AskUserQuestion call.
        import asyncio

        from claude_agent_sdk import PermissionResultAllow

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        runner._approval_mode = claude_options.APPROVAL_DENY_ALL
        tool_input = {
            "questions": [
                {
                    "question": "Which library?",
                    "header": "Library",
                    "options": [{"label": "requests", "description": ""}],
                }
            ]
        }
        with (
            patch.object(claude_worker, "_create_pending_user_input", return_value=7),
            patch.object(
                claude_worker,
                "_wait_for_user_input_response",
                return_value={"answers": {"q0": "requests"}},
            ),
        ):
            result = asyncio.run(
                runner._can_use_tool("AskUserQuestion", tool_input, None)
            )
        self.assertIsInstance(result, PermissionResultAllow)

    def test_hidden_system_agent_does_not_show_ask_ui(self) -> None:
        # A hidden run has no input UI; AskUserQuestion falls through to the
        # system-agent branch and is denied rather than parked on a never-shown UI.
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner()
        runner._instance.purpose = CodexInstance.PURPOSE_SYSTEM_AGENT
        runner._approval_mode = claude_options.APPROVAL_AUTO_REVIEW
        result = asyncio.run(
            runner._can_use_tool("AskUserQuestion", {"questions": []}, None)
        )
        self.assertIsInstance(result, PermissionResultDeny)

    def test_feedback_turn_routes_to_input_ui(self) -> None:
        # A QA/PR feedback turn runs visibly in the user's session, so a
        # clarification must reach the input UI like a user turn, not be denied.
        import asyncio

        from claude_agent_sdk import PermissionResultAllow

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        runner._instance.purpose = CodexInstance.PURPOSE_SYSTEM_FEEDBACK
        tool_input = {
            "questions": [
                {
                    "question": "Which?",
                    "header": "Pick",
                    "options": [{"label": "a", "description": ""}],
                }
            ]
        }
        with (
            patch.object(
                claude_worker, "_create_pending_user_input", return_value=9
            ) as mock_create,
            patch.object(
                claude_worker,
                "_wait_for_user_input_response",
                return_value={"answers": {"q0": "a"}},
            ),
        ):
            result = asyncio.run(
                runner._can_use_tool("AskUserQuestion", tool_input, None)
            )
        mock_create.assert_called_once()
        self.assertIsInstance(result, PermissionResultAllow)


class HiddenAutoReviewApprovalTests(TestCase):
    """Hidden auto-review runs auto-approve built-in mutating tools only under a
    write sandbox; an unsandboxed run, or a project/user MCP tool reaching
    ``can_use_tool``, must be denied since these runs have no approval UI."""

    def _runner(
        self,
        *,
        sandbox_policy: str | None = "workspaceWrite",
        approval_mode: str = claude_options.APPROVAL_AUTO_REVIEW,
    ) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_builtin_mutating_tools_auto_allowed_under_write_sandbox(self) -> None:
        import asyncio

        runner = self._runner(sandbox_policy="workspaceWrite")
        # ``Monitor`` runs under Bash rules and is bash-sandbox-confinable, so a
        # hidden write-sandbox run auto-approves it like Bash. ``BashOutput`` polls
        # a background Bash run, so it too must be observable to completion.
        for tool in (
            "Bash",
            "BashOutput",
            "Monitor",
            "Edit",
            "Write",
            "MultiEdit",
            "NotebookEdit",
        ):
            result = asyncio.run(runner._can_use_tool(tool, {}, None))
            self.assertIsInstance(result, PermissionResultAllow, tool)

    def test_mutating_tools_denied_without_write_sandbox(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        # No sandbox: an unsandboxed Bash/file edit in a hidden, prompt-injectable
        # agent must not be auto-run.
        runner = self._runner(sandbox_policy=None)
        for tool in ("Bash", "Edit"):
            result = asyncio.run(runner._can_use_tool(tool, {}, None))
            self.assertIsInstance(result, PermissionResultDeny, tool)

    def test_mcp_tool_is_denied_without_approval_ui(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner(sandbox_policy="workspaceWrite")
        result = asyncio.run(
            runner._can_use_tool("mcp__github__create_pr", {"title": "x"}, None)
        )
        self.assertIsInstance(result, PermissionResultDeny)

    def test_demo_agent_runs_setup_commands_without_write_sandbox(self) -> None:
        import asyncio

        from hitch.main import demo

        # The demo is a trusted Hitch setup agent that must run shell/podman
        # commands even with no write sandbox.
        runner = self._runner(sandbox_policy=None)
        runner._instance.agent_kind = demo.DEMO_AGENT_KIND
        result = asyncio.run(runner._can_use_tool("Bash", {"command": "podman ps"}, None))
        self.assertIsInstance(result, PermissionResultAllow)

    def test_demo_agent_runs_setup_commands_under_deny_all(self) -> None:
        import asyncio

        from hitch.main import demo

        # A saved ``deny_all`` preference must not break the opt-in web demo:
        # the trusted setup agent is exempted from the blanket denial and its
        # built-in setup commands are auto-approved (mirroring Codex's
        # never-ask ``deny_all``).
        runner = self._runner(
            sandbox_policy=None, approval_mode=claude_options.APPROVAL_DENY_ALL
        )
        runner._instance.agent_kind = demo.DEMO_AGENT_KIND
        for tool in ("Bash", "Edit", "Write"):
            result = asyncio.run(runner._can_use_tool(tool, {}, None))
            self.assertIsInstance(result, PermissionResultAllow, tool)

    def test_non_demo_agent_denied_under_deny_all(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        # A non-demo hidden run under ``deny_all`` is still blanket-denied; only
        # the trusted demo agent is exempted.
        runner = self._runner(
            sandbox_policy="workspaceWrite",
            approval_mode=claude_options.APPROVAL_DENY_ALL,
        )
        result = asyncio.run(runner._can_use_tool("Bash", {"command": "ls"}, None))
        self.assertIsInstance(result, PermissionResultDeny)


class WorkspaceWriteConfinementTests(TestCase):
    """Under ``workspaceWrite``, SandboxSettings only sandboxes bash, so the
    worker confines the SDK Write/Edit tools to ``cwd`` itself -- even under
    ``approve_all`` (which no longer fully bypasses for this sandbox)."""

    def _runner(self) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy="workspaceWrite",
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_edit_inside_cwd_is_allowed(self) -> None:
        import asyncio

        runner = self._runner()
        for path in ("/repo/a.py", "sub/b.py"):
            result = asyncio.run(
                runner._can_use_tool("Edit", {"file_path": path}, None)
            )
            self.assertIsInstance(result, PermissionResultAllow, path)

    def test_edit_outside_cwd_is_denied_even_under_approve_all(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner()
        for path in ("/etc/passwd", "/repo/../escape.py"):
            result = asyncio.run(
                runner._can_use_tool("Write", {"file_path": path}, None)
            )
            self.assertIsInstance(result, PermissionResultDeny, path)

    def test_bash_still_allowed_under_approve_all(self) -> None:
        import asyncio

        # Bash is confined by SandboxSettings; approve_all auto-approves it.
        runner = self._runner()
        result = asyncio.run(runner._can_use_tool("Bash", {"command": "ls"}, None))
        self.assertIsInstance(result, PermissionResultAllow)


class ReadOnlyMcpGuardTests(TestCase):
    """A read-only sandbox must stay authoritative: external MCP tools (which
    Claude's bash sandbox cannot constrain) are denied even under approve_all."""

    def _runner(self, approval_mode: str) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=claude_options.SANDBOX_READ_ONLY,
            approval_mode=approval_mode,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_external_mcp_tool_denied_under_read_only_approve_all(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner(claude_options.APPROVAL_APPROVE_ALL)
        result = asyncio.run(
            runner._can_use_tool("mcp__github__create_pr", {"title": "x"}, None)
        )
        self.assertIsInstance(result, PermissionResultDeny)

    def test_own_propose_tool_not_blocked_by_read_only_guard(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultAllow

        from hitch.main.runtime.claude_tools import PROPOSE_SESSION_TOOL_NAME

        # Our in-process propose tool is read-only-safe; the guard targets only
        # external MCP servers, so under approve_all it is allowed.
        runner = self._runner(claude_options.APPROVAL_APPROVE_ALL)
        result = asyncio.run(
            runner._can_use_tool(PROPOSE_SESSION_TOOL_NAME, {}, None)
        )
        self.assertIsInstance(result, PermissionResultAllow)


class WorkspaceWriteMcpGuardTests(TestCase):
    """A confining workspace-write sandbox bounds only built-in Bash/file tools,
    so under ``approve_all`` an external MCP tool must route to interactive
    approval rather than auto-run and escape the chosen confinement."""

    def _runner(self, tool_name: str) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=claude_options.SANDBOX_WORKSPACE_WRITE,
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_external_mcp_routes_to_interactive_under_approve_all(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker
        from hitch.main.models import ApprovalRequest

        runner = self._runner("mcp__github__create_pr")
        with (
            patch.object(
                claude_worker, "_create_pending_approval", return_value=9
            ) as mock_create,
            patch.object(
                claude_worker,
                "_wait_for_decision",
                return_value=ApprovalRequest.DECISION_ACCEPT,
            ) as mock_wait,
        ):
            result = asyncio.run(
                runner._can_use_tool("mcp__github__create_pr", {"title": "x"}, None)
            )
        mock_create.assert_called_once()
        mock_wait.assert_called_once()
        self.assertIsInstance(result, PermissionResultAllow)

    def test_own_propose_tool_still_auto_allowed_under_approve_all(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker
        from hitch.main.runtime.claude_tools import PROPOSE_SESSION_TOOL_NAME

        # The in-process propose tool is not an external MCP server, so the guard
        # does not touch it: approve_all still auto-allows without prompting.
        runner = self._runner(PROPOSE_SESSION_TOOL_NAME)
        with patch.object(
            claude_worker, "_create_pending_approval"
        ) as mock_create:
            result = asyncio.run(
                runner._can_use_tool(PROPOSE_SESSION_TOOL_NAME, {}, None)
            )
        mock_create.assert_not_called()
        self.assertIsInstance(result, PermissionResultAllow)


class DangerFullAccessApproveAllTests(TestCase):
    """``dangerFullAccess`` + ``approve_all`` no longer maps to
    ``bypassPermissions`` (the callback must stay live so ``AskUserQuestion`` can
    route to the input UI). The callback now makes such a turn "run everything":
    external MCP and unconfined PowerShell auto-approve since the user opted into
    full access -- but clarifications still surface."""

    def _runner(self) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=claude_options.SANDBOX_DANGER_FULL_ACCESS,
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_permission_mode_stays_default_not_bypass(self) -> None:
        # The whole fix hinges on the callback firing at all.
        self.assertEqual(
            claude_options.resolve_permission_mode(
                plan_mode=False,
                sandbox_policy=claude_options.SANDBOX_DANGER_FULL_ACCESS,
                approval_mode=claude_options.APPROVAL_APPROVE_ALL,
            ),
            "default",
        )

    def test_external_mcp_and_powershell_auto_approved(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultAllow

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        with patch.object(claude_worker, "_create_pending_approval") as mock_create:
            mcp = asyncio.run(
                runner._can_use_tool("mcp__github__create_pr", {"title": "x"}, None)
            )
            shell = asyncio.run(
                runner._can_use_tool("PowerShell", {"command": "Get-ChildItem"}, None)
            )
        mock_create.assert_not_called()
        self.assertIsInstance(mcp, PermissionResultAllow)
        self.assertIsInstance(shell, PermissionResultAllow)

    def test_ask_user_question_still_routes_to_input_ui(self) -> None:
        import asyncio

        # The crux of the fix: clarifications reach the input UI instead of being
        # bypassed and surfacing to no one.
        runner = self._runner()
        with patch.object(
            runner, "_ask_user_question", return_value=claude_options.allow_result()
        ) as mock_ask:
            asyncio.run(
                runner._can_use_tool(
                    "AskUserQuestion", {"questions": [{"question": "?"}]}, None
                )
            )
        mock_ask.assert_called_once()


class SteerPendingRollbackTests(TestCase):
    """A steered follow-up registers its pending response before scheduling the
    query, so a query that never runs must roll that registration back -- else the
    receive loop waits forever on a response that can't arrive."""

    def _runner(self) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_failed_steer_query_rolls_back_outstanding(self) -> None:
        from concurrent.futures import Future

        # The loop already consumed the steer's count into ``outstanding`` (=2:
        # the in-flight response plus this steer). A failed query must still roll
        # it back so the loop won't await a response that can't arrive.
        runner = self._runner()
        runner._outstanding_responses = 2
        future: Future[Any] = Future()
        future.set_exception(RuntimeError("client is closing"))
        runner._steer_query_done(future)
        self.assertEqual(runner._outstanding_responses, 1)

    def test_successful_steer_query_keeps_outstanding(self) -> None:
        from concurrent.futures import Future

        runner = self._runner()
        runner._outstanding_responses = 2
        future: Future[Any] = Future()
        future.set_result(None)
        runner._steer_query_done(future)
        self.assertEqual(runner._outstanding_responses, 2)

    def test_steered_image_arms_suppression_before_query(self) -> None:
        # The race fix: suppression must be armed before ``client.query`` is
        # awaited, since that await yields to the concurrent receive loop.
        import asyncio

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        armed: dict[str, int] = {}

        class _Client:
            async def query(self, _input: Any) -> None:
                armed["count"] = runner._translator._suppress_user_prompts

        runner._client = _Client()
        with patch.object(claude_worker, "_build_query_input", return_value=["<s>"]):
            asyncio.run(runner._send_steered_query("look", ["/img.png"], ""))
        self.assertEqual(armed["count"], 1)

    def test_failed_steered_image_query_releases_suppression(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker

        runner = self._runner()

        class _Client:
            async def query(self, _input: Any) -> None:
                raise RuntimeError("client refused concurrent follow-up")

        runner._client = _Client()
        with (
            patch.object(claude_worker, "_build_query_input", return_value=["<s>"]),
            patch.object(claude_worker, "discard_input_attachment_paths"),
            self.assertRaises(RuntimeError),
        ):
            asyncio.run(runner._send_steered_query("look", ["/img.png"], ""))
        self.assertEqual(runner._translator._suppress_user_prompts, 0)

    def _steer_acks(self, control_path: Any, steer_id: str) -> list[dict[str, Any]]:
        import json as _json

        lines = control_path.read_text(encoding="utf-8").splitlines()
        return [
            record
            for record in (_json.loads(line) for line in lines if line.strip())
            if record.get("op") == "steer_ack" and record.get("id") == steer_id
        ]

    def _runner_with_control(self, tmp: str) -> Any:
        from pathlib import Path

        runner = self._runner()
        runner._instance.events_path = str(Path(tmp) / "events.jsonl")
        return runner

    def test_steered_query_acks_delivery(self) -> None:
        # codex_pool._await_steer_ack polls the control file for this ack; without
        # it a successful Claude steer stalls the POST for the full timeout.
        import asyncio
        import tempfile

        from hitch.main.management.commands import claude_worker
        from hitch.main.runtime.codex_pool import control_path_for

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner_with_control(tmp)

            class _Client:
                async def query(self, _input: Any) -> None:
                    return None

            runner._client = _Client()
            with patch.object(claude_worker, "_build_query_input", return_value="hi"):
                asyncio.run(runner._send_steered_query("hi", [], "sid-ok"))
            acks = self._steer_acks(control_path_for(runner._instance), "sid-ok")
            self.assertEqual([a["delivered"] for a in acks], [True])

    def test_rejected_steered_query_acks_failure(self) -> None:
        # A rejected steer must ack delivered=False so the POST falls back to a
        # follow-up instead of hanging (and does not duplicate a delivered prompt).
        import asyncio
        import tempfile

        from hitch.main.management.commands import claude_worker
        from hitch.main.runtime.codex_pool import control_path_for

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner_with_control(tmp)

            class _Client:
                async def query(self, _input: Any) -> None:
                    raise RuntimeError("the turn already finished")

            runner._client = _Client()
            with (
                patch.object(claude_worker, "_build_query_input", return_value="hi"),
                self.assertRaises(RuntimeError),
            ):
                asyncio.run(runner._send_steered_query("hi", [], "sid-bad"))
            acks = self._steer_acks(control_path_for(runner._instance), "sid-bad")
            self.assertEqual([a["delivered"] for a in acks], [False])

    def test_failed_image_steer_writes_no_user_entry(self) -> None:
        # A rejected image steer re-delivers as a fresh follow-up, so it must not
        # leave an orphaned [image] user entry the model never processed.
        import asyncio
        import tempfile

        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner_with_control(tmp)

            class _Client:
                async def query(self, _input: Any) -> None:
                    raise RuntimeError("the turn already finished")

            runner._client = _Client()
            with (
                patch.object(
                    claude_worker, "_build_query_input", return_value=["<stream>"]
                ),
                patch.object(claude_worker, "discard_input_attachment_paths"),
                self.assertRaises(RuntimeError),
            ):
                asyncio.run(runner._send_steered_query("look", ["/i.png"], "sid"))
            self.assertNotIn("userMessage", runner._events_file.getvalue())
            self.assertEqual(runner._translator._suppress_user_prompts, 0)

    def test_delivered_image_steer_writes_user_entry(self) -> None:
        import asyncio
        import tempfile

        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner_with_control(tmp)

            class _Client:
                async def query(self, _input: Any) -> None:
                    return None

            runner._client = _Client()
            with patch.object(
                claude_worker, "_build_query_input", return_value=["<stream>"]
            ):
                asyncio.run(runner._send_steered_query("look", ["/i.png"], "sid"))
            written = runner._events_file.getvalue()
            self.assertIn("userMessage", written)
            self.assertIn("[image]", written)

    def test_rejected_image_steer_discards_attachments(self) -> None:
        # A rejected image steer must release its duplicated attachments from the
        # instance ledger, or repeated rejects exhaust the attachment limit.
        import asyncio
        import tempfile

        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner_with_control(tmp)

            class _Client:
                async def query(self, _input: Any) -> None:
                    raise RuntimeError("the turn already finished")

            runner._client = _Client()
            with (
                patch.object(
                    claude_worker, "_build_query_input", return_value=["<stream>"]
                ),
                patch.object(
                    claude_worker, "discard_input_attachment_paths"
                ) as mock_discard,
                self.assertRaises(RuntimeError),
            ):
                asyncio.run(runner._send_steered_query("look", ["/i.png"], "sid"))
            mock_discard.assert_called_once_with(runner._instance, ["/i.png"])


class ClaudeLiveApprovalModeTests(TransactionTestCase):
    """A visible turn's approval mode can change mid-turn from the session/global
    UI, so each tool decision re-reads the live ``CodexInstance.approval_mode``
    row rather than the value captured at worker startup -- matching the Codex
    approval handler. ``TransactionTestCase`` because the re-read runs on a worker
    thread (via ``asyncio.to_thread``) that needs the committed row."""

    def _runner(self, *, startup_mode: str) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance.objects.create(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
            approval_mode=startup_mode,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=claude_options.SANDBOX_WORKSPACE_WRITE,
            approval_mode=startup_mode,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_live_deny_all_overrides_startup_value(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        # Turn started under prompt_user (so can_use_tool runs); the user then
        # switched to deny_all mid-turn and the view updated the row.
        runner = self._runner(startup_mode="prompt_user")
        CodexInstance.objects.filter(pk=runner._instance.pk).update(
            approval_mode=claude_options.APPROVAL_DENY_ALL
        )
        result = asyncio.run(runner._can_use_tool("Bash", {"command": "ls"}, None))
        self.assertIsInstance(result, PermissionResultDeny)

    def test_live_approve_all_overrides_startup_value(self) -> None:
        import asyncio

        runner = self._runner(startup_mode="prompt_user")
        CodexInstance.objects.filter(pk=runner._instance.pk).update(
            approval_mode=claude_options.APPROVAL_APPROVE_ALL
        )
        result = asyncio.run(runner._can_use_tool("Bash", {"command": "ls"}, None))
        self.assertIsInstance(result, PermissionResultAllow)


class ClaudeMcpPrDetectionTests(TestCase):
    """A PR opened via a GitHub MCP tool on a normal Claude turn (outside the /pr
    workflow) is detected from the thread's event files, so the badge and
    ``/fix-pr`` find it even though the synthetic thread has no rollout turns."""

    def _events_file_with_pr(self, result_json: str) -> str:
        import json as _json
        import tempfile

        event = _json.dumps(
            {
                "method": "item/completed",
                "payload": {
                    "item": {
                        "id": "g1",
                        "type": "mcpToolCall",
                        "server": "github",
                        "tool": "create_pull_request",
                        "status": "completed",
                        "result": result_json,
                    }
                },
            }
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(event + "\n")
            name = fh.name
        self.addCleanup(Path(name).unlink, missing_ok=True)
        return name

    def _claude_instance(self, events_path: str) -> None:
        CodexInstance.objects.create(
            thread_id="claude-pr",
            cwd="/repo",
            prompt="open a pr",
            events_path=events_path,
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )

    def test_fix_pr_url_detects_mcp_opened_pr(self) -> None:
        from hitch.main import views

        path = self._events_file_with_pr(
            '{"url": "https://github.com/cberner/hitch/pull/500", '
            '"state": "open", "number": 500}'
        )
        self._claude_instance(path)
        self.assertEqual(
            views._claude_fix_pr_url("claude-pr"),
            "https://github.com/cberner/hitch/pull/500",
        )

    def test_observation_empty_without_pr_calls(self) -> None:
        from hitch.main import views

        path = self._events_file_with_pr("not json")
        self._claude_instance(path)
        observation = views._claude_pr_observation_for_session("claude-pr")
        self.assertIsNone(observation.snapshot)

    def _events_file(self, items: list[dict[str, Any]]) -> str:
        import json as _json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            for item in items:
                fh.write(
                    _json.dumps({"method": "item/completed", "payload": {"item": item}})
                    + "\n"
                )
            name = fh.name
        self.addCleanup(Path(name).unlink, missing_ok=True)
        return name

    def test_later_normal_turn_clears_stale_pr(self) -> None:
        # Turn 1 opens a PR; a later completed ordinary turn with no PR MCP calls
        # supersedes it, so the badge/`/fix-pr` must not keep targeting the stale PR.
        from hitch.main import views

        pr_path = self._events_file(
            [
                {
                    "id": "g1",
                    "type": "mcpToolCall",
                    "server": "github",
                    "tool": "create_pull_request",
                    "status": "completed",
                    "result": (
                        '{"url": "https://github.com/cberner/hitch/pull/500", '
                        '"state": "open", "number": 500}'
                    ),
                },
                {"id": "a1", "type": "agentMessage", "text": "opened the PR"},
            ]
        )
        self._claude_instance(pr_path)
        work_path = self._events_file(
            [
                {"id": "u2", "type": "userMessage", "text": "now refactor the parser"},
                {"id": "a2", "type": "agentMessage", "text": "done"},
            ]
        )
        self._claude_instance(work_path)

        observation = views._claude_pr_observation_for_session("claude-pr")
        self.assertIsNone(observation.snapshot)
        self.assertTrue(observation.superseded_by_lifecycle)
        self.assertIsNone(views._claude_fix_pr_url("claude-pr"))

    def test_pr_lookup_only_turn_does_not_set_session_pr(self) -> None:
        # Merely inspecting a PR (get_pr_info) must not make it the session PR;
        # only a create_pull_request opens the epoch.
        from hitch.main import views

        path = self._events_file(
            [
                {
                    "id": "g1",
                    "type": "mcpToolCall",
                    "server": "github",
                    "tool": "get_pr_info",
                    "status": "completed",
                    "result": (
                        '{"url": "https://github.com/cberner/hitch/pull/999", '
                        '"state": "open", "number": 999}'
                    ),
                },
                {"id": "a1", "type": "agentMessage", "text": "looked at PR 999"},
            ]
        )
        self._claude_instance(path)
        observation = views._claude_pr_observation_for_session("claude-pr")
        self.assertIsNone(observation.snapshot)
        self.assertIsNone(views._claude_fix_pr_url("claude-pr"))

    def test_pr_workflow_notice_turn_does_not_clear_pr(self) -> None:
        # A Hitch PR-monitor/QA feedback turn maintains the active PR, so it must
        # not clear the PR observation even with no PR-related MCP call (mirrors
        # the Codex replay's PR-workflow-notice exemption).
        from hitch.main import views

        pr_path = self._events_file(
            [
                {
                    "id": "g1",
                    "type": "mcpToolCall",
                    "server": "github",
                    "tool": "create_pull_request",
                    "status": "completed",
                    "result": (
                        '{"url": "https://github.com/cberner/hitch/pull/500", '
                        '"state": "open", "number": 500}'
                    ),
                },
                {"id": "a1", "type": "agentMessage", "text": "opened the PR"},
            ]
        )
        self._claude_instance(pr_path)
        notice_path = self._events_file(
            [
                {
                    "id": "u2",
                    "type": "userMessage",
                    "text": (
                        "Hitch PR monitor found follow-up work on the active PR. "
                        "Address the failing CI check."
                    ),
                },
                {"id": "a2", "type": "agentMessage", "text": "looking into CI"},
            ]
        )
        self._claude_instance(notice_path)

        observation = views._claude_pr_observation_for_session("claude-pr")
        self.assertIsNotNone(observation.snapshot)
        self.assertFalse(observation.superseded_by_lifecycle)
        # The session's recency must reach _workflow_after_main_lifecycle so a
        # later completed turn can strip a stale PR/QA workflow handoff (else
        # /fix-pr keeps targeting the obsolete PR).
        from datetime import UTC, datetime

        from hitch.main.models import SessionMetadata
        from hitch.main.runtime import codex_events
        from hitch.main.sessions import session_pr_plan

        ts = datetime(2026, 6, 15, tzinfo=UTC)
        SessionMetadata.objects.create(
            thread_id="claude-pr", cwd="/repo", codex_updated_at=ts
        )
        with (
            patch.object(
                session_pr_plan,
                "_claude_pr_observation_for_session",
                return_value=codex_events.PrObservationResult(snapshot=None),
            ),
            patch(
                "hitch.main.workflows.pr_stage._latest_pr_workflow_for_thread",
                return_value=None,
            ),
            patch.object(
                session_pr_plan,
                "_workflow_after_main_lifecycle",
                return_value=None,
            ) as mock_lifecycle,
            patch.object(
                session_pr_plan, "_current_pr_url_for_thread", return_value=None
            ),
        ):
            session_pr_plan._claude_fix_pr_url("claude-pr")
        self.assertEqual(mock_lifecycle.call_args.kwargs["main_updated_at"], ts)


class DemoSandboxOverrideTests(TestCase):
    """A Claude demo run forces full host access regardless of the user's sandbox
    so its podman/shell container setup is neither blocked nor confined."""

    def _runner(self, sandbox_policy: str | None) -> Any:
        import io

        from hitch.main import demo
        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            agent_kind=demo.DEMO_AGENT_KIND,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=sandbox_policy,
            approval_mode=claude_options.APPROVAL_AUTO_REVIEW,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_read_only_user_sandbox_is_upgraded_for_demo(self) -> None:
        runner = self._runner(claude_options.SANDBOX_READ_ONLY)
        self.assertEqual(
            runner._sandbox_policy, claude_options.SANDBOX_DANGER_FULL_ACCESS
        )
        # The options it builds therefore leave Bash enabled and unsandboxed.
        options = runner._build_options()
        self.assertNotIn("Bash", options.disallowed_tools)
        self.assertIsNone(getattr(options, "sandbox", None))

    def test_workspace_write_user_sandbox_is_upgraded_for_demo(self) -> None:
        runner = self._runner(claude_options.SANDBOX_WORKSPACE_WRITE)
        self.assertEqual(
            runner._sandbox_policy, claude_options.SANDBOX_DANGER_FULL_ACCESS
        )


class PlanModeApprovalTests(TestCase):
    """``ExitPlanMode`` is the plan-approval boundary, so it must always reach the
    interactive approval flow -- never be auto-approved by ``approve_all``."""

    def _runner(self) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
            web_search_mode=None,
            plan_mode=True,
        )

    def test_exit_plan_mode_is_reviewed_even_under_approve_all(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker
        from hitch.main.models import ApprovalRequest

        runner = self._runner()
        with (
            patch.object(
                claude_worker, "_create_pending_approval", return_value=11
            ) as mock_create,
            patch.object(
                claude_worker,
                "_wait_for_decision",
                return_value=ApprovalRequest.DECISION_ACCEPT,
            ) as mock_wait,
        ):
            result = asyncio.run(
                runner._can_use_tool("ExitPlanMode", {"plan": "do it"}, None)
            )
        # It went through the interactive approval flow, not the auto-allow.
        mock_create.assert_called_once()
        mock_wait.assert_called_once()
        self.assertIsInstance(result, PermissionResultAllow)

    def test_other_tools_still_auto_approved_under_approve_all(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        with patch.object(claude_worker, "_create_pending_approval") as mock_create:
            result = asyncio.run(runner._can_use_tool("Bash", {"command": "ls"}, None))
        mock_create.assert_not_called()
        self.assertIsInstance(result, PermissionResultAllow)

    def test_exit_plan_mode_is_reviewed_even_under_deny_all(self) -> None:
        # ``deny_all`` denies escalations, but leaving plan mode is the user's
        # call -- it must reach the interactive approval, not be denied (which
        # would trap the turn in plan mode).
        import asyncio

        from hitch.main.management.commands import claude_worker
        from hitch.main.models import ApprovalRequest

        runner = self._runner()
        runner._approval_mode = claude_options.APPROVAL_DENY_ALL
        with (
            patch.object(
                claude_worker, "_create_pending_approval", return_value=12
            ) as mock_create,
            patch.object(
                claude_worker,
                "_wait_for_decision",
                return_value=ApprovalRequest.DECISION_ACCEPT,
            ),
        ):
            result = asyncio.run(
                runner._can_use_tool("ExitPlanMode", {"plan": "p"}, None)
            )
        mock_create.assert_called_once()
        self.assertIsInstance(result, PermissionResultAllow)

    def test_hidden_run_cannot_leave_plan_mode(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner()
        runner._instance.purpose = CodexInstance.PURPOSE_SYSTEM_AGENT
        result = asyncio.run(runner._can_use_tool("ExitPlanMode", {"plan": "p"}, None))
        self.assertIsInstance(result, PermissionResultDeny)


class PowerShellGatingTests(TestCase):
    """``PowerShell`` runs host commands natively and Claude's bash sandbox can't
    confine it, so it is never auto-run under a confining sandbox: a visible
    session routes it to interactive approval, a hidden run is denied."""

    def _runner(
        self,
        *,
        purpose: str,
        sandbox_policy: str | None,
        approval_mode: str | None,
    ) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=purpose,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_powershell_uses_command_approval_method(self) -> None:
        from hitch.main.management.commands import claude_worker

        self.assertEqual(
            claude_worker._approval_method("PowerShell"),
            claude_worker._COMMAND_APPROVAL_METHOD,
        )

    def test_read_only_denies_powershell_via_disallow_list(self) -> None:
        _allowed, disallowed = claude_options.resolve_tool_lists(
            sandbox_policy=claude_options.SANDBOX_READ_ONLY, web_search_mode=None
        )
        self.assertIn("PowerShell", disallowed)

    def test_approve_all_visible_routes_powershell_to_interactive(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker
        from hitch.main.models import ApprovalRequest

        # Visible session, approve_all, default (workspace-write) sandbox: an
        # auto-allow would run PowerShell unconfined, so it must instead go through
        # the interactive approval flow.
        runner = self._runner(
            purpose=CodexInstance.PURPOSE_USER,
            sandbox_policy=None,
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
        )
        with (
            patch.object(
                claude_worker, "_create_pending_approval", return_value=7
            ) as mock_create,
            patch.object(
                claude_worker,
                "_wait_for_decision",
                return_value=ApprovalRequest.DECISION_ACCEPT,
            ) as mock_wait,
        ):
            result = asyncio.run(
                runner._can_use_tool(
                    "PowerShell", {"command": "Get-ChildItem"}, None
                )
            )
        mock_create.assert_called_once()
        mock_wait.assert_called_once()
        self.assertIsInstance(result, PermissionResultAllow)

    def test_approve_all_visible_still_auto_allows_bash(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker

        # Bash is confined by the workspace-write bash sandbox, so approve_all may
        # still auto-allow it -- only PowerShell is special-cased.
        runner = self._runner(
            purpose=CodexInstance.PURPOSE_USER,
            sandbox_policy=None,
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
        )
        with patch.object(claude_worker, "_create_pending_approval") as mock_create:
            result = asyncio.run(
                runner._can_use_tool("Bash", {"command": "ls"}, None)
            )
        mock_create.assert_not_called()
        self.assertIsInstance(result, PermissionResultAllow)

    def test_hidden_run_denies_unconfined_powershell(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            sandbox_policy=claude_options.SANDBOX_WORKSPACE_WRITE,
            approval_mode=claude_options.APPROVAL_AUTO_REVIEW,
        )
        result = asyncio.run(
            runner._can_use_tool("PowerShell", {"command": "Get-ChildItem"}, None)
        )
        self.assertIsInstance(result, PermissionResultDeny)

    def test_hidden_run_still_allows_bash(self) -> None:
        import asyncio

        runner = self._runner(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            sandbox_policy=claude_options.SANDBOX_WORKSPACE_WRITE,
            approval_mode=claude_options.APPROVAL_AUTO_REVIEW,
        )
        result = asyncio.run(
            runner._can_use_tool("Bash", {"command": "ls"}, None)
        )
        self.assertIsInstance(result, PermissionResultAllow)

    def test_powershell_unconfined_only_outside_danger_full_access(self) -> None:
        unconfined = self._runner(
            purpose=CodexInstance.PURPOSE_USER,
            sandbox_policy=claude_options.SANDBOX_WORKSPACE_WRITE,
            approval_mode=None,
        )
        self.assertTrue(unconfined._powershell_unconfined("PowerShell"))
        self.assertFalse(unconfined._powershell_unconfined("Bash"))
        confined = self._runner(
            purpose=CodexInstance.PURPOSE_USER,
            sandbox_policy=claude_options.SANDBOX_DANGER_FULL_ACCESS,
            approval_mode=None,
        )
        self.assertFalse(confined._powershell_unconfined("PowerShell"))


class EnterWorktreeGatingTests(TestCase):
    """``EnterWorktree`` is auto-approved by Claude and writes a worktree that can
    land outside ``cwd``, bypassing the cwd guard and bash sandbox. It is disallowed
    unless the user opted out of confinement with ``dangerFullAccess``."""

    def _disallowed(self, sandbox_policy: str | None) -> Any:
        _allowed, disallowed = claude_options.resolve_tool_lists(
            sandbox_policy=sandbox_policy, web_search_mode=None
        )
        return disallowed

    def test_disallowed_under_read_only(self) -> None:
        self.assertIn("EnterWorktree", self._disallowed(claude_options.SANDBOX_READ_ONLY))

    def test_disallowed_under_workspace_write(self) -> None:
        self.assertIn(
            "EnterWorktree", self._disallowed(claude_options.SANDBOX_WORKSPACE_WRITE)
        )

    def test_disallowed_under_empty_sandbox(self) -> None:
        self.assertIn("EnterWorktree", self._disallowed(None))

    def test_allowed_under_danger_full_access(self) -> None:
        self.assertNotIn(
            "EnterWorktree",
            self._disallowed(claude_options.SANDBOX_DANGER_FULL_ACCESS),
        )


class ClaudeArchiveUsageTests(TestCase):
    """Claude's token-usage cache row is authoritative (no rollout to recompute
    from), so archiving/unarchiving must not delete it the way it does for Codex."""

    def test_archiving_claude_session_preserves_token_usage(self) -> None:
        from django.urls import reverse

        from hitch.main import models
        from hitch.main.models import ArchivedSessionTokenUsage

        CodexInstance.objects.create(
            thread_id="thread-arch",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            backend=CodexInstance.BACKEND_CLAUDE,
            status=CodexInstance.STATUS_COMPLETED,
        )
        ArchivedSessionTokenUsage.objects.create(
            thread_id="thread-arch",
            rollout_path="",
            input_tokens=500,
            output_tokens=120,
            usage_logic_version=models.TOKEN_USAGE_LOGIC_VERSION,
        )
        for archived in ("true", "false"):
            response = self.client.post(
                reverse("set_session_archived", kwargs={"session_id": "thread-arch"}),
                {"archived": archived},
            )
            self.assertIn(response.status_code, (200, 204, 302))
            row = ArchivedSessionTokenUsage.objects.filter(
                thread_id="thread-arch"
            ).first()
            self.assertIsNotNone(row, archived)
            assert row is not None
            self.assertEqual(row.input_tokens, 500)


class ClaudeCandidateRenameTests(TestCase):
    """Accepting a Claude candidate proposal applies the title locally; there is
    no Codex app-server thread to ``thread_set_name``."""

    def test_claude_candidate_title_applied_without_codex(self) -> None:
        from types import SimpleNamespace

        from hitch.main import views

        CodexInstance.objects.create(
            thread_id="thread-claude-cand",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            backend=CodexInstance.BACKEND_CLAUDE,
            status=CodexInstance.STATUS_COMPLETED,
        )
        proposed = SimpleNamespace(title="Proposal Title")
        metadata = SimpleNamespace(thread_id="thread-claude-cand")
        with (
            patch("hitch.main.views.Codex") as mock_codex,
            patch("hitch.main.views.session_index.update_cached_name") as mock_name,
        ):
            result = views._rename_codex_thread_from_proposal(
                proposed_session=cast(Any, proposed),
                session_metadata=cast(Any, metadata),
                settings=cast(Any, SimpleNamespace(enable_memories=False)),
            )
        self.assertTrue(result)
        mock_codex.assert_not_called()
        mock_name.assert_called_once_with("thread-claude-cand", "Proposal Title")


class ClaudeUsageRefreshTests(TestCase):
    """An authoritative Claude usage cache (no rollout) is never a refresh/repair
    candidate, so /usage and /profile stop probing the Codex app-server for it."""

    def _claude_cache(self) -> Any:
        from hitch.main import models
        from hitch.main.models import ArchivedSessionTokenUsage

        return ArchivedSessionTokenUsage(
            thread_id="thread-claude-usage",
            rollout_path="",
            input_tokens=100,
            usage_logic_version=models.TOKEN_USAGE_LOGIC_VERSION,
        )

    def _metadata(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            thread_id="thread-claude-usage",
            codex_path="",
            usage_last_checked_at=None,
        )

    def test_authoritative_cache_is_not_refresh_pending(self) -> None:
        from hitch.main import views

        state = views._usage_token_cache_state(self._metadata(), self._claude_cache())
        self.assertFalse(state.refresh_pending)
        self.assertTrue(state.cache_usable)

    def test_authoritative_cache_does_not_need_refresh(self) -> None:
        from hitch.main import views

        self.assertFalse(
            views._usage_token_refresh_needed(self._metadata(), self._claude_cache())
        )

    def test_unidentified_empty_path_row_still_needs_refresh(self) -> None:
        # Without a known Claude backend, an empty-path uncached row is treated
        # as a fresh Codex thread awaiting path repair.
        from hitch.main import views

        self.assertTrue(views._usage_token_refresh_needed(self._metadata(), None))
        self.assertTrue(
            views._usage_token_cache_state(self._metadata(), None).refresh_pending
        )

    def test_uncached_claude_row_is_non_refreshable(self) -> None:
        # A known Claude row has no rollout to repair (the worker writes its
        # cache), so an uncached one must not be scheduled for Codex path repair
        # nor reported as refresh-pending -- otherwise it re-probes forever.
        from hitch.main import views

        self.assertFalse(
            views._usage_token_refresh_needed(self._metadata(), None, is_claude=True)
        )
        state = views._usage_token_cache_state(
            self._metadata(), None, is_claude=True
        )
        self.assertFalse(state.refresh_pending)
        self.assertFalse(state.cache_usable)

    def test_claude_thread_ids_resolves_backends(self) -> None:
        from hitch.main import views
        from hitch.main.models import CodexInstance

        def _instance(thread_id: str, backend: str) -> None:
            CodexInstance.objects.create(
                thread_id=thread_id,
                cwd="/repo",
                prompt="x",
                events_path="x",
                pid=0,
                status=CodexInstance.STATUS_COMPLETED,
                backend=backend,
            )

        _instance("claude-row", CodexInstance.BACKEND_CLAUDE)
        _instance("codex-row", CodexInstance.BACKEND_CODEX)
        self.assertEqual(
            views._claude_thread_ids(
                ["claude-row", "codex-row", "unknown-row", ""]
            ),
            {"claude-row"},
        )

    def test_uncached_claude_row_not_scheduled_for_path_repair(self) -> None:
        # End to end: a Claude-backed empty-path candidate with no cache is
        # neither selected for path repair nor left pending in the batcher.
        from hitch.main import views
        from hitch.main.models import CodexInstance

        CodexInstance.objects.create(
            thread_id="thread-claude-usage",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=CodexInstance.BACKEND_CLAUDE,
        )
        candidate = views._UsageTokenRefreshCandidate(
            thread_id="thread-claude-usage",
            codex_path="",
            usage_last_checked_at=None,
        )
        batches = list(views._usage_token_refresh_work_batches([candidate]))
        self.assertEqual(batches, [])


class ClaudeImageCleanupTests(TestCase):
    """Claude inlines uploaded images as base64, so the worker cleans them up at
    turn end unconditionally rather than only when a supersede flagged it."""

    def test_worker_clears_images_even_without_cleanup_flag(self) -> None:
        from django.test import override_settings

        from hitch.main.management.commands import claude_worker
        from hitch.main.runtime import codex_pool

        with (
            tempfile.TemporaryDirectory() as tmp,
            override_settings(CODEX_EVENTS_DIR=tmp),
        ):
            attachments = codex_pool.input_attachments_dir()
            attachments.mkdir(parents=True, exist_ok=True)
            image = attachments / "shot.png"
            image.write_bytes(b"img")
            events_path = Path(tmp) / "events.jsonl"
            instance = CodexInstance.objects.create(
                thread_id="thread-img",
                cwd=tmp,
                prompt="look",
                events_path=str(events_path),
                pid=0,
                status=CodexInstance.STATUS_RUNNING,
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_USER,
                input_image_paths=[str(image)],
                input_attachment_paths=[str(image)],
                # Ordinary uploads start with this False -- the bug was that
                # cleanup then never ran.
                input_attachment_cleanup_requested=False,
            )
            fake = _FakeClient([_assistant(TextBlock(text="ok")), _result()])

            def _factory(*, options: Any) -> _FakeClient:
                fake.options = options
                return fake

            with (
                patch.object(claude_worker, "ClaudeSDKClient", _factory),
                patch.object(claude_worker, "_apply_worker_oom_score_adjust"),
                patch("hitch.main.management.commands.claude_worker.signal.signal"),
                patch.object(claude_worker, "_build_query_input", return_value="look"),
            ):
                claude_worker.Command().handle(
                    instance_id=instance.pk,
                    model=None,
                    reasoning_effort=None,
                    sandbox_policy=None,
                    approval_mode=None,
                    web_search_mode=None,
                    plan_mode=False,
                )
            self.assertFalse(image.exists())


class ClaudeSessionRecencyTests(TestCase):
    """A finished Claude turn bumps the session's recency directly: the worker's
    metadata writes never reach the web index, so without this the session list
    ordering would lag real activity (mirrors the Codex worker)."""

    def test_handle_bumps_session_recency(self) -> None:
        from django.test import override_settings

        from hitch.main.management.commands import claude_worker
        from hitch.main.models import SessionMetadata

        with (
            tempfile.TemporaryDirectory() as tmp,
            override_settings(CODEX_EVENTS_DIR=tmp),
        ):
            events_path = Path(tmp) / "events.jsonl"
            instance = CodexInstance.objects.create(
                thread_id="thread-recency",
                cwd=tmp,
                prompt="hi",
                events_path=str(events_path),
                pid=0,
                status=CodexInstance.STATUS_RUNNING,
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_USER,
            )
            SessionMetadata.objects.create(thread_id="thread-recency", cwd=tmp)
            fake = _FakeClient([_assistant(TextBlock(text="ok")), _result()])

            def _factory(*, options: Any) -> _FakeClient:
                fake.options = options
                return fake

            with (
                patch.object(claude_worker, "ClaudeSDKClient", _factory),
                patch.object(claude_worker, "_apply_worker_oom_score_adjust"),
                patch("hitch.main.management.commands.claude_worker.signal.signal"),
                patch.object(claude_worker, "_build_query_input", return_value="hi"),
            ):
                claude_worker.Command().handle(
                    instance_id=instance.pk,
                    model=None,
                    reasoning_effort=None,
                    sandbox_policy=None,
                    approval_mode=None,
                    web_search_mode=None,
                    plan_mode=False,
                )
            metadata = SessionMetadata.objects.get(thread_id="thread-recency")
            self.assertIsNotNone(metadata.codex_updated_at)


class ClaudePrStageRefreshTests(TestCase):
    """The async PR stage refresh recovers a Claude session's PR from its events
    file (no Codex rollout) and persists the terminal stage at mtime 0, which the
    detail render -- also keyed on 0 for a no-rollout session -- then prefers."""

    def test_refresh_uses_claude_observation_and_caches_at_zero_mtime(self) -> None:
        from hitch.main import system_agents, views
        from hitch.main.models import SessionMetadata
        from hitch.main.runtime import codex_events
        from hitch.main.sessions import session_stage, session_stage_refresh
        from hitch.main.workflows import pr_stage

        SessionMetadata.objects.create(thread_id="claude-pr", cwd="/repo", codex_path="")
        snapshot: dict[str, Any] = {"pr_number": 7}  # gh-backed calls are mocked out
        observation = codex_events.PrObservationResult(snapshot=snapshot)
        with (
            patch.object(session_stage_refresh, "_session_is_claude", return_value=True),
            patch.object(
                session_stage_refresh,
                "_claude_pr_observation_for_session",
                return_value=observation,
            ) as mock_obs,
            patch.object(pr_stage, "_latest_pr_workflow_for_thread", return_value=None),
            patch.object(
                session_stage_refresh, "_workflow_after_main_lifecycle", return_value=None
            ),
            patch.object(
                system_agents, "pr_snapshot_stage_refresh_due", return_value=True
            ),
            patch.object(
                system_agents,
                "refreshed_pr_snapshot_for_stage",
                return_value=snapshot,
            ),
            patch.object(
                session_stage, "derive_stage", return_value=session_stage.DONE_MERGED
            ),
        ):
            views._refresh_session_pr_stage("claude-pr")
        mock_obs.assert_called_once_with("claude-pr")
        metadata = SessionMetadata.objects.get(thread_id="claude-pr")
        self.assertEqual(metadata.derived_stage, session_stage.DONE_MERGED.key)
        self.assertEqual(metadata.derived_stage_source_mtime_ns, 0)

    def test_session_list_serves_claude_cached_stage_at_zero_mtime(self) -> None:
        # The async refresh persists a Claude PR stage at mtime 0; the no-rollout
        # cache reader serves it (the list gates this on a reliable Claude-backend
        # check, since an empty codex_path alone also matches a fresh Codex row).
        from hitch.main.sessions import session_stage, session_stage_refresh

        row = {
            "stage_cache_key": session_stage.DONE_MERGED.key,
            "stage_cache_mtime_ns": 0,
        }
        self.assertEqual(
            session_stage_refresh._claude_cached_stage_for_row(row),
            session_stage.DONE_MERGED,
        )
        # A non-zero (rollout-keyed) mtime is not a Claude cache entry.
        self.assertIsNone(
            session_stage_refresh._claude_cached_stage_for_row(
                {**row, "stage_cache_mtime_ns": 123}
            )
        )
        # No cached stage key -> nothing to serve.
        self.assertIsNone(
            session_stage_refresh._claude_cached_stage_for_row(
                {"stage_cache_key": "", "stage_cache_mtime_ns": 0}
            )
        )

    def test_uncached_claude_list_row_recovers_pr_and_schedules_refresh(self) -> None:
        # First list render of a Claude session that opened a PR directly (no
        # /pr workflow, no cache yet): the list has no rollout to scan, so it
        # must recover the PR from the thread's events -- the same source the
        # detail page uses -- to show the PR stage and schedule the gh refresh
        # that populates the mtime-0 cache. Without the branch it would parse an
        # empty rollout and fall to activity/idle.
        from hitch.main.models import CodexInstance, SessionMetadata
        from hitch.main.runtime import codex_events
        from hitch.main.sessions import session_stage, session_stage_refresh
        from hitch.main.workflows import pr_qa

        SessionMetadata.objects.create(
            thread_id="claude-list-pr", cwd="/repo", codex_path=""
        )
        CodexInstance.objects.create(
            thread_id="claude-list-pr",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=CodexInstance.BACKEND_CLAUDE,
        )
        snapshot: dict[str, Any] = {"pr_number": 11}
        observation = codex_events.PrObservationResult(snapshot=snapshot)
        sessions: list[dict[str, Any]] = [
            {"id": "claude-list-pr", "cwd": "/repo", "has_activity": True}
        ]
        with (
            patch.object(
                session_stage_refresh,
                "_claude_pr_observation_for_session",
                return_value=observation,
            ) as mock_obs,
            patch.object(
                pr_qa, "pr_snapshot_stage_refresh_due", return_value=True
            ),
            patch.object(
                session_stage_refresh, "_schedule_pr_stage_refresh"
            ) as mock_schedule,
            patch.object(
                session_stage,
                "derive_stage",
                return_value=session_stage.PR,
            ) as mock_derive,
        ):
            session_stage_refresh._attach_session_stage_context(sessions)
        mock_obs.assert_called_once_with("claude-list-pr")
        # The recovered snapshot flows into derive_stage as the log snapshot.
        self.assertEqual(mock_derive.call_args.kwargs["pr_snapshot"], snapshot)
        # The first render schedules the gh-backed refresh for the PR snapshot.
        mock_schedule.assert_called_once_with("claude-list-pr")
        self.assertEqual(sessions[0]["stage"]["key"], session_stage.PR.key)
        # The PR number from the recovered snapshot lands on the badge label.
        self.assertEqual(sessions[0]["stage"]["label"], "PR #11")


class ClaudeDanglingRequestCleanupTests(TestCase):
    """A worker that fails after creating an approval/input row but before the
    wait path records a decision must close those rows out -- otherwise the
    session keeps rendering an actionable card for a dead worker and any user
    response is silently dropped (mirrors the Codex worker's failure path)."""

    def _run_handle(
        self,
        *,
        messages: list[Any],
        raise_in_stream: bool,
        stream_exc: type[BaseException] = RuntimeError,
    ) -> Any:
        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            instance = CodexInstance.objects.create(
                thread_id="thread-dangle",
                cwd=tmp,
                prompt="do it",
                events_path=str(events_path),
                pid=0,
                status=CodexInstance.STATUS_RUNNING,
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_USER,
            )
            approval = ApprovalRequest.objects.create(
                instance=instance,
                method="item/commandExecution/requestApproval",
                params={},
            )
            user_input = UserInputRequest.objects.create(
                instance=instance,
                method="session/request_user_input",
                params={},
            )

            class _FailingClient(_FakeClient):
                @override
                async def receive_response(self) -> Any:
                    if raise_in_stream:
                        raise stream_exc("events-file write failed")
                        yield  # pragma: no cover - unreachable, marks a generator
                    for message in messages:
                        yield message

            fake = _FailingClient([])

            def _factory(*, options: Any) -> _FakeClient:
                fake.options = options
                return fake

            with (
                patch.object(claude_worker, "ClaudeSDKClient", _factory),
                patch.object(claude_worker, "_apply_worker_oom_score_adjust"),
                patch("hitch.main.management.commands.claude_worker.signal.signal"),
                patch.object(claude_worker, "_build_query_input", return_value="do it"),
            ):
                if raise_in_stream:
                    with self.assertRaises(stream_exc):
                        claude_worker.Command().handle(
                            instance_id=instance.pk,
                            model=None,
                            reasoning_effort=None,
                            sandbox_policy=None,
                            approval_mode=None,
                            web_search_mode=None,
                            plan_mode=False,
                        )
                else:
                    claude_worker.Command().handle(
                        instance_id=instance.pk,
                        model=None,
                        reasoning_effort=None,
                        sandbox_policy=None,
                        approval_mode=None,
                        web_search_mode=None,
                        plan_mode=False,
                    )
            approval.refresh_from_db()
            user_input.refresh_from_db()
            instance.refresh_from_db()
            return instance, approval, user_input

    def test_exception_path_resolves_dangling_rows(self) -> None:
        instance, approval, user_input = self._run_handle(
            messages=[], raise_in_stream=True
        )
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(approval.decision, ApprovalRequest.DECISION_CANCEL)
        self.assertIsNotNone(approval.decided_at)
        self.assertEqual(user_input.response, {"answers": {}})

    def test_base_exception_path_commits_and_resolves(self) -> None:
        import asyncio

        # On Python 3.13 ``asyncio.CancelledError`` is a ``BaseException``; it must
        # still drive the terminal-status commit and dangling-row cleanup (and be
        # re-raised) rather than slipping past an ``except Exception``.
        instance, approval, user_input = self._run_handle(
            messages=[], raise_in_stream=True, stream_exc=asyncio.CancelledError
        )
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(approval.decision, ApprovalRequest.DECISION_CANCEL)
        self.assertEqual(user_input.response, {"answers": {}})
        self.assertIsNotNone(user_input.responded_at)

    def test_failed_result_resolves_dangling_rows(self) -> None:
        # The stream completes, but the ResultMessage is an error -> the turn is
        # marked failed, and the dangling rows still need closing.
        instance, approval, user_input = self._run_handle(
            messages=[_assistant(TextBlock(text="oops")), _result(subtype="error", is_error=True)],
            raise_in_stream=False,
        )
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(approval.decision, ApprovalRequest.DECISION_CANCEL)
        self.assertEqual(user_input.response, {"answers": {}})

    def test_completed_turn_leaves_resolved_rows_untouched(self) -> None:
        # A successful turn already resolved its own rows; the cleanup must not
        # run (status COMPLETED), so an *accepted* approval keeps its decision.
        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            instance = CodexInstance.objects.create(
                thread_id="thread-ok",
                cwd=tmp,
                prompt="do it",
                events_path=str(events_path),
                pid=0,
                status=CodexInstance.STATUS_RUNNING,
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_USER,
            )
            approval = ApprovalRequest.objects.create(
                instance=instance,
                method="item/commandExecution/requestApproval",
                params={},
                decision=ApprovalRequest.DECISION_ACCEPT,
            )
            fake = _FakeClient([_assistant(TextBlock(text="ok")), _result()])

            def _factory(*, options: Any) -> _FakeClient:
                fake.options = options
                return fake

            with (
                patch.object(claude_worker, "ClaudeSDKClient", _factory),
                patch.object(claude_worker, "_apply_worker_oom_score_adjust"),
                patch("hitch.main.management.commands.claude_worker.signal.signal"),
                patch.object(claude_worker, "_build_query_input", return_value="do it"),
            ):
                claude_worker.Command().handle(
                    instance_id=instance.pk,
                    model=None,
                    reasoning_effort=None,
                    sandbox_policy=None,
                    approval_mode=None,
                    web_search_mode=None,
                    plan_mode=False,
                )
            instance.refresh_from_db()
            approval.refresh_from_db()
            self.assertEqual(instance.status, CodexInstance.STATUS_COMPLETED)
            self.assertEqual(approval.decision, ApprovalRequest.DECISION_ACCEPT)


class ClaudeStopLatchTests(TestCase):
    """A Stop (SIGTERM) landing after the loop handlers are installed but before
    ``ClaudeSDKClient`` finishes connecting must latch the cancellation, not drop
    it -- otherwise the turn starts and the Stop is ignored until a second click."""

    def _runner(self) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            # System-agent purpose keeps ``_build_options`` from building the
            # in-process MCP server, which is irrelevant to the latch.
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=None,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_sigterm_before_client_ready_latches_cancel(self) -> None:
        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        self.assertIsNone(runner._client)
        # No event loop / no client: the handler must not try to schedule an
        # interrupt, only latch the flags.
        with patch.object(claude_worker, "request_cancel") as mock_cancel:
            runner._on_sigterm()
        self.assertTrue(runner._cancelled)
        mock_cancel.assert_called_once()

    def test_run_bails_without_query_when_cancelled(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker

        runner = self._runner()
        fake = _FakeClient([_assistant(TextBlock(text="hi")), _result()])

        def _factory(*, options: Any) -> _FakeClient:
            fake.options = options
            return fake

        with (
            patch.object(claude_worker, "ClaudeSDKClient", _factory),
            patch.object(claude_worker, "request_cancel"),
        ):
            # Simulate the latch set by ``_on_sigterm`` during ``__aenter__``.
            runner._cancelled = True
            asyncio.run(runner.run())
        # The query was never submitted, so the stopped turn never started.
        self.assertEqual(fake.queries, [])


class ClaudeTerminalStatusRaceTests(TransactionTestCase):
    """If the parent forced the row terminal (a Stop via ``_mark_failed`` or a
    reconcile sweep) while the worker was still draining, the worker's
    end-of-turn commit must adopt that state, not resurrect the row.

    Uses ``TransactionTestCase`` so the mid-turn flip (run on a worker thread via
    ``asyncio.to_thread``, with its own DB connection) commits without deadlocking
    against an enclosing test transaction."""

    def test_parent_failure_is_not_resurrected_to_completed(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            instance = CodexInstance.objects.create(
                thread_id="thread-race",
                cwd=tmp,
                prompt="do it",
                events_path=str(events_path),
                pid=0,
                status=CodexInstance.STATUS_RUNNING,
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_USER,
            )
            instance_pk = instance.pk

            def _force_failed() -> None:
                CodexInstance.objects.filter(pk=instance_pk).update(
                    status=CodexInstance.STATUS_FAILED,
                    error="stopped by parent",
                )

            class _ParentStopsMidTurn(_FakeClient):
                @override
                async def query(self, prompt: str) -> None:
                    await super().query(prompt)
                    # Parent flips the row terminal after the worker set RUNNING
                    # at start but before its own end-of-turn commit.
                    await asyncio.to_thread(_force_failed)

            fake = _ParentStopsMidTurn([_assistant(TextBlock(text="ok")), _result()])

            def _factory(*, options: Any) -> _FakeClient:
                fake.options = options
                return fake

            with (
                patch.object(claude_worker, "ClaudeSDKClient", _factory),
                patch.object(claude_worker, "_apply_worker_oom_score_adjust"),
                patch("hitch.main.management.commands.claude_worker.signal.signal"),
                patch.object(claude_worker, "_build_query_input", return_value="do it"),
            ):
                claude_worker.Command().handle(
                    instance_id=instance_pk,
                    model=None,
                    reasoning_effort=None,
                    sandbox_policy=None,
                    approval_mode=None,
                    web_search_mode=None,
                    plan_mode=False,
                )
            instance.refresh_from_db()
            # Even though the turn itself "succeeded", the parent's terminal
            # FAILED state is preserved -- not clobbered with COMPLETED.
            self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
            self.assertEqual(instance.error, "stopped by parent")


class ClaudeWorkflowBaseInstructionsTests(TestCase):
    """A Claude QA/PR/Spec-Critic workflow never carries Codex base instructions,
    even when the global provider was switched back to Codex."""

    def _settings(self, **overrides: Any) -> Any:
        from hitch.main.views import SettingsValues

        defaults: dict[str, Any] = {
            "model": "",
            "reasoning_effort": "",
            "sandbox_policy": "",
            "approval_mode": "auto_review",
            "coding_agent": coding_agents.CODING_AGENT_HITCH,
            "extra_system_prompt": "",
            "use_worktrees": False,
            "auto_pr_enabled": False,
            "auto_qa_enabled": False,
            "spec_critic_enabled": False,
            "web_search_mode": "",
            "show_archived_sessions": False,
            "last_selected_repo": "",
            "selected_project_id": None,
            "visible_session_project_ids": None,
            "show_no_project_sessions": True,
            "enable_memories": False,
            "provider": coding_agents.PROVIDER_CODEX,
        }
        defaults.update(overrides)
        return SettingsValues(**defaults)

    def test_codex_provider_settings_yield_base_instructions(self) -> None:
        # Guard the premise: these settings *would* attach base instructions on
        # the Codex path, so the Claude workflow omitting them is meaningful.
        from hitch.main import views

        self.assertIsNotNone(views._base_instructions_for_settings(self._settings()))

    def test_qa_workflow_omits_base_instructions(self) -> None:
        from hitch.main import views

        with (
            patch.object(
                views,
                "_claude_workflow_common",
                return_value=("/repo", "claude-opus-4-8", ""),
            ),
            patch.object(
                views, "_auto_merge_to_local_branch_for_session", return_value=(False, "")
            ),
            patch.object(views, "_claude_user_message_index", return_value=None),
            patch("hitch.main.views.system_agents.start_pr_qa_workflow") as mock_start,
        ):
            views._start_claude_qa_workflow(
                session_id="t",
                qa_activation=False,
                settings=self._settings(),
                input_image_paths=[],
            )
        self.assertNotIn("base_instructions", mock_start.call_args.kwargs)

    def test_spec_critic_follow_up_omits_base_instructions(self) -> None:
        from hitch.main import views

        with (
            patch.object(
                views,
                "_claude_workflow_common",
                return_value=("/repo", "claude-opus-4-8", ""),
            ),
            patch.object(views, "_auto_pr_enabled_for_session", return_value=False),
            patch.object(views, "_auto_qa_enabled_for_session", return_value=False),
            patch.object(
                views, "_auto_merge_to_local_branch_for_session", return_value=(False, "")
            ),
            patch.object(views, "_claude_user_message_index", return_value=None),
            patch("hitch.main.views.system_agents.start_spec_critic_workflow") as mock_spec,
        ):
            views._start_claude_spec_critic_follow_up(
                session_id="t",
                prompt="do it",
                settings=self._settings(),
                input_image_paths=[],
            )
        self.assertNotIn("base_instructions", mock_spec.call_args.kwargs)


class ClaudeSandboxDefaultTests(TestCase):
    """An empty/"Codex default" sandbox confines a *visible* Claude session to
    workspace-write so approve_all can't run fully unconfined; hidden runs keep
    their explicit (possibly empty) sandbox."""

    def _runner(
        self,
        *,
        purpose: str,
        sandbox_policy: str | None,
        approval_mode: str | None = None,
    ) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=purpose,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_visible_session_defaults_to_workspace_write(self) -> None:
        runner = self._runner(
            purpose=CodexInstance.PURPOSE_USER, sandbox_policy=None
        )
        self.assertEqual(
            runner._sandbox_policy, claude_options.SANDBOX_WORKSPACE_WRITE
        )

    def test_hidden_run_keeps_empty_sandbox(self) -> None:
        runner = self._runner(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT, sandbox_policy=None
        )
        self.assertIsNone(runner._sandbox_policy)

    def test_system_feedback_turn_defaults_to_workspace_write(self) -> None:
        # Feedback turns (QA/Spec feedback continued in the user's session) can
        # inherit approve_all but skip the hidden system-agent guard, so they need
        # the same safe default as user turns -- otherwise prompt-derived feedback
        # could auto-run unsandboxed host actions.
        runner = self._runner(
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK, sandbox_policy=None
        )
        self.assertEqual(
            runner._sandbox_policy, claude_options.SANDBOX_WORKSPACE_WRITE
        )

    def test_system_feedback_approve_all_confines_edits(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner(
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            sandbox_policy=None,
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
        )
        result = asyncio.run(
            runner._can_use_tool("Write", {"file_path": "/etc/passwd"}, None)
        )
        self.assertIsInstance(result, PermissionResultDeny)

    def test_system_feedback_keeps_explicit_sandbox(self) -> None:
        # An explicit read-only choice on the originating session must survive.
        runner = self._runner(
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            sandbox_policy=claude_options.SANDBOX_READ_ONLY,
        )
        self.assertEqual(
            runner._sandbox_policy, claude_options.SANDBOX_READ_ONLY
        )

    def test_approve_all_default_sandbox_confines_edits(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        # Visible session, approve_all, no explicit sandbox: the workspace-write
        # default's cwd guard still denies an edit outside the repo.
        runner = self._runner(
            purpose=CodexInstance.PURPOSE_USER,
            sandbox_policy=None,
            approval_mode=claude_options.APPROVAL_APPROVE_ALL,
        )
        result = asyncio.run(
            runner._can_use_tool("Write", {"file_path": "/etc/passwd"}, None)
        )
        self.assertIsInstance(result, PermissionResultDeny)


class ClaudeFilesystemSettingsGatingTests(TestCase):
    """Visible sessions load repo/user ``.claude`` settings (CLAUDE.md memory,
    project MCP) under every sandbox -- a visible session is the user's own repo,
    the same trust boundary their local ``claude`` honors. Hidden runs (which may
    target untrusted repos) never load them."""

    def _setting_sources(
        self, *, purpose: str, sandbox_policy: str | None
    ) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance(
            thread_id="t",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=purpose,
        )
        runner = claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=sandbox_policy,
            approval_mode=None,
            web_search_mode=None,
            plan_mode=False,
        )
        return runner._build_options().setting_sources

    def test_visible_session_loads_settings_under_every_sandbox(self) -> None:
        for sandbox in (
            claude_options.SANDBOX_WORKSPACE_WRITE,
            claude_options.SANDBOX_READ_ONLY,
            claude_options.SANDBOX_DANGER_FULL_ACCESS,
            None,
        ):
            self.assertEqual(
                self._setting_sources(
                    purpose=CodexInstance.PURPOSE_USER, sandbox_policy=sandbox
                ),
                ["user", "project", "local"],
                sandbox,
            )

    def test_feedback_turn_loads_settings(self) -> None:
        # A QA/PR feedback turn continues the user's own trusted session thread,
        # so it loads CLAUDE.md memory and project MCP like a user turn.
        self.assertEqual(
            self._setting_sources(
                purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
                sandbox_policy=claude_options.SANDBOX_WORKSPACE_WRITE,
            ),
            ["user", "project", "local"],
        )

    def test_hidden_run_blocks_settings(self) -> None:
        self.assertEqual(
            self._setting_sources(
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                sandbox_policy=claude_options.SANDBOX_WORKSPACE_WRITE,
            ),
            [],
        )


class ClaudePlanModeStateTests(TestCase):
    """Claude sessions have no rollout collaboration mode, so plan-mode state
    must come from the transcript -- not a sticky per-turn ``plan_mode`` flag
    that would keep follow-ups in plan mode after the plan is resolved."""

    def _claude_instance(self) -> None:
        CodexInstance.objects.create(
            thread_id="c",
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
            plan_mode=True,
        )

    def test_plan_mode_clears_after_final_answer(self) -> None:
        from types import SimpleNamespace

        from hitch.main import views

        self._claude_instance()
        entries = [{"kind": "plan", "text": "# plan"}, {"kind": "agent", "text": "done"}]
        state = views._thread_plan_mode_state(
            "c", SimpleNamespace(), entries, latest_collaboration_mode=None
        )
        self.assertFalse(state.active)

    def test_plan_mode_active_while_awaiting_approval(self) -> None:
        from types import SimpleNamespace

        from hitch.main import views

        self._claude_instance()
        entries = [{"kind": "plan", "text": "# plan"}]
        state = views._thread_plan_mode_state(
            "c", SimpleNamespace(), entries, latest_collaboration_mode=None
        )
        self.assertTrue(state.active)


class ClaudeSystemAgentIndexingTests(TestCase):
    """A Claude system-agent spawn must stamp the row hidden and resolve its
    project from ``cwd``; otherwise the ``codex_updated_at`` it writes makes the
    indexer skip the backfill, leaking the run into normal views and dropping it
    from project-filtered System Sessions."""

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_system_agent_session_is_hidden_and_project_scoped(
        self, mock_launch: MagicMock
    ) -> None:
        from hitch.main.models import SessionMetadata
        from hitch.main.test.support import _make_project

        mock_launch.return_value = codex_pool.WorkerLaunch(pid=7)
        with (
            tempfile.TemporaryDirectory() as repo,
            override_settings(CODEX_EVENTS_DIR=Path(repo)),
        ):
            project = _make_project(name="p", repo_path=repo)
            instance = codex_pool.spawn_new_session(
                cwd=repo,
                prompt="review this",
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                agent_kind="pr_qa_correctness",
            )
            row = SessionMetadata.objects.get(thread_id=instance.thread_id)
            self.assertTrue(row.is_hidden_system_session)
            self.assertEqual(row.project_id, project.pk)

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_user_session_is_not_flagged_hidden(self, mock_launch: MagicMock) -> None:
        from hitch.main.models import SessionMetadata

        mock_launch.return_value = codex_pool.WorkerLaunch(pid=8)
        with (
            tempfile.TemporaryDirectory() as repo,
            override_settings(CODEX_EVENTS_DIR=Path(repo)),
        ):
            instance = codex_pool.spawn_new_session(
                cwd=repo,
                prompt="hi",
                backend=CodexInstance.BACKEND_CLAUDE,
            )
            row = SessionMetadata.objects.get(thread_id=instance.thread_id)
            self.assertFalse(row.is_hidden_system_session)


class CreateClaudeSessionThreadTests(TestCase):
    """The Spec Critic preflight needs a Claude thread shell before the visible
    implementation turn: a metadata row plus a completed placeholder instance so
    the backend is recoverable from history."""

    def test_creates_metadata_and_claude_placeholder(self) -> None:
        from hitch.main.models import SessionMetadata

        thread_id = codex_pool.create_claude_session_thread(
            cwd="/repo",
            name="Spec preflight",
            model=claude_options.DEFAULT_CLAUDE_MODEL,
        )
        self.assertTrue(thread_id)
        row = SessionMetadata.objects.get(thread_id=thread_id)
        self.assertEqual(row.cwd, "/repo")
        placeholder = CodexInstance.objects.get(thread_id=thread_id)
        self.assertEqual(placeholder.backend, CodexInstance.BACKEND_CLAUDE)
        self.assertEqual(placeholder.status, CodexInstance.STATUS_COMPLETED)
        self.assertEqual(placeholder.pid, 0)
        # No worker is launched -- the placeholder only fixes the backend.
        self.assertEqual(placeholder.events_path, "")

    def test_developer_instructions_persist_on_placeholder(self) -> None:
        # A fresh ``/qa`` shell may finish its workflow without spawning a visible
        # turn, so the developer prompt has to live on the placeholder -- otherwise
        # a later follow-up inherits the latest instance's empty value and silently
        # drops the project/extra developer instructions.
        thread_id = codex_pool.create_claude_session_thread(
            cwd="/repo",
            name="QA review",
            model=claude_options.DEFAULT_CLAUDE_MODEL,
            developer_instructions="Always run the linter.",
        )
        placeholder = CodexInstance.objects.get(thread_id=thread_id)
        self.assertEqual(
            placeholder.developer_instructions, "Always run the linter."
        )

    def test_developer_instructions_default_to_empty(self) -> None:
        thread_id = codex_pool.create_claude_session_thread(
            cwd="/repo", name="Spec preflight"
        )
        placeholder = CodexInstance.objects.get(thread_id=thread_id)
        self.assertEqual(placeholder.developer_instructions, "")


class CodexFollowupModelTests(TestCase):
    """A resumed Codex thread must never be queued with a ``claude-*`` model id
    that leaked in from the global provider cookie."""

    def test_resumed_model_wins_and_claude_cookie_is_dropped(self) -> None:
        from types import SimpleNamespace

        from hitch.main import views

        claude_model = next(iter(claude_options.VALID_CLAUDE_MODELS))

        def _settings(model: str | None) -> Any:
            return cast(Any, SimpleNamespace(model=model))

        # The thread's own model takes priority over the cookie.
        self.assertEqual(
            views._codex_followup_model(
                SimpleNamespace(model="gpt-5-codex"), _settings(claude_model)
            ),
            "gpt-5-codex",
        )
        # No thread model + a Claude cookie -> drop it so Codex picks its default.
        self.assertIsNone(
            views._codex_followup_model(
                SimpleNamespace(model=""), _settings(claude_model)
            )
        )
        # A Codex cookie is preserved when the thread has no model yet.
        self.assertEqual(
            views._codex_followup_model(
                SimpleNamespace(model=None), _settings("gpt-5-codex")
            ),
            "gpt-5-codex",
        )


class SessionIndexInvalidationTests(TestCase):
    def test_codex_refresh_does_not_invalidate_claude_sessions(self) -> None:
        from django.utils import timezone

        from hitch.main.models import SessionMetadata
        from hitch.main.sessions import session_index

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


class CandidateBackendNormalizationTests(TestCase):
    """A candidate thread's per-turn model must match its fixed backend."""

    def _instance(self, thread_id: str, backend: str) -> None:
        CodexInstance.objects.create(
            thread_id=thread_id,
            cwd="/repo",
            prompt="x",
            events_path="x",
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=backend,
        )

    def test_thread_backend_resolved_from_history(self) -> None:
        from hitch.main import views

        self._instance("claude-cand", CodexInstance.BACKEND_CLAUDE)
        self._instance("codex-cand", CodexInstance.BACKEND_CODEX)
        self.assertEqual(
            views._candidate_thread_backend("claude-cand"),
            CodexInstance.BACKEND_CLAUDE,
        )
        self.assertEqual(
            views._candidate_thread_backend("codex-cand"),
            CodexInstance.BACKEND_CODEX,
        )
        # No history defaults to Codex.
        self.assertEqual(
            views._candidate_thread_backend("missing"),
            CodexInstance.BACKEND_CODEX,
        )

    def test_model_snapped_to_backend(self) -> None:
        from hitch.main import claude_options, views

        claude_model = next(iter(claude_options.VALID_CLAUDE_MODELS))
        # A Claude thread handed a Codex model id snaps to the Claude default.
        self.assertEqual(
            views._model_for_thread_backend(
                backend=CodexInstance.BACKEND_CLAUDE, model="gpt-5-codex"
            ),
            claude_options.DEFAULT_CLAUDE_MODEL,
        )
        # A valid Claude model id is preserved.
        self.assertEqual(
            views._model_for_thread_backend(
                backend=CodexInstance.BACKEND_CLAUDE, model=claude_model
            ),
            claude_model,
        )
        # A Codex thread must not be handed a Claude model id; drop to default.
        self.assertIsNone(
            views._model_for_thread_backend(
                backend=CodexInstance.BACKEND_CODEX, model=claude_model
            )
        )
        # A Codex thread handed a Claude model keeps its own prior Codex model
        # as the fallback so plan turns (which require a model) still have one.
        self.assertEqual(
            views._model_for_thread_backend(
                backend=CodexInstance.BACKEND_CODEX,
                model=claude_model,
                codex_fallback_model="gpt-5-codex",
            ),
            "gpt-5-codex",
        )
        # A Claude fallback is itself rejected for a Codex thread (drop to None).
        self.assertIsNone(
            views._model_for_thread_backend(
                backend=CodexInstance.BACKEND_CODEX,
                model=claude_model,
                codex_fallback_model=claude_model,
            )
        )
        # A Codex model id passes through untouched for a Codex thread.
        self.assertEqual(
            views._model_for_thread_backend(
                backend=CodexInstance.BACKEND_CODEX, model="gpt-5-codex"
            ),
            "gpt-5-codex",
        )


class ProposeSessionToolTests(TestCase):
    def test_build_server_exposes_propose_session(self) -> None:
        from hitch.main.runtime import claude_tools

        server = claude_tools.build_hitch_mcp_server(cwd="/repo", thread_id="t")
        self.assertEqual(server["name"], claude_tools.HITCH_MCP_SERVER_NAME)
        self.assertEqual(
            claude_tools.PROPOSE_SESSION_TOOL_NAME, "mcp__hitch__propose_session"
        )


def _result(
    subtype: str = "success",
    is_error: bool = False,
    structured_output: Any = None,
    usage: Any = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="sess-final",
        structured_output=structured_output,
        usage=usage,
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
        from hitch.main.sessions import claude_session_entries

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
        from hitch.main.sessions import claude_session_entries

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
            {
                "op": "steer",
                "id": "sid-1",
                "input": " go ",
                "inputImagePaths": ["/a.png", "", 3],
            }
        ).encode()
        text, paths, steer_id = claude_worker._steer_request(raw)
        self.assertEqual(text, "go")
        self.assertEqual(paths, ["/a.png"])
        self.assertEqual(steer_id, "sid-1")

    def test_steer_request_ignores_non_steer(self) -> None:
        import json

        from hitch.main.management.commands import claude_worker

        self.assertEqual(
            claude_worker._steer_request(json.dumps({"op": "other"}).encode()),
            ("", [], ""),
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

        from hitch.main.sessions import claude_session_entries

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
        from hitch.main.sessions import claude_session_entries

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

    def test_monitor_renders_as_command_execution(self) -> None:
        # Monitor runs a shell script under Bash rules, so it must surface its
        # command (auditable) rather than a generic dynamicToolCall.
        translator = claude_translate.EventTranslator()
        ev = translator.translate(
            _assistant(
                ToolUseBlock(id="m", name="Monitor", input={"command": "pytest -q"})
            )
        )
        item = ev[0][1]["item"]
        self.assertEqual(item["type"], "commandExecution")
        self.assertEqual(item["command"], "pytest -q")

    def test_powershell_renders_as_command_execution(self) -> None:
        # PowerShell runs host commands (gated like Bash in the worker), so its
        # command text must be auditable rather than a generic dynamicToolCall.
        translator = claude_translate.EventTranslator()
        ev = translator.translate(
            _assistant(
                ToolUseBlock(
                    id="p", name="PowerShell", input={"command": "Get-ChildItem"}
                )
            )
        )
        item = ev[0][1]["item"]
        self.assertEqual(item["type"], "commandExecution")
        self.assertEqual(item["command"], "Get-ChildItem")

    def test_repeated_message_id_yields_unique_item_ids(self) -> None:
        # Multiple assistant messages can share a message_id; their text item ids
        # must differ so the live renderer can't overwrite pre-tool text with the
        # final response.
        translator = claude_translate.EventTranslator()
        first = translator.translate(_assistant(TextBlock(text="pre-tool")))
        second = translator.translate(_assistant(TextBlock(text="final")))
        self.assertNotEqual(
            first[0][1]["item"]["id"], second[0][1]["item"]["id"]
        )
        # started/completed for a single block still share one id.
        self.assertEqual(first[0][1]["item"]["id"], first[1][1]["item"]["id"])

    def test_structured_output_result_becomes_agent_message(self) -> None:
        # With an output_schema the SDK returns the validated JSON on the
        # ResultMessage, not in an agentMessage; the translator must surface it
        # as a final agentMessage so the workflow's events-file parser sees it.
        translator = claude_translate.EventTranslator()
        verdict = {"verdict": "LGTM", "confidence": "high"}
        events = translator.translate(_result(structured_output=verdict))
        completed = [e for e in events if e[0] == "item/completed"]
        self.assertEqual(len(completed), 1)
        item = completed[0][1]["item"]
        self.assertEqual(item["type"], "agentMessage")
        self.assertEqual(json.loads(item["text"]), verdict)

    def test_structured_output_absent_emits_nothing(self) -> None:
        translator = claude_translate.EventTranslator()
        self.assertEqual(translator.translate(_result()), [])

    def test_structured_output_read_back_by_final_agent_text(self) -> None:
        # End-to-end: a worker stream whose only agent output is the structured
        # result must leave parseable JSON for system_agents._final_agent_text.
        from hitch.main import system_agents

        verdict = {"verdict": "request_changes", "confidence": "medium"}
        translator = claude_translate.EventTranslator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for method, payload in translator.translate(
                    _result(structured_output=verdict)
                ):
                    fh.write(json.dumps({"method": method, "payload": payload}) + "\n")
            text = system_agents._final_agent_text(str(path))
        self.assertEqual(json.loads(text), verdict)


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

    def test_image_turn_records_one_marked_user_message(self) -> None:
        # A mixed text+image turn: the worker records the turn itself (text plus an
        # [image] marker) because the SDK drops the image and echoes only the text.
        # The echo must be suppressed so the turn yields exactly one user entry.
        from hitch.main.management.commands import claude_worker

        messages = [
            # The SDK's echoed prompt for the image turn (image stripped, text kept).
            UserMessage(content=[TextBlock(text="please help")]),
            _assistant(TextBlock(text="done")),
            _result(),
        ]
        # A non-str query input is how _build_query_input signals image blocks.
        with patch.object(claude_worker, "_build_query_input", return_value=["<stream>"]):
            _runner, written = self._run(messages)
        user_texts = [
            json.loads(line)["payload"]["item"]["text"]
            for line in written.splitlines()
            if line.strip()
            and json.loads(line)["method"] == "item/completed"
            and json.loads(line)["payload"]["item"].get("type") == "userMessage"
        ]
        self.assertEqual(user_texts, ["please help\n[image]"])

    def test_pre_loop_sigterm_skips_the_turn(self) -> None:
        # A Stop that landed before the loop installed its handlers (recorded by
        # the protective handler) must abort the turn before it starts -- the
        # query is never submitted, so a stopped turn cannot start anyway.
        import asyncio

        from hitch.main.management.commands import claude_worker

        fake = _FakeClient([_assistant(TextBlock(text="hi")), _result()])

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
            claude_worker._PENDING_SIGTERM = True
            try:
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
            finally:
                claude_worker._PENDING_SIGTERM = False
        # The turn never started: no query was submitted and no result was seen,
        # so ``handle`` reconciles it as a stopped (failed) turn.
        self.assertEqual(fake.queries, [])
        self.assertFalse(runner.saw_result)
        self.assertTrue(runner._cancelled)

    def test_pre_loop_handlers_record_signals(self) -> None:
        from hitch.main.management.commands import claude_worker

        try:
            with patch.object(claude_worker, "request_cancel") as cancel:
                claude_worker._pre_loop_sigterm(15, None)
            self.assertTrue(claude_worker._PENDING_SIGTERM)
            cancel.assert_called_once()
            # SIGUSR1 handler is a no-op (the steer is already on disk).
            claude_worker._pre_loop_sigusr1(10, None)
        finally:
            claude_worker._PENDING_SIGTERM = False

    def test_stream_without_result_is_not_marked_completed(self) -> None:
        # A truncated/aborted stream (no ResultMessage) must not look successful.
        runner, _written = self._run([_assistant(TextBlock(text="partial"))])
        self.assertFalse(runner.saw_result)
        self.assertFalse(runner.failed)

    def test_steered_followup_response_is_drained(self) -> None:
        # A steer issued mid-turn schedules a second query; the runner must keep
        # draining so the steered prompt's response is translated, not dropped
        # when the first response's ResultMessage ends ``receive_response``.
        import asyncio

        from hitch.main.management.commands import claude_worker

        # First response: registers a pending steer (as the steer thread would)
        # then ends. Second response: the steered prompt's output.
        responses = [
            [_assistant(TextBlock(text="first")), _result()],
            [_assistant(TextBlock(text="steered reply")), _result()],
        ]

        class _SteeringClient(_FakeClient):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__([], **kwargs)
                self._scripts = list(responses)
                self._runner: Any = None

            @override
            async def receive_response(self) -> Any:
                script = self._scripts.pop(0)
                for message in script:
                    # Simulate a steer arriving during the first response.
                    if (
                        isinstance(message, ResultMessage)
                        and self._scripts
                        and self._runner is not None
                    ):
                        self._runner._add_outstanding(1)
                    yield message

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
            client = _SteeringClient()

            def _factory(*, options: Any) -> _SteeringClient:
                client.options = options
                return client

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
                client._runner = runner
                asyncio.run(runner.run())
            written = events_path.read_text(encoding="utf-8")
        # Both the original and the steered response were translated.
        self.assertIn("first", written)
        self.assertIn("steered reply", written)
        self.assertFalse(runner.failed)


class ClaudeTokenUsageTests(TestCase):
    def test_normalize_folds_cache_into_input(self) -> None:
        from hitch.main.runtime import claude_usage

        usage = claude_usage.normalize_turn_usage(
            {
                "input_tokens": 100,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
                "output_tokens": 40,
            },
            "claude-opus-4-8",
        )
        assert usage is not None
        # Codex shape: input includes cache; cached is read+creation.
        self.assertEqual(usage["input_tokens"], 150)
        self.assertEqual(usage["cached_input_tokens"], 50)
        self.assertEqual(usage["output_tokens"], 40)
        self.assertEqual(usage["context_tokens"], 190)
        self.assertEqual(usage["model_context_window"], 200_000)

    def test_normalize_returns_none_for_empty_usage(self) -> None:
        from hitch.main.runtime import claude_usage

        self.assertIsNone(claude_usage.normalize_turn_usage(None, "claude-opus-4-8"))
        self.assertIsNone(claude_usage.normalize_turn_usage({}, "claude-opus-4-8"))
        self.assertIsNone(
            claude_usage.normalize_turn_usage(
                {"input_tokens": 0, "output_tokens": 0}, None
            )
        )

    def test_unknown_model_uses_default_context_window(self) -> None:
        self.assertEqual(claude_options.context_window_for("nope"), 200_000)
        self.assertEqual(claude_options.context_window_for(None), 200_000)

    def test_record_turn_usage_accumulates(self) -> None:
        from hitch.main.models import ArchivedSessionTokenUsage
        from hitch.main.runtime import claude_usage

        raw = {
            "input_tokens": 100,
            "cache_read_input_tokens": 10,
            "output_tokens": 40,
        }
        claude_usage.record_turn_usage("thread-tok", raw, "claude-opus-4-8")
        claude_usage.record_turn_usage("thread-tok", raw, "claude-opus-4-8")
        row = ArchivedSessionTokenUsage.objects.get(thread_id="thread-tok")
        self.assertEqual(row.rollout_path, "")
        # input (110) + output (40) accumulated over two turns.
        self.assertEqual(row.input_tokens, 220)
        self.assertEqual(row.cached_input_tokens, 20)
        self.assertEqual(row.output_tokens, 80)
        self.assertEqual(row.total_tokens, 300)
        # Context occupancy reflects only the latest turn.
        self.assertEqual(row.context_tokens, 150)
        # Daily usage accumulated (non-cached input = 100 per turn).
        day = next(iter(row.daily_usage.values()))
        self.assertEqual(day["input"], 200)
        self.assertEqual(day["output"], 80)
        self.assertEqual(day["cached"], 20)

    def test_record_overwrites_stale_logic_row(self) -> None:
        from hitch.main import models
        from hitch.main.models import ArchivedSessionTokenUsage
        from hitch.main.runtime import claude_usage

        ArchivedSessionTokenUsage.objects.create(
            thread_id="thread-stale",
            rollout_path="",
            input_tokens=999,
            usage_logic_version=0,
        )
        claude_usage.record_turn_usage(
            "thread-stale",
            {"input_tokens": 50, "output_tokens": 10},
            "claude-opus-4-8",
        )
        row = ArchivedSessionTokenUsage.objects.get(thread_id="thread-stale")
        # Overwritten, not added to the stale 999.
        self.assertEqual(row.input_tokens, 50)
        self.assertEqual(
            row.usage_logic_version, models.TOKEN_USAGE_LOGIC_VERSION
        )

    def test_render_helper_formats_claude_cache_row(self) -> None:
        from hitch.main import views
        from hitch.main.models import ArchivedSessionTokenUsage
        from hitch.main.runtime import claude_usage

        claude_usage.record_turn_usage(
            "thread-render",
            {"input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 40},
            "claude-opus-4-8",
        )
        formatted = views._claude_token_usage_for("thread-render")
        assert formatted is not None
        self.assertEqual(formatted["input"], "100")
        self.assertEqual(formatted["cached"], "50")
        self.assertEqual(formatted["output"], "40")
        self.assertIn("context", formatted)

        # A Codex-shaped row (rollout_path set) is not served as Claude usage.
        ArchivedSessionTokenUsage.objects.filter(thread_id="thread-render").update(
            rollout_path="/some/rollout.jsonl"
        )
        self.assertIsNone(views._claude_token_usage_for("thread-render"))
        # Missing row → None.
        self.assertIsNone(views._claude_token_usage_for("thread-absent"))

    def test_worker_records_usage_on_completion(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker
        from hitch.main.models import ArchivedSessionTokenUsage

        messages = [
            _assistant(TextBlock(text="done")),
            _result(usage={"input_tokens": 200, "output_tokens": 60}),
        ]
        fake = _FakeClient(messages)

        def _factory(*, options: Any) -> _FakeClient:
            fake.options = options
            return fake

        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            instance = CodexInstance.objects.create(
                thread_id="thread-worker-tok",
                cwd=tmp,
                prompt="hi",
                events_path=str(events_path),
                pid=0,
                status=CodexInstance.STATUS_RUNNING,
                backend=CodexInstance.BACKEND_CLAUDE,
                purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
                model="claude-opus-4-8",
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
            claude_worker._record_token_usage(instance, runner)
        row = ArchivedSessionTokenUsage.objects.get(thread_id="thread-worker-tok")
        self.assertEqual(row.input_tokens, 200)
        self.assertEqual(row.output_tokens, 60)


class WorkerApprovalTests(TestCase):
    def _runner(
        self,
        *,
        purpose: str,
        approval_mode: str | None,
        sandbox_policy: str | None = "workspaceWrite",
    ) -> Any:
        import io

        from hitch.main.management.commands import claude_worker

        instance = CodexInstance.objects.create(
            thread_id="thread-x",
            cwd="/repo",
            prompt="p",
            events_path="/dev/null",
            pid=0,
            status=CodexInstance.STATUS_RUNNING,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=purpose,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model="claude-opus-4-8",
            reasoning_effort=None,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_auto_review_hidden_run_auto_approves_without_pending(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker

        runner = self._runner(
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT, approval_mode="auto_review"
        )
        with patch.object(claude_worker, "_create_pending_approval") as mock_pending:
            result = asyncio.run(runner._can_use_tool("Bash", {"command": "ls"}, None))
        # A hidden system-agent run must not block on an unanswerable browser
        # approval; it auto-approves under auto_review.
        self.assertIsInstance(result, PermissionResultAllow)
        mock_pending.assert_not_called()

    def test_auto_review_user_turn_still_requests_approval(self) -> None:
        import asyncio

        from hitch.main.management.commands import claude_worker

        runner = self._runner(
            purpose=CodexInstance.PURPOSE_USER, approval_mode="auto_review"
        )
        # A visible user turn still escalates: stub the pending-approval create
        # and decision so the call returns without real I/O.
        with (
            patch.object(
                claude_worker, "_create_pending_approval", return_value=7
            ) as mock_pending,
            patch.object(claude_worker, "_wait_for_decision", return_value="approved"),
        ):
            asyncio.run(runner._can_use_tool("Bash", {"command": "ls"}, None))
        mock_pending.assert_called_once()


class ClaudeFollowUpAutoQaTests(TestCase):
    def _settings(self, **overrides: Any) -> Any:
        from hitch.main import views

        base: dict[str, Any] = {
            "model": "claude-opus-4-8",
            "reasoning_effort": "",
            "sandbox_policy": "",
            "approval_mode": "auto_review",
            "coding_agent": "",
            "extra_system_prompt": "",
            "use_worktrees": False,
            "auto_pr_enabled": False,
            "auto_qa_enabled": True,
            "spec_critic_enabled": False,
            "web_search_mode": "",
            "show_archived_sessions": False,
            "last_selected_repo": "",
            "selected_project_id": None,
            "visible_session_project_ids": None,
            "show_no_project_sessions": False,
            "enable_memories": False,
            "provider": coding_agents.PROVIDER_CLAUDE,
        }
        base.update(overrides)
        return views.SettingsValues(**base)

    def _claude_instance(self, **overrides: Any) -> CodexInstance:
        from hitch.main.models import SessionMetadata

        SessionMetadata.objects.update_or_create(
            thread_id="claude-thread",
            defaults={"cwd": "/repo", "auto_qa_enabled": True},
        )
        defaults: dict[str, Any] = {
            "thread_id": "claude-thread",
            "cwd": "/repo",
            "prompt": "p",
            "events_path": "/dev/null",
            "pid": 0,
            "status": CodexInstance.STATUS_COMPLETED,
            "backend": CodexInstance.BACKEND_CLAUDE,
            "purpose": CodexInstance.PURPOSE_USER,
        }
        defaults.update(overrides)
        return CodexInstance.objects.create(**defaults)

    def test_follow_up_forwards_auto_qa(self) -> None:
        from hitch.main import views

        self._claude_instance()
        with (
            patch.object(codex_pool, "spawn_turn") as mock_spawn,
            patch.object(views, "_allowed_session_cwds", return_value={"/repo"}),
            patch.object(views, "_claude_user_message_index", return_value=2),
        ):
            response = views._send_claude_follow_up(
                session_id="claude-thread",
                prompt="next",
                plan_mode=False,
                settings=self._settings(),
                input_image_paths=[],
            )
        self.assertEqual(response.status_code, 302)
        kwargs = mock_spawn.call_args.kwargs
        self.assertTrue(kwargs["auto_qa_enabled"])
        self.assertEqual(kwargs["user_message_index"], 2)

    def test_follow_up_uses_session_approval_mode_override(self) -> None:
        from hitch.main import views
        from hitch.main.models import SessionMetadata

        self._claude_instance()
        # A per-session approval override (set from the session header) must win
        # over the global settings value, as the Codex follow-up path does.
        SessionMetadata.objects.update_or_create(
            thread_id="claude-thread",
            defaults={"cwd": "/repo", "approval_mode": "deny_all"},
        )
        with (
            patch.object(codex_pool, "spawn_turn") as mock_spawn,
            patch.object(views, "_allowed_session_cwds", return_value={"/repo"}),
            patch.object(views, "_claude_user_message_index", return_value=0),
        ):
            views._send_claude_follow_up(
                session_id="claude-thread",
                prompt="next",
                plan_mode=False,
                settings=self._settings(),  # global approval_mode is "auto_review"
                input_image_paths=[],
            )
        self.assertEqual(mock_spawn.call_args.kwargs["approval_mode"], "deny_all")

    def test_follow_up_in_plan_mode_skips_auto_qa(self) -> None:
        from hitch.main import views

        self._claude_instance()
        with (
            patch.object(codex_pool, "spawn_turn") as mock_spawn,
            patch.object(views, "_allowed_session_cwds", return_value={"/repo"}),
        ):
            views._send_claude_follow_up(
                session_id="claude-thread",
                prompt="next",
                plan_mode=True,
                settings=self._settings(),
                input_image_paths=[],
            )
        self.assertNotIn("auto_qa_enabled", mock_spawn.call_args.kwargs)

    def test_follow_up_prefers_prior_claude_model_over_codex_cookie(self) -> None:
        from hitch.main import views

        self._claude_instance(model="claude-sonnet-4-6")
        with (
            patch.object(codex_pool, "spawn_turn") as mock_spawn,
            patch.object(views, "_allowed_session_cwds", return_value={"/repo"}),
            patch.object(views, "_claude_user_message_index", return_value=0),
        ):
            views._send_claude_follow_up(
                session_id="claude-thread",
                prompt="next",
                plan_mode=False,
                # Settings cookie holds a Codex model id (provider switched back).
                settings=self._settings(model="gpt-5-codex"),
                input_image_paths=[],
            )
        # The session's prior Claude model is preserved rather than defaulting.
        self.assertEqual(mock_spawn.call_args.kwargs["model"], "claude-sonnet-4-6")

    def test_manual_qa_starts_workflow_on_claude_session(self) -> None:
        from hitch.main import system_agents, views

        self._claude_instance(model="claude-sonnet-4-6")
        with (
            patch.object(views, "_is_allowed_session_cwd", return_value=True),
            patch.object(views, "_claude_user_message_index", return_value=2),
            patch.object(system_agents, "start_pr_qa_workflow") as mock_start,
        ):
            response = views._start_claude_qa_workflow(
                session_id="claude-thread",
                qa_activation=True,
                settings=self._settings(model="claude-sonnet-4-6"),
                input_image_paths=[],
            )
        self.assertEqual(response.status_code, 302)
        kwargs = mock_start.call_args.kwargs
        self.assertEqual(kwargs["main_thread_id"], "claude-thread")
        self.assertEqual(kwargs["model"], "claude-sonnet-4-6")
        self.assertEqual(kwargs["initial_user_message_index"], 2)
        # /qa reviews without opening a PR; the workflow records the Claude
        # backend from the thread and runs its sub-agents as Claude workers.
        self.assertFalse(kwargs["open_pr_on_lgtm"])

    def test_manual_pr_opens_pr_on_claude_session(self) -> None:
        from hitch.main import system_agents, views

        self._claude_instance(model="claude-sonnet-4-6")
        with (
            patch.object(views, "_is_allowed_session_cwd", return_value=True),
            patch.object(views, "_claude_user_message_index", return_value=0),
            patch.object(system_agents, "start_pr_qa_workflow") as mock_start,
        ):
            views._start_claude_qa_workflow(
                session_id="claude-thread",
                qa_activation=False,  # /pr
                settings=self._settings(),
                input_image_paths=[],
            )
        # /pr leaves open_pr_on_lgtm at its default (True) so hitch opens the PR.
        self.assertNotIn("open_pr_on_lgtm", mock_start.call_args.kwargs)

    def test_follow_up_forwards_auto_pr(self) -> None:
        from hitch.main import views
        from hitch.main.models import SessionMetadata

        self._claude_instance()
        SessionMetadata.objects.filter(thread_id="claude-thread").update(
            auto_pr_enabled=True, auto_qa_enabled=False
        )
        with (
            patch.object(codex_pool, "spawn_turn") as mock_spawn,
            patch.object(views, "_allowed_session_cwds", return_value={"/repo"}),
            patch.object(views, "_claude_user_message_index", return_value=1),
        ):
            views._send_claude_follow_up(
                session_id="claude-thread",
                prompt="next",
                plan_mode=False,
                settings=self._settings(),
                input_image_paths=[],
            )
        kwargs = mock_spawn.call_args.kwargs
        # Auto-PR now rides every follow-up; it supersedes Auto-QA.
        self.assertTrue(kwargs.get("auto_pr_enabled"))
        self.assertNotIn("auto_qa_enabled", kwargs)

    def test_spec_critic_runs_on_claude_follow_up(self) -> None:
        from hitch.main import system_agents, views

        self._claude_instance()
        with (
            patch.object(views, "_is_allowed_session_cwd", return_value=True),
            patch.object(views, "_claude_user_message_index", return_value=2),
            patch.object(system_agents, "start_spec_critic_workflow") as mock_start,
        ):
            response = views._start_claude_spec_critic_follow_up(
                session_id="claude-thread",
                prompt="add a parser feature",
                settings=self._settings(model="claude-sonnet-4-6"),
                input_image_paths=[],
            )
        self.assertEqual(response.status_code, 302)
        kwargs = mock_start.call_args.kwargs
        self.assertEqual(kwargs["main_thread_id"], "claude-thread")
        self.assertEqual(kwargs["prompt"], "add a parser feature")
        self.assertEqual(kwargs["model"], "claude-sonnet-4-6")
        self.assertEqual(kwargs["initial_user_message_index"], 2)

    def test_fix_pr_routes_to_monitor_workflow(self) -> None:
        from hitch.main import system_agents, views

        self._claude_instance(model="claude-sonnet-4-6")
        pr_url = "https://github.com/cberner/hitch/pull/7"
        with (
            patch.object(views, "_is_allowed_session_cwd", return_value=True),
            patch.object(views, "_claude_user_message_index", return_value=4),
            patch.object(views, "_claude_fix_pr_url", return_value=pr_url),
            patch.object(system_agents, "start_pr_monitor_workflow") as mock_monitor,
        ):
            response = views._start_claude_fix_pr_workflow(
                session_id="claude-thread",
                settings=self._settings(model="claude-sonnet-4-6"),
                input_image_paths=[],
            )
        self.assertEqual(response.status_code, 302)
        kwargs = mock_monitor.call_args.kwargs
        # /fix-pr targets the already-open PR via the monitor workflow, so it
        # never opens a second PR; the URL comes from the PR workflow handoff.
        self.assertEqual(kwargs["pr_url"], pr_url)
        self.assertEqual(kwargs["main_thread_id"], "claude-thread")
        self.assertEqual(kwargs["model"], "claude-sonnet-4-6")
        self.assertEqual(kwargs["initial_user_message_index"], 4)

    def test_fix_pr_requires_an_opened_pr(self) -> None:
        from hitch.main import system_agents, views

        self._claude_instance()
        with (
            patch.object(views, "_claude_fix_pr_url", return_value=None),
            patch.object(system_agents, "start_pr_monitor_workflow") as mock_monitor,
        ):
            response = views._start_claude_fix_pr_workflow(
                session_id="claude-thread",
                settings=self._settings(),
                input_image_paths=[],
            )
        self.assertEqual(response.status_code, 400)
        mock_monitor.assert_not_called()

    def test_workflows_honor_session_approval_mode_override(self) -> None:
        # /qa, /pr, /fix-pr and the Spec Critic follow-up must honor a per-session
        # approval override (as the Codex follow-up path does), not the global
        # settings value -- otherwise their hidden Claude agents run under a
        # different policy than the session header advertises.
        from hitch.main import system_agents, views
        from hitch.main.models import SessionMetadata

        self._claude_instance(model="claude-sonnet-4-6")
        SessionMetadata.objects.filter(thread_id="claude-thread").update(
            approval_mode="deny_all"
        )
        settings = self._settings(model="claude-sonnet-4-6")  # global "auto_review"
        pr_url = "https://github.com/cberner/hitch/pull/7"
        with (
            patch.object(views, "_is_allowed_session_cwd", return_value=True),
            patch.object(views, "_claude_user_message_index", return_value=0),
            patch.object(views, "_claude_fix_pr_url", return_value=pr_url),
            patch.object(system_agents, "start_pr_qa_workflow") as mock_qa,
            patch.object(system_agents, "start_pr_monitor_workflow") as mock_monitor,
            patch.object(system_agents, "start_spec_critic_workflow") as mock_spec,
        ):
            views._start_claude_qa_workflow(
                session_id="claude-thread",
                qa_activation=True,
                settings=settings,
                input_image_paths=[],
            )
            views._start_claude_fix_pr_workflow(
                session_id="claude-thread",
                settings=settings,
                input_image_paths=[],
            )
            views._start_claude_spec_critic_follow_up(
                session_id="claude-thread",
                prompt="add a feature",
                settings=settings,
                input_image_paths=[],
            )
        self.assertEqual(mock_qa.call_args.kwargs["approval_mode"], "deny_all")
        self.assertEqual(mock_monitor.call_args.kwargs["approval_mode"], "deny_all")
        self.assertEqual(mock_spec.call_args.kwargs["approval_mode"], "deny_all")

    def test_send_message_routes_fix_pr_to_monitor_not_qa(self) -> None:
        from django.http import HttpResponse
        from django.urls import reverse

        from hitch.main import views

        self._claude_instance()
        with (
            patch.object(
                views, "_stored_settings", return_value=self._settings()
            ),
            patch.object(
                views,
                "_start_claude_fix_pr_workflow",
                return_value=HttpResponse(status=204),
            ) as mock_fix,
            patch.object(views, "_start_claude_qa_workflow") as mock_qa,
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "claude-thread"}),
                data={"prompt": "/fix-pr"},
            )
        self.assertEqual(response.status_code, 204)
        mock_fix.assert_called_once()
        mock_qa.assert_not_called()

    def test_send_message_unarchives_archived_claude_session(self) -> None:
        # A follow-up to an archived Claude session must clear the archived bit
        # (the Claude helpers only bump recency), or the accepted turn stays
        # hidden from the session list until a manual unarchive.
        from django.http import HttpResponse
        from django.urls import reverse

        from hitch.main import views
        from hitch.main.models import SessionMetadata

        self._claude_instance()
        # cwd="" keeps _metadata_cwd_is_disallowed False without an allowlist patch.
        SessionMetadata.objects.filter(thread_id="claude-thread").update(
            codex_archived=True, cwd=""
        )
        with (
            patch.object(views, "_stored_settings", return_value=self._settings()),
            patch.object(
                views, "_send_claude_follow_up", return_value=HttpResponse(status=204)
            ),
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "claude-thread"}),
                data={"prompt": "keep going"},
            )
        self.assertEqual(response.status_code, 204)
        row = SessionMetadata.objects.get(thread_id="claude-thread")
        self.assertFalse(row.codex_archived)

    def test_send_message_spec_critic_followup_is_not_preclassified(self) -> None:
        from django.http import HttpResponse
        from django.urls import reverse

        from hitch.main import system_agents, views

        self._claude_instance()
        with (
            patch.object(
                views,
                "_stored_settings",
                return_value=self._settings(spec_critic_enabled=True),
            ),
            patch.object(system_agents, "spec_critic_should_run") as mock_should_run,
            patch.object(
                views,
                "_start_claude_spec_critic_follow_up",
                return_value=HttpResponse(status=204),
            ) as mock_spec,
        ):
            response = self.client.post(
                reverse("send_message", kwargs={"session_id": "claude-thread"}),
                data={"prompt": "add a parser feature"},
            )
        self.assertEqual(response.status_code, 204)
        # The workflow classifies in the background; the request path must not run
        # the synchronous Codex classifier.
        mock_should_run.assert_not_called()
        mock_spec.assert_called_once()


class ClaudeReasoningEffortSettingsTests(TestCase):
    """Claude accepts a fixed effort set (no ``minimal``), so the settings dialog
    must advertise it and ``update_settings`` must reject an unsupported effort
    rather than store one the Claude worker silently drops at turn time."""

    def _claude_settings(self, **overrides: Any) -> Any:
        from hitch.main import views

        base: dict[str, Any] = {
            "model": claude_options.DEFAULT_CLAUDE_MODEL,
            "reasoning_effort": "",
            "sandbox_policy": "",
            "approval_mode": "auto_review",
            "coding_agent": "",
            "extra_system_prompt": "",
            "use_worktrees": False,
            "auto_pr_enabled": False,
            "auto_qa_enabled": False,
            "spec_critic_enabled": False,
            "web_search_mode": "",
            "show_archived_sessions": False,
            "last_selected_repo": "",
            "selected_project_id": None,
            "visible_session_project_ids": None,
            "show_no_project_sessions": False,
            "enable_memories": False,
            "provider": coding_agents.PROVIDER_CLAUDE,
        }
        base.update(overrides)
        return views.SettingsValues(**base)

    def test_settings_dialog_hides_minimal_effort_for_claude(self) -> None:
        from hitch.main import views

        ctx = views._settings_context(self._claude_settings(), [])
        claude_opts = ctx["model_options_by_provider"][coding_agents.PROVIDER_CLAUDE]
        advertised = claude_opts[0]["supported_efforts"].split()
        self.assertNotIn("minimal", advertised)
        self.assertIn("high", advertised)
        # The initial render (Claude provider) marks minimal unsupported so the
        # dropdown hides it, while a real Claude effort stays selectable.
        supported = {o["value"]: o["supported"] for o in ctx["effort_options"]}
        self.assertFalse(supported["minimal"])
        self.assertTrue(supported["high"])

    def test_update_settings_rejects_minimal_effort_for_claude(self) -> None:
        from django.urls import reverse

        response = self.client.post(
            reverse("update_settings"),
            data={
                "provider": "claude",
                "model": claude_options.DEFAULT_CLAUDE_MODEL,
                "reasoning_effort": "minimal",
                "approval_mode": "auto_review",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_update_settings_accepts_supported_effort_for_claude(self) -> None:
        from django.urls import reverse

        response = self.client.post(
            reverse("update_settings"),
            data={
                "provider": "claude",
                "model": claude_options.DEFAULT_CLAUDE_MODEL,
                "reasoning_effort": "high",
                "approval_mode": "auto_review",
            },
        )
        self.assertEqual(response.status_code, 302)

    def test_settings_dialog_offers_claude_only_max_effort(self) -> None:
        # ``max`` is a Claude-only effort (absent from Codex's ReasoningEffort
        # enum); the dropdown must still render a selectable option for it, marked
        # supported under the Claude provider.
        from hitch.main import views

        ctx = views._settings_context(self._claude_settings(), [])
        supported = {o["value"]: o["supported"] for o in ctx["effort_options"]}
        self.assertIn("max", supported)
        self.assertTrue(supported["max"])

    def test_update_settings_accepts_max_effort_for_claude(self) -> None:
        # A direct POST of the Claude-only ``max`` must not 400 in the early
        # (Codex-enum) effort guard before the Claude-specific validation runs.
        from django.urls import reverse

        response = self.client.post(
            reverse("update_settings"),
            data={
                "provider": "claude",
                "model": claude_options.DEFAULT_CLAUDE_MODEL,
                "reasoning_effort": "max",
                "approval_mode": "auto_review",
            },
        )
        self.assertEqual(response.status_code, 302)


class CandidateThreadIndexTests(TestCase):
    """A Claude candidate thread is local-only, so its user-message index must
    come from the worker events, not a Codex ``thread_resume``."""

    def test_claude_candidate_index_skips_codex_resume(self) -> None:
        from hitch.main import views
        from hitch.main.sessions import session_pr_plan
        from hitch.main.views import common

        CodexInstance.objects.create(
            thread_id="cand",
            cwd="/repo",
            prompt="x",
            events_path="",
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=CodexInstance.BACKEND_CLAUDE,
        )
        with (
            patch.object(common, "Codex") as mock_codex,
            patch.object(
                session_pr_plan, "_claude_user_message_index", return_value=3
            ) as mock_count,
        ):
            result = views._candidate_thread_user_message_index("cand", MagicMock())
        self.assertEqual(result, 3)
        mock_codex.assert_not_called()
        mock_count.assert_called_once_with("cand")
