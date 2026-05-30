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

    def test_image_only_user_message_records_image_marker(self) -> None:
        # The SDK strips image blocks while parsing the echoed user message, so an
        # image-only turn arrives empty; it must still produce a userMessage event
        # so the turn is recorded and the auto-QA turn count stays accurate.
        translator = claude_translate.EventTranslator()
        events = translator.translate(UserMessage(content=[]))
        self.assertEqual(events[-1][1]["item"]["type"], "userMessage")
        self.assertEqual(events[-1][1]["item"]["text"], "[image]")


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


class HiddenAutoReviewApprovalTests(TestCase):
    """Hidden auto-review runs auto-approve only built-in mutating tools; a
    project/user MCP tool reaching ``can_use_tool`` must be denied, since these
    runs have no approval UI to gate it."""

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
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
        )
        return claude_worker._TurnRunner(
            instance=instance,
            events_file=io.StringIO(),
            model=None,
            reasoning_effort=None,
            sandbox_policy=None,
            approval_mode=claude_options.APPROVAL_AUTO_REVIEW,
            web_search_mode=None,
            plan_mode=False,
        )

    def test_builtin_mutating_tools_are_auto_allowed(self) -> None:
        import asyncio

        runner = self._runner()
        for tool in ("Bash", "Edit", "Write", "MultiEdit", "NotebookEdit"):
            result = asyncio.run(runner._can_use_tool(tool, {}, None))
            self.assertIsInstance(result, PermissionResultAllow, tool)

    def test_mcp_tool_is_denied_without_approval_ui(self) -> None:
        import asyncio

        from claude_agent_sdk import PermissionResultDeny

        runner = self._runner()
        result = asyncio.run(
            runner._can_use_tool("mcp__github__create_pr", {"title": "x"}, None)
        )
        self.assertIsInstance(result, PermissionResultDeny)


class ClaudeSystemAgentIndexingTests(TestCase):
    """A Claude system-agent spawn must stamp the row hidden and resolve its
    project from ``cwd``; otherwise the ``codex_updated_at`` it writes makes the
    indexer skip the backfill, leaking the run into normal views and dropping it
    from project-filtered System Sessions."""

    @patch("hitch.main.codex_pool._launch_worker_process")
    def test_system_agent_session_is_hidden_and_project_scoped(
        self, mock_launch: MagicMock
    ) -> None:
        from hitch.main.models import Project, SessionMetadata

        mock_launch.return_value = codex_pool.WorkerLaunch(pid=7)
        with (
            tempfile.TemporaryDirectory() as repo,
            override_settings(CODEX_EVENTS_DIR=Path(repo)),
        ):
            project = Project.objects.create(name="p", repo_path=repo)
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
        from hitch.main import claude_tools

        server = claude_tools.build_hitch_mcp_server(cwd="/repo", thread_id="t")
        self.assertEqual(server["name"], claude_tools.HITCH_MCP_SERVER_NAME)
        self.assertEqual(
            claude_tools.PROPOSE_SESSION_TOOL_NAME, "mcp__hitch__propose_session"
        )


def _result(
    subtype: str = "success",
    is_error: bool = False,
    structured_output: Any = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="sess-final",
        structured_output=structured_output,
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
                        with self._runner._steer_lock:
                            self._runner._steer_pending += 1
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


class WorkerApprovalTests(TestCase):
    def _runner(self, *, purpose: str, approval_mode: str | None) -> Any:
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
            sandbox_policy=None,
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

        base: dict[str, Any] = dict(
            model="claude-opus-4-8",
            reasoning_effort="",
            sandbox_policy="",
            approval_mode="auto_review",
            coding_agent="",
            extra_system_prompt="",
            use_worktrees=False,
            auto_pr_enabled=False,
            auto_qa_enabled=True,
            qa_panel_enabled=False,
            spec_critic_enabled=False,
            web_search_mode="",
            show_archived_sessions=False,
            last_selected_repo="",
            selected_project_id=None,
            enable_memories=False,
            provider=coding_agents.PROVIDER_CLAUDE,
        )
        base.update(overrides)
        return views.SettingsValues(**base)

    def _claude_instance(self, **overrides: Any) -> CodexInstance:
        from hitch.main.models import SessionMetadata

        SessionMetadata.objects.update_or_create(
            thread_id="claude-thread",
            defaults={"cwd": "/repo", "auto_qa_enabled": True},
        )
        defaults: dict[str, Any] = dict(
            thread_id="claude-thread",
            cwd="/repo",
            prompt="p",
            events_path="/dev/null",
            pid=0,
            status=CodexInstance.STATUS_COMPLETED,
            backend=CodexInstance.BACKEND_CLAUDE,
            purpose=CodexInstance.PURPOSE_USER,
        )
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
