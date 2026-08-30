import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from django.http import HttpResponse
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone
from openai_codex.generated.v2_all import ReasoningThreadItem

from hitch.main import caches
from hitch.main.diffs import DiffFile, DiffLine, DiffView
from hitch.main.models import (
    CodexInstance,
    SessionMetadata,
    SessionPullRequest,
)
from hitch.main.runtime import codex_events
from hitch.main.sessions.entry_render import tool_call_detail, tool_call_status
from hitch.main.sessions.session_pr_plan import _fix_pr_url_for_thread
from hitch.main.test.support import (
    _make_project,
    _rollout_line,
    _seed_cookies,
)

# Used for active-worker rendering tests so the session view's
# ``reconcile_dead`` sweep doesn't mark the row failed before the assertions
# run; the current process pid is by definition alive.
_LIVE_PID = os.getpid()


def _worker_is_live_for_test(instance: CodexInstance) -> bool:
    return instance.pid == _LIVE_PID


def _root(item: Any) -> SimpleNamespace:
    """Wrap an item to look like a pydantic RootModel from the codex SDK."""
    return SimpleNamespace(root=item)


def _user_message(*texts: str) -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="userMessage",
            content=[_root(SimpleNamespace(type="text", text=t)) for t in texts],
        )
    )


def _agent_message(
    text: str,
    phase: str | None = None,
    memory_citation: SimpleNamespace | dict[str, object] | None = None,
) -> SimpleNamespace:
    # The SDK surfaces phase as a MessagePhase enum (with `.value`); mirror
    # that shape so tests match production deserialization.
    phase_obj = SimpleNamespace(value=phase) if phase is not None else None
    return _root(
        SimpleNamespace(
            type="agentMessage",
            text=text,
            phase=phase_obj,
            memory_citation=memory_citation,
        )
    )


def _command(command: str, status: str = "completed") -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="commandExecution",
            command=command,
            status=SimpleNamespace(value=status),
        )
    )


def _reasoning(text: str) -> SimpleNamespace:
    return _root(
        ReasoningThreadItem(
            id="reasoning-item",
            type="reasoning",
            summary=[text],
            content=[],
        )
    )


def _web_search(query: str) -> SimpleNamespace:
    return _root(SimpleNamespace(type="webSearch", query=query))


def _tool_call(item_type: str) -> SimpleNamespace:
    return _root(SimpleNamespace(type=item_type))


def _mcp_tool_call(
    server: str,
    tool: str,
    result: object | None = None,
) -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="mcpToolCall",
            server=server,
            tool=tool,
            result=result,
            status=SimpleNamespace(value="completed"),
        )
    )


def _turn(items: list[SimpleNamespace], started_at: int | None = 1700000000) -> SimpleNamespace:
    return SimpleNamespace(items=items, started_at=started_at)


def _thread(turns: list[SimpleNamespace], **overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "id": "thread-1",
        "name": "Sample session",
        "preview": "first message",
        "cwd": "/tmp/repo",
        "updated_at": 1700000000,
        "turns": turns,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _write_rollout_tempfile(lines: list[str], *, binary: bytes | None = None) -> Path:
    if binary is not None:
        with tempfile.NamedTemporaryFile(prefix="rollout-", suffix=".jsonl", mode="wb", delete=False) as fh:
            fh.write(binary)
            return Path(fh.name)
    with tempfile.NamedTemporaryFile(prefix="rollout-", suffix=".jsonl", mode="w", delete=False) as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
        return Path(fh.name)


def _patch_thread(test: TestCase, mock_codex: MagicMock, thread: SimpleNamespace) -> None:
    client = mock_codex.return_value.__enter__.return_value
    client._client.thread_resume.return_value.thread = thread


def _clear_models_cache() -> None:
    with caches._MODELS_REFRESH_LOCK:
        caches._MODELS_CACHE_VALUE = {}
        caches._MODELS_CACHE_FETCHED_AT = {}
        caches._MODELS_REFRESH_IN_FLIGHT = set()


def _seed_models_cache(models: list[SimpleNamespace]) -> None:
    with caches._MODELS_REFRESH_LOCK:
        caches._MODELS_CACHE_VALUE[False] = list(models)
        caches._MODELS_CACHE_FETCHED_AT[False] = timezone.now()
        caches._MODELS_REFRESH_IN_FLIGHT.discard(False)


def _make_rollout(test: TestCase, lines: list[str], *, binary: bytes | None = None) -> Path:
    path = _write_rollout_tempfile(lines, binary=binary)
    test.addCleanup(path.unlink, missing_ok=True)
    return path


def _get_session(client: Client, session_id: str = "thread-1") -> HttpResponse:
    response = client.get(reverse("session", kwargs={"session_id": session_id}))
    assert isinstance(response, HttpResponse)
    return response


def _diff_view() -> DiffView:
    return DiffView(
        files=[
            DiffFile(
                path="hitch/main/views.py",
                old_path="hitch/main/views.py",
                status="Modified",
                additions=1,
                deletions=1,
                lines=[
                    DiffLine("hunk", None, None, "@@ -1 +1 @@"),
                    DiffLine("remove", 1, None, '<span class="k">return</span> 1'),
                    DiffLine("add", None, 1, '<span class="k">return</span> 2'),
                ],
            )
        ],
        additions=1,
        deletions=1,
    )


class PrUrlDetectionTests(TestCase):
    def test_fix_pr_uses_registered_repo_and_number_identity(self) -> None:
        SessionPullRequest.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            state={
                "pr_handoff": {
                    "repository_full_name": "cberner/hitch",
                    "pr_number": 94,
                }
            },
        )

        self.assertEqual(
            _fix_pr_url_for_thread("thread-1"),
            "https://github.com/cberner/hitch/pull/94",
        )

class SessionViewTests(TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        patcher = patch(
            "hitch.main.runtime.codex_pool.worker_is_alive",
            side_effect=_worker_is_live_for_test,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(_clear_models_cache)

    @patch(
        "hitch.main.repos.discover_repos",
        return_value=[Path("/tmp/other"), Path("/tmp/hitch")],
    )
    @patch("hitch.main.views.common.Codex")
    def test_action_menu_prefers_hitch_project_for_debug_chat_link(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        session_project = _make_project(name="Other", repo_path="/tmp/other")
        hitch_project = _make_project(repo_path="/tmp/hitch")
        SessionMetadata.objects.create(thread_id="thread-1", cwd="/tmp/other", project=session_project)
        thread = _thread([_turn([_user_message("hi")])], cwd="/tmp/other")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        query = parse_qs(urlparse(debug_url).query)
        self.assertEqual(query["project"], [str(hitch_project.pk)])
        self.assertNotEqual(query["project"], [str(session_project.pk)])

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/tmp/other")])
    @patch("hitch.main.views.common.Codex")
    def test_action_menu_ignores_undiscovered_hitch_project_for_debug_chat_link(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        session_project = _make_project(name="Other", repo_path="/tmp/other")
        hitch_project = _make_project(name="Hitch", repo_path="/tmp/missing-hitch")
        SessionMetadata.objects.create(thread_id="thread-1", cwd="/tmp/other", project=session_project)
        thread = _thread([_turn([_user_message("hi")])], cwd="/tmp/other")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        query = parse_qs(urlparse(debug_url).query)
        self.assertEqual(query["project"], [str(session_project.pk)])
        self.assertNotEqual(query["project"], [str(hitch_project.pk)])

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/tmp/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_action_menu_includes_cwd_for_bare_repo_debug_chat_link(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        _make_project(name="Other", repo_path="/tmp/repo")
        SessionMetadata.objects.create(thread_id="thread-1", cwd="/tmp/repo", project_cleared=True)
        thread = _thread([_turn([_user_message("hi")])], cwd="/tmp/repo")
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        debug_url = cast(str, cast(Any, response).context["debug_chat_url"])
        query = parse_qs(urlparse(debug_url).query)
        self.assertNotIn("project", query)
        self.assertEqual(query["cwd"], ["/tmp/repo"])

    @patch("hitch.main.views.common.Codex")
    def test_set_session_project_moves_and_clears_project(self, mock_codex: MagicMock) -> None:
        project = _make_project(repo_path="/tmp/repo")
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        self.assertEqual(metadata.project, project)
        self.assertEqual(metadata.cwd, "/tmp/repo")
        self.assertFalse(metadata.project_cleared)

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": ""},
        )

        self.assertEqual(response.status_code, 302)
        metadata.refresh_from_db()
        self.assertIsNone(metadata.project)
        self.assertTrue(metadata.project_cleared)

    @patch("hitch.main.views.common.Codex")
    def test_next_message_model_comes_only_from_resumed_thread(self, mock_codex: MagicMock) -> None:
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_thread([_turn([_user_message("hi")])]),
            reasoning_effort=SimpleNamespace(value="medium"),
        )
        _seed_cookies(self.client, hitch_model="gpt-stale")

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ">model<")
        self.assertContains(response, 'data-normal-value="Unknown"')
        self.assertContains(response, 'data-plan-value="gpt-stale"')

    @patch("hitch.main.views.common.Codex")
    def test_archived_session_menu_offers_unarchive(self, mock_codex: MagicMock) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        rollout_path = Path(temp_dir.name) / "archived_sessions" / "2026" / "05" / "15" / "rollout-thread-1.jsonl"
        rollout_path.parent.mkdir(parents=True)
        rollout_path.write_text(_rollout_line("event_msg", {"type": "user_message", "message": "hi"}) + "\n")
        thread = _thread(
            [_turn([_user_message("hi")])],
            path=str(rollout_path),
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="archived" value="false"')
        self.assertContains(response, 'role="menuitem">Unarchive</button>')
        self.assertNotContains(response, 'role="menuitem">Archive</button>')

    @patch("hitch.main.views.common.Codex")
    def test_sdk_memory_citation_renders_details(self, mock_codex: MagicMock) -> None:
        memory_citation = SimpleNamespace(
            entries=[
                SimpleNamespace(
                    path="MEMORY.md",
                    line_start=1,
                    line_end=2,
                    note="project convention",
                )
            ],
            thread_ids=["019cc2ea-1dff-7902-8d40-c8f6e5d83cc4"],
        )
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Use prior context"),
                        _agent_message(
                            "Applied the remembered convention.",
                            memory_citation=memory_citation,
                        ),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Applied the remembered convention.")
        # 1 file-line entry + 1 prior-session id; both render in the popover,
        # so the summary count must include both kinds.
        self.assertContains(response, "Memories used: 2")
        self.assertContains(response, "MEMORY.md:1-2")
        self.assertContains(response, "project convention")
        self.assertContains(response, "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4")
        self.assertContains(response, "function renderMemoryCitation")
        self.assertContains(response, "item.memoryCitation || item.memory_citation")


class RolloutFileViewTests(TestCase):
    """When ``thread.path`` points at a rollout file, the view parses it
    directly to recover the commandExecution items codex strips from
    ``thread/read``; failure modes (missing, malformed, empty, or
    contentless rollouts) fall back to the SDK-derived entries instead of
    bubbling the error to the response.
    """

    @patch("hitch.main.views.common.Codex")
    def test_rollout_memory_citation_renders_details(self, mock_codex: MagicMock) -> None:
        raw_text = (
            "Used prior context."
            "<oai-mem-citation>"
            "<citation_entries>\n"
            "MEMORY.md:3-4|note=[repo preference]\n"
            "</citation_entries>\n"
            "<rollout_ids>\n"
            "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4\n"
            "</rollout_ids>"
            "</oai-mem-citation>"
        )
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "remember?"}),
            _rollout_line(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": raw_text}],
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Used prior context.")
        # 1 file-line entry + 1 prior-session id; both render in the popover,
        # so the summary count must include both kinds.
        self.assertContains(response, "Memories used: 2")
        self.assertContains(response, "MEMORY.md:3-4")
        self.assertContains(response, "repo preference")
        self.assertNotContains(response, "oai-mem-citation")

    @patch("hitch.main.views.common.Codex")
    def test_rejected_command_renders_approval_choice(self, mock_codex: MagicMock) -> None:
        rejection = (
            "exec_command failed for `/bin/bash -lc 'printf Reason: command && git push origin master'`: "
            'CreateProcess { message: "Rejected(\\"This action was rejected'
            "\\\\nReason: Pushing directly to origin/master is risky."
            '\\\\nStop and request user input.\\\\")" }'
        )
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "push it"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "git push origin master"}),
                    "call_id": "push",
                },
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call_output",
                    "call_id": "push",
                    "output": rejection,
                },
            ),
            _rollout_line(
                "event_msg",
                {
                    "type": "agent_message",
                    "message": "Please confirm explicitly.",
                    "phase": "final_answer",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Approval declined")
        self.assertContains(response, "git push origin master")
        self.assertContains(response, "Pushing directly to origin/master is risky.")
        self.assertNotContains(response, "I explicitly approve")
        self.assertNotContains(response, "No, do not run this command")
        self.assertLess(body.index("git push origin master"), body.index("Approval declined"))
        self.assertLess(body.index("Approval declined"), body.index("Please confirm explicitly."))

    @patch("hitch.main.views.common.Codex")
    def test_resolved_plan_card_hides_approval_actions(self, mock_codex: MagicMock) -> None:
        plan = "# Fix Login CSRF Redirect Failure\n\n## Summary\nRead the CSRF cookie at submit time."
        rollout_lines = [
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "Debug the login CSRF issue"},
            ),
            _rollout_line(
                "event_msg",
                {
                    "type": "item_completed",
                    "item": {"type": "Plan", "id": "turn-plan", "text": plan},
                },
            ),
            _rollout_line(
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
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "Implement the plan."},
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                    "phase": "final_answer",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="plan-card"')
        self.assertContains(response, "Fix Login CSRF Redirect Failure")
        self.assertNotContains(response, "Approve plan")
        self.assertNotContains(response, 'name="plan_action" value="approve"')
        self.assertContains(response, 'data-initial-plan-mode="false"')
        self.assertContains(response, 'name="default_plan_mode" value=""')

    @patch("hitch.main.views.common.Codex")
    def test_falls_back_to_sdk_on_rollout_failure_modes(self, mock_codex: MagicMock) -> None:
        """Unreadable / unparseable / message-less / empty rollouts each fall
        through to the SDK-derived turns rather than crashing or dropping
        the page back to an empty state."""
        sdk_turns = [_turn([_user_message("hi"), _agent_message("hello")])]

        # (label, thread.path) — each path is a different failure mode.
        cases: list[tuple[str, str]] = [
            ("missing path", "/nonexistent/rollout.jsonl"),
            (
                "binary garbage that survives line splitting",
                str(_make_rollout(self, [], binary=b"\xff\xfe\x00not json\n")),
            ),
            (
                "tool-only rollout (no user/agent messages)",
                str(
                    _make_rollout(
                        self,
                        [
                            _rollout_line(
                                "response_item",
                                {
                                    "type": "function_call",
                                    "name": "exec_command",
                                    "arguments": json.dumps({"cmd": "uname -a"}),
                                    "call_id": "c1",
                                },
                            ),
                        ],
                    )
                ),
            ),
            ("empty rollout file", str(_make_rollout(self, []))),
        ]

        for label, path in cases:
            with self.subTest(label=label):
                _patch_thread(self, mock_codex, _thread(sdk_turns, path=path))
                response = _get_session(self.client)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "hi")
                self.assertContains(response, "hello")
                # The tool-only rollout's commandExecution must NOT leak into
                # the response — the SDK path took over entirely.
                if "tool-only" in label:
                    self.assertNotContains(response, "uname -a")

    @patch("hitch.main.views.common.Codex")
    def test_empty_rollout_with_no_sdk_turns_renders_placeholder(self, mock_codex: MagicMock) -> None:
        # Both the rollout and Thread.turns are empty — the page should still
        # render its empty-state placeholder rather than fall back to a
        # second SDK call.
        rollout_path = _make_rollout(self, [])
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")

    @patch("hitch.main.views.common.Codex")
    def test_token_usage_renders_under_title(self, mock_codex: MagicMock) -> None:
        # The most recent token_count event's non-cached input and output
        # surface under the session title with thousands separators, while
        # context usage comes from the non-cumulative active context token
        # count. Threads with no token_count event in their rollout hide the
        # section entirely.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "go"}),
            _rollout_line(
                "event_msg",
                {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 12345,
                            "cached_input_tokens": 678,
                            "output_tokens": 9012,
                            "reasoning_output_tokens": 3,
                            "total_tokens": 22038,
                        },
                        "last_token_usage": {
                            "input_tokens": 10000,
                            "cached_input_tokens": 500,
                            "output_tokens": 8000,
                            "reasoning_output_tokens": 3,
                            "total_tokens": 22038,
                        },
                        "model_context_window": 200000,
                    },
                },
            ),
        ]
        path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="usage"')
        self.assertContains(response, ">context<")
        self.assertContains(response, "11%")
        self.assertContains(response, "22,038 of 200,000 tokens in current context")
        self.assertContains(response, ">in<")
        self.assertContains(response, ">out<")
        self.assertContains(response, ">cached<")
        self.assertContains(response, "11,667")
        self.assertNotContains(response, "12,345")
        self.assertContains(response, "9,012")
        self.assertContains(response, "678")

        # Same view with no token_count events: usage row is omitted.
        empty_path = _make_rollout(
            self,
            [_rollout_line("event_msg", {"type": "user_message", "message": "hi"})],
        )
        _patch_thread(self, mock_codex, _thread([], path=str(empty_path)))
        response = _get_session(self.client)
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'class="usage"')

    @patch("hitch.main.views.common.Codex")
    def test_rollout_commentary_phase_never_treated_as_final(self, mock_codex: MagicMock) -> None:
        # Scanning from the end must skip the trailing tool call AND both
        # commentary messages, falling through to "no final agent".
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "go"}),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "preamble", "phase": "commentary"},
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "narrating", "phase": "commentary"},
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "ls"}),
                    "call_id": "c1",
                },
            ),
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "preamble")
        self.assertContains(response, "narrating")
        self.assertNotContains(response, ">Agent<")
        self.assertContains(response, ">Agent (thinking)<", count=2)
        self.assertNotContains(response, '<details class="intermediate">')


class IntermediateCollapseTests(TestCase):
    """Consecutive command/reasoning/web-search activity collapses."""

    @patch("hitch.main.views.common.Codex")
    def test_thinking_messages_stay_visible_and_split_activity_groups(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Help me out"),
                        _agent_message("Let me look at this."),
                        _reasoning("Inspecting the code"),
                        _command("./scripts/check.sh"),
                        _agent_message("Trying something else."),
                        _command("./scripts/check-again.sh"),
                        _reasoning("Reviewing the result"),
                        _agent_message("Here is the answer."),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Here is the answer.")
        self.assertContains(response, '<details class="intermediate">', count=2)
        self.assertContains(response, "1 reasoning message and 1 command message", count=2)
        self.assertContains(response, "Let me look at this.")
        self.assertContains(response, "Trying something else.")
        self.assertContains(response, "./scripts/check.sh")
        self.assertContains(response, ">Agent (thinking)<", count=2)
        self.assertLess(body.index("Let me look at this."), body.index("<details"))
        self.assertLess(body.index("./scripts/check.sh"), body.index("Trying something else."))

    @patch("hitch.main.views.common.Codex")
    def test_web_search_collapses_with_commands_and_reasoning(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Research it"),
                        _command("check local sources"),
                        _web_search("current documentation"),
                        _reasoning("Compare the results"),
                        _agent_message("Done."),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<details class="intermediate">', count=1)
        self.assertContains(
            response,
            "1 reasoning message, 1 command message, and 1 web search",
        )
        self.assertContains(response, "current documentation")

    @patch("hitch.main.views.common.Codex")
    def test_commentary_phase_is_never_final(self, mock_codex: MagicMock) -> None:
        """All-commentary turns keep every Thinking message top-level."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Working on it"),
                        _agent_message("Step 1", phase="commentary"),
                        _agent_message("Step 2", phase="commentary"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<details class="intermediate">')
        self.assertNotContains(response, ">Agent<")
        self.assertContains(response, ">Agent (thinking)<", count=2)

    @patch("hitch.main.views.common.Codex")
    def test_phase_accepts_raw_string_shape(self, mock_codex: MagicMock) -> None:
        """Robustness: phase as a raw wire string (e.g. data deserialized
        without pydantic) is still recognized."""
        commentary = _root(SimpleNamespace(type="agentMessage", text="Thinking.", phase="commentary"))
        final = _root(SimpleNamespace(type="agentMessage", text="42.", phase="final_answer"))
        thread = _thread([_turn([_user_message("Q"), commentary, final])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertLess(body.index(">Agent<"), body.index("42."))
        self.assertContains(response, ">Agent (thinking)<")


class FinalAgentMarkdownTests(TestCase):
    """The turn's final agent reply is rendered as markdown when the body
    contains a high-confidence markdown construct; thinking entries and user
    messages always stay plain-text.
    """

    @patch("hitch.main.views.common.Codex")
    def test_final_agent_markdown_is_rendered(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Plan it"),
                        _agent_message("# Plan\n\n- step one\n- step two"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<h1>Plan</h1>", html=False)
        self.assertContains(response, "<li>step one</li>", html=False)
        self.assertContains(response, '<div class="body markdown">', html=False)

    @patch("hitch.main.views.common.Codex")
    def test_plain_final_agent_is_not_treated_as_markdown(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("Hi"), _agent_message("Hello, friend.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hello, friend.")
        self.assertNotContains(response, '<div class="body markdown">')


class ToolCallDetailTests(TestCase):
    """Exercise every branch of ``tool_call_detail`` and ``tool_call_status``
    so the per-type description and status badge stay stable.
    """

    def test_detail_per_tool_type(self) -> None:
        cases: list[tuple[str, str, SimpleNamespace, str]] = [
            ("commandExecution", "with command", SimpleNamespace(command="ls -la"), "ls -la"),
            ("commandExecution", "missing command", SimpleNamespace(command=None), ""),
            (
                "mcpToolCall",
                "server and tool",
                SimpleNamespace(server="github", tool="create_pr"),
                "github / create_pr",
            ),
            (
                "dynamicToolCall",
                "with namespace",
                SimpleNamespace(namespace="codex", tool="apply_patch"),
                "codex::apply_patch",
            ),
            (
                "dynamicToolCall",
                "without namespace",
                SimpleNamespace(namespace=None, tool="apply_patch"),
                "apply_patch",
            ),
            ("fileChange", "empty changes", SimpleNamespace(changes=[]), ""),
            (
                "fileChange",
                "single change",
                SimpleNamespace(changes=[SimpleNamespace(path="a.py")]),
                "a.py",
            ),
            (
                "fileChange",
                "multiple changes",
                SimpleNamespace(changes=[SimpleNamespace(path="a.py"), SimpleNamespace(path="b.py")]),
                "a.py (+1 more)",
            ),
            ("webSearch", "query", SimpleNamespace(query="how to django"), "how to django"),
            (
                "plan",
                "first line only",
                SimpleNamespace(text="Step 1\nStep 2\nStep 3"),
                "Step 1",
            ),
            ("imageView", "path", SimpleNamespace(path="/tmp/x.png"), "/tmp/x.png"),
            (
                "imageGeneration",
                "prefers revised_prompt",
                SimpleNamespace(revised_prompt="a cat", saved_path="/tmp/y.png"),
                "a cat",
            ),
            (
                "imageGeneration",
                "falls back to saved_path",
                SimpleNamespace(revised_prompt=None, saved_path="/tmp/y.png"),
                "/tmp/y.png",
            ),
            ("somethingNew", "unknown type", SimpleNamespace(), ""),
        ]
        for tool_type, label, item, expected in cases:
            with self.subTest(tool_type=tool_type, case=label):
                self.assertEqual(tool_call_detail(item, tool_type), expected)

    def test_collab_agent_detail(self) -> None:
        # Two cases share an assertion shape but differ in what they expose.
        with_receiver = SimpleNamespace(tool=SimpleNamespace(value="spawn"), receiver_thread_ids=["child-thread"])
        without_receiver = SimpleNamespace(tool=SimpleNamespace(value="spawn"), receiver_thread_ids=[])
        result = tool_call_detail(with_receiver, "collabAgentToolCall")
        self.assertIn("child-thread", result)
        self.assertIn("spawn", result)
        self.assertEqual(tool_call_detail(without_receiver, "collabAgentToolCall"), "spawn")

    def test_status_badge(self) -> None:
        # ``completed`` and missing status both render no badge; ``failed``
        # surfaces verbatim so the UI can highlight it.
        self.assertIsNone(tool_call_status(SimpleNamespace(status=SimpleNamespace(value="completed"))))
        self.assertIsNone(tool_call_status(SimpleNamespace()))
        self.assertEqual(
            tool_call_status(SimpleNamespace(status=SimpleNamespace(value="failed"))),
            "failed",
        )


def _make_codex_instance(**kwargs: object) -> CodexInstance:
    """Helper used by the active-worker session view tests."""
    defaults: dict[str, object] = {
        "pid": 0,
        "thread_id": "thread-1",
        "cwd": "/repo",
        "prompt": "do work",
        "events_path": "/tmp/some.jsonl",
        "status": CodexInstance.STATUS_RUNNING,
    }
    defaults.update(kwargs)
    return CodexInstance.objects.create(**defaults)


class SessionViewActiveWorkerTests(TestCase):
    """How the session detail view surfaces an in-progress turn: the
    streaming UI guard, the pending user bubble, the in-progress turn
    trim, and the dead-worker reconciliation sweep.
    """

    @override
    def setUp(self) -> None:
        super().setUp()
        patcher = patch(
            "hitch.main.runtime.codex_pool.worker_is_alive",
            side_effect=_worker_is_live_for_test,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("hitch.main.views.common.Codex")
    def test_inactive_thread_surfaces_latest_user_turn_failure(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        ended_at = timezone.now().replace(microsecond=0)
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_FAILED,
            error="This content was flagged for possible cybersecurity risk.",
            purpose=CodexInstance.PURPOSE_USER,
        )
        CodexInstance.objects.filter(pk=instance.pk).update(ended_at=ended_at)

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "data-turn-failure")
        self.assertContains(response, "Agent turn failed")
        self.assertContains(
            response,
            "This content was flagged for possible cybersecurity risk.",
        )
        self.assertContains(response, f'data-ts="{int(ended_at.timestamp())}"')


    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_finished_worker_rebuilds_task_plan_on_page_load(self, mock_codex: MagicMock, mock_diff: MagicMock) -> None:
        # The task-plan widget must rebuild from the thread's persisted worker
        # logs after a reload, when no worker is running, just as the goal
        # objective does.
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "explanation": "Persist plan past the worker",
                            "plan": [
                                {"step": "Inspect current live UI", "status": "completed"},
                                {"step": "Render saved task plan", "status": "in_progress"},
                            ],
                        },
                        "recordedAt": 30,
                        "eventSeq": 3,
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_COMPLETED,
            prompt="finished work",
            pid=0,
            events_path=events_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        # No active worker: the page renders without a live stream root.
        self.assertNotContains(response, "data-live-root></div>")
        self.assertContains(response, 'class="has-task-plan"')
        self.assertContains(response, "Render saved task plan")
        self.assertContains(response, 'class="plan-step inProgress"')
        self.assertNotIn('aria-label="Current task plan" hidden', body)
        self.assertContains(
            response,
            '<span class="task-plan-current" data-task-plan-current>Render saved task plan</span>',
            html=False,
        )

    @patch("hitch.main.views.common.build_worktree_diff")
    @patch("hitch.main.views.common.Codex")
    def test_active_worker_preserves_cleared_task_plan_order_after_refresh(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "plan": [
                                {"step": "Stale task", "status": "in_progress"},
                            ],
                        },
                        "recordedAt": 10,
                        "eventSeq": 1,
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {"threadId": "thread-1", "plan": []},
                        "recordedAt": 20,
                        "eventSeq": 2,
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
            events_path=events_path,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertContains(response, "data-task-plan")
        self.assertContains(response, 'data-stream-url="')
        self.assertContains(response, 'class="">')
        self.assertContains(response, 'data-recorded-at="20"')
        self.assertContains(response, 'data-event-seq="2"')
        self.assertContains(response, 'data-fallback-order="2"')
        self.assertRegex(
            body,
            r'<aside class="task-plan"[\s\S]*?data-recorded-at="20"'
            r'[\s\S]*?data-fallback-order="2"'
            r'[\s\S]*?aria-label="Current task plan"\s+hidden>',
        )
        self.assertNotIn("Stale task", body)
        self.assertContains(
            response,
            '<span class="task-plan-current" data-task-plan-current></span>',
            html=False,
        )

    @patch("hitch.main.views.common.Codex")
    def test_inactive_thread_omits_stop_button(self, mock_codex: MagicMock) -> None:
        # Stop is only meaningful while a turn is in progress; otherwise it
        # would just be dead chrome cluttering the composer.
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, reverse("stop_session", kwargs={"session_id": "thread-1"}))










    def test_system_session_detail_requires_system_run(self) -> None:
        response = self.client.get(reverse("system_session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 404)
