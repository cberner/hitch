import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse

from hitch.main.views import _tool_call_detail, _tool_call_status


def _root(item: SimpleNamespace) -> SimpleNamespace:
    """Wrap an item to look like a pydantic RootModel from the codex SDK."""
    return SimpleNamespace(root=item)


def _user_message(*texts: str) -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="userMessage",
            content=[_root(SimpleNamespace(type="text", text=t)) for t in texts],
        )
    )


def _agent_message(text: str, phase: str | None = None) -> SimpleNamespace:
    # The SDK surfaces phase as a MessagePhase enum (with `.value`); mirror
    # that shape so tests match production deserialization.
    phase_obj = SimpleNamespace(value=phase) if phase is not None else None
    return _root(SimpleNamespace(type="agentMessage", text=text, phase=phase_obj))


def _command(command: str, status: str = "completed") -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="commandExecution",
            command=command,
            status=SimpleNamespace(value=status),
        )
    )


def _file_change(*paths: str, status: str = "completed") -> SimpleNamespace:
    return _root(
        SimpleNamespace(
            type="fileChange",
            changes=[SimpleNamespace(path=p) for p in paths],
            status=SimpleNamespace(value=status),
        )
    )


def _tool_call(item_type: str) -> SimpleNamespace:
    return _root(SimpleNamespace(type=item_type))


def _turn(items: list[SimpleNamespace], started_at: int | None = 1700000000) -> SimpleNamespace:
    return SimpleNamespace(items=items, started_at=started_at)


def _thread(turns: list[SimpleNamespace], **overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "id": "thread-1",
        "name": "Demo session",
        "preview": "first message",
        "cwd": "/tmp/demo",
        "updated_at": 1700000000,
        "turns": turns,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _rollout_line(
    line_type: str,
    payload: dict[str, object],
    *,
    timestamp: str = "2025-01-05T12:00:00Z",
) -> str:
    return json.dumps({"timestamp": timestamp, "type": line_type, "payload": payload})


def _write_rollout_tempfile(lines: list[str]) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="rollout-", suffix=".jsonl", mode="w", delete=False
    ) as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
        return Path(fh.name)


class SessionViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_renders_user_and_agent_messages(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Refactor the login flow"),
                        _agent_message("Sure, here is the plan."),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        client._client.thread_read.assert_called_once_with("thread-1", include_turns=True)
        self.assertContains(response, "Demo session")
        self.assertContains(response, "Refactor the login flow")
        self.assertContains(response, "Sure, here is the plan.")
        self.assertContains(response, ">User<")
        self.assertContains(response, ">Agent<")

    @patch("hitch.main.views.Codex")
    def test_each_tool_call_is_rendered_individually(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Do the thing"),
                        _command("./scripts/build.sh"),
                        _command("./scripts/test.sh"),
                        _file_change("hitch/main/views.py"),
                        _agent_message("Done."),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "./scripts/build.sh")
        self.assertContains(response, "./scripts/test.sh")
        self.assertContains(response, "hitch/main/views.py")
        # Three separate tool-call rows (now inside the collapsed intermediate
        # block), not a single aggregate row.
        self.assertEqual(response.content.decode().count('class="tool-call"'), 3)

    @patch("hitch.main.views.Codex")
    def test_trailing_tool_calls_are_rendered(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Investigate"),
                        _agent_message("Looking into it."),
                        _command("rg --files"),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "rg --files")

    @patch("hitch.main.views.Codex")
    def test_unknown_tool_types_are_still_rendered(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Run something"),
                        _tool_call("brandNewTool"),
                        _agent_message("Ok."),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        # Unmapped types fall back to the raw type tag so nothing is hidden.
        self.assertContains(response, "brandNewTool")

    @patch("hitch.main.views.Codex")
    def test_failed_tool_calls_show_status(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Run tests"),
                        _command("just test", status="failed"),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "failed")

    @patch("hitch.main.views.Codex")
    def test_messages_carry_turn_timestamp(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Hi"),
                        _agent_message("Hello."),
                    ],
                    started_at=1700000123,
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-ts="1700000123"')

    @patch("hitch.main.views.Codex")
    def test_empty_session_shows_placeholder(self, mock_codex: MagicMock) -> None:
        thread = _thread([])
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")

    @patch("hitch.main.views.Codex")
    def test_prefers_rollout_file_when_path_is_set(self, mock_codex: MagicMock) -> None:
        # When thread.path points at a readable rollout file, the view parses
        # it directly to recover the commandExecution items codex strips from
        # `thread/read`. The shell call lands inside the intermediate
        # <details> block, alongside the mid-turn agent commentary.
        rollout_lines = [
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "build it"},
                timestamp="2026-04-14T23:05:00Z",
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "looking into it"},
                timestamp="2026-04-14T23:05:05Z",
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "cargo build --release"}),
                    "call_id": "c1",
                },
                timestamp="2026-04-14T23:05:10Z",
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "done"},
                timestamp="2026-04-14T23:05:30Z",
            ),
        ]
        rollout_path = _write_rollout_tempfile(rollout_lines)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread([], path=str(rollout_path))
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "build it")
        self.assertContains(response, "cargo build --release")
        self.assertContains(response, "done")
        self.assertContains(response, '<details class="intermediate">')
        self.assertContains(response, "1 thinking message and 1 tool call")

    @patch("hitch.main.views.Codex")
    def test_falls_back_to_sdk_when_rollout_path_missing(self, mock_codex: MagicMock) -> None:
        # An unreadable thread.path must not crash the view; the SDK-derived
        # entries take over instead.
        thread = _thread(
            [_turn([_user_message("hi"), _agent_message("hello")])],
            path="/nonexistent/rollout.jsonl",
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "hi")
        self.assertContains(response, "hello")

    @patch("hitch.main.views.Codex")
    def test_falls_back_to_sdk_when_rollout_parse_raises(self, mock_codex: MagicMock) -> None:
        # The rollout file is unreadable as JSONL (whole file is binary
        # garbage that survives line splitting). The view must still render
        # the SDK-derived entries instead of bubbling the parse error.
        with tempfile.NamedTemporaryFile(
            prefix="rollout-", suffix=".jsonl", mode="wb", delete=False
        ) as fh:
            fh.write(b"\xff\xfe\x00not json\n")
            rollout_path = Path(fh.name)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread(
            [_turn([_user_message("hi"), _agent_message("hello")])],
            path=str(rollout_path),
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "hi")
        self.assertContains(response, "hello")

    @patch("hitch.main.views.Codex")
    def test_falls_back_to_sdk_when_rollout_has_no_messages(
        self, mock_codex: MagicMock
    ) -> None:
        # The rollout parses to tool-only entries (no user_message /
        # agent_message). Under schema drift the SDK may still know how to
        # surface the conversation, so prefer it over a tool-only view.
        rollout_lines = [
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "uname -a"}),
                    "call_id": "c1",
                },
            ),
        ]
        rollout_path = _write_rollout_tempfile(rollout_lines)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread(
            [_turn([_user_message("hi"), _agent_message("hello")])],
            path=str(rollout_path),
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "hi")
        self.assertContains(response, "hello")
        # Tool-only entries from the rollout aren't merged in; the SDK path
        # took over entirely, so the command from the rollout isn't shown.
        self.assertNotContains(response, "uname -a")

    @patch("hitch.main.views.Codex")
    def test_falls_back_to_sdk_when_rollout_is_empty(self, mock_codex: MagicMock) -> None:
        # If the rollout parses to nothing but the SDK has turns, prefer the
        # SDK output — schema drift or a truncated file shouldn't wipe the
        # session detail page.
        with tempfile.NamedTemporaryFile(
            prefix="rollout-", suffix=".jsonl", mode="w", delete=False
        ) as fh:
            rollout_path = Path(fh.name)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread(
            [_turn([_user_message("hi"), _agent_message("hello")])],
            path=str(rollout_path),
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "hi")
        self.assertContains(response, "hello")

    @patch("hitch.main.views.Codex")
    def test_empty_rollout_with_no_sdk_turns_renders_placeholder(
        self, mock_codex: MagicMock
    ) -> None:
        # Both the rollout and Thread.turns are empty — the page should still
        # render its empty-state placeholder rather than fall back to a
        # second SDK call.
        with tempfile.NamedTemporaryFile(
            prefix="rollout-", suffix=".jsonl", mode="w", delete=False
        ) as fh:
            rollout_path = Path(fh.name)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread([], path=str(rollout_path))
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")

    @patch("hitch.main.views.Codex")
    def test_rollout_groups_multiple_turns_separately(self, mock_codex: MagicMock) -> None:
        # Two user messages in the rollout should produce two independent
        # intermediate blocks — one per turn — rather than one giant block.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "first ask"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "echo a"}),
                    "call_id": "c1",
                },
            ),
            _rollout_line("event_msg", {"type": "agent_message", "message": "first reply"}),
            _rollout_line("event_msg", {"type": "user_message", "message": "second ask"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "echo b"}),
                    "call_id": "c2",
                },
            ),
            _rollout_line("event_msg", {"type": "agent_message", "message": "second reply"}),
        ]
        rollout_path = _write_rollout_tempfile(rollout_lines)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread([], path=str(rollout_path))
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode().count('<details class="intermediate">'), 2)

    @patch("hitch.main.views.Codex")
    def test_rollout_turn_with_only_tool_calls_has_no_final_agent(
        self, mock_codex: MagicMock
    ) -> None:
        # An interrupted turn that produced no agent reply should still
        # render, with the tool call(s) inside the intermediate block and no
        # top-level agent message.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "try this"}),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "sleep 1"}),
                    "call_id": "c1",
                },
            ),
        ]
        rollout_path = _write_rollout_tempfile(rollout_lines)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread([], path=str(rollout_path))
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "try this")
        self.assertContains(response, "sleep 1")
        self.assertContains(response, '<details class="intermediate">')
        # No final agent message exists, so nothing renders outside the block.
        self.assertNotContains(response, ">Agent<")

    @patch("hitch.main.views.Codex")
    def test_rollout_commentary_phase_never_treated_as_final(
        self, mock_codex: MagicMock
    ) -> None:
        # All agent messages are commentary, and the last entry is a tool
        # call. Scanning from the end must skip the trailing tool call AND
        # both commentary messages, falling through to "no final agent".
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
        rollout_path = _write_rollout_tempfile(rollout_lines)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread([], path=str(rollout_path))
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "preamble")
        self.assertContains(response, "narrating")
        # Neither commentary becomes the final agent reply.
        self.assertNotContains(response, ">Agent<")
        self.assertContains(response, "2 thinking messages and 1 tool call")

    @patch("hitch.main.views.Codex")
    def test_rollout_final_answer_phase_wins_over_later_unphased(
        self, mock_codex: MagicMock
    ) -> None:
        # The explicit final_answer is the final agent reply even when an
        # un-phased agent message follows it; the trailing message folds
        # into the post-final intermediate block alongside any later tool
        # call.
        rollout_lines = [
            _rollout_line("event_msg", {"type": "user_message", "message": "go"}),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "the answer", "phase": "final_answer"},
            ),
            _rollout_line(
                "event_msg",
                {"type": "agent_message", "message": "post-answer note"},
            ),
        ]
        rollout_path = _write_rollout_tempfile(rollout_lines)
        self.addCleanup(rollout_path.unlink, missing_ok=True)

        thread = _thread([], path=str(rollout_path))
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # The final_answer text renders outside the <details> block...
        final_idx = body.index("the answer")
        details_idx = body.index("<details")
        self.assertLess(final_idx, details_idx)
        # ...while the trailing un-phased message goes inside it.
        self.assertContains(response, "post-answer note")
        self.assertContains(response, "1 thinking message")


class IntermediateCollapseTests(TestCase):
    """Non-final agent messages and tool calls fold into a <details> block so
    the page renders only the user/final-agent conversation by default."""

    @patch("hitch.main.views.Codex")
    def test_intermediate_thinking_and_tool_calls_collapse(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Help me out"),
                        _agent_message("Let me look at this."),
                        _command("./scripts/check.sh"),
                        _agent_message("Trying something else."),
                        _agent_message("Here is the answer."),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Here is the answer.")
        self.assertContains(response, '<details class="intermediate">')
        self.assertContains(response, "2 thinking messages and 1 tool call")
        # Intermediate content stays in the document (collapsed, not removed).
        self.assertContains(response, "Let me look at this.")
        self.assertContains(response, "Trying something else.")
        self.assertContains(response, "./scripts/check.sh")
        # <details> appears before the final agent message in source order.
        self.assertLess(body.index("<details"), body.index("Here is the answer."))

    @patch("hitch.main.views.Codex")
    def test_no_collapse_when_nothing_intermediate(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread(
            [_turn([_user_message("Hi"), _agent_message("Hello.")])]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "<details")

    @patch("hitch.main.views.Codex")
    def test_summary_with_only_tool_calls(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Run it"),
                        _command("./scripts/run.sh"),
                        _agent_message("Done."),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<summary>1 tool call</summary>", html=False)
        self.assertNotContains(response, "thinking message")

    @patch("hitch.main.views.Codex")
    def test_summary_with_only_thinking(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Think it through"),
                        _agent_message("Step 1."),
                        _agent_message("Final."),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<summary>1 thinking message</summary>", html=False)

    @patch("hitch.main.views.Codex")
    def test_phase_final_answer_wins_over_position(
        self, mock_codex: MagicMock
    ) -> None:
        """An explicit final_answer phase is the final reply even if later
        agent messages have no phase set."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Question"),
                        _agent_message("The answer is 42.", phase="final_answer"),
                        _agent_message("post-answer noise"),
                    ]
                ),
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # "The answer is 42." renders as the top-level Agent block.
        agent_pos = body.index(">Agent<")
        answer_pos = body.index("The answer is 42.")
        noise_pos = body.index("post-answer noise")
        self.assertLess(agent_pos, answer_pos)
        self.assertLess(answer_pos, noise_pos)
        # The trailing message lives inside an intermediate block.
        self.assertContains(response, ">Agent (thinking)<")

    @patch("hitch.main.views.Codex")
    def test_commentary_phase_is_never_final(self, mock_codex: MagicMock) -> None:
        """All-commentary turns (in-progress) collapse entirely; no top-level
        Agent block."""
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
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<details class="intermediate">')
        self.assertContains(response, "2 thinking messages")
        self.assertNotContains(response, ">Agent<")
        self.assertContains(response, ">Agent (thinking)<")

    @patch("hitch.main.views.Codex")
    def test_phase_accepts_raw_string_shape(self, mock_codex: MagicMock) -> None:
        """Robustness: phase as a raw wire string (e.g. data deserialized
        without pydantic) is still recognized."""
        commentary = _root(
            SimpleNamespace(type="agentMessage", text="Thinking.", phase="commentary")
        )
        final = _root(
            SimpleNamespace(type="agentMessage", text="42.", phase="final_answer")
        )
        thread = _thread(
            [_turn([_user_message("Q"), commentary, final])]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        agent_pos = body.index(">Agent<")
        answer_pos = body.index("42.")
        self.assertLess(agent_pos, answer_pos)
        self.assertContains(response, ">Agent (thinking)<")


class ToolCallDetailTests(TestCase):
    """Exercise every branch of _tool_call_detail so the per-type description
    surfaced in the UI stays stable.
    """

    def test_command_execution(self) -> None:
        item = SimpleNamespace(command="ls -la")
        self.assertEqual(_tool_call_detail(item, "commandExecution"), "ls -la")

    def test_command_execution_missing_command(self) -> None:
        self.assertEqual(_tool_call_detail(SimpleNamespace(command=None), "commandExecution"), "")

    def test_mcp_tool_call(self) -> None:
        item = SimpleNamespace(server="github", tool="create_pr")
        self.assertEqual(_tool_call_detail(item, "mcpToolCall"), "github / create_pr")

    def test_dynamic_tool_call_with_namespace(self) -> None:
        item = SimpleNamespace(namespace="codex", tool="apply_patch")
        self.assertEqual(_tool_call_detail(item, "dynamicToolCall"), "codex::apply_patch")

    def test_dynamic_tool_call_without_namespace(self) -> None:
        item = SimpleNamespace(namespace=None, tool="apply_patch")
        self.assertEqual(_tool_call_detail(item, "dynamicToolCall"), "apply_patch")

    def test_file_change_empty(self) -> None:
        self.assertEqual(_tool_call_detail(SimpleNamespace(changes=[]), "fileChange"), "")

    def test_file_change_single(self) -> None:
        item = SimpleNamespace(changes=[SimpleNamespace(path="a.py")])
        self.assertEqual(_tool_call_detail(item, "fileChange"), "a.py")

    def test_file_change_multiple(self) -> None:
        item = SimpleNamespace(
            changes=[SimpleNamespace(path="a.py"), SimpleNamespace(path="b.py")]
        )
        self.assertEqual(_tool_call_detail(item, "fileChange"), "a.py (+1 more)")

    def test_web_search(self) -> None:
        self.assertEqual(
            _tool_call_detail(SimpleNamespace(query="how to django"), "webSearch"),
            "how to django",
        )

    def test_plan_takes_first_line(self) -> None:
        item = SimpleNamespace(text="Step 1\nStep 2\nStep 3")
        self.assertEqual(_tool_call_detail(item, "plan"), "Step 1")

    def test_image_view(self) -> None:
        self.assertEqual(
            _tool_call_detail(SimpleNamespace(path="/tmp/x.png"), "imageView"),
            "/tmp/x.png",
        )

    def test_image_generation_prefers_revised_prompt(self) -> None:
        item = SimpleNamespace(revised_prompt="a cat", saved_path="/tmp/y.png")
        self.assertEqual(_tool_call_detail(item, "imageGeneration"), "a cat")

    def test_image_generation_falls_back_to_saved_path(self) -> None:
        item = SimpleNamespace(revised_prompt=None, saved_path="/tmp/y.png")
        self.assertEqual(_tool_call_detail(item, "imageGeneration"), "/tmp/y.png")

    def test_collab_agent_tool_call_with_receiver(self) -> None:
        item = SimpleNamespace(
            tool=SimpleNamespace(value="spawn"),
            receiver_thread_ids=["child-thread"],
        )
        self.assertIn("child-thread", _tool_call_detail(item, "collabAgentToolCall"))
        self.assertIn("spawn", _tool_call_detail(item, "collabAgentToolCall"))

    def test_collab_agent_tool_call_without_receiver(self) -> None:
        item = SimpleNamespace(
            tool=SimpleNamespace(value="spawn"),
            receiver_thread_ids=[],
        )
        self.assertEqual(_tool_call_detail(item, "collabAgentToolCall"), "spawn")

    def test_unknown_type_returns_empty(self) -> None:
        self.assertEqual(_tool_call_detail(SimpleNamespace(), "somethingNew"), "")

    def test_status_completed_is_hidden(self) -> None:
        item = SimpleNamespace(status=SimpleNamespace(value="completed"))
        self.assertIsNone(_tool_call_status(item))

    def test_status_failed_is_surfaced(self) -> None:
        item = SimpleNamespace(status=SimpleNamespace(value="failed"))
        self.assertEqual(_tool_call_status(item), "failed")

    def test_missing_status_is_none(self) -> None:
        self.assertIsNone(_tool_call_status(SimpleNamespace()))
