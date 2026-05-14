from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse


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


def _agent_message(text: str) -> SimpleNamespace:
    return _root(SimpleNamespace(type="agentMessage", text=text))


def _tool_call(item_type: str) -> SimpleNamespace:
    return _root(SimpleNamespace(type=item_type))


def _thread(turns: list[list[SimpleNamespace]], **overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "id": "thread-1",
        "name": "Demo session",
        "preview": "first message",
        "cwd": "/tmp/demo",
        "updated_at": 1700000000,
        "turns": [SimpleNamespace(items=items) for items in turns],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class SessionViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_renders_user_and_agent_messages(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                [
                    _user_message("Refactor the login flow"),
                    _agent_message("Sure, here is the plan."),
                ],
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
    def test_summarizes_tool_calls_between_messages(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                [
                    _user_message("Do the thing"),
                    _tool_call("commandExecution"),
                    _tool_call("commandExecution"),
                    _tool_call("fileChange"),
                    _agent_message("Done."),
                ],
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "3 tool calls")
        self.assertContains(response, "Command execution &times; 2", html=False)
        self.assertContains(response, "File change &times; 1", html=False)

    @patch("hitch.main.views.Codex")
    def test_trailing_tool_calls_are_summarized(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                [
                    _user_message("Investigate"),
                    _agent_message("Looking into it."),
                    _tool_call("commandExecution"),
                ],
            ]
        )
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "1 tool call:")
        self.assertContains(response, "Command execution &times; 1", html=False)

    @patch("hitch.main.views.Codex")
    def test_empty_session_shows_placeholder(self, mock_codex: MagicMock) -> None:
        thread = _thread([])
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_read.return_value.thread = thread

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")
