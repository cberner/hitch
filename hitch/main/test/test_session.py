import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.core import signing
from django.http import HttpResponse
from django.test import Client, TestCase
from django.urls import reverse
from openai_codex.errors import AppServerError

from hitch.main import codex_events, system_agents
from hitch.main.diffs import DiffFile, DiffLine, DiffView
from hitch.main.models import (
    CodexInstance,
    KeyResult,
    Objective,
    Project,
    SessionDemo,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.views import _pr_url_for_thread, _tool_call_detail, _tool_call_status

# Used for active-worker rendering tests so the session view's
# ``reconcile_dead`` sweep doesn't mark the row failed before the assertions
# run; the current process pid is by definition alive.
_LIVE_PID = os.getpid()


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


def _write_rollout_tempfile(lines: list[str], *, binary: bytes | None = None) -> Path:
    if binary is not None:
        with tempfile.NamedTemporaryFile(
            prefix="rollout-", suffix=".jsonl", mode="wb", delete=False
        ) as fh:
            fh.write(binary)
            return Path(fh.name)
    with tempfile.NamedTemporaryFile(
        prefix="rollout-", suffix=".jsonl", mode="w", delete=False
    ) as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
        return Path(fh.name)


def _patch_thread(test: TestCase, mock_codex: MagicMock, thread: SimpleNamespace) -> None:
    client = mock_codex.return_value.__enter__.return_value
    client._client.thread_resume.return_value.thread = thread


def _make_rollout(test: TestCase, lines: list[str], *, binary: bytes | None = None) -> Path:
    path = _write_rollout_tempfile(lines, binary=binary)
    test.addCleanup(path.unlink, missing_ok=True)
    return path


def _get_session(client: Client, session_id: str = "thread-1") -> HttpResponse:
    response = client.get(reverse("session", kwargs={"session_id": session_id}))
    assert isinstance(response, HttpResponse)
    return response


def _sign(name: str, value: str) -> str:
    return signing.get_cookie_signer(salt=name).sign(value)


def _seed_cookies(client: Client, **values: str) -> None:
    for name, value in values.items():
        client.cookies[name] = _sign(name, value)


def _cookie_value(response: HttpResponse, name: str) -> str:
    return signing.get_cookie_signer(salt=name).unsign(response.cookies[name].value)


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
    def test_detects_pr_url_from_latest_pr_turn_github_mcp_result(self) -> None:
        earlier = "https://github.com/cberner/hitch/pull/93"
        latest = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("ordinary follow-up"),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {"structuredContent": {"display_url": earlier}},
                        ),
                        _agent_message("Done."),
                    ]
                ),
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call(
                            "github",
                            "_create_pull_request",
                            {
                                "content": [
                                    {
                                        "text": json.dumps(
                                            {"url": earlier, "display_url": latest}
                                        )
                                    }
                                ],
                            },
                        ),
                        _agent_message("Opened the PR."),
                    ]
                ),
            ]
        )

        self.assertEqual(_pr_url_for_thread(thread), latest)

    def test_ignores_non_pr_turns_and_non_github_tools(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        non_pr_thread = _thread(
            [
                _turn(
                    [
                        _user_message("ordinary follow-up"),
                        _mcp_tool_call("github", "_create_pull_request", {"url": url}),
                        _agent_message("Done."),
                    ]
                )
            ]
        )
        non_github_thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("linear", "create_issue", {"url": url}),
                        _agent_message("Done."),
                    ]
                )
            ]
        )

        self.assertIsNone(_pr_url_for_thread(non_pr_thread))
        self.assertIsNone(_pr_url_for_thread(non_github_thread))

    def test_ignores_incomplete_pr_turns(self) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"url": url}),
                    ]
                )
            ]
        )

        self.assertIsNone(_pr_url_for_thread(thread))

    def test_latest_completed_pr_turn_without_url_stops_search(self) -> None:
        stale_url = "https://github.com/cberner/hitch/pull/93"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"url": stale_url}),
                        _agent_message("Opened the PR."),
                    ]
                ),
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"content": []}),
                        _agent_message("No PR was opened."),
                    ]
                ),
            ]
        )

        self.assertIsNone(_pr_url_for_thread(thread))


class SessionViewTests(TestCase):
    @patch("hitch.main.views.Codex")
    def test_renders_primary_nav_menu_instead_of_back_link(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-nav-menu")
        self.assertContains(response, "data-nav-menu-open")
        self.assertContains(response, "data-nav-menu-panel")
        self.assertContains(response, 'aria-label="Navigation menu"')
        self.assertContains(response, f'href="{reverse("index")}#new-session"')
        self.assertContains(response, ">new session<")
        self.assertContains(response, f'href="{reverse("index")}"')
        self.assertNotContains(
            response, f'href="{reverse("index")}" aria-current="page"'
        )
        self.assertContains(response, f'href="{reverse("usage")}"')
        self.assertContains(response, ">settings<")
        self.assertContains(response, "data-settings-dialog")
        self.assertNotContains(response, 'class="back-link"')

    @patch("hitch.main.views.Codex")
    def test_settings_dialog_uses_resolved_settings(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi")])])
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value.thread = thread
        client.models.return_value.data = [
            SimpleNamespace(
                id="gpt-current",
                display_name="GPT Current",
                is_default=True,
                default_reasoning_effort=SimpleNamespace(value="medium"),
                supported_reasoning_efforts=[],
            )
        ]
        _seed_cookies(
            self.client,
            hitch_model="stale-model",
            hitch_reasoning_effort="high",
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'value="gpt-current" selected')
        self.assertEqual(_cookie_value(response, "hitch_model"), "gpt-current")
        self.assertEqual(_cookie_value(response, "hitch_reasoning_effort"), "medium")

    @patch("hitch.main.views.Codex")
    def test_renders_edit_title_form(self, mock_codex: MagicMock) -> None:
        """The edit form is pre-populated with the current name when set, and
        empty when not — so the user can revise without retyping from scratch."""
        for name, expected_value in (("Custom title", "Custom title"), (None, "")):
            with self.subTest(name=name):
                thread = _thread([_turn([_user_message("hi")])], name=name)
                _patch_thread(self, mock_codex, thread)
                response = _get_session(self.client)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, 'name="name"')
                self.assertContains(response, f'value="{expected_value}"')
                self.assertContains(
                    response,
                    reverse("set_session_name", kwargs={"session_id": "thread-1"}),
                )
                self.assertContains(
                    response,
                    reverse("set_session_archived", kwargs={"session_id": "thread-1"}),
                )
                self.assertContains(response, 'aria-label="Session actions"')
                self.assertContains(response, 'role="menuitem" data-edit-title-open>Rename')
                self.assertContains(response, 'name="archived" value="true"')
                self.assertContains(response, 'role="menuitem">Archive</button>')
                self.assertNotContains(response, ">Edit</button>")

    @patch("hitch.main.views.Codex")
    def test_renders_move_to_project_menu_and_dialog(self, mock_codex: MagicMock) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/tmp/demo")
        SessionMetadata.objects.create(thread_id="thread-1", cwd="/tmp/demo", project=project)
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Move to project")
        self.assertContains(
            response,
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
        )
        self.assertContains(response, 'name="project"')
        self.assertContains(response, f'value="{project.pk}" selected')

    @patch("hitch.main.views.Codex")
    def test_set_session_project_moves_and_clears_project(
        self, mock_codex: MagicMock
    ) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/tmp/demo")
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": str(project.pk)},
        )

        self.assertEqual(response.status_code, 302)
        metadata = SessionMetadata.objects.get(thread_id="thread-1")
        self.assertEqual(metadata.project, project)
        self.assertEqual(metadata.cwd, "/tmp/demo")
        self.assertFalse(metadata.project_cleared)

        response = self.client.post(
            reverse("set_session_project", kwargs={"session_id": "thread-1"}),
            data={"project": ""},
        )

        self.assertEqual(response.status_code, 302)
        metadata.refresh_from_db()
        self.assertIsNone(metadata.project)
        self.assertTrue(metadata.project_cleared)

    @patch("hitch.main.views.Codex")
    def test_renders_open_pr_menu_link_when_detected(
        self, mock_codex: MagicMock
    ) -> None:
        url = "https://github.com/cberner/hitch/pull/94"
        thread = _thread(
            [
                _turn(
                    [
                        _user_message(system_agents.PR_SLASH_PROMPT),
                        _mcp_tool_call("github", "_create_pull_request", {"url": url}),
                        _agent_message("Opened the PR."),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'<a href="{url}" role="menuitem" target="_blank" rel="noopener noreferrer">Open PR</a>',
            html=True,
        )

    @patch("hitch.main.views.Codex")
    def test_hides_open_pr_menu_link_without_detected_pr(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi"), _agent_message("Hello.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Open PR")

    @patch("hitch.main.views.Codex")
    def test_next_message_settings_render_under_title(
        self, mock_codex: MagicMock
    ) -> None:
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.return_value = SimpleNamespace(
            thread=_thread([_turn([_user_message("hi")])]),
            model="gpt-5.4",
            reasoning_effort=SimpleNamespace(value="high"),
        )
        _seed_cookies(
            self.client,
            hitch_sandbox_policy="workspaceWrite",
            hitch_approval_mode="prompt_user",
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'aria-label="Settings for the next message"')
        self.assertContains(response, ">model<")
        self.assertContains(response, "gpt-5.4")
        self.assertContains(response, ">reasoning<")
        self.assertContains(response, "high")
        self.assertContains(response, 'data-normal-value="high"')
        self.assertContains(response, 'data-plan-value="medium"')
        self.assertContains(response, ">sandbox<")
        self.assertContains(response, "Workspace write")
        self.assertContains(response, ">approval<")
        self.assertContains(response, "Always prompt for approval")

    @patch("hitch.main.views.Codex")
    def test_system_feedback_renders_with_display_author(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Feedback from Hitch QA agent:\n\nFix the failing flow."
        thread = _thread([_turn([_user_message(prompt), _agent_message("fixed")])])
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            display_author="QA agent",
            user_message_index=0,
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>')
        self.assertNotContains(response, '<span class="role">User</span>')

    @patch("hitch.main.views.Codex")
    def test_system_feedback_author_uses_turn_index_not_text(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "Feedback from Hitch QA agent:\n\nFix the failing flow."
        thread = _thread(
            [
                _turn([_user_message(prompt), _agent_message("fixed")]),
                _turn([_user_message(prompt), _agent_message("explained")]),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        CodexInstance.objects.create(
            pid=1,
            thread_id="thread-1",
            cwd="/tmp/demo",
            prompt=prompt,
            events_path="/dev/null",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            display_author="QA agent",
            user_message_index=0,
        )

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>', count=1)
        self.assertContains(response, '<span class="role">User</span>', count=1)

    @patch("hitch.main.views.Codex")
    def test_next_message_model_comes_only_from_resumed_thread(
        self, mock_codex: MagicMock
    ) -> None:
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

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_renders_diff_viewer_entry_points_and_modal(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'role="menuitem" data-diff-open')
        self.assertContains(response, 'class="diff-fab"')
        self.assertContains(response, '<dialog class="diff-modal"', html=False)
        self.assertContains(response, "Working tree diff")
        self.assertContains(response, "hitch/main/views.py")
        self.assertContains(response, '<span class="k">return</span> 2', html=False)

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_diff_menu_item_is_disabled_without_changes(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = DiffView(files=[])
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertContains(response, "data-diff-open disabled")
        self.assertNotContains(response, 'class="diff-fab"')
        self.assertNotContains(response, '<dialog class="diff-modal"', html=False)

    @patch("hitch.main.views.Codex")
    def test_archived_session_menu_offers_unarchive(self, mock_codex: MagicMock) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        rollout_path = (
            Path(temp_dir.name)
            / "archived_sessions"
            / "2026"
            / "05"
            / "15"
            / "rollout-thread-1.jsonl"
        )
        rollout_path.parent.mkdir(parents=True)
        rollout_path.write_text(
            _rollout_line("event_msg", {"type": "user_message", "message": "hi"}) + "\n"
        )
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

    @patch("hitch.main.views.Codex")
    def test_archived_menu_label_does_not_depend_on_thread_list(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("hi")])])
        _patch_thread(self, mock_codex, thread)
        client = mock_codex.return_value.__enter__.return_value
        client.thread_list.side_effect = AppServerError("archived list unavailable")

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'role="menuitem">Archive</button>')
        client.thread_list.assert_not_called()

    @patch("hitch.main.views.Codex")
    def test_session_menu_offers_demo_actions(self, mock_codex: MagicMock) -> None:
        SessionDemo.objects.create(
            thread_id="thread-1",
            host="127.0.0.1",
            port=45678,
            status=SessionDemo.STATUS_ACTIVE,
        )
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'action="{reverse("start_session_demo", kwargs={"session_id": "thread-1"})}"',
        )
        self.assertContains(response, "Start demo</button>")
        self.assertNotContains(response, "disabled>Start demo</button>")
        self.assertContains(response, 'href="http://testserver/sessions/thread-1/demo/"')
        self.assertContains(response, 'role="menuitem" target="_blank" rel="noopener noreferrer">Open demo</a>')

    @patch("hitch.main.views.Codex")
    def test_start_demo_menu_item_disabled_during_system_workflow(
        self, mock_codex: MagicMock
    ) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_RUNNING,
        )
        _patch_thread(self, mock_codex, _thread([_turn([_user_message("hi")])]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "disabled>Start demo</button>")

    @patch("hitch.main.views.Codex")
    def test_topbar_title_truncates_long_preview(self, mock_codex: MagicMock) -> None:
        # The session page topbar title uses the same `_display_title` as the
        # index, so an unnamed thread with a long preview gets clipped here too.
        thread = _thread([_turn([_user_message("x" * 200)])], name=None, preview="x" * 200)
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # The topbar title contains the clipped title, not the full 200-char preview.
        self.assertIn('<div class="topbar-title">' + "x" * 80 + "...</div>", body)

    @patch("hitch.main.views.Codex")
    def test_renders_messages_tool_calls_and_timestamps(
        self, mock_codex: MagicMock
    ) -> None:
        """A representative turn: user prompt, agent reply, several tool
        calls (each rendered as its own row, including an unknown type),
        and the turn's started_at flowing through as ``data-ts``."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Refactor the login flow"),
                        _command("./scripts/build.sh"),
                        _command("./scripts/test.sh"),
                        _file_change("hitch/main/views.py"),
                        _tool_call("brandNewTool"),
                        _agent_message("Sure, here is the plan."),
                    ],
                    started_at=1700000123,
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # ``thread/read`` requires the thread to already be loaded into the
        # codex app-server; the view uses ``thread/resume`` so a brand-new
        # session (or any thread the request's app-server hasn't seen) can
        # be loaded from disk in the same call that fetches the turns.
        client = mock_codex.return_value.__enter__.return_value
        client._client.thread_resume.assert_called_once_with("thread-1")
        self.assertContains(response, "Demo session")
        self.assertContains(response, "Refactor the login flow")
        self.assertContains(response, "Sure, here is the plan.")
        self.assertContains(response, ">User<")
        self.assertContains(response, ">Agent<")
        self.assertContains(response, "./scripts/build.sh")
        self.assertContains(response, "./scripts/test.sh")
        self.assertContains(response, "hitch/main/views.py")
        # Unmapped types fall back to the raw type tag so nothing is hidden.
        self.assertContains(response, "brandNewTool")
        # Four separate tool-call rows (inside the collapsed intermediate
        # block), not a single aggregate row.
        self.assertEqual(body.count('class="tool-call"'), 4)
        self.assertContains(response, 'data-ts="1700000123"')

    @patch("hitch.main.views.Codex")
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
        self.assertContains(response, "Memories used: 1")
        self.assertContains(response, "MEMORY.md:1-2")
        self.assertContains(response, "project convention")
        self.assertContains(response, "019cc2ea-1dff-7902-8d40-c8f6e5d83cc4")
        self.assertContains(response, "function renderMemoryCitation")
        self.assertContains(response, "item.memoryCitation || item.memory_citation")

    @patch("hitch.main.views.Codex")
    def test_messages_are_copyable_by_long_press(self, mock_codex: MagicMock) -> None:
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Copy my prompt"),
                        _agent_message("Thinking", phase="commentary"),
                        _agent_message("Copy my answer"),
                    ]
                )
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertContains(
            response,
            '<div class="message user" data-copyable-message>',
            html=False,
        )
        self.assertContains(
            response,
            '<div class="message thinking" data-copyable-message>',
            html=False,
        )
        self.assertContains(
            response,
            '<div class="message agent" data-copyable-message>',
            html=False,
        )
        self.assertContains(response, 'target.closest("[data-copyable-message]")')
        self.assertContains(response, "navigator.clipboard.writeText(text)")
        self.assertContains(response, 'document.addEventListener("pointerdown"')
        self.assertContains(response, 'msg.dataset.copyableMessage = "";')

    @patch("hitch.main.views.Codex")
    def test_trailing_tool_calls_are_rendered(self, mock_codex: MagicMock) -> None:
        # Mid-turn agent commentary followed by a tool call (no final agent
        # message) still surfaces the tool call.
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
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "rg --files")

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
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "failed")

    @patch("hitch.main.views.Codex")
    def test_empty_session_shows_placeholder(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")


class RolloutFileViewTests(TestCase):
    """When ``thread.path`` points at a rollout file, the view parses it
    directly to recover the commandExecution items codex strips from
    ``thread/read``; failure modes (missing, malformed, empty, or
    contentless rollouts) fall back to the SDK-derived entries instead of
    bubbling the error to the response.
    """

    @patch("hitch.main.views.Codex")
    def test_prefers_rollout_file_when_path_is_set(self, mock_codex: MagicMock) -> None:
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
        rollout_path = _make_rollout(self, rollout_lines)

        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "build it")
        self.assertContains(response, "cargo build --release")
        self.assertContains(response, "done")
        self.assertContains(response, '<details class="intermediate">')
        self.assertContains(response, "1 thinking message and 1 tool call")

    @patch("hitch.main.views.Codex")
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
        self.assertContains(response, "Memories used: 1")
        self.assertContains(response, "MEMORY.md:3-4")
        self.assertContains(response, "repo preference")
        self.assertNotContains(response, "oai-mem-citation")

    @patch("hitch.main.views.Codex")
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
        self.assertLess(body.index("</details>"), body.index("Approval declined"))
        self.assertLess(body.index("Approval declined"), body.index("Please confirm explicitly."))

    @patch("hitch.main.views.Codex")
    def test_plan_mode_rollout_renders_final_plan_card(
        self, mock_codex: MagicMock
    ) -> None:
        plan = "# Fix Login CSRF Redirect Failure\n\n## Summary\nRead the CSRF cookie at submit time."
        rollout_lines = [
            _rollout_line(
                "event_msg",
                {"type": "user_message", "message": "Debug the login CSRF issue"},
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call",
                    "name": "request_user_input",
                    "arguments": json.dumps(
                        {
                            "questions": [
                                {
                                    "id": "failure_point",
                                    "header": "Failure",
                                    "question": "Where do you see the CSRF failure?",
                                    "options": [
                                        {
                                            "label": "Main-page action (Recommended)",
                                            "description": "The next unsafe action fails.",
                                        },
                                        {
                                            "label": "Immediate page",
                                            "description": "Login lands on the failure page.",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    "call_id": "ask",
                },
            ),
            _rollout_line(
                "response_item",
                {
                    "type": "function_call_output",
                    "call_id": "ask",
                    "output": json.dumps({"answers": {}}),
                },
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
        ]
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="plan-card"')
        self.assertContains(response, "Fix Login CSRF Redirect Failure")
        self.assertContains(response, "Read the CSRF cookie at submit time.")
        self.assertContains(response, "Approve plan")
        self.assertContains(response, 'name="plan_action" value="approve"')
        self.assertContains(response, 'name="collaboration_mode" value="default"')
        self.assertContains(response, "Implement the plan.")
        self.assertContains(response, 'name="plan_action" value="revise"')
        self.assertContains(response, 'data-initial-plan-mode="true"')
        self.assertContains(response, 'name="plan_mode" value="true" data-plan-mode-input')
        self.assertContains(response, 'name="default_plan_mode" value="true"')
        self.assertContains(
            response,
            'name="plan_mode_explicit" value="" data-plan-mode-explicit-input',
        )
        self.assertNotContains(response, "&lt;proposed_plan&gt;")
        self.assertNotContains(response, "request_user_input")
        self.assertNotContains(response, '<details class="intermediate">')
        self.assertIn("Debug the login CSRF issue", body)

    @patch("hitch.main.views.Codex")
    def test_resolved_plan_card_hides_approval_actions(
        self, mock_codex: MagicMock
    ) -> None:
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

    @patch("hitch.main.views.Codex")
    def test_falls_back_to_sdk_on_rollout_failure_modes(
        self, mock_codex: MagicMock
    ) -> None:
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

    @patch("hitch.main.views.Codex")
    def test_empty_rollout_with_no_sdk_turns_renders_placeholder(
        self, mock_codex: MagicMock
    ) -> None:
        # Both the rollout and Thread.turns are empty — the page should still
        # render its empty-state placeholder rather than fall back to a
        # second SDK call.
        rollout_path = _make_rollout(self, [])
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No messages in this session yet.")

    @patch("hitch.main.views.Codex")
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
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

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
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "try this")
        self.assertContains(response, "sleep 1")
        self.assertContains(response, '<details class="intermediate">')
        self.assertNotContains(response, ">Agent<")

    @patch("hitch.main.views.Codex")
    def test_rollout_commentary_phase_never_treated_as_final(
        self, mock_codex: MagicMock
    ) -> None:
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
        self.assertContains(response, "2 thinking messages and 1 tool call")

    @patch("hitch.main.views.Codex")
    def test_rollout_final_answer_phase_wins_over_later_unphased(
        self, mock_codex: MagicMock
    ) -> None:
        # The explicit final_answer is the final agent reply even when an
        # un-phased agent message follows it; the trailing message folds
        # into the post-final intermediate block.
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
        rollout_path = _make_rollout(self, rollout_lines)
        _patch_thread(self, mock_codex, _thread([], path=str(rollout_path)))

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # The final_answer text renders outside the <details> block...
        self.assertLess(body.index("the answer"), body.index('<details class="intermediate"'))
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
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
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
        thread = _thread([_turn([_user_message("Hi"), _agent_message("Hello.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, '<details class="intermediate"')

    @patch("hitch.main.views.Codex")
    def test_summary_pluralization(self, mock_codex: MagicMock) -> None:
        """The summary shows just the relevant kind(s) and pluralizes
        correctly: 1 tool call, 1 thinking message, etc."""
        # The "must_not_contain" pattern is anchored to ``</summary>`` so it
        # matches only the rendered summary text — the streaming script
        # also mentions ``tool call`` and ``thinking`` in its tool-label
        # map and would otherwise trigger a false positive.
        cases: list[tuple[list[SimpleNamespace], str, str]] = [
            (
                [_user_message("Run it"), _command("./scripts/run.sh"), _agent_message("Done.")],
                "<summary>1 tool call</summary>",
                "thinking message</summary>",
            ),
            (
                [_user_message("Think it through"), _agent_message("Step 1."), _agent_message("Final.")],
                "<summary>1 thinking message</summary>",
                "tool call</summary>",
            ),
        ]
        for items, expected, must_not_contain in cases:
            with self.subTest(expected=expected):
                _patch_thread(self, mock_codex, _thread([_turn(items)]))
                response = _get_session(self.client)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, expected, html=False)
                self.assertNotContains(response, must_not_contain)

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
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
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
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

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

    @patch("hitch.main.views.Codex")
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

    @patch("hitch.main.views.Codex")
    def test_plain_final_agent_is_not_treated_as_markdown(
        self, mock_codex: MagicMock
    ) -> None:
        thread = _thread([_turn([_user_message("Hi"), _agent_message("Hello, friend.")])])
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Hello, friend.")
        self.assertNotContains(response, '<div class="body markdown">')

    @patch("hitch.main.views.Codex")
    def test_markdown_styling_is_only_for_final_agent(
        self, mock_codex: MagicMock
    ) -> None:
        """Mid-turn agent commentary and user messages keep their literal
        markdown characters even when the text looks like a heading: only
        the turn's final agent reply gets markdown rendering."""
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("# my heading"),
                        _agent_message("# Thinking..."),
                        _agent_message("ok"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "# my heading")
        self.assertContains(response, "# Thinking...")
        self.assertNotContains(response, "<h1>my heading</h1>")
        self.assertNotContains(response, "<h1>Thinking")

    @patch("hitch.main.views.Codex")
    def test_agent_html_is_escaped(self, mock_codex: MagicMock) -> None:
        # Even when the body is detected as markdown, raw HTML in the
        # source is escaped, so an agent reply can't smuggle a <script>
        # tag into the page.
        thread = _thread(
            [
                _turn(
                    [
                        _user_message("Q"),
                        _agent_message("# Heading\n\n<script>alert(1)</script>"),
                    ]
                ),
            ]
        )
        _patch_thread(self, mock_codex, thread)

        response = _get_session(self.client)
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("<script>alert(1)</script>", body)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", body)


class ToolCallDetailTests(TestCase):
    """Exercise every branch of ``_tool_call_detail`` and ``_tool_call_status``
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
                SimpleNamespace(
                    changes=[SimpleNamespace(path="a.py"), SimpleNamespace(path="b.py")]
                ),
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
                self.assertEqual(_tool_call_detail(item, tool_type), expected)

    def test_collab_agent_detail(self) -> None:
        # Two cases share an assertion shape but differ in what they expose.
        with_receiver = SimpleNamespace(
            tool=SimpleNamespace(value="spawn"), receiver_thread_ids=["child-thread"]
        )
        without_receiver = SimpleNamespace(
            tool=SimpleNamespace(value="spawn"), receiver_thread_ids=[]
        )
        result = _tool_call_detail(with_receiver, "collabAgentToolCall")
        self.assertIn("child-thread", result)
        self.assertIn("spawn", result)
        self.assertEqual(
            _tool_call_detail(without_receiver, "collabAgentToolCall"), "spawn"
        )

    def test_status_badge(self) -> None:
        # ``completed`` and missing status both render no badge; ``failed``
        # surfaces verbatim so the UI can highlight it.
        self.assertIsNone(_tool_call_status(SimpleNamespace(status=SimpleNamespace(value="completed"))))
        self.assertIsNone(_tool_call_status(SimpleNamespace()))
        self.assertEqual(
            _tool_call_status(SimpleNamespace(status=SimpleNamespace(value="failed"))),
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

    @patch("hitch.main.views.Codex")
    def test_inactive_thread_renders_status_pill_idle_with_no_live_root(
        self, mock_codex: MagicMock
    ) -> None:
        # Without an active worker the live-streaming insertion anchor
        # (``data-live-root``) must not be in the DOM, so streamed item
        # events have nowhere to land — but the live-status pill still
        # renders (in its hidden idle state) so the JS heartbeat handler
        # can surface reconnecting/fatal connection states if frames stop
        # arriving.
        # The pill-text checks use the ``>...</span`` anchor so they
        # don't false-match the same string literal inside the JS map.
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "data-live-root></div>")
        self.assertContains(response, "data-live-status")
        self.assertContains(response, 'data-state="idle"')
        self.assertContains(response, ">Connected</span>")
        self.assertNotContains(response, ">Codex is working")
        self.assertNotContains(response, 'class="jump-latest" data-jump-latest')

    @patch("hitch.main.views.Codex")
    def test_connection_indicator_retries_before_showing_fatal_loss(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'reconnecting: "Reconnecting')
        self.assertContains(response, "const RECONNECT_FATAL_MS")
        self.assertContains(response, '"disconnected" : "reconnecting"')
        self.assertContains(response, 'indicator.addEventListener("click"')
        self.assertContains(response, "function retryNow()")
        self.assertContains(response, "function clearRetryTimer()")
        self.assertContains(response, "new EventSource(streamUrl)")
        self.assertContains(response, "const seenEvents = new Set()")
        self.assertContains(response, "if (seenEvents.has(key)) return;")

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_active_worker_renders_status_pill_working(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        # With an active worker the pill renders in its "working" state
        # so the user sees the pulsing-green indicator immediately, even
        # before the first heartbeat lands.
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "data-live-status")
        self.assertContains(response, 'data-state="working"')
        self.assertContains(response, ">Codex is working")
        self.assertContains(response, 'class="jump-latest" data-jump-latest')
        self.assertContains(response, 'aria-label="Jump to latest message"')
        self.assertContains(response, '<dialog class="diff-modal"', html=False)
        self.assertNotContains(response, 'class="diff-fab"')
        self.assertNotContains(response, 'aria-label="Settings for the next message"')

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_active_qa_worker_renders_token_progress(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "goal": {
                                "objective": "Apply QA feedback",
                                "tokens_used": 1234,
                            },
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_FEEDBACK,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            events_path=events_path,
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "QA agent working...1.2K tokens")
        self.assertContains(response, 'data-working-text="QA agent working...1.2K tokens"')

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_active_worker_renders_latest_goal_near_status_pill(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = _diff_view()
        _patch_thread(self, mock_codex, _thread([]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "goal": {"objective": "Keep the live goal visible"},
                        },
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

        self.assertContains(response, 'data-live-work data-state="working"')
        self.assertContains(response, 'data-live-goal')
        self.assertContains(response, ">Goal<")
        self.assertContains(response, "Keep the live goal visible")

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_active_worker_renders_latest_task_plan_after_refresh(
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
                            "explanation": "Keep progress visible after refresh",
                            "plan": [
                                {"step": "Inspect current live UI", "status": "completed"},
                                {"step": "Render saved task plan", "status": "in_progress"},
                                {"step": "Run the refresh tests", "status": "pending"},
                            ],
                        },
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

        self.assertContains(response, 'data-task-plan')
        self.assertContains(response, "<main data-session-main data-stream-url=")
        self.assertContains(response, 'class="has-task-plan"')
        self.assertContains(response, 'data-recorded-at="20"')
        self.assertContains(response, 'data-event-seq="2"')
        self.assertContains(response, 'data-fallback-order="1"')
        self.assertNotIn('aria-label="Current task plan" hidden', body)
        self.assertContains(response, "Keep progress visible after refresh")
        self.assertContains(response, "Render saved task plan")
        self.assertContains(response, 'class="plan-step inProgress"')
        self.assertContains(
            response,
            '<span class="task-plan-current" data-task-plan-current>Render saved task plan</span>',
            html=False,
        )

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
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

        self.assertContains(response, 'data-task-plan')
        self.assertContains(response, "<main data-session-main data-stream-url=")
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

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_active_worker_preserves_fallback_task_plan_order_after_refresh(
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
                                {"step": "Fallback stale task", "status": "in_progress"},
                            ],
                        },
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                        "payload": {
                            "threadId": "thread-1",
                            "plan": [
                                {"step": "Fallback latest task", "status": "pending"},
                            ],
                        },
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

        self.assertContains(response, 'data-recorded-at="0"')
        self.assertContains(response, 'data-event-seq="0"')
        self.assertContains(response, 'data-fallback-order="2"')
        self.assertContains(response, "Fallback latest task")
        self.assertNotIn("Fallback stale task", body)

    @patch("hitch.main.views.Codex")
    def test_active_worker_renders_live_section_and_stream_url(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="please refactor",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-live-root")
        instance = CodexInstance.objects.get(thread_id="thread-1")
        self.assertContains(
            response, reverse("session_stream", kwargs={"session_id": "thread-1"})
        )
        # The pending user message is surfaced as a regular user bubble so
        # the user sees their own prompt immediately, even before the
        # worker's user_message event reaches the rollout.
        self.assertContains(response, "please refactor")
        self.assertContains(response, "data-pending-user>")
        self.assertContains(
            response,
            '<div class="message user pending" data-copyable-message>',
            html=False,
        )
        # Stop button is wired to the interrupt endpoint via formaction so
        # the user can abort an in-progress turn without leaving the page.
        # The regular composer submit carries the same active instance in a
        # hidden field so it steers this running turn instead of spawning an
        # overlapping follow-up worker.
        # The button's name/value carries the *specific* worker the page
        # is showing — clicking Stop on a stale tab must not accidentally
        # abort a newer overlapping worker the user can't see.
        stop_url = reverse("stop_session", kwargs={"session_id": "thread-1"})
        self.assertContains(response, f'name="active_instance" value="{instance.id}"')
        self.assertContains(response, ">Steer</button>")
        self.assertContains(response, f'formaction="{stop_url}"')
        self.assertContains(response, f'name="instance" value="{instance.id}"')

    @patch("hitch.main.views.Codex")
    def test_inactive_thread_omits_stop_button(
        self, mock_codex: MagicMock
    ) -> None:
        # Stop is only meaningful while a turn is in progress; otherwise it
        # would just be dead chrome cluttering the composer.
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(
            response, reverse("stop_session", kwargs={"session_id": "thread-1"})
        )

    @patch("hitch.main.views.Codex")
    def test_hidden_system_workflow_renders_busy_state(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([]))
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_RUNNING,
            step="qa_running",
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as fh:
            fh.write(
                json.dumps(
                    {
                        "method": codex_events.GOAL_UPDATED_METHOD,
                        "payload": {
                            "threadId": "hidden-thread",
                            "goal": {
                                "objective": "Review the diff",
                                "tokensUsed": 4200,
                            },
                        },
                    }
                )
                + "\n"
            )
            events_path = fh.name
        self.addCleanup(Path(events_path).unlink, missing_ok=True)
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            events_path=events_path,
            pid=_LIVE_PID,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        stream_path = reverse("session_stream", kwargs={"session_id": "thread-1"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "QA agent working...4.2K tokens")
        self.assertContains(response, 'data-workflow-locked="true"')
        self.assertContains(response, "QA workflow is running")
        self.assertContains(response, 'aria-label="Stop the QA workflow"')
        self.assertContains(
            response,
            f'data-stream-url="{stream_path}?baseline=&amp;active=&amp;workflow={workflow.pk}"',
        )

    @patch("hitch.main.views.Codex")
    def test_completed_qa_approval_is_shown_in_transcript(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(
            self,
            mock_codex,
            _thread(
                [
                    _turn([_user_message("Change it"), _agent_message("Done")]),
                    _turn(
                        [
                            _user_message(system_agents.PR_SLASH_PROMPT),
                            _agent_message("Opened PR"),
                        ],
                        started_at=1700000010,
                    ),
                ]
            ),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_PR_PROMPT_SPAWNED,
            state={
                "next_user_message_index": 2,
                "last_feedback": "No qualifying findings.",
            },
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "No qualifying findings.", "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>')
        self.assertContains(response, "QA agent approved the diff.")
        self.assertContains(response, "No qualifying findings.")
        self.assertLess(
            body.index("QA agent approved the diff."),
            body.index(system_agents.PR_SLASH_PROMPT),
        )

    @patch("hitch.main.views.Codex")
    def test_completed_qa_only_approval_is_appended_to_transcript(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(
            self,
            mock_codex,
            _thread([_turn([_user_message("Change it"), _agent_message("Done")])]),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/tmp/demo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_QA_APPROVED,
            state={
                "next_user_message_index": 1,
                "last_feedback": "No qualifying findings.",
            },
        )
        instance = _make_codex_instance(
            thread_id="hidden-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="hidden-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
            output={"feedback": "No qualifying findings.", "lgtm": True},
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        body = response.content.decode()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<span class="role">QA agent</span>')
        self.assertContains(response, "QA agent approved the diff.")
        self.assertContains(response, "No qualifying findings.")
        self.assertLess(body.index("Done"), body.index("QA agent approved the diff."))

    @patch("hitch.main.views.Codex")
    def test_okr_task_generation_log_is_read_only_session_view(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(
            self,
            mock_codex,
            _thread([], id="task-thread", name="Task thread", cwd="/repo"),
        )
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_OKR_TASKS_RUNNING,
            state={"key_result_id": key_result.pk},
        )
        instance = _make_codex_instance(
            thread_id="task-thread",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            prompt="generate tasks",
            pid=_LIVE_PID,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.OKR_TASK_AGENT_KIND,
            thread_id="task-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.get(
            reverse("okr_task_generation_log", kwargs={"workflow_id": workflow.pk})
        )
        stream_path = reverse("session_stream", kwargs={"session_id": "task-thread"})
        stop_url = reverse("stop_session", kwargs={"session_id": "task-thread"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<body class="read-only">')
        self.assertContains(response, "Task generation log")
        self.assertContains(response, "data-live-root")
        self.assertContains(response, 'data-read-only="true"')
        self.assertContains(
            response,
            f'data-stream-url="{stream_path}?baseline={instance.pk}&amp;active={instance.pk}&amp;workflow="',
        )
        html = response.content.decode()
        self.assertIn(
            "approvals.set(id, { node: wrap, box, actions: null, resolved: false });",
            html,
        )
        self.assertIn("actions: null,\n                        answers: {},", html)
        self.assertIn("resolved: false,\n                    });", html)
        self.assertEqual(
            html.count(
                "if (entry.actions && entry.actions.parentNode) entry.actions.remove();"
            ),
            2,
        )
        self.assertNotContains(response, 'aria-label="Session actions"')
        self.assertNotContains(response, 'class="composer"')
        self.assertNotContains(response, f'formaction="{stop_url}"')

    @patch("hitch.main.views.Codex")
    def test_okr_task_generation_log_requires_selected_project(
        self, mock_codex: MagicMock
    ) -> None:
        _patch_thread(self, mock_codex, _thread([], id="task-thread"))
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        other = Project.objects.create(name="Other", repo_path="/other")
        objective = Objective.objects.create(project=project, title="Improve planning")
        key_result = KeyResult.objects.create(objective=objective, title="Draft plan")
        _seed_cookies(self.client, hitch_selected_project_id=str(other.pk))
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.OKR_TASK_AGENT_KIND,
            main_thread_id=system_agents._okr_task_main_thread_id(key_result.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            state={"key_result_id": key_result.pk},
        )

        response = self.client.get(
            reverse("okr_task_generation_log", kwargs={"workflow_id": workflow.pk})
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.views.Codex")
    def test_system_session_detail_is_read_only_and_shows_system_prompt(
        self, mock_codex: MagicMock
    ) -> None:
        prompt = "You are Hitch's QA agent.\nReview <diff>."
        _patch_thread(
            self,
            mock_codex,
            _thread([], id="qa-thread", name="QA thread", cwd="/repo"),
        )
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="thread-1",
            cwd="/repo",
        )
        instance = _make_codex_instance(
            thread_id="qa-thread",
            status=CodexInstance.STATUS_COMPLETED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            display_author=system_agents.QA_DISPLAY_AUTHOR,
            prompt=prompt,
        )
        SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.PR_QA_AGENT_KIND,
            thread_id="qa-thread",
            instance=instance,
            status=SystemAgentRun.STATUS_COMPLETED,
        )

        response = self.client.get(
            reverse("system_session", kwargs={"session_id": "qa-thread"})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<body class="read-only">')
        self.assertContains(response, "QA agent log")
        self.assertContains(response, '<details class="system-prompt">', html=False)
        self.assertContains(response, "<summary>System prompt</summary>", html=False)
        self.assertContains(response, "Review &lt;diff&gt;.")
        self.assertNotContains(response, "No messages in this session yet.")
        self.assertNotContains(response, 'class="composer"')

    def test_system_session_detail_requires_system_run(self) -> None:
        response = self.client.get(
            reverse("system_session", kwargs={"session_id": "thread-1"})
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.views.Codex")
    def test_indicator_stream_url_lives_on_composer_form(
        self, mock_codex: MagicMock
    ) -> None:
        # The connection indicator's EventSource reads ``data-stream-url``
        # off the composer form so it works whether or not a live worker
        # is active. The URL is tagged with the page's render-time view
        # of the session state so the SSE endpoint can detect a stale
        # render and force a reload before any item events flow.
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        stream_path = reverse("session_stream", kwargs={"session_id": "thread-1"})
        # No workers exist for this session, so both query params are
        # empty — empty is the canonical encoding of "page knows of no
        # prior state".
        self.assertContains(response, "data-composer")
        self.assertContains(
            response,
            f'data-stream-url="{stream_path}?baseline=&amp;active=&amp;workflow="',
        )

    @patch("hitch.main.views.Codex")
    def test_composer_supports_super_enter_submit(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "event.metaKey")
        self.assertContains(response, 'event.key === "Enter"')
        self.assertContains(response, 'event.getModifierState("Meta")')
        self.assertContains(response, 'event.getModifierState("OS")')
        self.assertContains(response, "requestSubmit(composer, submit)")

    @patch("hitch.main.views.Codex")
    def test_composer_adjusts_for_mobile_keyboard(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, "--composer-keyboard-offset")
        self.assertContains(response, "window.visualViewport")
        self.assertContains(response, 'window.matchMedia("(max-width: 640px)")')
        self.assertContains(response, "viewport.scale")
        self.assertContains(response, "document.documentElement.style.setProperty")
        self.assertContains(response, "scheduleComposerKeyboardOffset")

    @patch("hitch.main.views.Codex")
    def test_composer_exposes_plan_slash_command(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'class="slash-trigger"')
        self.assertContains(response, 'name="plan_mode"')
        self.assertContains(response, "Plan mode")
        self.assertContains(response, "PR")
        self.assertContains(response, "QA")
        self.assertContains(response, "data-slash-pr")
        self.assertContains(response, "data-slash-qa")
        self.assertContains(
            response,
            "Do a thorough review of the diff. Rebase on master, clean it up, and then open a PR",
        )
        self.assertContains(
            response,
            "Run the QA agent on the current diff and fix anything it finds",
        )
        self.assertContains(response, "syncNextMessageConfig")
        self.assertContains(response, 'parsePlanCommand() !== null')
        self.assertContains(response, "parsePrCommand")
        self.assertContains(response, "parseQaCommand")
        self.assertContains(response, 'case "turn/plan/updated"')
        self.assertContains(response, 'case "item/plan/delta"')
        self.assertContains(response, "activateFinalPlanText")
        self.assertContains(response, "input-option-description")
        self.assertContains(response, 'other.placeholder = "Other"')
        self.assertContains(response, "delete entry.answers[key]")

    @patch("hitch.main.views.Codex")
    def test_live_task_plan_uses_responsive_panel(self, mock_codex: MagicMock) -> None:
        _patch_thread(self, mock_codex, _thread([]))

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'class="task-plan"')
        self.assertContains(response, 'data-task-plan')
        self.assertContains(response, 'data-task-plan-current')
        self.assertContains(response, 'data-fallback-order=')
        self.assertContains(response, "@media (min-width: 1120px)")
        self.assertContains(response, "main.has-task-plan")
        self.assertContains(
            response,
            "grid-template-columns: minmax(0, 820px) minmax(260px, 320px)",
        )
        self.assertContains(response, "function updateTaskPlan")
        self.assertContains(response, "function applyTaskPlanOrder")
        self.assertContains(response, "let streamFallbackOrder = 0")
        self.assertContains(response, "Number(taskPlan.dataset.fallbackOrder)")
        self.assertContains(response, "streamFallbackOrder += 1")
        self.assertContains(response, "order[2] < latestTaskPlanOrder[2]")
        self.assertContains(
            response,
            'case "turn/plan/updated": handlePlanUpdated(payload, order); break;',
        )
        self.assertContains(response, 'status === "in_progress"')
        self.assertContains(response, "taskPlan.dataset.expanded")
        self.assertNotContains(response, "ensurePlanView(turnId, null)")

    @patch("hitch.main.views.build_worktree_diff")
    @patch("hitch.main.views.Codex")
    def test_open_slash_menu_offsets_live_status_pill(
        self, mock_codex: MagicMock, mock_diff: MagicMock
    ) -> None:
        mock_diff.return_value = DiffView(files=[])
        _patch_thread(self, mock_codex, _thread([]))
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertContains(response, 'data-live-work data-state="working"')
        self.assertContains(response, 'data-slash-menu-open="true"')
        self.assertContains(response, "syncSlashMenuStatusOffset")
        self.assertContains(response, 'liveWork.dataset.slashMenuOpen = "true"')

    @patch("hitch.main.views.Codex")
    def test_stream_url_carries_baseline_and_active_when_worker_present(
        self, mock_codex: MagicMock
    ) -> None:
        # When a worker is running, the page emits its pk on both query
        # params so ``session_stream`` can confirm the SSE-time DB state
        # still matches what was rendered. A mismatch on either param
        # is the trigger for the reload path.
        _patch_thread(self, mock_codex, _thread([]))
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="warming up",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))
        stream_path = reverse("session_stream", kwargs={"session_id": "thread-1"})
        self.assertContains(
            response,
            f'data-stream-url="{stream_path}?baseline={instance.pk}&amp;active={instance.pk}&amp;workflow="',
        )

    @patch("hitch.main.views.Codex")
    def test_in_progress_turn_is_trimmed_when_worker_active(
        self, mock_codex: MagicMock
    ) -> None:
        # The rollout may already have the in-progress turn's user (and
        # possibly some early agent commentary) by the time the session
        # page loads. The view must hide that range from the rollout-
        # rendered entries so the SSE stream — which replays from byte 0
        # of the events file — doesn't render every item twice.
        prior_user = _user_message("earlier")
        prior_agent = _agent_message("earlier reply")
        in_progress_user = _user_message("run tests")
        partial_agent = _agent_message("working on it")
        thread = _thread(
            [
                _turn([prior_user, prior_agent], started_at=1700000000),
                _turn([in_progress_user, partial_agent], started_at=1700000100),
            ]
        )
        _patch_thread(self, mock_codex, thread)
        _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="run tests",
            pid=_LIVE_PID,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        # The earlier (completed) turn is still rendered server-side.
        self.assertIn("earlier", body)
        self.assertIn("earlier reply", body)
        # The in-progress turn's content is *not* server-rendered. The
        # stream owns that range; the pending bubble covers the user's
        # message until the userMessage event arrives.
        self.assertNotIn("working on it", body)
        self.assertContains(response, "data-pending-user>")
        self.assertContains(response, "run tests")
        self.assertContains(response, "data-live-root")

    @patch("hitch.main.views.Codex")
    def test_active_worker_picked_over_newer_terminal_row(
        self, mock_codex: MagicMock
    ) -> None:
        # ``send_message`` can stack workers: a newer row may flip to
        # FAILED quickly while an older row is still RUNNING. The session
        # must stay in streaming mode until *all* workers are terminal.
        _patch_thread(self, mock_codex, _thread([]))
        older = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            prompt="still running",
            pid=_LIVE_PID,
        )
        newer = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_FAILED,
            prompt="bailed fast",
            pid=_LIVE_PID,
        )
        # Force a strictly later started_at on the newer terminal row so
        # ``latest_for_thread`` would return it; the active-aware helper
        # must skip past it.
        CodexInstance.objects.filter(pk=newer.pk).update(
            started_at=older.started_at + timedelta(seconds=1)
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-live-root")
        # The pending bubble carries the still-running worker's prompt,
        # not the bailed one — proves we picked the active row.
        self.assertContains(response, "still running")
        self.assertNotContains(response, "bailed fast")

    @patch("hitch.main.views.codex_pool.is_alive", return_value=False)
    @patch("hitch.main.views.Codex")
    def test_dead_worker_is_reconciled_before_render(
        self, mock_codex: MagicMock, _mock_alive: MagicMock
    ) -> None:
        # A worker that died without writing a terminal status would leave
        # the page in "streaming" mode permanently. The session view
        # sweeps such rows before reading status so the live UI doesn't
        # appear.
        _patch_thread(self, mock_codex, _thread([]))
        instance = _make_codex_instance(
            thread_id="thread-1",
            status=CodexInstance.STATUS_RUNNING,
            pid=99999999,
        )

        response = self.client.get(reverse("session", kwargs={"session_id": "thread-1"}))

        instance.refresh_from_db()
        self.assertEqual(instance.status, CodexInstance.STATUS_FAILED)
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "data-live-root></div>")
