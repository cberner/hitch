from __future__ import annotations

import json
from io import StringIO
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from hitch.main.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    registered_dynamic_tool_specs,
)
from hitch.main.goals.proposed_sessions import (
    ProposedSessionError,
    ProposedSessionInput,
    create_proposed_session,
)
from hitch.main.models import AutonomousGoal, Project, ProposedSession, SessionMetadata


class ProposedSessionServiceTests(TestCase):
    def test_create_proposed_session_resolves_project_and_source_session(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        source = SessionMetadata.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            project=project,
        )

        proposal = create_proposed_session(
            ProposedSessionInput(
                title="Add focused coverage",
                summary="This would cover the proposal service.",
                prompt="Add tests for proposed sessions.",
                cwd="/repo",
                relevant_files=["hitch/main/proposed_sessions.py", ""],
                confidence=AutonomousGoal.CONFIDENCE_HIGH,
                source_thread_id=source.thread_id,
            )
        )

        self.assertEqual(proposal.project, project)
        self.assertIsNone(proposal.autonomous_goal)
        self.assertEqual(proposal.source_session, source)
        self.assertEqual(proposal.relevant_files, ["hitch/main/proposed_sessions.py"])

    def test_create_proposed_session_rejects_unknown_project(self) -> None:
        with self.assertRaisesRegex(
            ProposedSessionError, "cwd does not match a Hitch project"
        ):
            create_proposed_session(
                ProposedSessionInput(
                    title="Add focused coverage",
                    summary="This would cover the proposal service.",
                    prompt="Add tests for proposed sessions.",
                    cwd="/repo",
                    relevant_files=[],
                )
            )

    def test_create_proposed_session_rejects_invalid_values(self) -> None:
        Project.objects.create(name="Hitch", repo_path="/repo")
        cases = [
            ("", "summary", "prompt", "/repo", "medium", "title is required"),
            ("x" * 201, "summary", "prompt", "/repo", "medium", "title is too long"),
            ("title", "", "prompt", "/repo", "medium", "summary is required"),
            ("title", "summary", "", "/repo", "medium", "prompt is required"),
            ("title", "summary", "prompt", "", "medium", "cwd is required"),
            ("title", "summary", "prompt", "/repo", "huge", "confidence is invalid"),
        ]
        for title, summary, prompt, cwd, confidence, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ProposedSessionError, message),
            ):
                create_proposed_session(
                    ProposedSessionInput(
                        title=title,
                        summary=summary,
                        prompt=prompt,
                        cwd=cwd,
                        relevant_files=[],
                        confidence=confidence,
                    )
                )


class ProposeSessionCommandTests(TestCase):
    def test_command_creates_json_response(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        out = StringIO()

        call_command(
            "propose_session",
            "--title",
            "Add focused coverage",
            "--summary",
            "This would cover the proposal service.",
            "--prompt",
            "Add tests for proposed sessions.",
            "--cwd",
            "/repo",
            "--relevant-file",
            "hitch/main/proposed_sessions.py",
            "--json",
            stdout=out,
        )

        payload = json.loads(out.getvalue())
        proposal = ProposedSession.objects.get(pk=payload["id"])
        self.assertEqual(payload["project_id"], project.pk)
        self.assertEqual(proposal.prompt, "Add tests for proposed sessions.")
        self.assertEqual(proposal.relevant_files, ["hitch/main/proposed_sessions.py"])

    def test_command_rejects_invalid_request(self) -> None:
        with self.assertRaisesRegex(CommandError, "summary is required"):
            call_command(
                "propose_session",
                "--title",
                "Add focused coverage",
                "--summary",
                "",
                "--prompt",
                "Add tests for proposed sessions.",
                "--cwd",
                "/repo",
            )


class CodexToolTests(TestCase):
    def test_registered_specs_include_propose_session(self) -> None:
        specs = registered_dynamic_tool_specs()

        self.assertEqual(specs[0]["namespace"], "hitch")
        self.assertEqual(specs[0]["name"], "propose_session")
        self.assertIn("inputSchema", specs[0])

    def test_dynamic_tool_call_creates_proposal(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")
        source = SessionMetadata.objects.create(
            thread_id="thread-1",
            cwd="/repo",
            project=project,
        )

        response = handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "propose_session",
                "arguments": {
                    "title": "Add focused coverage",
                    "summary": "This would cover the proposal service.",
                    "prompt": "Add tests for proposed sessions.",
                    "relevant_files": ["hitch/main/proposed_sessions.py"],
                    "confidence": AutonomousGoal.CONFIDENCE_VERY_HIGH,
                },
            },
            ToolContext(cwd="/repo", thread_id=source.thread_id),
        )

        self.assertTrue(response["success"])
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.project, project)
        self.assertEqual(proposal.source_session, source)
        self.assertEqual(proposal.confidence, AutonomousGoal.CONFIDENCE_VERY_HIGH)

    def test_dynamic_tool_call_reports_invalid_input(self) -> None:
        response = handle_dynamic_tool_call(
            {"namespace": "hitch", "tool": "missing", "arguments": {}},
            ToolContext(cwd="/repo", thread_id="thread-1"),
        )

        self.assertFalse(response["success"])
        self.assertIn("unknown Hitch tool", response["contentItems"][0]["text"])

    def test_dynamic_tool_call_accepts_namespace_less_payload(self) -> None:
        project = Project.objects.create(name="Hitch", repo_path="/repo")

        response = handle_dynamic_tool_call(
            {
                "tool": "propose_session",
                "arguments": {
                    "title": "Add parser tests",
                    "summary": "Parser coverage is thin.",
                    "prompt": "Implement parser tests.",
                },
            },
            ToolContext(cwd="/repo", thread_id="thread-1"),
        )

        self.assertTrue(response["success"])
        proposal = ProposedSession.objects.get()
        self.assertEqual(proposal.project, project)

    @patch("hitch.main.codex_tools.connection.close")
    def test_dynamic_tool_call_closes_thread_connection(
        self, mock_close: MagicMock
    ) -> None:
        Project.objects.create(name="Hitch", repo_path="/repo")

        response = handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "propose_session",
                "arguments": {
                    "title": "Add parser tests",
                    "summary": "Parser coverage is thin.",
                    "prompt": "Implement parser tests.",
                },
            },
            ToolContext(cwd="/repo", thread_id="thread-1"),
        )

        self.assertTrue(response["success"])
        mock_close.assert_called_once_with()

    def test_dynamic_tool_call_rejects_malformed_requests(self) -> None:
        cases = [
            (None, "tool call params are required"),
            ({"namespace": 1, "tool": "propose_session"}, "tool namespace"),
            (
                {"namespace": "hitch", "tool": "propose_session", "arguments": []},
                "tool arguments must be an object",
            ),
            (
                {
                    "namespace": "hitch",
                    "tool": "propose_session",
                    "arguments": {"title": []},
                },
                "title must be a string",
            ),
            (
                {
                    "namespace": "hitch",
                    "tool": "propose_session",
                    "arguments": {
                        "title": "Title",
                        "summary": "Summary",
                        "prompt": "Prompt",
                        "relevant_files": "file.py",
                    },
                },
                "relevant_files must be a list",
            ),
        ]
        for params, message in cases:
            with self.subTest(message=message):
                response = handle_dynamic_tool_call(
                    cast(dict[str, Any] | None, params),
                    ToolContext(cwd="/repo", thread_id="thread-1"),
                )

                self.assertFalse(response["success"])
                self.assertIn(message, response["contentItems"][0]["text"])
