from __future__ import annotations

import json
from io import StringIO
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from hitch.main.goals.proposed_sessions import (
    ProposedSessionError,
    ProposedSessionInput,
    ProposedSessionUpdateInput,
    create_proposed_session,
    update_proposed_session,
)
from hitch.main.models import AutonomousGoal, ProposedSession, SessionMetadata
from hitch.main.runtime.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    registered_dynamic_tool_specs,
)
from hitch.main.test.support import _make_project


class ProposedSessionServiceTests(TestCase):
    def test_create_proposed_session_resolves_project_and_source_session(self) -> None:
        project = _make_project()
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
        _make_project()
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

    def test_update_proposed_session_edits_unresolved_project_proposal(self) -> None:
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Old title",
            summary="Old summary",
            prompt="Old prompt",
            confidence=AutonomousGoal.CONFIDENCE_MEDIUM,
            relevant_files=["old.py"],
        )

        updated = update_proposed_session(
            ProposedSessionUpdateInput(
                proposal_id=proposal.pk,
                title=" New title ",
                cwd="/repo",
                relevant_files=["new.py", "", "new.py"],
                confidence=AutonomousGoal.CONFIDENCE_HIGH,
            )
        )

        self.assertEqual(updated.title, "New title")
        self.assertEqual(updated.summary, "Old summary")
        self.assertEqual(updated.prompt, "Old prompt")
        self.assertEqual(updated.confidence, AutonomousGoal.CONFIDENCE_HIGH)
        self.assertEqual(updated.relevant_files, ["new.py"])
        self.assertEqual(ProposedSession.objects.count(), 1)

    def test_update_proposed_session_rejects_invalid_targets(self) -> None:
        project = _make_project()
        resolved = ProposedSession.objects.create(
            project=project,
            title="Resolved",
            summary="Summary",
            prompt="Prompt",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
        )
        notice = ProposedSession.objects.create(
            project=project,
            title="Notice",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        )
        other_project = _make_project(name="Other", repo_path="/other")
        other_project_proposal = ProposedSession.objects.create(
            project=other_project,
            title="Other",
            summary="Summary",
            prompt="Prompt",
        )
        cases = [
            (
                resolved.pk,
                ProposedSessionUpdateInput(
                    proposal_id=resolved.pk,
                    cwd="/repo",
                    title="Updated",
                ),
                "proposal has already been resolved",
            ),
            (
                notice.pk,
                ProposedSessionUpdateInput(
                    proposal_id=notice.pk,
                    cwd="/repo",
                    title="Updated",
                ),
                "proposal item is not editable",
            ),
            (
                other_project_proposal.pk,
                ProposedSessionUpdateInput(
                    proposal_id=other_project_proposal.pk,
                    cwd="/repo",
                    title="Updated",
                ),
                "proposal does not match current Hitch project",
            ),
            (
                0,
                ProposedSessionUpdateInput(
                    proposal_id=0,
                    cwd="/repo",
                    title="Updated",
                ),
                "proposal does not match current Hitch project",
            ),
        ]
        for proposal_id, values, message in cases:
            with (
                self.subTest(proposal_id=proposal_id),
                self.assertRaisesRegex(ProposedSessionError, message),
            ):
                update_proposed_session(values)

    def test_update_proposed_session_requires_an_edit(self) -> None:
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Title",
            summary="Summary",
            prompt="Prompt",
        )

        with self.assertRaisesRegex(
            ProposedSessionError, "at least one editable field is required"
        ):
            update_proposed_session(
                ProposedSessionUpdateInput(proposal_id=proposal.pk, cwd="/repo")
            )

    def test_update_proposed_session_rechecks_unresolved_status_on_write(self) -> None:
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Old title",
            summary="Old summary",
            prompt="Old prompt",
        )
        real_filter = ProposedSession.objects.filter
        filter_calls = 0

        def filter_with_resolution_race(*args: Any, **kwargs: Any) -> Any:
            nonlocal filter_calls
            filter_calls += 1
            if filter_calls == 2:
                real_filter(pk=proposal.pk).update(
                    outcome_status=ProposedSession.OUTCOME_ACCEPTED
                )
            return real_filter(*args, **kwargs)

        with (
            patch.object(
                ProposedSession.objects,
                "filter",
                side_effect=filter_with_resolution_race,
            ),
            self.assertRaisesRegex(
                ProposedSessionError, "proposal has already been resolved"
            ),
        ):
            update_proposed_session(
                ProposedSessionUpdateInput(
                    proposal_id=proposal.pk,
                    cwd="/repo",
                    title="New title",
                )
            )

        proposal.refresh_from_db()
        self.assertEqual(proposal.title, "Old title")
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)


class ProposeSessionCommandTests(TestCase):
    def test_command_creates_json_response(self) -> None:
        project = _make_project()
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

    def test_command_updates_json_response(self) -> None:
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Old title",
            summary="Old summary",
            prompt="Old prompt",
            relevant_files=["old.py"],
        )
        out = StringIO()

        call_command(
            "propose_session",
            "--proposal-id",
            str(proposal.pk),
            "--title",
            "New title",
            "--clear-relevant-files",
            "--cwd",
            "/repo",
            "--json",
            stdout=out,
        )

        payload = json.loads(out.getvalue())
        proposal.refresh_from_db()
        self.assertEqual(payload["action"], "updated")
        self.assertEqual(payload["id"], proposal.pk)
        self.assertEqual(payload["project_id"], project.pk)
        self.assertEqual(proposal.title, "New title")
        self.assertEqual(proposal.summary, "Old summary")
        self.assertEqual(proposal.relevant_files, [])


class CodexToolTests(TestCase):
    def test_registered_specs_include_propose_session(self) -> None:
        specs = registered_dynamic_tool_specs()

        self.assertEqual(specs[0]["namespace"], "hitch")
        self.assertEqual(specs[0]["name"], "propose_session")
        self.assertIn("inputSchema", specs[0])
        self.assertIn("proposal_id", specs[0]["inputSchema"]["properties"])

    def test_dynamic_tool_call_creates_proposal(self) -> None:
        project = _make_project()
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

    def test_dynamic_tool_call_updates_proposal(self) -> None:
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Old title",
            summary="Old summary",
            prompt="Old prompt",
            confidence=AutonomousGoal.CONFIDENCE_VERY_HIGH,
            relevant_files=["old.py"],
        )

        response = handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "propose_session",
                "arguments": {
                    "proposal_id": proposal.pk,
                    "summary": "New summary",
                    "confidence": None,
                    "relevant_files": [],
                },
            },
            ToolContext(cwd="/repo", thread_id="thread-1"),
        )

        self.assertTrue(response["success"])
        self.assertIn("Updated proposed session", response["contentItems"][0]["text"])
        proposal.refresh_from_db()
        self.assertEqual(proposal.title, "Old title")
        self.assertEqual(proposal.summary, "New summary")
        self.assertEqual(proposal.confidence, AutonomousGoal.CONFIDENCE_VERY_HIGH)
        self.assertEqual(proposal.relevant_files, [])
        self.assertEqual(ProposedSession.objects.count(), 1)

    def test_dynamic_tool_call_rejects_non_string_relevant_file_entries(self) -> None:
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Old title",
            summary="Old summary",
            prompt="Old prompt",
            relevant_files=["old.py"],
        )

        response = handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "propose_session",
                "arguments": {
                    "proposal_id": proposal.pk,
                    "summary": "New summary",
                    "relevant_files": [None],
                },
            },
            ToolContext(cwd="/repo", thread_id="thread-1"),
        )

        self.assertFalse(response["success"])
        self.assertIn(
            "relevant_files entries must be strings",
            response["contentItems"][0]["text"],
        )
        proposal.refresh_from_db()
        self.assertEqual(proposal.summary, "Old summary")
        self.assertEqual(proposal.relevant_files, ["old.py"])

    def test_dynamic_tool_call_reports_invalid_input(self) -> None:
        response = handle_dynamic_tool_call(
            {"namespace": "hitch", "tool": "missing", "arguments": {}},
            ToolContext(cwd="/repo", thread_id="thread-1"),
        )

        self.assertFalse(response["success"])
        self.assertIn("unknown Hitch tool", response["contentItems"][0]["text"])

    def test_dynamic_tool_call_accepts_namespace_less_payload(self) -> None:
        project = _make_project()

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

    @patch("hitch.main.runtime.codex_tools.connection.close")
    def test_dynamic_tool_call_closes_thread_connection(
        self, mock_close: MagicMock
    ) -> None:
        _make_project()

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
            (
                {
                    "namespace": "hitch",
                    "tool": "propose_session",
                    "arguments": {"proposal_id": "1", "title": "Title"},
                },
                "proposal_id must be an integer",
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
