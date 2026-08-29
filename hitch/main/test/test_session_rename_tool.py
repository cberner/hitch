"""Codex tool for renaming the invoking Hitch session."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from django.test import TestCase
from openai_codex.errors import InvalidRequestError

from hitch.main.models import SessionMetadata
from hitch.main.runtime.codex_tools import (
    ToolContext,
    handle_dynamic_tool_call,
    registered_dynamic_tool_specs,
)


class RenameSessionToolTests(TestCase):
    def test_registered_spec_targets_current_session(self) -> None:
        specs = registered_dynamic_tool_specs()

        rename = next(spec for spec in specs if spec["name"] == "rename_session")
        self.assertEqual(rename["namespace"], "hitch")
        self.assertEqual(rename["inputSchema"]["required"], ["name"])
        self.assertEqual(
            set(rename["inputSchema"]["properties"]),
            {"name"},
        )
        self.assertFalse(rename["inputSchema"]["additionalProperties"])

    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_renames_context_thread_and_updates_cached_name(
        self, mock_run: MagicMock
    ) -> None:
        codex = MagicMock()

        def run_operation(
            _factory: Any, operation: Any, **_kwargs: Any
        ) -> Any:
            return operation(codex)

        mock_run.side_effect = run_operation
        metadata = SessionMetadata.objects.create(
            thread_id="current-thread",
            cwd="/repo",
            codex_name="Old name",
            codex_display_title="Old name",
        )

        response = handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "rename_session",
                "arguments": {"name": "  New name  "},
            },
            ToolContext(
                cwd="/repo",
                thread_id="current-thread",
                enable_memories=True,
                web_search_mode="live",
            ),
        )

        self.assertTrue(response["success"])
        self.assertIn("New name", response["contentItems"][0]["text"])
        codex._client.thread_set_name.assert_called_once_with(
            "current-thread", "New name"
        )
        self.assertEqual(mock_run.call_args.kwargs["enable_memories"], True)
        self.assertEqual(mock_run.call_args.kwargs["web_search_mode"], "live")
        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_name, "New name")
        self.assertEqual(metadata.codex_display_title, "New name")

    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_rejects_invalid_names_without_renaming(
        self, mock_run: MagicMock
    ) -> None:
        cases: list[tuple[str, object, str]] = [
            ("empty", "", "name is required"),
            ("whitespace", "   ", "name is required"),
            ("too long", "x" * 201, "name is too long"),
            ("not a string", 1, "name must be a string"),
        ]

        for label, name, message in cases:
            with self.subTest(label=label):
                response = handle_dynamic_tool_call(
                    {
                        "namespace": "hitch",
                        "tool": "rename_session",
                        "arguments": {"name": name},
                    },
                    ToolContext(cwd="/repo", thread_id="current-thread"),
                )

                self.assertFalse(response["success"])
                self.assertIn(message, response["contentItems"][0]["text"])
        mock_run.assert_not_called()

    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_codex_rejection_does_not_update_cached_name(
        self, mock_run: MagicMock
    ) -> None:
        metadata = SessionMetadata.objects.create(
            thread_id="current-thread",
            cwd="/repo",
            codex_name="Old name",
            codex_display_title="Old name",
        )
        mock_run.side_effect = InvalidRequestError(
            code=-32600, message="thread is archived"
        )

        response = handle_dynamic_tool_call(
            {
                "namespace": "hitch",
                "tool": "rename_session",
                "arguments": {"name": "New name"},
            },
            ToolContext(cwd="/repo", thread_id="current-thread"),
        )

        self.assertFalse(response["success"])
        self.assertIn(
            "current session is archived or unknown",
            response["contentItems"][0]["text"],
        )
        metadata.refresh_from_db()
        self.assertEqual(metadata.codex_name, "Old name")
        self.assertEqual(metadata.codex_display_title, "Old name")
