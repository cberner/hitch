from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.urls import reverse

from hitch.main.goals.proposed_sessions import (
    ProposedSessionError,
    ProposedSessionInput,
    ProposedSessionUpdateInput,
    create_proposed_session,
    update_proposed_session,
)


class Command(BaseCommand):
    help = "Create or edit a proposed session inbox item for the Hitch project containing cwd."

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--proposal-id", type=int, default=None)
        parser.add_argument("--title", default=None)
        parser.add_argument("--summary", default=None)
        parser.add_argument("--prompt", default=None)
        parser.add_argument("--cwd", default="")
        parser.add_argument("--relevant-file", action="append", default=None)
        parser.add_argument("--clear-relevant-files", action="store_true")
        parser.add_argument("--confidence", default=None)
        parser.add_argument("--source-thread-id", default="")
        parser.add_argument("--json", action="store_true", dest="json_output")

    @override
    def handle(self, *args: Any, **options: Any) -> str | None:
        cwd = options["cwd"] or str(Path.cwd())
        action = "created"
        try:
            if options["proposal_id"] is None:
                proposal = create_proposed_session(
                    ProposedSessionInput(
                        title=options["title"] or "",
                        summary=options["summary"] or "",
                        prompt=options["prompt"] or "",
                        cwd=cwd,
                        relevant_files=list(options["relevant_file"] or []),
                        confidence=options["confidence"] or "",
                        source_thread_id=options["source_thread_id"],
                    )
                )
            else:
                action = "updated"
                proposal = update_proposed_session(
                    ProposedSessionUpdateInput(
                        proposal_id=options["proposal_id"],
                        title=options["title"],
                        summary=options["summary"],
                        prompt=options["prompt"],
                        cwd=cwd,
                        relevant_files=_update_relevant_files(options),
                        confidence=options["confidence"],
                    )
                )
        except ProposedSessionError as exc:
            raise CommandError(str(exc)) from exc
        payload = {
            "action": action,
            "id": proposal.pk,
            "title": proposal.title,
            "project_id": proposal.project_id,
            "inbox_url": reverse("inbox"),
        }
        if options["json_output"]:
            return json.dumps(payload, sort_keys=True)
        if action == "updated":
            return (
                f"Updated proposed session #{proposal.pk}: {proposal.title}\n"
                f"Inbox: {payload['inbox_url']}"
            )
        return (
            f"Created proposed session #{proposal.pk}: {proposal.title}\n"
            f"Inbox: {payload['inbox_url']}"
        )


def _update_relevant_files(options: dict[str, Any]) -> list[str] | None:
    relevant_files = options["relevant_file"]
    if options["clear_relevant_files"] and relevant_files:
        raise ProposedSessionError(
            "--clear-relevant-files cannot be combined with --relevant-file"
        )
    if options["clear_relevant_files"]:
        return []
    if relevant_files is None:
        return None
    return list(relevant_files)
