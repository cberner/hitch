from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.urls import reverse

from hitch.main.proposed_sessions import (
    ProposedSessionError,
    ProposedSessionInput,
    create_proposed_session,
)


class Command(BaseCommand):
    help = "Create a proposed session inbox item for the Hitch project containing cwd."

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--title", required=True)
        parser.add_argument("--summary", required=True)
        parser.add_argument("--prompt", required=True)
        parser.add_argument("--cwd", default="")
        parser.add_argument("--relevant-file", action="append", default=[])
        parser.add_argument("--confidence", default="")
        parser.add_argument("--source-thread-id", default="")
        parser.add_argument("--json", action="store_true", dest="json_output")

    @override
    def handle(self, *args: Any, **options: Any) -> str | None:
        cwd = options["cwd"] or str(Path.cwd())
        try:
            proposal = create_proposed_session(
                ProposedSessionInput(
                    title=options["title"],
                    summary=options["summary"],
                    prompt=options["prompt"],
                    cwd=cwd,
                    relevant_files=list(options["relevant_file"]),
                    confidence=options["confidence"],
                    source_thread_id=options["source_thread_id"],
                )
            )
        except ProposedSessionError as exc:
            raise CommandError(str(exc)) from exc
        payload = {
            "id": proposal.pk,
            "title": proposal.title,
            "project_id": proposal.project_id,
            "inbox_url": reverse("standing_orders"),
        }
        if options["json_output"]:
            return json.dumps(payload, sort_keys=True)
        return (
            f"Created proposed session #{proposal.pk}: {proposal.title}\n"
            f"Inbox: {payload['inbox_url']}"
        )
