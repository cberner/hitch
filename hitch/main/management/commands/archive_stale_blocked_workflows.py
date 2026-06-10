from __future__ import annotations

from argparse import ArgumentParser
from datetime import timedelta
from typing import Any, override

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from hitch.main.workflows import system_agents


class Command(BaseCommand):
    help = (
        "Archive blocked system workflows older than a cutoff so stale failures "
        "stop surfacing as a Blocked stage in the session inbox."
    )

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--days",
            type=int,
            default=7,
            help="Only archive blocked workflows last updated this many days ago "
            "or earlier (default: 7).",
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Persist the changes. Without this flag the command only reports "
            "what it would archive.",
        )

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        days = options["days"]
        if days < 0:
            raise CommandError("--days must not be negative")
        apply = options["apply"]
        cutoff = timezone.now() - timedelta(days=days)
        archived_ids = system_agents.archive_stale_blocked_workflows(
            older_than=cutoff, apply=apply
        )
        verb = "Archived" if apply else "Would archive"
        self.stdout.write(
            f"{verb} {len(archived_ids)} blocked workflow(s) last updated before "
            f"{cutoff.isoformat()}."
        )
        if archived_ids:
            self.stdout.write("Workflow ids: " + ", ".join(map(str, archived_ids)))
        if not apply and archived_ids:
            self.stdout.write("Re-run with --apply to perform the archive.")
