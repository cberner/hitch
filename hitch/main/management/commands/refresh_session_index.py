from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, override

from django.core.management.base import BaseCommand
from openai_codex import Codex

from hitch.main import codex_pool, session_index
from hitch.main.models import Project


class Command(BaseCommand):
    help = "Refresh Hitch's cached Codex session index."

    @override
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--full-scan",
            action="store_true",
            help="Allow Codex to scan rollouts to repair its state DB before caching.",
        )
        parser.add_argument(
            "--active-only",
            action="store_true",
            help="Only refresh active sessions.",
        )

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        projects = list(Project.objects.all())
        config = codex_pool.app_server_config(enable_memories=False)
        with Codex(config=config) as codex:
            result = session_index.refresh_from_codex(
                codex,
                projects=projects,
                include_archived=not options["active_only"],
                use_state_db_only=not options["full_scan"],
                max_pages=None,
            )
        status = "with errors" if result.failed else "successfully"
        self.stdout.write(f"Refreshed {result.synced} sessions {status}.")
