from __future__ import annotations

from typing import Any, override

from django.core.management.base import BaseCommand, CommandError, CommandParser

from hitch.main.models import Project
from hitch.main.runtime import codex_pool
from hitch.main.workflows import system_agents


class Command(BaseCommand):
    help = "Start eligible auto-proposal autonomous goal workflows."

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--project-id", type=int, default=None)

    @override
    def handle(self, *args: Any, **options: Any) -> str:
        project: Project | None = None
        project_id = options["project_id"]
        if project_id is not None:
            project = Project.objects.filter(pk=project_id).first()
            if project is None:
                raise CommandError("project not found")
        codex_pool.reconcile_dead()
        started = system_agents.maybe_start_auto_proposal_workflows(project=project)
        return f"Started {started} auto-proposal workflow(s)."
