from typing import override

from django.apps import AppConfig


class MainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "hitch.main"

    @override
    def ready(self) -> None:
        from hitch.main.auto_proposals import start_auto_proposal_scheduler
        from hitch.main.codex_pool import start_codex_pool_keepalive
        from hitch.main.workflow_maintenance import (
            start_workflow_maintenance_scheduler,
        )

        # The maintenance scheduler's gate already encodes "this is a real
        # server process" (gunicorn/uvicorn/... or the runserver child), which is
        # exactly where the shared app-server pool is used. Piggyback on it so the
        # keepalive never starts under management commands, migrations, or tests.
        started_server_scheduler = start_workflow_maintenance_scheduler()
        start_auto_proposal_scheduler()
        if started_server_scheduler:
            start_codex_pool_keepalive()
