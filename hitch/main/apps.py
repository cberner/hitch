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

        # The keepalive self-gates to real server processes (where the shared
        # app-server pool is used), independent of whether the maintenance
        # scheduler is enabled -- a server that runs maintenance elsewhere still
        # needs its request-path pool kept warm.
        start_workflow_maintenance_scheduler()
        start_auto_proposal_scheduler()
        start_codex_pool_keepalive()
