from typing import override

from django.apps import AppConfig


class MainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "hitch.main"

    @override
    def ready(self) -> None:
        from hitch.main.runtime.app_server_pool import start_codex_pool_keepalive
        from hitch.main.runtime.maintenance import (
            start_maintenance_scheduler,
        )

        # The keepalive self-gates to real server processes (where the shared
        # app-server pool is used), independent of whether the maintenance
        # scheduler is enabled -- a server that runs maintenance elsewhere still
        # needs its request-path pool kept warm.
        start_maintenance_scheduler()
        start_codex_pool_keepalive()
