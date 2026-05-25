from typing import override

from django.apps import AppConfig


class MainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "hitch.main"

    @override
    def ready(self) -> None:
        from hitch.main.auto_proposals import start_auto_proposal_scheduler

        start_auto_proposal_scheduler()
