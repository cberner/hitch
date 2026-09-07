"""Bounded history of accepted browser prompts."""

import logging

from django.db import DatabaseError, transaction

from hitch.main.models import RecentPrompt

logger = logging.getLogger(__name__)


def remember_prompt(prompt: str) -> None:
    if not prompt.strip():
        return
    # History is optional bookkeeping after the worker has accepted the input.
    try:
        with transaction.atomic():
            RecentPrompt.objects.create(prompt=prompt)
            RecentPrompt.objects.filter(
                pk__in=RecentPrompt.objects.order_by("-pk").values("pk")[20:]
            ).delete()
    except DatabaseError:
        logger.exception("failed to save recent prompt")
