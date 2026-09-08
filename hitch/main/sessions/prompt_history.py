"""Bounded history of accepted browser prompts."""

import logging

from django.db import DatabaseError, transaction

from hitch.main.models import CodexInstance, Project, RecentPrompt, SessionMetadata
from hitch.main.sessions.session_index import _project_for_cwd

logger = logging.getLogger(__name__)


def remember_prompt(prompt: str, thread_id: str = "", *, cwd: str = "") -> None:
    if not prompt.strip():
        return
    # History is optional bookkeeping after the worker has accepted the input.
    try:
        with transaction.atomic():
            metadata = SessionMetadata.objects.filter(thread_id=thread_id).first()
            if metadata is not None and metadata.is_hidden_system_session:
                return
            project_id = metadata.project_id if metadata is not None else None
            if project_id is None and not (metadata and metadata.project_cleared):
                session_cwd = cwd or (metadata.cwd if metadata is not None else "")
                if not session_cwd and thread_id:
                    session_cwd = (
                        CodexInstance.objects.filter(thread_id=thread_id)
                        .order_by("-pk").values_list("cwd", flat=True).first() or ""
                    )
                project = _project_for_cwd(session_cwd, Project.objects.all())
                project_id = project.pk if project is not None else None
            RecentPrompt.objects.create(prompt=prompt, project_id=project_id)
            history = RecentPrompt.objects.filter(project=project_id)
            history.filter(pk__in=history.order_by("-pk").values("pk")[20:]).delete()
    except DatabaseError:
        logger.exception("failed to save recent prompt")
