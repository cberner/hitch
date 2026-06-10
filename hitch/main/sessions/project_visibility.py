"""Project-visibility filtering and display-context helpers for session lists.

Pure helpers that filter session/proposed-session querysets by the viewer's
project visibility and build the visibility display context (labels, titles).
"""

from __future__ import annotations

from typing import Any

from django.db.models import Q, QuerySet
from django.urls import reverse

from hitch.main.models import Project, ProposedSession, SessionMetadata
from hitch.main.sessions.settings_cookies import (
    SessionProjectVisibility,
    SettingsValues,
    _visible_session_project_ids_cookie_fits,
)


def _settings_with_visible_selected_project(
    values: SettingsValues, project: Project | None, *, cookie_required: bool
) -> SettingsValues:
    if project is None or values.visible_session_project_ids is None:
        return values
    if project.pk in values.visible_session_project_ids:
        return values
    visible_project_ids = (*values.visible_session_project_ids, project.pk)
    if cookie_required and not _visible_session_project_ids_cookie_fits(
        visible_project_ids
    ):
        return values._replace(visible_session_project_ids=None)
    return values._replace(visible_session_project_ids=visible_project_ids)


def _session_project_is_visible(
    project: Project | None, visibility: SessionProjectVisibility
) -> bool:
    if project is None:
        return visibility.include_no_project
    return visibility.project_ids is None or project.pk in visibility.project_ids


def _filter_session_metadata_by_project_visibility(
    rows: QuerySet[SessionMetadata], visibility: SessionProjectVisibility
) -> QuerySet[SessionMetadata]:
    if visibility.project_ids is None:
        if visibility.include_no_project:
            return rows
        return rows.exclude(project__isnull=True)
    project_filter = Q(project_id__in=visibility.project_ids)
    if visibility.include_no_project:
        project_filter |= Q(project__isnull=True)
    return rows.filter(project_filter)


def _filter_proposed_sessions_by_project_visibility(
    rows: QuerySet[ProposedSession], visibility: SessionProjectVisibility
) -> QuerySet[ProposedSession]:
    if visibility.project_ids is None:
        if visibility.include_no_project:
            return rows
        return rows.exclude(project__isnull=True)
    project_filter = Q(project_id__in=visibility.project_ids)
    if visibility.include_no_project:
        project_filter |= Q(project__isnull=True)
    return rows.filter(project_filter)


def _session_project_visibility_context(
    visibility: SessionProjectVisibility, projects: list[Project]
) -> dict[str, Any]:
    return {
        "visible_session_projects_url": reverse("update_visible_session_projects"),
        "visible_session_projects": [
            {
                "id": project.pk,
                "name": project.name,
                "visible": (
                    visibility.project_ids is None or project.pk in visibility.project_ids
                ),
            }
            for project in projects
        ],
        "visible_session_no_project": visibility.include_no_project,
    }


def _session_list_title(
    visibility: SessionProjectVisibility, projects: list[Project]
) -> str:
    if visibility.project_ids is None:
        return "Codex sessions"
    if len(visibility.project_ids) == 1 and not visibility.include_no_project:
        project_id = next(iter(visibility.project_ids))
        project = next(
            (project for project in projects if project.pk == project_id), None
        )
        if project is not None:
            return f"{project.name} sessions"
    return "Codex sessions"


def _project_visibility_label(
    visibility: SessionProjectVisibility, projects: list[Project]
) -> str:
    if visibility.project_ids is None:
        return "All projects"
    if len(visibility.project_ids) == 1 and not visibility.include_no_project:
        project_id = next(iter(visibility.project_ids))
        project = next(
            (project for project in projects if project.pk == project_id), None
        )
        if project is not None:
            return project.name
    if visibility.project_ids:
        return "Visible projects"
    if visibility.include_no_project:
        return "No repo"
    return "No projects"


def _project_visibility_shows_project_names(
    visibility: SessionProjectVisibility,
) -> bool:
    if visibility.project_ids is None:
        return True
    return visibility.include_no_project or len(visibility.project_ids) != 1


def _metadata_by_thread_id(threads: list[Any]) -> dict[str, SessionMetadata]:
    thread_ids = [
        thread.id
        for thread in threads
        if isinstance(getattr(thread, "id", None), str) and thread.id
    ]
    if not thread_ids:
        return {}
    return SessionMetadata.objects.select_related("project").in_bulk(
        thread_ids, field_name="thread_id"
    )
