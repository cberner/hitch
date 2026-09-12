"""URL configuration for hitch project."""

from django.conf import settings
from django.contrib import admin
from django.urls import path

from hitch.main.sessions import session_approval
from hitch.main.views import (
    account,
    messages,
    new_session,
    proposals,
    session_actions,
    session_detail,
    session_goals,
    session_list,
)
from hitch.main.views import settings as settings_views

urlpatterns = [
    path("", session_list.index, name="index"),
    path("register/", account.register, name="register"),
    path("login/", account.login, name="login"),
    path("profile/", account.profile, name="profile"),
    path("logout/", account.logout, name="logout"),
    path("nuke-codex/", account.nuke_codex, name="nuke_codex"),
    path("health/", account.health_dashboard, name="health_dashboard"),
    path("usage/", session_list.usage, name="usage"),
    path("usage/refresh/", session_list.usage_refresh, name="usage_refresh"),
    path("inbox/", session_list.inbox, name="inbox"),
    path(
        "inbox/proposed-sessions/<int:proposed_session_id>/outcome/",
        proposals.update_proposed_session_outcome,
        name="update_proposed_session_outcome",
    ),
    path("settings/", settings_views.update_settings, name="update_settings"),
    path(
        "settings/archived-sessions/",
        settings_views.update_archived_session_visibility,
        name="update_archived_session_visibility",
    ),
    path(
        "settings/visible-session-projects/",
        settings_views.update_visible_session_projects,
        name="update_visible_session_projects",
    ),
    path("projects/new/", settings_views.new_project, name="new_project"),
    path("projects/edit/", settings_views.edit_project, name="edit_project"),
    path("system-sessions/", session_list.system_sessions, name="system_sessions"),
    path(
        "system-sessions/<str:session_id>/",
        session_list.system_session,
        name="system_session",
    ),
    path("sessions/new/", new_session.new_session, name="new_session"),
    path(
        "sessions/<str:session_id>/history/",
        session_detail.session_history,
        name="session_history",
    ),
    path("sessions/<str:session_id>/", session_detail.session, name="session"),
    path("sessions/<str:session_id>/goal/", session_goals.update_goal, name="set_session_goal"),
    path("sessions/<str:session_id>/agents/", session_detail.session_agents, name="session_agents"),
    path(
        "sessions/<str:session_id>/intermediate/<int:entry_index>/",
        session_detail.session_intermediate,
        name="session_intermediate",
    ),
    path(
        "sessions/<str:session_id>/name/",
        session_actions.set_session_name,
        name="set_session_name",
    ),
    path(
        "sessions/<str:session_id>/archive/",
        session_actions.set_session_archived,
        name="set_session_archived",
    ),
    path(
        "sessions/<str:session_id>/project/",
        session_actions.set_session_project,
        name="set_session_project",
    ),
    path(
        "sessions/<str:session_id>/approval-mode/",
        session_actions.set_session_approval_mode,
        name="set_session_approval_mode",
    ),
    path(
        "sessions/<str:session_id>/model/",
        session_actions.set_session_model,
        name="set_session_model",
    ),
    path(
        "sessions/<str:session_id>/message/",
        messages.send_message,
        name="send_message",
    ),
    path(
        "sessions/<str:session_id>/stream/",
        session_detail.session_stream,
        name="session_stream",
    ),
    path(
        "approval/<int:approval_id>/",
        session_approval.resolve_approval,
        name="resolve_approval",
    ),
    path(
        "input/<int:input_id>/",
        session_approval.resolve_input_request,
        name="resolve_input_request",
    ),
    path(
        "sessions/<str:session_id>/stop/",
        session_approval.stop_session,
        name="stop_session",
    ),
    path("admin/", admin.site.urls),
]

if "debug_toolbar" in settings.INSTALLED_APPS:
    from debug_toolbar.toolbar import debug_toolbar_urls

    urlpatterns += debug_toolbar_urls()
