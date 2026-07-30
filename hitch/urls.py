"""URL configuration for hitch project."""

from django.conf import settings
from django.contrib import admin
from django.urls import path

from hitch.main.sessions import session_approval
from hitch.main.views import (
    account,
    goals,
    messages,
    new_session,
    session_actions,
    session_detail,
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
    path("inbox/", session_list.inbox, name="inbox"),
    path("autonomous-goals/", goals.autonomous_goals, name="autonomous_goals"),
    path(
        "autonomous-goals/create/",
        goals.create_autonomous_goal,
        name="create_autonomous_goal",
    ),
    path(
        "autonomous-goals/<int:autonomous_goal_id>/edit/",
        goals.edit_autonomous_goal,
        name="edit_autonomous_goal",
    ),
    path(
        "autonomous-goals/<int:autonomous_goal_id>/delete/",
        goals.delete_autonomous_goal,
        name="delete_autonomous_goal",
    ),
    path(
        "autonomous-goals/<int:autonomous_goal_id>/run/",
        goals.run_autonomous_goal,
        name="run_autonomous_goal",
    ),
    path(
        "autonomous-goals/run-all/",
        goals.run_autonomous_goals,
        name="run_autonomous_goals",
    ),
    path(
        "autonomous-goals/runs/<int:workflow_id>/log/",
        goals.autonomous_goal_run_log,
        name="autonomous_goal_run_log",
    ),
    path(
        "autonomous-goals/proposed-sessions/<int:proposed_session_id>/outcome/",
        goals.update_proposed_session_outcome,
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
        "sessions/<str:session_id>/demo/start/",
        session_actions.start_session_demo,
        name="start_session_demo",
    ),
    path(
        "sessions/<str:session_id>/demo/register/",
        session_actions.register_session_demo,
        name="session_demo_register",
    ),
    path(
        "sessions/<str:session_id>/demo/",
        session_actions.session_demo_proxy_root,
        name="session_demo_proxy_root",
    ),
    path(
        "sessions/<str:session_id>/demo/<path:path>",
        session_actions.session_demo_proxy,
        name="session_demo_proxy",
    ),
    path("sessions/<str:session_id>/", session_detail.session, name="session"),
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
