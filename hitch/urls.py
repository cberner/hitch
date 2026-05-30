"""URL configuration for hitch project."""

from django.conf import settings
from django.contrib import admin
from django.urls import path

import hitch.main.views

urlpatterns = [
    path("", hitch.main.views.index, name="index"),
    path("register/", hitch.main.views.register, name="register"),
    path("login/", hitch.main.views.login, name="login"),
    path("profile/", hitch.main.views.profile, name="profile"),
    path("logout/", hitch.main.views.logout, name="logout"),
    path("usage/", hitch.main.views.usage, name="usage"),
    path("inbox/", hitch.main.views.inbox, name="inbox"),
    path("autonomous-goals/", hitch.main.views.autonomous_goals, name="autonomous_goals"),
    path(
        "autonomous-goals/create/",
        hitch.main.views.create_autonomous_goal,
        name="create_autonomous_goal",
    ),
    path(
        "autonomous-goals/<int:autonomous_goal_id>/edit/",
        hitch.main.views.edit_autonomous_goal,
        name="edit_autonomous_goal",
    ),
    path(
        "autonomous-goals/<int:autonomous_goal_id>/run/",
        hitch.main.views.run_autonomous_goal,
        name="run_autonomous_goal",
    ),
    path(
        "autonomous-goals/run-all/",
        hitch.main.views.run_autonomous_goals,
        name="run_autonomous_goals",
    ),
    path(
        "autonomous-goals/runs/<int:workflow_id>/log/",
        hitch.main.views.autonomous_goal_run_log,
        name="autonomous_goal_run_log",
    ),
    path(
        "autonomous-goals/proposed-sessions/<int:proposed_session_id>/outcome/",
        hitch.main.views.update_proposed_session_outcome,
        name="update_proposed_session_outcome",
    ),
    path("settings/", hitch.main.views.update_settings, name="update_settings"),
    path(
        "settings/archived-sessions/",
        hitch.main.views.update_archived_session_visibility,
        name="update_archived_session_visibility",
    ),
    path("projects/new/", hitch.main.views.new_project, name="new_project"),
    path("projects/edit/", hitch.main.views.edit_project, name="edit_project"),
    path("system-sessions/", hitch.main.views.system_sessions, name="system_sessions"),
    path(
        "system-sessions/<str:session_id>/",
        hitch.main.views.system_session,
        name="system_session",
    ),
    path("sessions/new/", hitch.main.views.new_session, name="new_session"),
    path(
        "sessions/<str:session_id>/demo/start/",
        hitch.main.views.start_session_demo,
        name="start_session_demo",
    ),
    path(
        "sessions/<str:session_id>/demo/register/",
        hitch.main.views.register_session_demo,
        name="session_demo_register",
    ),
    path(
        "sessions/<str:session_id>/demo/",
        hitch.main.views.session_demo_proxy_root,
        name="session_demo_proxy_root",
    ),
    path(
        "sessions/<str:session_id>/demo/<path:path>",
        hitch.main.views.session_demo_proxy,
        name="session_demo_proxy",
    ),
    path("sessions/<str:session_id>/", hitch.main.views.session, name="session"),
    path(
        "sessions/<str:session_id>/intermediate/<int:entry_index>/",
        hitch.main.views.session_intermediate,
        name="session_intermediate",
    ),
    path(
        "sessions/<str:session_id>/name/",
        hitch.main.views.set_session_name,
        name="set_session_name",
    ),
    path(
        "sessions/<str:session_id>/archive/",
        hitch.main.views.set_session_archived,
        name="set_session_archived",
    ),
    path(
        "sessions/<str:session_id>/project/",
        hitch.main.views.set_session_project,
        name="set_session_project",
    ),
    path(
        "sessions/<str:session_id>/open-pr/",
        hitch.main.views.open_session_pr,
        name="open_session_pr",
    ),
    path(
        "sessions/<str:session_id>/message/",
        hitch.main.views.send_message,
        name="send_message",
    ),
    path(
        "sessions/<str:session_id>/stream/",
        hitch.main.views.session_stream,
        name="session_stream",
    ),
    path(
        "approval/<int:approval_id>/",
        hitch.main.views.resolve_approval,
        name="resolve_approval",
    ),
    path(
        "input/<int:input_id>/",
        hitch.main.views.resolve_input_request,
        name="resolve_input_request",
    ),
    path(
        "sessions/<str:session_id>/stop/",
        hitch.main.views.stop_session,
        name="stop_session",
    ),
    path("admin/", admin.site.urls),
]

if "debug_toolbar" in settings.INSTALLED_APPS:
    from debug_toolbar.toolbar import debug_toolbar_urls

    urlpatterns += debug_toolbar_urls()
