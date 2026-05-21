"""URL configuration for hitch project."""

from django.contrib import admin
from django.urls import path

import hitch.main.views

urlpatterns = [
    path("", hitch.main.views.index, name="index"),
    path("register/", hitch.main.views.register, name="register"),
    path("login/", hitch.main.views.login, name="login"),
    path("logout/", hitch.main.views.logout, name="logout"),
    path("usage/", hitch.main.views.usage, name="usage"),
    path("standing-orders/", hitch.main.views.standing_orders, name="standing_orders"),
    path(
        "standing-orders/create/",
        hitch.main.views.create_standing_order,
        name="create_standing_order",
    ),
    path(
        "standing-orders/run-all/",
        hitch.main.views.run_standing_orders,
        name="run_standing_orders",
    ),
    path(
        "standing-orders/runs/<int:workflow_id>/log/",
        hitch.main.views.standing_order_run_log,
        name="standing_order_run_log",
    ),
    path(
        "standing-orders/proposed-sessions/<int:proposed_session_id>/outcome/",
        hitch.main.views.update_proposed_session_outcome,
        name="update_proposed_session_outcome",
    ),
    path("okrs/", hitch.main.views.okrs, name="okrs"),
    path("okrs/objectives/", hitch.main.views.create_objective, name="create_objective"),
    path(
        "okrs/objectives/<int:objective_id>/key-results/",
        hitch.main.views.create_key_result,
        name="create_key_result",
    ),
    path(
        "okrs/key-results/<int:key_result_id>/generate-tasks/",
        hitch.main.views.generate_key_result_tasks,
        name="generate_key_result_tasks",
    ),
    path(
        "okrs/task-generation/<int:workflow_id>/log/",
        hitch.main.views.okr_task_generation_log,
        name="okr_task_generation_log",
    ),
    path(
        "okrs/proposed-tasks/<int:task_id>/outcome/",
        hitch.main.views.update_proposed_task_outcome,
        name="update_proposed_task_outcome",
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
