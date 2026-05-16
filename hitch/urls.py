"""URL configuration for hitch project."""

from django.contrib import admin
from django.urls import path

import hitch.main.views

urlpatterns = [
    path("", hitch.main.views.index, name="index"),
    path("register/", hitch.main.views.register, name="register"),
    path("login/", hitch.main.views.login, name="login"),
    path("logout/", hitch.main.views.logout, name="logout"),
    path("settings/", hitch.main.views.update_settings, name="update_settings"),
    path("sessions/new/", hitch.main.views.new_session, name="new_session"),
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
        "sessions/<str:session_id>/stop/",
        hitch.main.views.stop_session,
        name="stop_session",
    ),
    path("admin/", admin.site.urls),
]
