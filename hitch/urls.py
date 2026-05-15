"""URL configuration for hitch project."""

from django.contrib import admin
from django.urls import path

import hitch.main.views

urlpatterns = [
    path("", hitch.main.views.index, name="index"),
    path("sessions/new/", hitch.main.views.new_session, name="new_session"),
    path("sessions/<str:session_id>/", hitch.main.views.session, name="session"),
    path(
        "sessions/<str:session_id>/name/",
        hitch.main.views.set_session_name,
        name="set_session_name",
    ),
    path(
        "sessions/<str:session_id>/message/",
        hitch.main.views.send_message,
        name="send_message",
    ),
    path("admin/", admin.site.urls),
]
