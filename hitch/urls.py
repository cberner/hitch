"""URL configuration for hitch project."""

from django.contrib import admin
from django.urls import path

import hitch.main.views

urlpatterns = [
    path("", hitch.main.views.index, name="index"),
    path("admin/", admin.site.urls),
]
