"""WSGI entry point that serves application static files in production."""

from django.contrib.staticfiles.handlers import StaticFilesHandler

from hitch.wsgi import application as django_application

application = StaticFilesHandler(django_application)
