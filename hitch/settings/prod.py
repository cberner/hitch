import os

from django.core.exceptions import ImproperlyConfigured

from hitch.settings.common import *  # noqa: F403

# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# Production settings fail closed instead of falling back to the public
# development key or accepting arbitrary hosts.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "").strip()
if not SECRET_KEY:
    raise ImproperlyConfigured("DJANGO_SECRET_KEY is required in production")
if (
    len(SECRET_KEY) < 50
    or len(set(SECRET_KEY)) < 5
    or SECRET_KEY.startswith("django-insecure-")
):
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY must be a strong production signing key"
    )

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("ADDITIONAL_ALLOWED_HOSTS", "").split(",")
    if host.strip()
]
if not ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "ADDITIONAL_ALLOWED_HOSTS must contain at least one production host"
    )
if "*" in ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "ADDITIONAL_ALLOWED_HOSTS cannot contain '*' in production"
    )

# The systemd installer is intended to sit behind HTTPS on its public domain.
# Trust only configured production hosts and only over HTTPS for unsafe requests.
CSRF_TRUSTED_ORIGINS = [
    f"https://{'*' + host if host.startswith('.') else host}"
    for host in ALLOWED_HOSTS
]
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True
