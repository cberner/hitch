import os

from hitch.settings import common
from hitch.settings.common import *  # noqa: F403

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
# A deployment (e.g. the install-systemd unit) injects a strong, host-unique key
# via DJANGO_SECRET_KEY. Without it the published "insecure" default is used,
# which is fine for local dev but would let anyone forge Hitch's signed cookies
# (model/sandbox/approval settings) on a publicly reachable host.
SECRET_KEY = (
    os.environ.get("DJANGO_SECRET_KEY")
    or "django-insecure-_+&3jq$3=lc^c7t9p#sahy0v6=l8u@wfs+0nf!d2zo)kk1m_v0"
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1", ".demo.localhost"]
ALLOWED_HOSTS += common.additional_allowed_hosts()
INTERNAL_IPS = ["localhost", "127.0.0.1"]

DEBUG_TOOLBAR_CONFIG = {
    "SHOW_COLLAPSED": True,
    # Session pages carry large parsed transcripts. Rendering the template
    # context in Debug Toolbar can expand those objects into multi-GB strings.
    "SHOW_TEMPLATE_CONTEXT": False,
}

_DEBUG_TOOLBAR_DEFAULT = (
    "0" if os.environ.get("ADDITIONAL_ALLOWED_HOSTS", "").strip() else "1"
)
_DEBUG_TOOLBAR_ENABLED = (
    os.environ.get("HITCH_ENABLE_DEBUG_TOOLBAR", _DEBUG_TOOLBAR_DEFAULT).strip().lower()
    not in {"0", "false", "no", "off"}
)

if not common.TESTING and _DEBUG_TOOLBAR_ENABLED:
    INSTALLED_APPS = [
        *common.INSTALLED_APPS,
        "debug_toolbar",
    ]
    MIDDLEWARE = [
        *common.MIDDLEWARE[:2],
        "debug_toolbar.middleware.DebugToolbarMiddleware",
        *common.MIDDLEWARE[2:],
    ]

# Trust every allowed host for CSRF over either scheme: a developer fronting
# the dev server with an HTTPS tunnel or LAN proxy shouldn't have to repeat
# the host list to satisfy Django's Origin check on unsafe methods. Django's
# leading-dot subdomain pattern (".example.com") maps to CSRF's "*.example.com"
# form; the bare "*" allow-all has no CSRF equivalent and is dropped.
CSRF_TRUSTED_ORIGINS = common.csrf_trusted_origins(ALLOWED_HOSTS)
