import os

from hitch.settings import common
from hitch.settings.common import *  # noqa: F403

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-_+&3jq$3=lc^c7t9p#sahy0v6=l8u@wfs+0nf!d2zo)kk1m_v0"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1", ".demo.localhost"]
ALLOWED_HOSTS += [h.strip() for h in os.environ.get("ADDITIONAL_ALLOWED_HOSTS", "").split(",") if h.strip()]
INTERNAL_IPS = ["localhost", "127.0.0.1"]

if not common.TESTING:
    INSTALLED_APPS = [
        *common.INSTALLED_APPS,
        "debug_toolbar",
    ]
    MIDDLEWARE = [
        *common.MIDDLEWARE[:2],
        "debug_toolbar.middleware.DebugToolbarMiddleware",
        *common.MIDDLEWARE[2:],
    ]
    DEBUG_TOOLBAR_CONFIG = {
        "SHOW_COLLAPSED": True,
    }

# Trust every allowed host for CSRF over either scheme: a developer fronting
# the dev server with an HTTPS tunnel or LAN proxy shouldn't have to repeat
# the host list to satisfy Django's Origin check on unsafe methods. Django's
# leading-dot subdomain pattern (".example.com") maps to CSRF's "*.example.com"
# form; the bare "*" allow-all has no CSRF equivalent and is dropped.
CSRF_TRUSTED_ORIGINS = [
    f"{scheme}://{'*' + host if host.startswith('.') else host}"
    for host in ALLOWED_HOSTS
    if host != "*"
    for scheme in ("http", "https")
]
