import os

from hitch.settings.common import *  # noqa: F403

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-_+&3jq$3=lc^c7t9p#sahy0v6=l8u@wfs+0nf!d2zo)kk1m_v0"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
ALLOWED_HOSTS += [h.strip() for h in os.environ.get("ADDITIONAL_ALLOWED_HOSTS", "").split(",") if h.strip()]
INTERNAL_IPS = ["localhost", "127.0.0.1"]
