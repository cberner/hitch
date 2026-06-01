"""
Django settings for hitch project.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.2/ref/settings/
"""

import os
import sys
from pathlib import Path

# For mypy type checking
import django_stubs_ext

django_stubs_ext.monkeypatch()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

TESTING = "test" in sys.argv

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "hitch.main",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "hitch.main.demo_middleware.DemoProxyMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "hitch.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.csrf",
                "hitch.main.context_processors.server_revision",
            ],
        },
    },
]

WSGI_APPLICATION = "hitch.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {
            # Session pages refresh cached metadata while worker subprocesses
            # update state rows. WAL keeps readers from blocking writers, and
            # the busy timeout lets brief write contention settle instead of
            # surfacing intermittent "database is locked" errors.
            "init_command": "PRAGMA journal_mode=WAL",
            "timeout": 30,
        },
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# Per-worker JSONL event logs (see hitch.main.codex_pool). Workers detach from
# the Django process and write here; the dir must be writable by the user
# running Django and live somewhere that survives restarts. Kept outside the
# project tree so setuptools' flat-layout discovery doesn't pick it up.
CODEX_EVENTS_DIR = Path.home() / ".hitch" / "codex_events"

# Worker isolation policy: "auto" uses systemd scopes only when the user
# manager is reachable, "systemd" fails closed, and "direct" preserves the
# legacy process-group launch path for non-systemd environments. The defaults
# reserve memory for the Django server on a 16G host even when QA panel lanes run
# concurrently.
CODEX_WORKER_ISOLATION = os.environ.get("HITCH_CODEX_WORKER_ISOLATION", "auto")
CODEX_WORKER_SLICE = os.environ.get(
    "HITCH_CODEX_WORKER_SLICE", "hitch-codex-workers.slice"
)
CODEX_WORKER_SLICE_MEMORY_HIGH = os.environ.get(
    "HITCH_CODEX_WORKER_SLICE_MEMORY_HIGH", "8G"
)
CODEX_WORKER_SLICE_MEMORY_MAX = os.environ.get(
    "HITCH_CODEX_WORKER_SLICE_MEMORY_MAX", "10G"
)
CODEX_WORKER_MEMORY_HIGH = os.environ.get("HITCH_CODEX_WORKER_MEMORY_HIGH", "2G")
CODEX_WORKER_MEMORY_MAX = os.environ.get("HITCH_CODEX_WORKER_MEMORY_MAX", "4G")
# Swap caps make MemoryMax a true ceiling: cgroup v2 only counts RAM toward
# MemoryMax, so without a swap cap a runaway worker is pushed to swap rather
# than OOM-killed and thrashes the host indefinitely instead of failing the
# turn. "0" forbids swap so the cap fires; raise it to grant a swap cushion.
CODEX_WORKER_MEMORY_SWAP_MAX = os.environ.get(
    "HITCH_CODEX_WORKER_MEMORY_SWAP_MAX", "0"
)
CODEX_WORKER_SLICE_MEMORY_SWAP_MAX = os.environ.get(
    "HITCH_CODEX_WORKER_SLICE_MEMORY_SWAP_MAX", "0"
)
CODEX_WORKER_OOM_SCORE_ADJ = os.environ.get(
    "HITCH_CODEX_WORKER_OOM_SCORE_ADJ", "0" if TESTING else "1000"
)

# Managed git worktrees for new Codex sessions when the user opts into
# isolating agent changes from the source checkout selected in the UI.
HITCH_WORKTREES_DIR = Path.home() / ".hitch" / "worktrees"
