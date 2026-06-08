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
            # IMMEDIATE transactions avoid SQLite's lock-upgrade failure mode
            # when workflow code reads state and then writes inside atomic().
            # A single global write lock still serializes writers, so the
            # larger page cache and memory-mapped reads exist to keep each
            # write transaction short: fewer page faults under the lock means
            # writers release it sooner, so contended writers are far less
            # likely to exhaust busy_timeout and raise "database is locked".
            # Django splits init_command on ';' and runs each statement
            # separately, so every PRAGMA below is applied per connection.
            "init_command": (
                "PRAGMA journal_mode=WAL;"
                "PRAGMA synchronous=NORMAL;"
                "PRAGMA busy_timeout=60000;"
                "PRAGMA cache_size=-65536;"
                "PRAGMA mmap_size=268435456"
            ),
            "timeout": 60,
            "transaction_mode": "IMMEDIATE",
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


# Shared Hitch runtime state. By default this is ``~/.hitch``; keep the setting
# separate so tests and deployments can relocate the whole state tree.
HITCH_HOME_DIR = Path(os.environ.get("HITCH_HOME_DIR", Path.home() / ".hitch"))
HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT = os.environ.get(
    "HITCH_MAX_ALLOWED_DISK_SPACE_PERCENT", "20"
)

# Per-worker JSONL event logs (see hitch.main.codex_pool). Workers detach from
# the Django process and write here; the dir must be writable by the user
# running Django and live somewhere that survives restarts. Kept outside the
# project tree so setuptools' flat-layout discovery doesn't pick it up.
CODEX_EVENTS_DIR = HITCH_HOME_DIR / "codex_events"
# Optional override for detached worker stderr logs. When unset, Hitch stores
# them under HITCH_HOME_DIR / "worker_logs" for crash forensics.
_CODEX_WORKER_LOG_DIR = os.environ.get("HITCH_CODEX_WORKER_LOG_DIR")
CODEX_WORKER_LOG_DIR = (
    Path(_CODEX_WORKER_LOG_DIR) if _CODEX_WORKER_LOG_DIR else None
)

# Codex keeps its own SQLite databases (state_5.sqlite, logs_2.sqlite) under
# ``sqlite_home`` (default ``$CODEX_HOME``). Many app-servers sharing one home
# serialize on a single SQLite writer lock and race the one-time init/backfill,
# surfacing as "database is locked" (openai/codex#20213). We split the home by
# role: in-process request + scheduler app-servers share ``web`` (so
# ``use_state_db_only`` thread listing reads a populated index), while detached
# per-turn workers shard across a bounded pool of homes. The databases are a
# derived index over the rollout JSONL under CODEX_HOME, so relocating them here
# loses nothing -- a fresh home is rebuilt by Codex's backfill on first use.
CODEX_SQLITE_HOME_BASE = Path(
    os.environ.get("HITCH_CODEX_SQLITE_HOME_BASE", HITCH_HOME_DIR / "codex_sqlite")
)
# Number of distinct ``sqlite_home`` directories detached workers round-robin
# across. Defaults to 2x CPU cores: enough headroom that concurrent turns rarely
# share a home (and so rarely contend on its writer lock) while keeping the
# one-time backfill cost amortized across a small, reused set of homes.
_CODEX_WORKER_SQLITE_POOL_SIZE = os.environ.get("HITCH_CODEX_WORKER_SQLITE_POOL_SIZE")
CODEX_WORKER_SQLITE_POOL_SIZE = (
    int(_CODEX_WORKER_SQLITE_POOL_SIZE)
    if _CODEX_WORKER_SQLITE_POOL_SIZE
    else max(1, 2 * (os.cpu_count() or 1))
)
# Codex's log DB is diagnostic and self-pruned per thread, but a shared worker
# home accumulates rows across every worker that lands on it. After a worker
# finishes its turn, Hitch deletes that home's log DB if it has grown past this
# size so a hot home cannot grow without bound. The state DB is never touched.
CODEX_WORKER_LOGS_DB_MAX_BYTES = int(
    os.environ.get("HITCH_CODEX_WORKER_LOGS_DB_MAX_BYTES", str(100 * 1024 * 1024))
)

# Worker isolation policy: "auto" uses systemd units only when the user manager
# is reachable, "systemd" fails closed, and "direct" preserves the legacy
# process-group launch path for non-systemd environments. When Hitch itself is
# already running as a systemd unit, default to fail-closed systemd workers so a
# webserver restart cannot silently fall back to kill-prone direct children.
_CODEX_WORKER_ISOLATION_DEFAULT = "systemd" if os.environ.get("INVOCATION_ID") else "auto"
CODEX_WORKER_ISOLATION = os.environ.get(
    "HITCH_CODEX_WORKER_ISOLATION", _CODEX_WORKER_ISOLATION_DEFAULT
)
CODEX_WORKER_SLICE = os.environ.get(
    "HITCH_CODEX_WORKER_SLICE", "hitch-codex-workers.slice"
)
# Parent slice CPU bias. cgroup-v2 cpu.weight is per-level and relative to
# siblings, so to make the user-facing runserver win CPU contests against a
# busy worker pool the weight must sit on the deepest slice Hitch owns whose
# parent also hosts the runserver (hitch.slice, sibling to app.slice under
# user@.service) — NOT on the workers/codex leaf slices, which have no siblings
# at their level and would bias workers only against each other. Default 20 vs
# app.slice's 100 gives the runserver subtree ~5x share when contested while
# leaving uncontested workloads at full speed. Empty disables the parent-slice
# configuration (deployments not under systemd-user); an empty weight resets to
# the cgroup-v2 default of 100.
CODEX_PARENT_SLICE = os.environ.get("HITCH_CODEX_PARENT_SLICE", "hitch.slice")
CODEX_PARENT_SLICE_CPU_WEIGHT = os.environ.get(
    "HITCH_CODEX_PARENT_SLICE_CPU_WEIGHT", "20"
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
HITCH_WORKTREES_DIR = HITCH_HOME_DIR / "worktrees"
