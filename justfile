run: pre
  uv run ./manage.py migrate --settings hitch.settings.dev
  uv run ./manage.py runserver --settings hitch.settings.dev

# Local-dev mirror of the CI ollama+qwen setup: starts `ollama serve` in the
# background if nothing is already listening on 11434, pulls the model, and
# points Codex at it via a CODEX_HOME under $TMPDIR so the user's
# ~/.codex/config.toml is left alone.
run_qwen: pre
  if ! curl -fsS http://localhost:11434/api/tags > /dev/null 2>&1; then \
    nohup ollama serve > /tmp/hitch-ollama.log 2>&1 & \
    for _ in $(seq 1 30); do \
      curl -fsS http://localhost:11434/api/tags > /dev/null 2>&1 && break; \
      sleep 1; \
    done; \
    curl -fsS http://localhost:11434/api/tags > /dev/null 2>&1 || \
      { echo "ollama did not become ready within 30s; see /tmp/hitch-ollama.log" >&2; exit 1; }; \
  fi
  ollama pull qwen2.5-coder:0.5b
  mkdir -p "${TMPDIR:-/tmp}/hitch-codex-home"
  printf '%s\n' \
    'model = "qwen2.5-coder:0.5b"' \
    'model_provider = "ollama"' > "${TMPDIR:-/tmp}/hitch-codex-home/config.toml"
  CODEX_HOME="${TMPDIR:-/tmp}/hitch-codex-home" uv run ./manage.py migrate --settings hitch.settings.dev
  CODEX_HOME="${TMPDIR:-/tmp}/hitch-codex-home" uv run ./manage.py runserver --settings hitch.settings.dev

pre: sync
  uv run ruff check .
  uv run mypy .

test-pre:
  scripts/test-in-podman pre

# Test sessions may inherit the production service's systemd worker isolation.
# Run the suite in Podman with an isolated HOME/HITCH_HOME_DIR/CODEX_HOME so QA
# cannot touch the deployment database, Codex state, or worker units.
test: test-pre
  scripts/test-in-podman test

test-integration: test-pre
  scripts/test-in-podman test-integration

coverage: test-pre
  scripts/test-in-podman coverage

format:
  uv run ruff format .
  uv run ruff check --select I --fix

qa-browser-setup: sync
  uv run playwright install --with-deps chromium

qa-browser-check: qa-browser-setup
  uv run python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); browser = p.chromium.launch(headless=True); browser.close(); p.stop()"

sync:
  uv sync --all-groups

# Install a per-user systemd unit that serves Hitch from this repo. Prompts for
# the public domain name, wires it into ADDITIONAL_ALLOWED_HOSTS, and re-pulls
# the install-time branch + applies migrations on every (re)start so a crash
# loop self-heals.
install-systemd:
  #!/usr/bin/env bash
  set -euo pipefail
  read -r -p "Domain name (e.g. hitch.example.com): " DOMAIN
  if [ -z "${DOMAIN}" ]; then
    echo "A domain name is required." >&2
    exit 1
  fi
  REPO_DIR="$(pwd)"
  UV_BIN="$(command -v uv)"
  GIT_BIN="$(command -v git)"
  if [ -z "${UV_BIN}" ] || [ -z "${GIT_BIN}" ]; then
    echo "uv and git must both be on PATH." >&2
    exit 1
  fi
  BRANCH="$("${GIT_BIN}" -C "${REPO_DIR}" symbolic-ref --quiet --short HEAD || true)"
  if [ -z "${BRANCH}" ]; then
    echo "HEAD is detached; check out the branch to deploy from before installing." >&2
    exit 1
  fi
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "${UNIT_DIR}"
  UNIT_PATH="${UNIT_DIR}/hitch.service"
  # Generate a strong, host-unique signing key once and keep it across
  # reinstalls so existing signed cookies/sessions stay valid. Without this the
  # public host would fall back to the published "insecure" dev key, letting
  # anyone forge Hitch's signed setting cookies (sandbox/approval modes).
  ENV_PATH="${UNIT_DIR}/hitch.env"
  if ! grep -q '^DJANGO_SECRET_KEY=' "${ENV_PATH}" 2>/dev/null; then
    SECRET="$("${UV_BIN}" run --no-project python -c 'import secrets; print(secrets.token_urlsafe(64))')"
    umask 077
    printf 'DJANGO_SECRET_KEY=%s\n' "${SECRET}" > "${ENV_PATH}"
    chmod 600 "${ENV_PATH}"
  fi
  cat > "${UNIT_PATH}" <<EOF
  [Unit]
  Description=Hitch server (${REPO_DIR})
  After=network-online.target
  Wants=network-online.target

  [Service]
  Type=simple
  WorkingDirectory=${REPO_DIR}
  EnvironmentFile=${ENV_PATH}
  Environment=ADDITIONAL_ALLOWED_HOSTS=${DOMAIN}
  Environment=HITCH_ENABLE_DEBUG_TOOLBAR=0
  Environment=HITCH_CODEX_WORKER_ISOLATION=systemd
  # Re-sync to the install-time branch and apply migrations on every (re)start
  # so a crash loop picks up fixes pushed since the last successful boot.
  ExecStartPre="${GIT_BIN}" -C "${REPO_DIR}" pull --ff-only origin ${BRANCH}
  ExecStartPre="${UV_BIN}" run ./manage.py migrate --settings hitch.settings.dev
  ExecStart="${UV_BIN}" run ./manage.py runserver --noreload --settings hitch.settings.dev
  Restart=always
  RestartSec=2
  RestartSteps=5
  RestartMaxDelaySec=30

  [Install]
  WantedBy=default.target
  EOF
  systemctl --user daemon-reload
  systemctl --user enable hitch.service
  # Without lingering the user manager exits at logout and never restarts at
  # boot, so the deployed service would die with the install session.
  if ! loginctl show-user "$(id -un)" --property=Linger --value 2>/dev/null | grep -qx yes; then
    if ! loginctl enable-linger "$(id -un)" 2>/dev/null; then
      echo "Could not enable lingering automatically. Run:" >&2
      echo "  sudo loginctl enable-linger $(id -un)" >&2
      echo "so the service survives logout and starts at boot." >&2
    fi
  fi
  echo "Installed ${UNIT_PATH}."
  echo "Start with: systemctl --user start hitch.service"
