# hitch
Human Interface for Taming Coding Helpers

## Architecture

Hitch is a server-rendered Django application that provides a local browser UI
for Codex. The `hitch.main` app owns the page views, form endpoints, SSE
streaming endpoints, templates, and data model. Django's ORM stores durable
application state in SQLite by default, including spawned Codex workers,
approval and user-input handoffs, user settings, archived usage metadata, and
Hitch-managed system-agent workflows. Codex conversation history still lives in
Codex rollout files; Hitch reads those files directly when it needs details the
SDK does not expose in the normal thread view.

Runtime work is split between the Django request process and detached worker
subprocesses. When a user starts or resumes a session, the view layer uses the
`openai-codex` SDK to create or resume a Codex thread, then `codex_pool` starts
a `codex_worker` Django management command for a single turn. The worker talks
to the Codex app-server, writes raw SDK events to per-worker JSONL files,
updates its `CodexInstance` row as it progresses, and uses database rows to
bridge browser approvals and structured input requests. The browser follows
active turns through `EventSource`/Server-Sent Events, while optional managed
git worktrees isolate agent edits under `~/.hitch/worktrees`.

## Major frameworks and dependencies

- Python 3.13 or newer
- Django 6 for the web app, ORM, templates, auth, and tests
- SQLite via Django's built-in database backend
- `openai-codex` for Codex app-server and thread APIs
- `markdown-it-py` and Pygments for rendering formatted model output
- `uv` for dependency management and environment setup
- `just` for local development, test, and formatting tasks
- Ruff, mypy, `django-stubs`, and `django-stubs-ext` for linting and type checks
- Coverage.py and Django's test runner for test coverage
- Playwright for browser-oriented QA checks
- Ollama as the optional local model provider used by `just run_qwen`

## Local development

The fastest path to a working dev server — and the only one that does not
require a ChatGPT/OpenAI account — is to point Codex at a local
`qwen2.5-coder:0.5b` running under `ollama`:

```sh
just run_qwen
```

Prerequisites:

- [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/) on PATH
- [`ollama`](https://ollama.com) on PATH (the recipe will start `ollama serve`
  in the background if nothing is listening on `:11434`)
- a `codex` binary on PATH; the version must match the `openai-codex` SDK
  pinned in `pyproject.toml` (see `.github/workflows/ci.yml` for the exact
  npm version CI installs)
- the [`gh`](https://cli.github.com) CLI on PATH and authenticated; the auto-PR
  workflow uses it to open the pull request once the work agent finishes

What the recipe does:

1. Starts `ollama serve` if it is not already running, then `ollama pull
   qwen2.5-coder:0.5b`.
2. Writes a Codex config to `${TMPDIR:-/tmp}/hitch-codex-home/config.toml`
   that sets `model = "qwen2.5-coder:0.5b"` and `model_provider =
   "ollama"`. Your real `~/.codex/config.toml` is untouched.
3. Runs `manage.py migrate` and `manage.py runserver` with `CODEX_HOME`
   pointed at that temp dir, so every Codex worker the app spawns uses
   the local model.

If you have a ChatGPT account and would rather use the model Codex picks
up from your own `~/.codex/config.toml`, run `just run` instead.
