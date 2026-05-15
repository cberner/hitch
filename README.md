# hitch
Human Interface for Taming Coding Helpers

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
