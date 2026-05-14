run: pre
  uv run ./manage.py migrate --settings hitch.settings.dev
  uv run ./manage.py runserver --settings hitch.settings.dev

pre: sync
  uv run ruff check .
  uv run mypy .

test: pre
  uv run python -Wa ./manage.py test --settings hitch.settings.dev

coverage: pre
  uv run coverage run ./manage.py test --settings hitch.settings.dev
  uv run coverage report
  uv run coverage xml
  uv run coverage html

format:
  uv run ruff format .
  uv run ruff check --select I --fix

sync:
  uv sync --all-groups
