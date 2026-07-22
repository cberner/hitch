"""Shared helpers for the hitch.main test suite.

Single home for the fixtures that several test modules previously each
carried a private copy of: signed settings-cookie seeding, the Codex SDK
context-manager mock, model stubs, and rollout-line builders.
"""

import base64
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from django.core import signing
from django.test import Client
from openai_codex.errors import MethodNotFoundError

from hitch.main.models import Project


def _sign(name: str, value: str) -> str:
    return signing.get_cookie_signer(salt=name).sign(value)


def _seed_cookies(client: Client, **values: str) -> None:
    for name, value in values.items():
        client.cookies[name] = _sign(name, value)


def _cookie_value(response: object, name: str) -> str:
    """Pull a signed cookie's plaintext value out of a TestClient response."""
    raw = response.cookies[name].value  # type: ignore[attr-defined]
    return signing.get_cookie_signer(salt=name).unsign(raw)


def _encode_extra_system_prompt(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode("ascii")


def _decode_extra_system_prompt(value: str) -> str:
    return base64.urlsafe_b64decode(value.encode("ascii")).decode()


def _setup_codex(
    mock_codex: MagicMock,
    *,
    threads: list[Any] | None = None,
    archived_threads: list[Any] | None = None,
    models: list[Any] | None = None,
) -> MagicMock:
    """Configure the Codex mock with ``thread_list`` and ``models``.

    The index view reads both active and, when enabled, archived thread
    lists. Also stubs ``_client.request`` to raise
    MethodNotFound so the rate-limits fetch falls through its
    unsupported-endpoint branch — tests that care set their own value."""
    ctx: MagicMock = mock_codex.return_value.__enter__.return_value

    def thread_list(*, archived: bool | None = None, **_: Any) -> SimpleNamespace:
        data = archived_threads if archived else threads
        return SimpleNamespace(data=data or [])

    ctx.thread_list.side_effect = thread_list
    ctx.models.return_value.data = models or []
    ctx._client.request.side_effect = MethodNotFoundError(
        -32601, "method not found", None
    )
    return ctx


def _make_model(model_id: str, *, is_default: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        id=model_id,
        display_name=model_id,
        is_default=is_default,
        default_reasoning_effort=SimpleNamespace(value="medium"),
        supported_reasoning_efforts=[
            SimpleNamespace(reasoning_effort=SimpleNamespace(value=v), description=v)
            for v in ("low", "medium", "high")
        ],
    )


def _make_project(name: str = "Hitch", repo_path: str = "/repo", **kwargs: Any) -> Project:
    """Create a Project with the defaults the test suite overwhelmingly uses."""
    kwargs.setdefault("auto_pull_enabled", False)
    return Project.objects.create(name=name, repo_path=repo_path, **kwargs)


def _rollout_line(
    line_type: str,
    payload: dict[str, object],
    *,
    timestamp: str = "2025-01-05T12:00:00Z",
) -> str:
    return json.dumps({"timestamp": timestamp, "type": line_type, "payload": payload})


def _git(repo: Path, *args: str) -> str:
    """Run git in a test repo, isolated from the host's git configuration.

    Pins author/committer identity and masks the user/system gitconfig so
    host-specific settings (e.g. commit signing) cannot leak into tests.
    """
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Hitch Tests",
        "GIT_AUTHOR_EMAIL": "hitch@example.com",
        "GIT_COMMITTER_NAME": "Hitch Tests",
        "GIT_COMMITTER_EMAIL": "hitch@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(
    repo: Path, *, initial_branch: str = "master", configure_user: bool = False
) -> None:
    """Create a git repo with one README commit.

    Always disables commit signing in the repo config so production code
    paths that commit inside the repo are also immune to host settings.
    ``configure_user`` additionally persists the test identity for those
    same production-side commits.
    """
    subprocess.run(
        ["git", "init", f"--initial-branch={initial_branch}", str(repo)],
        check=True,
        capture_output=True,
    )
    _git(repo, "config", "commit.gpgsign", "false")
    if configure_user:
        _git(repo, "config", "user.name", "Hitch Tests")
        _git(repo, "config", "user.email", "hitch@example.com")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
