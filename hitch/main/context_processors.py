import subprocess
from functools import cache

from django.conf import settings
from django.http import HttpRequest


def server_version(_request: HttpRequest) -> dict[str, str]:
    return {"server_git_hash": _server_git_hash()}


@cache
def _server_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(settings.BASE_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()[:6]
