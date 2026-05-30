import subprocess
from functools import cache

from django.conf import settings
from django.http import HttpRequest


@cache
def server_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(settings.BASE_DIR), "rev-parse", "--short=6", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            # Bound the call like every other git invocation in the codebase: a
            # hung filesystem or an unexpected git prompt would otherwise block
            # the first request that renders any template forever, since this
            # runs inside a context processor.
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def server_revision(_request: HttpRequest) -> dict[str, str]:
    return {"server_git_hash": server_git_hash()}
