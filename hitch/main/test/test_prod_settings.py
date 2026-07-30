import json
import os
import subprocess
import sys
from pathlib import Path

from django.test import SimpleTestCase

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STRONG_SECRET = "test-production-secret-0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ProductionSettingsTests(SimpleTestCase):
    def test_requires_secret_key(self) -> None:
        result = _run_prod_settings_import(
            {"ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com"}
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DJANGO_SECRET_KEY is required", result.stderr)

    def test_requires_allowed_hosts(self) -> None:
        result = _run_prod_settings_import({"DJANGO_SECRET_KEY": _STRONG_SECRET})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ADDITIONAL_ALLOWED_HOSTS must contain", result.stderr)

    def test_rejects_weak_secret_key(self) -> None:
        result = _run_prod_settings_import(
            {
                "DJANGO_SECRET_KEY": "test-secret",
                "ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com",
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be a strong production signing key", result.stderr)

    def test_rejects_wildcard_allowed_host(self) -> None:
        result = _run_prod_settings_import(
            {
                "DJANGO_SECRET_KEY": _STRONG_SECRET,
                "ADDITIONAL_ALLOWED_HOSTS": "*",
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot contain '*'", result.stderr)

    def test_uses_production_key_hosts_and_https_csrf_origins(self) -> None:
        result = _run_prod_settings_import(
            {
                "DJANGO_SECRET_KEY": _STRONG_SECRET,
                "ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com, .internal.example.com",
            },
            snapshot=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "secret_key": _STRONG_SECRET,
                "debug": False,
                "allowed_hosts": ["hitch.example.com", ".internal.example.com"],
                "csrf_trusted_origins": [
                    "https://hitch.example.com",
                    "https://*.internal.example.com",
                ],
                "csrf_cookie_secure": True,
                "session_cookie_secure": True,
            },
        )


def _run_prod_settings_import(
    env_overrides: dict[str, str], *, snapshot: bool = False
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("DJANGO_SECRET_KEY", None)
    env.pop("ADDITIONAL_ALLOWED_HOSTS", None)
    env.update(env_overrides)
    expression = "import hitch.settings.prod"
    if snapshot:
        expression = (
            "import json; import hitch.settings.prod as prod; "
            "print(json.dumps({"
            "'secret_key': prod.SECRET_KEY, "
            "'debug': prod.DEBUG, "
            "'allowed_hosts': prod.ALLOWED_HOSTS, "
            "'csrf_trusted_origins': prod.CSRF_TRUSTED_ORIGINS, "
            "'csrf_cookie_secure': prod.CSRF_COOKIE_SECURE, "
            "'session_cookie_secure': prod.SESSION_COOKIE_SECURE"
            "}))"
        )
    return subprocess.run(
        [sys.executable, "-c", expression],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
