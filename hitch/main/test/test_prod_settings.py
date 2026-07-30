import os
import runpy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

_PROD_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings" / "prod.py"
_STRONG_SECRET = "test-production-secret-0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ProductionSettingsTests(SimpleTestCase):
    def test_requires_secret_key(self) -> None:
        with self.assertRaisesRegex(
            ImproperlyConfigured, "DJANGO_SECRET_KEY is required"
        ):
            _load_prod_settings({"ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com"})

    def test_requires_allowed_hosts(self) -> None:
        with self.assertRaisesRegex(
            ImproperlyConfigured, "ADDITIONAL_ALLOWED_HOSTS must contain"
        ):
            _load_prod_settings({"DJANGO_SECRET_KEY": _STRONG_SECRET})

    def test_rejects_weak_secret_key(self) -> None:
        with self.assertRaisesRegex(
            ImproperlyConfigured, "must be a strong production signing key"
        ):
            _load_prod_settings(
                {
                    "DJANGO_SECRET_KEY": "test-secret",
                    "ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com",
                }
            )

    def test_rejects_wildcard_allowed_host(self) -> None:
        with self.assertRaisesRegex(ImproperlyConfigured, "cannot contain '\\*'"):
            _load_prod_settings(
                {
                    "DJANGO_SECRET_KEY": _STRONG_SECRET,
                    "ADDITIONAL_ALLOWED_HOSTS": "*",
                }
            )

    def test_uses_production_key_hosts_and_https_csrf_origins(self) -> None:
        prod = _load_prod_settings(
            {
                "DJANGO_SECRET_KEY": _STRONG_SECRET,
                "ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com, .internal.example.com",
            }
        )

        self.assertEqual(prod["SECRET_KEY"], _STRONG_SECRET)
        self.assertIs(prod["DEBUG"], False)
        self.assertEqual(
            prod["ALLOWED_HOSTS"], ["hitch.example.com", ".internal.example.com"]
        )
        self.assertEqual(
            prod["CSRF_TRUSTED_ORIGINS"],
            ["https://hitch.example.com", "https://*.internal.example.com"],
        )
        self.assertIs(prod["CSRF_COOKIE_SECURE"], True)
        self.assertIs(prod["SESSION_COOKIE_SECURE"], True)


def _load_prod_settings(env_overrides: dict[str, str]) -> dict[str, Any]:
    with patch.dict(os.environ):
        os.environ.pop("DJANGO_SECRET_KEY", None)
        os.environ.pop("ADDITIONAL_ALLOWED_HOSTS", None)
        os.environ.update(env_overrides)
        return runpy.run_path(str(_PROD_SETTINGS_PATH))
