import json
import os
import subprocess
import sys
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

_REPO_ROOT = Path(__file__).resolve().parents[3]


class DebugToolbarSettingsTests(SimpleTestCase):
    def test_debug_toolbar_is_disabled_during_tests(self) -> None:
        self.assertNotIn("debug_toolbar", settings.INSTALLED_APPS)
        self.assertNotIn(
            "debug_toolbar.middleware.DebugToolbarMiddleware",
            settings.MIDDLEWARE,
        )

    def test_debug_toolbar_does_not_render_template_context(self) -> None:
        self.assertFalse(settings.DEBUG_TOOLBAR_CONFIG["SHOW_TEMPLATE_CONTEXT"])

    def test_debug_toolbar_env_gate_in_non_test_settings_imports(self) -> None:
        cases: tuple[tuple[dict[str, str], bool], ...] = (
            ({}, True),
            ({"ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com"}, False),
            ({"HITCH_ENABLE_DEBUG_TOOLBAR": "0"}, False),
            (
                {
                    "ADDITIONAL_ALLOWED_HOSTS": "hitch.example.com",
                    "HITCH_ENABLE_DEBUG_TOOLBAR": "1",
                },
                True,
            ),
        )
        for env_overrides, expected_enabled in cases:
            with self.subTest(env_overrides=env_overrides):
                snapshot = _dev_settings_snapshot(env_overrides)
                self.assertEqual(snapshot["toolbar_app_enabled"], expected_enabled)
                self.assertEqual(snapshot["toolbar_middleware_enabled"], expected_enabled)
                self.assertFalse(snapshot["show_template_context"])


def _dev_settings_snapshot(env_overrides: dict[str, str]) -> dict[str, bool]:
    env = os.environ.copy()
    env.pop("ADDITIONAL_ALLOWED_HOSTS", None)
    env.pop("HITCH_ENABLE_DEBUG_TOOLBAR", None)
    env.update(env_overrides)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "import hitch.settings.dev as dev; "
                "print(json.dumps({"
                "'toolbar_app_enabled': 'debug_toolbar' in dev.INSTALLED_APPS, "
                "'toolbar_middleware_enabled': "
                "'debug_toolbar.middleware.DebugToolbarMiddleware' in dev.MIDDLEWARE, "
                "'show_template_context': "
                "dev.DEBUG_TOOLBAR_CONFIG['SHOW_TEMPLATE_CONTEXT']"
                "}))"
            ),
        ],
        check=True,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return {
        "toolbar_app_enabled": bool(payload["toolbar_app_enabled"]),
        "toolbar_middleware_enabled": bool(payload["toolbar_middleware_enabled"]),
        "show_template_context": bool(payload["show_template_context"]),
    }
