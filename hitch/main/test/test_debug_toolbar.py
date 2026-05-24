from django.conf import settings
from django.test import SimpleTestCase


class DebugToolbarSettingsTests(SimpleTestCase):
    def test_debug_toolbar_is_disabled_during_tests(self) -> None:
        self.assertNotIn("debug_toolbar", settings.INSTALLED_APPS)
        self.assertNotIn("debug_toolbar.middleware.DebugToolbarMiddleware", settings.MIDDLEWARE)
