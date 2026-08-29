from django.conf import settings
from django.test import SimpleTestCase


class DebugToolbarSettingsTests(SimpleTestCase):
    def test_debug_toolbar_does_not_render_template_context(self) -> None:
        self.assertFalse(settings.DEBUG_TOOLBAR_CONFIG["SHOW_TEMPLATE_CONTEXT"])
