from unittest.mock import patch

from django.test import SimpleTestCase
from django.urls import reverse


class SessionAssetTests(SimpleTestCase):
    def test_assets_revalidate_current_contents_including_shared_snippets(self) -> None:
        for name, content_type, included in (
            ("session.css", "text/css", "--text-muted"),
            ("session.js", "text/javascript", "applySupportedEffortFilter"),
        ):
            with self.subTest(asset=name):
                url = reverse("session_asset", args=[name])
                first = self.client.get(url)
                self.assertContains(first, included)
                self.assertNotContains(first, "<script>")
                self.assertNotContains(first, "{%")
                self.assertNotContains(first, "{{")
                self.assertTrue(first["Content-Type"].startswith(content_type))
                self.assertEqual(first["Cache-Control"], "no-cache")
                unchanged = self.client.get(url, HTTP_IF_NONE_MATCH="W/" + first["ETag"])
                self.assertEqual(unchanged.status_code, 304)
                self.assertEqual(unchanged.content, b"")
                self.assertEqual(unchanged["Cache-Control"], "no-cache")
                self.assertEqual(unchanged["ETag"], first["ETag"])
                with patch("hitch.main.views.assets.render_to_string", return_value="/* Updated shared snippet */"):
                    changed = self.client.get(url, HTTP_IF_NONE_MATCH=first["ETag"])
                self.assertContains(changed, "Updated shared snippet")
                self.assertNotEqual(changed["ETag"], first["ETag"])
                self.assertEqual(self.client.head(url).status_code, 200)
                self.assertEqual(self.client.post(url).status_code, 405)
        self.assertEqual(self.client.get(reverse("session_asset", args=["unknown.js"])).status_code, 404)
