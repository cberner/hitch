from django.test import TestCase
from django.urls import reverse


class IndexViewTests(TestCase):
    def test_index_renders_hitch(self) -> None:
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HITCH")
