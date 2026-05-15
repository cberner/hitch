from django.test import SimpleTestCase

from hitch.main.formatting import looks_like_markdown, render_markdown


class LooksLikeMarkdownTests(SimpleTestCase):
    """High-confidence markdown signals; plain prose must stay unflagged."""

    def test_empty_text(self) -> None:
        self.assertFalse(looks_like_markdown(""))

    def test_plain_sentence(self) -> None:
        self.assertFalse(looks_like_markdown("Sure, here is the plan."))

    def test_plain_multiline(self) -> None:
        self.assertFalse(looks_like_markdown("Done.\nLet me know if you need anything else."))

    def test_stray_asterisk(self) -> None:
        self.assertFalse(looks_like_markdown("It uses *args style positional args."))

    def test_stray_inline_backticks(self) -> None:
        # Inline `code` alone is too common in prose to be a confident signal.
        self.assertFalse(looks_like_markdown("Use `print` to send output."))

    def test_single_bullet(self) -> None:
        # One isolated dash is not a list -- could just be a sentence.
        self.assertFalse(looks_like_markdown("- only one thing"))

    def test_single_numbered_item(self) -> None:
        self.assertFalse(looks_like_markdown("1. only one"))

    def test_hash_without_space(self) -> None:
        # "#123" is an issue reference, not a heading.
        self.assertFalse(looks_like_markdown("Filed as #123."))

    def test_fenced_code_block(self) -> None:
        self.assertTrue(looks_like_markdown("Here:\n```python\nprint('hi')\n```"))

    def test_atx_heading(self) -> None:
        self.assertTrue(looks_like_markdown("# Overview\n\nSome text."))

    def test_atx_subheading(self) -> None:
        self.assertTrue(looks_like_markdown("Intro.\n\n### Details\n\nMore."))

    def test_two_bullet_items(self) -> None:
        self.assertTrue(looks_like_markdown("- first\n- second"))

    def test_two_numbered_items(self) -> None:
        self.assertTrue(looks_like_markdown("1. first\n2. second"))

    def test_asterisk_bullet_list(self) -> None:
        self.assertTrue(looks_like_markdown("* a\n* b"))

    def test_markdown_link_https(self) -> None:
        self.assertTrue(looks_like_markdown("See [docs](https://example.com/d) for more."))

    def test_markdown_link_http(self) -> None:
        self.assertTrue(looks_like_markdown("See [docs](http://example.com)."))

    def test_array_indexing_is_not_a_link(self) -> None:
        # `Array[0](foo)` looks superficially link-shaped but the URL part is
        # neither http(s) nor mailto, so it shouldn't trip detection.
        self.assertFalse(looks_like_markdown("Call Array[0](foo) to fetch."))

    def test_table_with_separator(self) -> None:
        self.assertTrue(looks_like_markdown("| a | b |\n|---|---|\n| 1 | 2 |"))

    def test_single_pipe_line_is_not_a_table(self) -> None:
        # A lone `| x |` row without the separator below it can be ASCII art
        # or stray prose; we want the table delimiter row before we commit.
        self.assertFalse(looks_like_markdown("| just text |"))


class RenderMarkdownTests(SimpleTestCase):
    """Rendering produces safe HTML; agent-supplied script tags stay inert."""

    def test_heading_renders_as_h1(self) -> None:
        html = render_markdown("# Title")
        self.assertIn("<h1>Title</h1>", html)

    def test_bullet_list_renders_as_ul(self) -> None:
        html = render_markdown("- one\n- two")
        self.assertIn("<ul>", html)
        self.assertIn("<li>one</li>", html)
        self.assertIn("<li>two</li>", html)

    def test_fenced_code_preserves_content(self) -> None:
        html = render_markdown("```\nprint('hi')\n```")
        self.assertIn("<pre><code>", html)
        self.assertIn("print(", html)

    def test_script_tag_is_escaped(self) -> None:
        html = render_markdown("<script>alert(1)</script>")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_javascript_url_does_not_become_link(self) -> None:
        html = render_markdown("[click](javascript:alert(1))")
        # No <a href="javascript:..."> should be emitted; the raw text
        # survives but cannot be clicked into running JS.
        self.assertNotIn('href="javascript:', html)

    def test_https_link_is_rendered(self) -> None:
        html = render_markdown("[ok](https://example.com)")
        self.assertIn('<a href="https://example.com">ok</a>', html)

    def test_image_syntax_does_not_emit_img_tag(self) -> None:
        # Auto-fetched <img> tags would leak the viewer's IP/referrer to any
        # URL an agent (or prompt-injected content) chose; markdown image
        # syntax must degrade to a plain clickable link, never an <img>.
        html = render_markdown("![pixel](https://attacker.example/x.png)")
        self.assertNotIn("<img", html)
        self.assertNotIn("src=", html)

    def test_table_is_rendered(self) -> None:
        html = render_markdown("| a | b |\n|---|---|\n| 1 | 2 |")
        self.assertIn("<table>", html)
        self.assertIn("<th>a</th>", html)
        self.assertIn("<td>1</td>", html)
