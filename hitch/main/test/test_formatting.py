from pathlib import Path

from django.test import SimpleTestCase

from hitch.main.formatting import looks_like_markdown, render_markdown


class LooksLikeMarkdownTests(SimpleTestCase):
    """High-confidence markdown signals; plain prose must stay unflagged."""

    def test_detection_cases(self) -> None:
        cases: list[tuple[str, str, bool]] = [
            ("empty text", "", False),
            ("plain sentence", "Sure, here is the plan.", False),
            ("plain multiline", "Done.\nLet me know if you need anything else.", False),
            ("stray asterisk", "It uses *args style positional args.", False),
            ("stray inline backticks", "Use `print` to send output.", False),
            ("single bullet", "- only one thing", False),
            ("single numbered item", "1. only one", False),
            ("single paren-numbered item", "1) only one", False),
            ("hash without space", "Filed as #123.", False),
            ("fenced code block", "Here:\n```python\nprint('hi')\n```", True),
            ("ATX heading", "# Overview\n\nSome text.", True),
            ("ATX subheading", "Intro.\n\n### Details\n\nMore.", True),
            ("two bullet items", "- first\n- second", True),
            ("two numbered items", "1. first\n2. second", True),
            ("two paren-numbered items", "1) first\n2) second", True),
            (
                "paren-numbered items under intro line",
                "Steps:\n1) Set up the repo\n2) Run the tests",
                True,
            ),
            # An ordered list can only interrupt a paragraph when it starts at 1,
            # so a run beginning above 1 glued onto a prose line stays inside the
            # paragraph -- the renderer emits no <ol>, so the detector must not
            # flag it (true for both "." and ")" markers).
            (
                "numbered run starting above 1 after prose is not a list",
                "Steps:\n2. do the second thing\n3. do the third thing",
                False,
            ),
            (
                "paren run starting above 1 after prose is not a list",
                "Steps:\n2) do the second thing\n3) do the third thing",
                False,
            ),
            (
                "numbered run starting above 1 after a blank line is a list",
                "Steps:\n\n2) do the second thing\n3) do the third thing",
                True,
            ),
            (
                "numbered run starting above 1 at the top is a list",
                "2) do the second thing\n3) do the third thing",
                True,
            ),
            (
                "multi-line list items still count as a list",
                "1. first item\n   continued on the next line\n2. second item",
                True,
            ),
            ("asterisk bullet list", "* a\n* b", True),
            (
                "indented bullets under intro line",
                "Here are the items:\n  - first\n  - second",
                True,
            ),
            (
                "indented numbered items under intro line",
                "Sequence:\n  1. first\n  2. second",
                True,
            ),
            (
                "four-space indented bullets are a code block, not a list",
                "Code:\n    - this is\n    - rendered as code",
                False,
            ),
            ("tab after bullet marker", "-\tfirst\n-\tsecond", True),
            ("two spaces after bullet marker", "-  first\n-  second", True),
            ("markdown HTTPS link", "See [docs](https://example.com/d) for more.", True),
            ("markdown HTTP link", "See [docs](http://example.com).", True),
            ("array indexing is not a link", "Call Array[0](foo) to fetch.", False),
            ("table with separator", "| a | b |\n|---|---|\n| 1 | 2 |", True),
            (
                "table with single-dash delimiter cells",
                "| a | b |\n| - | - |\n| 1 | 2 |",
                True,
            ),
            # Table detection defers to the renderer, so it tracks markdown-it
            # exactly: a pipeless delimiter behind a matching header renders as a
            # table and is flagged, while ``- | -`` renders as a list and is not.
            (
                "pipeless delimiter row behind a header is a table",
                "a | b\n--- | ---\n1 | 2",
                True,
            ),
            (
                "pipeless single-dash row renders as a list, not a table",
                "a | b\n- | -\n1 | 2",
                False,
            ),
            ("single pipe line is not a table", "| just text |", False),
            ("horizontal rule is not a table", "---", False),
            ("rule then pipe line is not a table", "---\n| ---", False),
            ("prose with a pipe is not a table", "Choose cats | dogs.", False),
            # A delimiter row only renders as a table behind a header row, so a
            # standalone one (any dash count) must not switch text to markdown.
            (
                "standalone single-dash delimiter row is not a table",
                "| - | - |",
                False,
            ),
            (
                "standalone multi-dash delimiter row is not a table",
                "| --- | --- |",
                False,
            ),
            (
                "delimiter row after a pipeless prose line is not a table",
                "See below.\n| - | - |",
                False,
            ),
            # The renderer rejects these, so the detector must too: a header with
            # a different column count, and a header indented as a code block.
            (
                "header/delimiter column-count mismatch is not a table",
                "a | b | c\n| - | - |",
                False,
            ),
            (
                "code-block-indented header is not a table",
                "    a | b\n| - | - |",
                False,
            ),
            # The candidate gate is line-ending agnostic and column-count
            # agnostic, so renderer-supported CRLF and one-column tables detect.
            (
                "CRLF table is detected",
                "| a | b |\r\n| - | - |\r\n| 1 | 2 |",
                True,
            ),
            (
                "one-column table is detected",
                "| value |\n| - |\n| 1 |",
                True,
            ),
        ]
        for label, text, expected in cases:
            with self.subTest(label=label):
                self.assertIs(looks_like_markdown(text), expected)


class RenderMarkdownTests(SimpleTestCase):
    """Rendering produces safe HTML; agent-supplied script tags stay inert."""

    def test_codex_response_keeps_absolute_file_citation_non_navigable(self) -> None:
        text = """Changed:

- [formatting.py](/root/hitch/hitch/main/formatting.py:38)
- [worker.py](/opt/hitch/hitch/main/worker.py:12)
- [deployment guide](https://example.com/deployment)
"""

        self.assertTrue(looks_like_markdown(text))
        html = render_markdown(text)

        self.assertIn(
            "[formatting.py](/root/hitch/hitch/main/formatting.py:38)", html
        )
        self.assertNotIn('href="/root/hitch/hitch/main/formatting.py:38"', html)
        self.assertIn("[worker.py](/opt/hitch/hitch/main/worker.py:12)", html)
        self.assertNotIn('href="/opt/hitch/hitch/main/worker.py:12"', html)
        self.assertIn(
            '<a href="https://example.com/deployment">deployment guide</a>', html
        )

    def test_markdown_preserves_explicit_tex_for_client_rendering(self) -> None:
        html = render_markdown(
            "# Bound\n\n"
            "Inline \\(x_1 \\in \\{1, 2\\}\\), display "
            "\\[x_1 < y & z\\], and $$z_2 > 0$$."
        )

        self.assertIn(r"\(x_1 \in \{1, 2\}\)", html)
        self.assertIn(r"\[x_1 &lt; y &amp; z\]", html)
        self.assertIn("$$z_2 &gt; 0$$", html)

    def test_tex_protection_does_not_restore_unsafe_html(self) -> None:
        html = render_markdown(
            r"# Bound" "\n\n" r"\(\text{<img src=x onerror=alert(1)>}\)"
        )

        self.assertNotIn("<img", html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)

    def test_rendering_cases(self) -> None:
        cases: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
            ("heading renders as h1", "# Title", ("<h1>Title</h1>",), ()),
            (
                "bullet list renders as ul",
                "- one\n- two",
                ("<ul>", "<li>one</li>", "<li>two</li>"),
                (),
            ),
            (
                "fenced code preserves content",
                "```\nprint('hi')\n```",
                ("<pre><code>", "print("),
                (),
            ),
            (
                "script tag is escaped",
                "<script>alert(1)</script>",
                ("&lt;script&gt;",),
                ("<script>",),
            ),
            (
                "javascript URL does not become link",
                "[click](javascript:alert(1))",
                (),
                ('href="javascript:',),
            ),
            (
                # markdown-it percent-encodes the smuggled control char, so a
                # naive ``^javascript:`` check passes ``java%09script:`` even
                # though browsers strip the tab and execute it.
                "entity-encoded tab javascript URL does not become link",
                "[click](java&#9;script:alert(1))",
                (),
                ("<a ", "href="),
            ),
            (
                "entity-encoded newline javascript URL does not become link",
                "# Heading\n\n[click](java&#10;script:alert(document.cookie))",
                (),
                ("<a ", "href="),
            ),
            (
                "vbscript URL does not become link",
                "[click](vbscript:msgbox(1))",
                (),
                ("<a ",),
            ),
            (
                "HTTPS link is rendered",
                "[ok](https://example.com)",
                ('<a href="https://example.com">ok</a>',),
                (),
            ),
            (
                "arbitrary absolute file link does not navigate through Hitch",
                "[operator guide](/srv/nompiler/docs/operator.md:48)",
                ("[operator guide](/srv/nompiler/docs/operator.md:48)",),
                ("<a ", "href="),
            ),
            (
                "encoded absolute file link does not navigate through Hitch",
                "[worker](%2Fworkspace%2Fhitch%2Fworker.py%3A12)",
                ("[worker](%2Fworkspace%2Fhitch%2Fworker.py%3A12)",),
                ("<a ", "href="),
            ),
            (
                "image syntax does not emit img tag",
                "![pixel](https://attacker.example/x.png)",
                (),
                ("<img", "src="),
            ),
            (
                "table is rendered",
                "| a | b |\n|---|---|\n| 1 | 2 |",
                ("<table>", "<th>a</th>", "<td>1</td>"),
                (),
            ),
            (
                "paren-numbered list renders as ol",
                "1) one\n2) two",
                ("<ol>", "<li>one</li>", "<li>two</li>"),
                (),
            ),
            (
                "single-dash delimiter table is rendered",
                "| a | b |\n| - | - |\n| 1 | 2 |",
                ("<table>", "<th>a</th>", "<td>1</td>"),
                (),
            ),
        ]
        for label, text, expected_present, expected_absent in cases:
            with self.subTest(label=label):
                html = render_markdown(text)
                for snippet in expected_present:
                    self.assertIn(snippet, html)
                for snippet in expected_absent:
                    self.assertNotIn(snippet, html)


class SessionMathBrowserTests(SimpleTestCase):
    def test_renders_explicit_agent_math_only_after_message_completion(self) -> None:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self.skipTest(f"playwright unavailable: {exc}")

        static_root = Path(__file__).resolve().parent.parent / "static"
        html = r"""<!doctype html>
            <div class="message thinking"><div class="body">
                Inline \(x^2 + \varepsilon\).
                <code>Literal \(not_math\)</code>
            </div></div>
            <div class="message agent"><div class="body">
                Display \[\sum_{n=1}^{\infty} n^{-2}\].
            </div></div>
            <div class="message agent"><div class="body">Cost $5 only.</div></div>
            <div class="message agent malformed"><div class="body">
                Bad \(\definitelyUnknownCommand{x}\).
            </div></div>
            <div class="message agent untrusted"><div class="body">
                Link \(\href{https://attacker.example}{x}\).
            </div></div>
            <div class="message user"><div class="body">User \(x\).</div></div>
            <div class="message agent streaming"><div class="body">Live \(y\).</div></div>
        """

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                page.add_style_tag(
                    path=str(static_root / "vendor" / "katex" / "katex.min.css")
                )
                page.add_script_tag(
                    path=str(static_root / "vendor" / "katex" / "katex.min.js")
                )
                page.add_script_tag(
                    path=str(
                        static_root
                        / "vendor"
                        / "katex"
                        / "contrib"
                        / "auto-render.min.js"
                    )
                )
                page.add_script_tag(path=str(static_root / "session_math.js"))

                self.assertEqual(page.locator(".message.thinking .katex").count(), 1)
                self.assertEqual(page.locator(".message.agent .katex-display").count(), 1)
                self.assertEqual(page.locator("code .katex").count(), 0)
                self.assertEqual(page.locator(".message.malformed .katex").count(), 0)
                self.assertIn(
                    r"\definitelyUnknownCommand",
                    page.locator(".message.malformed").inner_text(),
                )
                self.assertEqual(page.locator(".message.untrusted a").count(), 0)
                self.assertEqual(page.locator(".message.user .katex").count(), 0)
                self.assertEqual(page.locator(".message.streaming .katex").count(), 0)
                self.assertIn(
                    r"Inline \(x^2 + \varepsilon\).",
                    page.locator(".message.thinking .body").get_attribute(
                        "data-copy-text"
                    )
                    or "",
                )

                page.locator(".message.streaming").evaluate(
                    "(message) => message.classList.remove('streaming')"
                )
                page.evaluate("window.hitchMath.render(document)")
                self.assertEqual(page.locator(".message.agent .katex").count(), 3)
                self.assertNotIn("katex", page.locator("body").inner_text().lower())
            finally:
                browser.close()
