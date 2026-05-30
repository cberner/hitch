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
