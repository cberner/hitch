"""Detect and render markdown in agent replies on the session detail page.

Only the final agent message of each turn is eligible -- intermediate
"thinking" entries, user messages, and tool calls stay plain-text. Detection
demands at least one unambiguous markdown construct (fenced code block, ATX
heading, multi-item list, explicit ``[label](http(s)://...)`` link, or a
table delimiter row) so agent replies that merely contain a stray asterisk
or backtick are not reformatted.

The CommonMark renderer is configured with ``html=False`` because the
``commonmark`` preset enables raw HTML by default; without that override the
agent could inject ``<script>`` tags into the page. ``javascript:`` URLs are
dropped by markdown-it-py's built-in link validator, so a separate output
sanitiser is not required. The ``image`` rule is also disabled so that an
``![alt](https://attacker.example/pixel)`` in an agent reply doesn't make
the browser fetch a third-party URL the moment the session page is viewed
(IP/referrer leakage); image syntax degrades to a clickable link instead.
"""

import re

from django.utils.safestring import SafeString, mark_safe
from markdown_it import MarkdownIt

_RENDERER = (
    MarkdownIt("commonmark", {"html": False, "breaks": False})
    .enable("table")
    .disable("image")
)

_FENCED_CODE = re.compile(r"^```", re.MULTILINE)
_ATX_HEADING = re.compile(r"^#{1,6} \S", re.MULTILINE)
_BULLET_LIST = re.compile(r"^[-*+] \S", re.MULTILINE)
_NUMBERED_LIST = re.compile(r"^\d+\. \S", re.MULTILINE)
_MARKDOWN_LINK = re.compile(r"\[[^\]\n]+\]\((?:https?://|mailto:)[^\s)]+\)")
_TABLE_SEPARATOR = re.compile(r"^\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", re.MULTILINE)


def looks_like_markdown(text: str) -> bool:
    """True iff text contains an unambiguous markdown construct.

    Lists and tables require multiple rows because a single ``- foo`` or
    ``| x |`` line is too easy to hit by accident in normal prose.
    """
    if not text:
        return False
    if _FENCED_CODE.search(text):
        return True
    if _ATX_HEADING.search(text):
        return True
    if len(_BULLET_LIST.findall(text)) >= 2:
        return True
    if len(_NUMBERED_LIST.findall(text)) >= 2:
        return True
    if _MARKDOWN_LINK.search(text):
        return True
    return bool(_TABLE_SEPARATOR.search(text))


def render_markdown(text: str) -> SafeString:
    """Convert ``text`` to safe HTML.

    Raw HTML in the input is escaped, ``javascript:`` URLs are stripped, so
    agent-supplied content can't inject script or click-handler payloads.
    """
    html: str = _RENDERER.render(text)
    return mark_safe(html)
