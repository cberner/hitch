"""Detect and render markdown in agent replies on the session detail page.

Only the final agent message of each turn is eligible -- intermediate
"thinking" entries, user messages, and tool calls stay plain-text. Detection
demands at least one unambiguous markdown construct (fenced code block, ATX
heading, multi-item list, explicit ``[label](http(s)://...)`` link, or a
table delimiter row) so agent replies that merely contain a stray asterisk
or backtick are not reformatted.

The CommonMark renderer is configured with ``html=False`` because the
``commonmark`` preset enables raw HTML by default; without that override the
agent could inject ``<script>`` tags into the page. markdown-it-py's built-in
link validator drops bare ``javascript:`` URLs, but it runs *after* the
destination is percent-encoded, so an entity-encoded control character
(``[x](java&#9;script:alert(1))`` -> ``java%09script:alert(1)``) slips past it
while browsers still strip the tab and execute it; ``_is_safe_link`` re-checks
the decoded scheme to close that hole. The ``image`` rule is also disabled so that an
``![alt](https://attacker.example/pixel)`` in an agent reply doesn't make
the browser fetch a third-party URL the moment the session page is viewed
(IP/referrer leakage); image syntax degrades to a clickable link instead.
"""

import re
from urllib.parse import unquote

from django.utils.safestring import SafeString, mark_safe
from markdown_it import MarkdownIt

# markdown-it normalizes (percent-encodes) link destinations before validating
# them, so control characters smuggled in via HTML entities survive as ``%09``
# / ``%0A`` and defeat its literal ``^javascript:`` check. Browsers strip those
# characters from the scheme and execute the link, so decode and strip them the
# same way before deciding whether a scheme is dangerous.
_LINK_CONTROL_CHARS = re.compile(r"[\x00-\x20\x7f]+")
_DANGEROUS_LINK_SCHEMES = ("javascript:", "vbscript:", "data:", "file:")


def _is_safe_link(url: str) -> bool:
    cleaned = _LINK_CONTROL_CHARS.sub("", unquote(url)).strip().lower()
    return not cleaned.startswith(_DANGEROUS_LINK_SCHEMES)


_RENDERER = (
    MarkdownIt("commonmark", {"html": False, "breaks": False})
    .enable("table")
    .disable("image")
)
# Override the built-in validator (see module docstring) to also reject
# encoded-control-character scheme smuggling. markdown-it looks this up on the
# instance, so assigning the attribute is the supported override hook.
_RENDERER.validateLink = _is_safe_link  # type: ignore[method-assign]

_FENCED_CODE = re.compile(r"^```", re.MULTILINE)
_ATX_HEADING = re.compile(r"^#{1,6} \S", re.MULTILINE)
# CommonMark renders list items indented 0-3 spaces -- 4+ spaces is a code
# block -- with any mix of spaces and tabs between the marker and the first
# non-blank character. Matching that here keeps the detector in sync with the
# renderer so a nested or under-a-heading list does not silently fall back to
# plain-text rendering.
_BULLET_LIST = re.compile(r"^ {0,3}[-*+][ \t]+\S", re.MULTILINE)
# CommonMark ordered lists accept either ``.`` or ``)`` after the number, so the
# detector must match both delimiters; otherwise a ``1)``-style list (a very
# common agent phrasing) renders as raw plain text while the renderer would have
# produced an ``<ol>``. The captured number drives the paragraph-interruption
# rule in ``_has_ordered_list``.
_NUMBERED_LIST = re.compile(r"^ {0,3}(\d{1,9})[.)][ \t]+\S")
_MARKDOWN_LINK = re.compile(r"\[[^\]\n]+\]\((?:https?://|mailto:)[^\s)]+\)")
# Characters that may appear on a GFM table delimiter row (cells of dashes with
# optional alignment colons, separated/bounded by pipes, plus padding). A line
# built only from these and containing both a pipe and a dash is a delimiter-row
# *candidate* -- see ``_has_table`` for why detection ultimately defers to the
# renderer rather than validating the row precisely here.
_TABLE_DELIMITER_CHARS = frozenset("|:- \t")


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
    if _has_ordered_list(text):
        return True
    if _MARKDOWN_LINK.search(text):
        return True
    return _has_table(text)


def _has_table(text: str) -> bool:
    """True iff the renderer actually emits a ``<table>`` for ``text``.

    GFM's table rule hinges on matching header/delimiter column counts, the
    header's indentation, surrounding block context, one- vs. multi-column
    layout, and leading-pipe vs. pipeless delimiters -- mirroring all of that by
    hand drifts from the renderer and resurfaces as detector/renderer-parity
    bugs. Instead we cheaply spot a delimiter-row candidate (so ordinary prose
    never triggers a render) and let markdown-it itself decide. The page renders
    the reply again for display, but only when a candidate is present, which is
    fine for a single-user tool.
    """
    if not any(_is_table_delimiter_candidate(line) for line in text.split("\n")):
        return False
    return "<table" in render_markdown(text)


def _is_table_delimiter_candidate(line: str) -> bool:
    # ``strip`` drops the trailing ``\r`` of a CRLF line and any padding, so the
    # gate is line-ending agnostic. A pipe rules out a bare ``---`` rule and a
    # dash rules out a plain ``| cell |`` data/header row; both are required so
    # only a delimiter-shaped line opens the (renderer-backed) table check.
    line = line.strip()
    return bool(line) and "|" in line and "-" in line and set(line) <= _TABLE_DELIMITER_CHARS


def _has_ordered_list(text: str) -> bool:
    """True iff ``text`` holds an ordered list of >=2 items the renderer emits.

    CommonMark only lets an ordered list interrupt a paragraph when its first
    item is numbered 1, so a run like ``2)``/``3)`` glued straight onto a
    preceding prose line stays *inside* that paragraph rather than becoming an
    ``<ol>``. Mirroring that rule keeps the detector from reflowing such text as
    markdown, which would otherwise collapse its intended line breaks. A blank
    line (or the start of the text) removes the paragraph to interrupt, so an
    item may then begin at any number; once a list is open, later items continue
    it regardless of their number.
    """
    count = 0
    in_list = False
    paragraph_open = False
    for line in text.split("\n"):
        match = _NUMBERED_LIST.match(line)
        if match is not None:
            if in_list or not paragraph_open or int(match.group(1)) == 1:
                count += 1
                if count >= 2:
                    return True
                in_list = True
                paragraph_open = False
            else:
                # Can't interrupt the open paragraph; the marker is plain text.
                in_list = False
            continue
        if not line.strip():
            # A blank line ends both the paragraph and any open list run; the
            # next item starts fresh (and may legitimately begin above 1).
            in_list = False
            paragraph_open = False
        elif not in_list:
            # Ordinary prose; a numbered marker on the next line could only
            # start a list if it begins at 1. (Lines while ``in_list`` are lazy
            # continuations and leave the list open.)
            paragraph_open = True
    return False


def render_markdown(text: str) -> SafeString:
    """Convert ``text`` to safe HTML.

    Raw HTML in the input is escaped, ``javascript:`` URLs are stripped, so
    agent-supplied content can't inject script or click-handler payloads.
    """
    html: str = _RENDERER.render(text)
    return mark_safe(html)
