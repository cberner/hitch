"""Small, dependency-free sequence helpers shared across the app."""

from collections.abc import Iterable
from typing import TypeVar

_T = TypeVar("_T")


def unique_nonempty(values: Iterable[_T]) -> list[_T]:
    """Order-preserving de-duplication that drops falsy items.

    ``dict.fromkeys`` keeps first-seen order, and the truthiness filter drops
    empty strings and ``None``. This is the canonical cleanup for lists of
    thread ids, working directories, and file paths gathered from several
    sources, where both duplicates and empties are common.
    """
    return [value for value in dict.fromkeys(values) if value]
