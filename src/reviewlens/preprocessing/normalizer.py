"""Persian text normalization.

``normalize_text`` runs a fixed sequence of small steps. hazm does the Persian-specific work
(Arabic -> Persian letters, half-spaces, diacritics, Persian digits, repeated letters); the
regex steps around it handle what hazm does not cover or would damage:

1. remove URLs           (hazm would split them into meaningless pieces)
2. hazm normalization
3. lowercase Latin text  ("GOOD" and "good" become one feature)
4. collapse repeated punctuation ("!!!!!" -> "!")
5. collapse whitespace   (newlines and tabs become single spaces)

Emojis are kept on purpose: they are strong sentiment signals in reviews.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from hazm import Normalizer

_URL_RE = re.compile(r"(https?://|www\.)\S+", flags=re.IGNORECASE)
_REPEATED_PUNCT_RE = re.compile(r"([!?.,؟،؛:])\1+")
_WHITESPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _hazm_normalizer() -> Normalizer:
    """Build the hazm normalizer once; constructing it compiles many regexes."""
    return Normalizer()


def normalize_text(text: str) -> str:
    """Normalize one review. Non-string input (e.g. a missing value) becomes ``""``."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _hazm_normalizer().normalize(text)
    text = text.lower()
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_texts(texts: Iterable[str]) -> list[str]:
    """Normalize many reviews."""
    return [normalize_text(text) for text in texts]
