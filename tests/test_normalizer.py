from __future__ import annotations

import math

import pytest

from reviewlens.preprocessing import normalize_text, normalize_texts

ZWNJ = "‌"  # Persian half-space (zero-width non-joiner)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("كيفيت عالي بود", "کیفیت عالی بود"),  # Arabic kaf/yeh -> Persian
        ("3 بار", "۳ بار"),  # Latin digits -> Persian digits
        ("خیییییلی خوب", "خیلی خوب"),  # repeated letters
        ("عالی بود!!!!!", "عالی بود!"),  # repeated punctuation
        ("مُتَشَکِّرَم", "متشکرم"),  # diacritics
        ("Pizza خیلی GOOD بود", "pizza خیلی good بود"),  # Latin lowercased
        ("  خوب   بود \n\t نه  ", "خوب بود نه"),  # whitespace collapsed
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_half_space_is_inserted() -> None:
    assert normalize_text("کتاب ها") == f"کتاب{ZWNJ}ها"
    assert normalize_text("می خواهم") == f"می{ZWNJ}خواهم"


def test_urls_are_removed() -> None:
    assert normalize_text("ببینید https://example.com/x?a=1 را") == "ببینید را"
    assert normalize_text("سایت www.example.com خوبه") == "سایت خوبه"


def test_emojis_are_kept() -> None:
    assert "😍" in normalize_text("غذا عالی بود 😍")


@pytest.mark.parametrize("value", [None, math.nan, 5])
def test_non_string_becomes_empty(value: object) -> None:
    assert normalize_text(value) == ""  # type: ignore[arg-type]


def test_normalize_texts_maps_each_item() -> None:
    assert normalize_texts(["كيك", "  خوب  "]) == ["کیک", "خوب"]
