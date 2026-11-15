from __future__ import annotations

import pandas as pd
import pytest

from reviewlens.data.validate import (
    DataValidationError,
    check_prepared,
    drop_duplicates,
    drop_short_texts,
    encode_labels,
)

LABELS = ["negative", "positive"]
LABEL_MAP = {"SAD": "negative", "HAPPY": "positive"}


def test_encode_labels_maps_to_ids_and_drops_unknown() -> None:
    frame = pd.DataFrame({"text": ["a", "b", "c"], "raw_label": ["HAPPY", "SAD", "NEUTRAL"]})

    result, n_unknown = encode_labels(frame, "raw_label", "label", LABEL_MAP, LABELS)

    assert n_unknown == 1
    assert list(result.columns) == ["text", "label"]
    assert result["label"].tolist() == [1, 0]


def test_drop_short_texts_handles_missing_values() -> None:
    frame = pd.DataFrame({"text": ["خوب", "!", "", None], "label": [1, 0, 0, 1]})

    result, n_dropped = drop_short_texts(frame, "text", min_chars=2)

    assert n_dropped == 3
    assert result["text"].tolist() == ["خوب"]


def test_drop_duplicates_keeps_one_copy_and_removes_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "text": ["a", "a", "b", "b", "c"],
            "label": [1, 1, 0, 1, 0],  # "a" duplicated, "b" conflicting
        }
    )

    result, n_dup, n_conflict = drop_duplicates(frame, "text", "label")

    assert result["text"].tolist() == ["a", "c"]
    assert (n_dup, n_conflict) == (1, 2)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})


def test_check_prepared_accepts_valid_frame() -> None:
    check_prepared(_valid_frame(), "text", "label", n_labels=2)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"text": [], "label": []}),
        pd.DataFrame({"text": ["a", ""], "label": [0, 1]}),
        pd.DataFrame({"text": ["a", "b"], "label": [0, 2]}),  # label out of range
        pd.DataFrame({"text": ["a", "b"], "label": [1, 1]}),  # one class only
        pd.DataFrame({"text": ["a", "a"], "label": [0, 1]}),  # duplicate text
    ],
)
def test_check_prepared_rejects_broken_frames(frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        check_prepared(frame, "text", "label", n_labels=2)
