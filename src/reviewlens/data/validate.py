"""Data validation and cleaning steps.

Each step is a small function that returns the cleaned frame *and* how many rows it removed,
so the preparation report can show exactly what happened to the data. Rows are dropped, never
silently "fixed": a review with an unknown label is not guessed into a class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


class DataValidationError(ValueError):
    """The prepared data breaks an invariant the rest of the pipeline relies on."""


def encode_labels(
    frame: pd.DataFrame,
    raw_label_column: str,
    label_column: str,
    label_map: Mapping[str, str],
    labels: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Map raw labels to integer ids (``labels.index(name)``); drop rows with unknown labels."""
    label_ids = {raw: labels.index(name) for raw, name in label_map.items()}
    encoded = frame[raw_label_column].map(label_ids)
    known = encoded.notna()
    result = frame.loc[known].drop(columns=[raw_label_column])
    result[label_column] = encoded[known].astype(int)
    return result.reset_index(drop=True), int((~known).sum())


def drop_short_texts(
    frame: pd.DataFrame, text_column: str, min_chars: int
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose text is missing or shorter than ``min_chars`` characters."""
    lengths = frame[text_column].fillna("").astype(str).str.len()
    keep = lengths >= min_chars
    return frame.loc[keep].reset_index(drop=True), int((~keep).sum())


def drop_duplicates(
    frame: pd.DataFrame, text_column: str, label_column: str
) -> tuple[pd.DataFrame, int, int]:
    """Remove duplicate texts.

    - Same text, same label: keep the first copy (the copies are counted as duplicates).
    - Same text, different labels: drop every copy, because the true label is ambiguous
      (counted as conflicting rows).

    Returns ``(frame, n_duplicates_removed, n_conflicting_removed)``.
    """
    labels_per_text = frame.groupby(text_column)[label_column].transform("nunique")
    conflicting = labels_per_text > 1
    consistent = frame.loc[~conflicting]
    deduplicated = consistent.drop_duplicates(subset=text_column, keep="first")
    n_duplicates = len(consistent) - len(deduplicated)
    return deduplicated.reset_index(drop=True), n_duplicates, int(conflicting.sum())


def check_prepared(frame: pd.DataFrame, text_column: str, label_column: str, n_labels: int) -> None:
    """Raise ``DataValidationError`` if the prepared frame breaks a pipeline invariant."""
    if frame.empty:
        raise DataValidationError("No rows left after cleaning.")
    if frame[text_column].isna().any() or (frame[text_column].str.len() == 0).any():
        raise DataValidationError("Prepared data contains empty texts.")
    labels = frame[label_column]
    if labels.isna().any() or not labels.between(0, n_labels - 1).all():
        raise DataValidationError(f"Labels must be integers in [0, {n_labels - 1}].")
    if labels.nunique() < 2:
        raise DataValidationError("Prepared data must contain at least two classes.")
    if frame[text_column].duplicated().any():
        raise DataValidationError("Prepared data contains duplicate texts.")
