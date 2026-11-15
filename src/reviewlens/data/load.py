"""Read raw CSV files and rename their columns to the project's standard names.

After loading, every dataset looks the same to the rest of the code: a text column
(``task.text_column``) and a raw label column (``RAW_LABEL``), whatever the source called them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from reviewlens.config import DataConfig, TaskConfig

RAW_LABEL = "raw_label"


def read_raw_csv(path: Path, data_cfg: DataConfig, task: TaskConfig) -> pd.DataFrame:
    """Read one raw CSV and keep only the text and label columns, renamed."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {path}. Run `reviewlens data download`.")
    required = [data_cfg.raw_text_column, data_cfg.raw_label_column]
    # Read everything as text: a review like "5" must stay the string "5", not a number.
    frame = pd.read_csv(path, encoding="utf-8", dtype=str)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(frame.columns)}")
    return frame[required].rename(
        columns={data_cfg.raw_text_column: task.text_column, data_cfg.raw_label_column: RAW_LABEL}
    )


def read_raw_files(paths: Sequence[Path], data_cfg: DataConfig, task: TaskConfig) -> pd.DataFrame:
    """Read several raw CSVs and stack them into one frame."""
    frames = [read_raw_csv(path, data_cfg, task) for path in paths]
    return pd.concat(frames, ignore_index=True)
