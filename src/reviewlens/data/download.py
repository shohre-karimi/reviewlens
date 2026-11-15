"""Download the raw dataset files.

Only the standard library is used (``urllib``): downloading a few CSV files does not justify
an extra dependency. Each file is written to ``<name>.part`` first and renamed when complete,
so an interrupted download never leaves a truncated file that looks valid.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from reviewlens.config import DataConfig

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 60


class DownloadError(RuntimeError):
    """A raw data file could not be downloaded."""


def download_file(url: str, target: Path) -> None:
    """Download ``url`` to ``target`` atomically (via a temporary ``.part`` file)."""
    partial = target.with_name(target.name + ".part")
    try:
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except (URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Could not download {url}: {error}") from error
    partial.replace(target)


def download_raw_files(data_cfg: DataConfig, raw_dir: Path, force: bool = False) -> list[Path]:
    """Download every configured file into ``raw_dir``; existing files are kept unless ``force``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in data_cfg.all_files:
        target = raw_dir / filename
        if target.exists() and not force:
            logger.info("Raw file exists, skipping file=%s", target)
        else:
            url = data_cfg.file_url(filename)
            logger.info("Downloading url=%s", url)
            download_file(url, target)
            logger.info("Saved file=%s bytes=%d", target, target.stat().st_size)
        paths.append(target)
    return paths
