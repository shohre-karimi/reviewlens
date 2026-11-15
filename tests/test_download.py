"""Download tests with a fake ``urlopen``: no network access."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.error import URLError

import pytest

from reviewlens.config import DataConfig
from reviewlens.data import download
from reviewlens.data.download import DownloadError, download_raw_files

DATA_CFG = DataConfig()


def _fake_urlopen(url: str, timeout: float) -> io.BytesIO:
    return io.BytesIO(f"comment,label\nfrom {url.rsplit('/', 1)[-1]},HAPPY\n".encode())


def test_downloads_every_configured_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    def recording_urlopen(url: str, timeout: float) -> io.BytesIO:
        requested.append(url)
        return _fake_urlopen(url, timeout)

    monkeypatch.setattr(download, "urlopen", recording_urlopen)

    paths = download_raw_files(DATA_CFG, tmp_path)

    assert [p.name for p in paths] == ["train.csv", "validation.csv", "test.csv"]
    assert all(DATA_CFG.revision in url for url in requested)
    assert "from test.csv" in (tmp_path / "test.csv").read_text(encoding="utf-8")


def test_existing_files_are_skipped_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "urlopen", _fake_urlopen)
    for name in DATA_CFG.all_files:
        (tmp_path / name).write_text("old", encoding="utf-8")

    download_raw_files(DATA_CFG, tmp_path)
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == "old"

    download_raw_files(DATA_CFG, tmp_path, force=True)
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") != "old"


def test_failed_download_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_urlopen(url: str, timeout: float) -> io.BytesIO:
        raise URLError("network unreachable")

    monkeypatch.setattr(download, "urlopen", failing_urlopen)

    with pytest.raises(DownloadError, match="network unreachable"):
        download_raw_files(DATA_CFG, tmp_path)
    assert list(tmp_path.iterdir()) == []
