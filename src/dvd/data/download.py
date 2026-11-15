"""Download + extract helpers with a persistent zip cache (Drive) and fast local extraction."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm


def download(url: str, dest: str | Path) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    shutil.move(tmp, dest)
    return dest


def download_and_extract(url: str, cache_dir: str | Path, extract_to: str | Path,
                         marker: str | Path) -> Path:
    """Download `url` into `cache_dir` (once) and extract into `extract_to` unless `marker` exists."""
    marker = Path(marker)
    if marker.exists():
        return marker
    zpath = download(url, Path(cache_dir) / url.rsplit("/", 1)[-1])
    with zipfile.ZipFile(zpath) as zf:
        for m in tqdm(zf.infolist(), desc=f"extract {zpath.name}"):
            zf.extract(m, extract_to)
    return marker
