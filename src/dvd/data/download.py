"""Download + extract helpers with a persistent zip cache (Drive) and fast local extraction."""
# Postpone evaluation of type hints (compatibility with older Python versions).
from __future__ import annotations

# shutil: `move` renames the finished temp file into place.
import shutil
# time: `sleep` between retries.
import time
# urllib.request: standard-library HTTP client (no extra dependency).
import urllib.request
# zipfile: read the dataset zips and extract single members.
import zipfile
# pathlib.Path: portable file paths.
from pathlib import Path

# tqdm: progress bars for long downloads/extractions.
from tqdm.auto import tqdm


def download(url: str, dest: str | Path, quiet: bool = False, retries: int = 6) -> Path:
    """Download `url` to `dest` (skipped if it exists). `quiet` hides the progress bar (many small files).
    Failures (e.g. HTTP 503 "Slow Down", the COCO server's rate limit on shared cloud IPs) are retried with a growing
    pause (5, 10, 20, 40, 60 s), alternating between the http:// and https:// form of the URL."""
    dest = Path(dest)
    # Already downloaded (non-empty file): nothing to do.
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Download into a temporary ".part" file so a half-finished download is never mistaken for a complete one.
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Same file over https as a second route (some networks / rate limits treat the two differently).
    urls = [url] + ([url.replace("http://", "https://", 1)] if url.startswith("http://") else [])
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urls[attempt % len(urls)], timeout=60) as r, open(tmp, "wb") as f:
                # Total size from the HTTP header (None if the server does not send it).
                total = int(r.headers.get("Content-Length", 0)) or None
                with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, disable=quiet) as bar:
                    while True:
                        # Read 1 MiB at a time so memory stays small even for the 778 MB zip.
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except OSError:  # URLError / timeouts are OSError subclasses
            # Re-raise only after the last attempt; earlier failures are retried after a growing pause.
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))
    # Atomically put the finished file at its final name.
    shutil.move(tmp, dest)
    return dest


def download_and_extract(url: str, cache_dir: str | Path, extract_to: str | Path,
                         marker: str | Path, members: list[str] | None = None) -> Path:
    """Download `url` into `cache_dir` (once) and extract into `extract_to` unless `marker` exists.
    `members` restricts the extraction to those archive entries (e.g. one JSON out of a big zip)."""
    marker = Path(marker)
    # The marker file proves a previous extraction finished, so skip everything.
    if marker.exists():
        return marker
    # The zip lives in the persistent cache (Drive on Colab), so it is downloaded only once.
    zpath = download(url, Path(cache_dir) / url.rsplit("/", 1)[-1])
    with zipfile.ZipFile(zpath) as zf:
        # Either the named members only, or the whole archive.
        infos = [zf.getinfo(m) for m in members] if members else zf.infolist()
        for m in tqdm(infos, desc=f"extract {zpath.name}"):
            zf.extract(m, extract_to)
    return marker
