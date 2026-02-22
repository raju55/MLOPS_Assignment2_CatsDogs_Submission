"""
catsdogs.data_ingest

Download & extract the Kaggle Cats vs Dogs dataset (PetImages).

Expected raw layout after extraction:
  data/raw/PetImages/Cat/*.jpg
  data/raw/PetImages/Dog/*.jpg

Why we validate later:
The PetImages dataset is known to contain a number of corrupted images. The preprocessing step
removes unreadable files automatically and logs how many were dropped.

CLI:
  python -m catsdogs.data_ingest --help
"""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import requests

from .config import DEFAULT_DATASET_URL, Paths, project_root
from .logging_config import setup_logging
from .utils import ensure_dir, human_bytes

log = logging.getLogger(__name__)


def download_dataset(url: str, dest_zip: Path, timeout_s: int = 60) -> None:
    ensure_dir(dest_zip.parent)
    if dest_zip.exists() and dest_zip.stat().st_size > 0:
        log.info("Dataset zip already exists: %s (%s)", dest_zip, human_bytes(dest_zip.stat().st_size))
        return

    log.info("Downloading dataset from: %s", url)
    with requests.get(url, stream=True, timeout=timeout_s, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        with dest_zip.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    if downloaded % (20 * 1024 * 1024) < 1024 * 1024:
                        log.info("... %.1f%% (%s/%s)", pct, human_bytes(downloaded), human_bytes(total))
    log.info("Downloaded: %s (%s)", dest_zip, human_bytes(dest_zip.stat().st_size))


def extract_dataset(zip_path: Path, raw_dir: Path) -> Path:
    """
    Extract zip under raw_dir and return PetImages folder path.
    """
    ensure_dir(raw_dir)
    petimages_dir = raw_dir / "PetImages"
    if petimages_dir.exists() and any(petimages_dir.rglob("*.jpg")):
        log.info("PetImages already extracted: %s", petimages_dir)
        return petimages_dir

    log.info("Extracting %s -> %s", zip_path, raw_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    if not petimages_dir.exists():
        raise FileNotFoundError(f"Expected {petimages_dir} after extraction, but not found.")
    return petimages_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download & extract Cats vs Dogs dataset.")
    parser.add_argument("--url", default=DEFAULT_DATASET_URL, help="Dataset URL (zip).")
    parser.add_argument("--out", default="data/raw/kagglecatsanddogs_5340.zip", help="Output zip path.")
    args = parser.parse_args(argv)

    setup_logging()
    paths = Paths(project_root())
    zip_path = paths.project_root / args.out

    try:
        download_dataset(args.url, zip_path)
        petimages = extract_dataset(zip_path, paths.raw_dir)
        log.info("Raw dataset ready at: %s", petimages)
        return 0
    except Exception as e:
        log.exception("Data ingest failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
