"""
Unit test: preprocessing

- Builds a tiny synthetic PetImages folder:
    raw/PetImages/Cat/*.jpg
    raw/PetImages/Dog/*.jpg
  plus one corrupt file.
- Runs catsdogs.preprocess.preprocess()
- Validates:
    - corrupt file is dropped
    - output images are 224x224 RGB
    - splits.csv exists with expected columns
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.catsdogs.preprocess import preprocess
from src.catsdogs.config import IMAGE_SIZE


def _make_img(path: Path) -> None:
    img = Image.new("RGB", (300, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG")


def test_preprocess_resizes_and_splits(tmp_path: Path):
    raw = tmp_path / "data/raw/PetImages"
    processed = tmp_path / "data/processed"

    # 4 cats + 4 dogs
    for i in range(4):
        _make_img(raw / "Cat" / f"cat_{i}.jpg")
        _make_img(raw / "Dog" / f"dog_{i}.jpg")

    # corrupt file
    bad = raw / "Cat" / "bad.jpg"
    bad.write_bytes(b"not an image")

    summary = preprocess(raw, processed, max_per_class=0)

    assert summary.total_found == 9
    assert summary.total_dropped_corrupt >= 1
    assert (processed / "splits.csv").exists()

    df = pd.read_csv(processed / "splits.csv")
    assert set(df.columns) == {"split", "label", "src_path", "rel_path"}

    # check that at least one output file exists and is correct size
    out_img = processed / df.iloc[0]["rel_path"]
    img = Image.open(out_img).convert("RGB")
    assert img.size == IMAGE_SIZE
