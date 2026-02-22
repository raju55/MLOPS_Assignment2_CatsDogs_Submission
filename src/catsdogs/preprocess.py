"""
catsdogs.preprocess

Validate images, resize to 224x224 RGB, and create train/val/test splits.

Assignment requirements:
- 224x224 RGB images for standard CNNs
- Train/Val/Test split (default 80/10/10)
- Track processed artifacts (split manifest + summary)

CLI examples:
  python -m catsdogs.preprocess
  python -m catsdogs.preprocess --max-per-class 200  # helpful for quick smoke runs / CI
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError

from .config import CLASSES, DEFAULT_SPLITS, IMAGE_SIZE, Paths, RANDOM_SEED, project_root
from .logging_config import setup_logging
from .utils import ensure_dir, save_json, set_seed

log = logging.getLogger(__name__)


@dataclass
class PreprocessSummary:
    raw_dir: str
    processed_dir: str
    image_size: Tuple[int, int]
    splits: Dict[str, float]
    seed: int
    total_found: int
    total_valid: int
    total_dropped_corrupt: int
    per_class_valid: Dict[str, int]
    per_split_counts: Dict[str, Dict[str, int]]


def _is_image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()  # quick integrity check
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _load_and_resize(path: Path, size: Tuple[int, int]) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size, resample=Image.BILINEAR)
        return img


def _gather_files(raw_petimages_dir: Path) -> List[Tuple[Path, str]]:
    files: List[Tuple[Path, str]] = []
    for cls in CLASSES:
        cls_dir = raw_petimages_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Expected class folder not found: {cls_dir}")
        for p in cls_dir.rglob("*.jpg"):
            files.append((p, cls))
    return files


def preprocess(
    raw_petimages_dir: Path,
    processed_dir: Path,
    splits: Dict[str, float] = DEFAULT_SPLITS,
    seed: int = RANDOM_SEED,
    max_per_class: int = 0,
) -> PreprocessSummary:
    """
    Main preprocessing function (unit-tested).
    """
    set_seed(seed)
    ensure_dir(processed_dir)

    all_files = _gather_files(raw_petimages_dir)
    total_found = len(all_files)

    # validate images first (drop corrupt)
    valid: List[Tuple[Path, str]] = []
    dropped = 0
    per_class: Dict[str, List[Tuple[Path, str]]] = {c: [] for c in CLASSES}
    for p, cls in all_files:
        if _is_image_ok(p):
            per_class[cls].append((p, cls))
        else:
            dropped += 1

    # optional limit (useful for CI / fast iteration)
    for cls in CLASSES:
        items = per_class[cls]
        if max_per_class and len(items) > max_per_class:
            items = items[:max_per_class]
        valid.extend(items)

    # shuffle consistently
    import random
    random.shuffle(valid)

    # split by class to keep class balance across splits
    per_split_counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in splits}
    manifest_rows: List[Dict[str, str]] = []

    for split in splits:
        ensure_dir(processed_dir / split / "Cat")
        ensure_dir(processed_dir / split / "Dog")

    # group by class for stratified splitting
    for cls in CLASSES:
        cls_items = [x for x in valid if x[1] == cls]
        n = len(cls_items)
        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])
        n_test = n - n_train - n_val
        split_items = (
            [("train", x) for x in cls_items[:n_train]]
            + [("val", x) for x in cls_items[n_train:n_train + n_val]]
            + [("test", x) for x in cls_items[n_train + n_val:]]
        )
        assert len(split_items) == n_train + n_val + n_test

        for split_name, (src_path, _) in split_items:
            out_name = src_path.name
            out_path = processed_dir / split_name / cls / out_name

            # Some filenames repeat; avoid overwrite by prefixing with a short hash if needed
            if out_path.exists():
                import hashlib
                h = hashlib.md5(str(src_path).encode("utf-8")).hexdigest()[:8]
                out_path = processed_dir / split_name / cls / f"{h}_{out_name}"

            img = _load_and_resize(src_path, IMAGE_SIZE)
            img.save(out_path, format="JPEG", quality=95)

            per_split_counts[split_name][cls] += 1
            manifest_rows.append(
                {
                    "split": split_name,
                    "label": cls,
                    "src_path": str(src_path.relative_to(raw_petimages_dir)),
                    "rel_path": str(out_path.relative_to(processed_dir)),
                }
            )

    # write manifest
    splits_csv = processed_dir / "splits.csv"
    ensure_dir(splits_csv.parent)
    with splits_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "src_path", "rel_path"])
        w.writeheader()
        w.writerows(manifest_rows)

    summary = PreprocessSummary(
        raw_dir=str(raw_petimages_dir),
        processed_dir=str(processed_dir),
        image_size=IMAGE_SIZE,
        splits=splits,
        seed=seed,
        total_found=total_found,
        total_valid=len(valid),
        total_dropped_corrupt=dropped,
        per_class_valid={c: len([x for x in valid if x[1] == c]) for c in CLASSES},
        per_split_counts=per_split_counts,
    )
    save_json(asdict(summary), processed_dir / "preprocess_summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess PetImages into 224x224 train/val/test sets.")
    parser.add_argument("--raw", default="data/raw/PetImages", help="Raw PetImages directory.")
    parser.add_argument("--out", default="data/processed", help="Processed output directory.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit images per class (0 = no limit).")
    args = parser.parse_args(argv)

    setup_logging()
    paths = Paths(project_root())
    raw_dir = paths.project_root / args.raw
    out_dir = paths.project_root / args.out

    try:
        summary = preprocess(raw_dir, out_dir, seed=args.seed, max_per_class=args.max_per_class)
        log.info("Preprocessing complete. Valid images: %s", summary.total_valid)
        log.info("Split counts: %s", summary.per_split_counts)
        return 0
    except Exception as e:
        log.exception("Preprocess failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
