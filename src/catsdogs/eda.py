"""
catsdogs.eda

Quick exploratory analysis for the processed dataset.
Saves plots to artifacts/plots.

Plots:
- Class distribution by split (bar chart)
- Sample images grid (few cats + dogs)

CLI:
  python -m catsdogs.eda --processed data/processed
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import CLASSES, Paths, project_root
from .logging_config import setup_logging
from .utils import ensure_dir

log = logging.getLogger(__name__)


def run_eda(processed_dir: Path, plots_dir: Path, seed: int = 42) -> None:
    ensure_dir(plots_dir)
    splits_csv = processed_dir / "splits.csv"
    if not splits_csv.exists():
        raise FileNotFoundError(f"Missing split manifest: {splits_csv}. Run preprocess first.")

    df = pd.read_csv(splits_csv)

    # 1) Class distribution
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar")
    ax.set_title("Cats vs Dogs - Class Distribution by Split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image Count")
    fig = ax.get_figure()
    fig.tight_layout()
    out1 = plots_dir / "class_distribution_by_split.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    log.info("Saved: %s", out1)

    # 2) Sample grid (balanced)
    random.seed(seed)
    samples = []
    for cls in CLASSES:
        cls_rows = df[df["label"] == cls].sample(n=min(8, (df["label"] == cls).sum()), random_state=seed)
        samples.extend([(cls, processed_dir / p) for p in cls_rows["rel_path"].tolist()])

    # keep deterministic order: cats first then dogs
    samples = sorted(samples, key=lambda x: x[0])

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes]  # type: ignore

    # flatten axes
    ax_list = []
    for r in range(rows):
        for c in range(cols):
            ax_list.append(axes[r][c] if rows > 1 else axes[c])

    for ax, (cls, img_path) in zip(ax_list, samples):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")

    for ax in ax_list[len(samples):]:
        ax.axis("off")

    fig.suptitle("Sample Images (After 224x224 RGB Preprocessing)", y=0.98)
    fig.tight_layout()
    out2 = plots_dir / "sample_grid.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    log.info("Saved: %s", out2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EDA for Cats vs Dogs processed dataset.")
    parser.add_argument("--processed", default="data/processed", help="Processed dataset dir.")
    parser.add_argument("--plots", default="artifacts/plots", help="Output plots dir.")
    args = parser.parse_args(argv)

    setup_logging()
    paths = Paths(project_root())
    try:
        run_eda(paths.project_root / args.processed, paths.project_root / args.plots)
        return 0
    except Exception as e:
        log.exception("EDA failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
