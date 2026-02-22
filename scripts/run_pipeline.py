"""scripts.run_pipeline

Single entrypoint to run the Assignment-2 end-to-end pipeline:

  Data ingest -> Preprocess -> EDA -> Train (+ optional MLflow logging)

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --skip-ingest   (if dataset already downloaded)
  python scripts/run_pipeline.py --skip-eda
  python scripts/run_pipeline.py --skip-train
  python scripts/run_pipeline.py --max-per-class 200   (fast smoke run)

Notes:
- The raw dataset is NOT committed to git due to size.
- DVC is recommended to version the dataset artifacts (see dvc.yaml + README).
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--max-per-class", type=int, default=0, help="Limit images per class (0 = no limit).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="Limit batches per epoch (0 = full). Useful for CI/tests.")
    args = p.parse_args(argv)

    try:
        if not args.skip_ingest:
            _run([sys.executable, "-m", "src.catsdogs.data_ingest"])

        _run([sys.executable, "-m", "src.catsdogs.preprocess", "--max-per-class", str(args.max_per_class)])

        if not args.skip_eda:
            _run([sys.executable, "-m", "src.catsdogs.eda"])

        if not args.skip_train:
            _run([sys.executable, "-m", "src.catsdogs.train", "--epochs", str(args.epochs), "--max-steps", str(args.max_steps)])

        print("\nPipeline finished successfully.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed with exit code: {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
