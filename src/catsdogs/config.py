"""
catsdogs.config

Central configuration values for the Cats vs Dogs pipeline.

Assignment 2 requirements:
- Resize images to 224x224 RGB
- Split into train/val/test (default 80/10/10)
- Provide reproducible paths for artifacts/models

Note:
The Kaggle Cats & Dogs dataset is also mirrored by Microsoft as:
kagglecatsanddogs_5340.zip
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def plots_dir(self) -> Path:
        return self.artifacts_dir / "plots"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"


def project_root() -> Path:
    """
    Resolve repo root from this file location: <repo>/src/catsdogs/config.py
    """
    return Path(__file__).resolve().parents[2]


# Default dataset download (direct link).
# This is commonly used in TF/keras examples and community threads.
DEFAULT_DATASET_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)

IMAGE_SIZE = (224, 224)  # width, height
CHANNELS = 3

DEFAULT_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
RANDOM_SEED = 42
CLASSES = ["Cat", "Dog"]
