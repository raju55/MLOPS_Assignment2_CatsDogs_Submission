"""
catsdogs.train

Train baseline CNN using the processed dataset.

Outputs:
- models/catsdogs_cnn.pt                       (weights)
- artifacts/metrics/metrics.json               (summary metrics)
- artifacts/plots/confusion_matrix.png         (artifact)
- artifacts/plots/loss_curve.png               (artifact)
- artifacts/metrics/classification_report.txt  (artifact)

Experiment tracking:
- MLflow is used if installed (mlflow==... in requirements.txt).
  The tracking URI is controlled by MLFLOW_TRACKING_URI (file store by default).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from .config import CLASSES, IMAGE_SIZE, Paths, RANDOM_SEED, project_root
from .logging_config import setup_logging
from .model import SimpleCNN
from .utils import ensure_dir, save_json, set_seed

log = logging.getLogger(__name__)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image into a float32 CHW tensor in [0, 1].

    NOTE:
    Some Windows PyTorch wheels fail to initialize the NumPy bridge when NumPy
    2.x is installed, which causes runtime errors such as:
    "RuntimeError: Numpy is not available" when calling torch.from_numpy()
    or Tensor.numpy().

    To keep the training/inference pipeline robust across environments, we
    avoid the NumPy bridge and use the Python buffer protocol instead.
    """
    img = img.convert("RGB")
    w, h = img.size
    raw = img.tobytes()  # RGBRGB...
    if hasattr(torch, "frombuffer"):
        x = torch.frombuffer(raw, dtype=torch.uint8)
    else:
        x = torch.ByteTensor(torch.ByteStorage.from_buffer(raw))
    x = x.view(h, w, 3).permute(2, 0, 1).contiguous()  # CHW
    return x.to(dtype=torch.float32).div_(255.0)


def _augment(img: Image.Image) -> Image.Image:
    # Random horizontal flip
    if np.random.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Small rotation
    angle = float(np.random.uniform(-15, 15))
    img = img.rotate(angle)
    # Brightness / contrast jitter
    if np.random.rand() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(float(np.random.uniform(0.8, 1.2)))
    if np.random.rand() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(float(np.random.uniform(0.8, 1.2)))
    return img


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, augment: bool = False) -> None:
        self.root = root
        self.augment = augment
        self.samples: List[Tuple[Path, int]] = []
        for idx, cls in enumerate(CLASSES):
            for p in (root / cls).rglob("*.jpg"):
                self.samples.append((p, idx))
        if not self.samples:
            raise ValueError(f"No images found under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        # images should already be resized in preprocess, but we enforce for safety
        img = img.resize(IMAGE_SIZE, resample=Image.BILINEAR)
        if self.augment:
            img = _augment(img)
        x = _pil_to_tensor(img)
        y = torch.tensor([float(label)], dtype=torch.float32)
        return x, y


@dataclass
class TrainResult:
    epochs: int
    train_loss: float
    val_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]


def _evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, max_batches: int = 0) -> Tuple[float, List[int], List[int]]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        b = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits).detach().cpu().view(-1).tolist()
            preds = [1 if p >= 0.5 else 0 for p in probs]
            y_pred.extend(preds)
            y_true.extend([int(v) for v in y.detach().cpu().view(-1).tolist()])
            b += 1
            if max_batches and b >= max_batches:
                break
    return (sum(losses) / len(losses)) if losses else math.nan, y_true, y_pred


def train(
    processed_dir: Path,
    models_dir: Path,
    artifacts_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = RANDOM_SEED,
    max_steps_per_epoch: int = 0,
) -> TrainResult:
    set_seed(seed)
    ensure_dir(models_dir)
    ensure_dir(artifacts_dir / "plots")
    ensure_dir(artifacts_dir / "metrics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    train_ds = ImageFolderDataset(processed_dir / "train", augment=True)
    val_ds = ImageFolderDataset(processed_dir / "val", augment=False)
    test_ds = ImageFolderDataset(processed_dir / "test", augment=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        t0 = time.time()
        step = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
            step += 1
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
        train_loss = float(np.mean(ep_losses))
        val_loss, _, _ = _evaluate(model, val_loader, device, max_batches=max_steps_per_epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        log.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | time=%.1fs", ep, epochs, train_loss, val_loss, time.time() - t0)

    # final test evaluation
    test_loss, y_true, y_pred = _evaluate(model, test_loader, device, max_batches=max_steps_per_epoch)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # save weights
    weights_path = models_dir / "catsdogs_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
        },
        weights_path,
    )
    log.info("Saved model weights: %s", weights_path)

    # plots
    fig = plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="train")
    plt.plot(range(1, epochs + 1), val_losses, label="val")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    out_loss = artifacts_dir / "plots" / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(out_loss, dpi=200)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(np.array(cm), interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.xticks([0, 1], CLASSES)
    plt.yticks([0, 1], CLASSES)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    out_cm = artifacts_dir / "plots" / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_cm, dpi=200)
    plt.close(fig)

    # report text + metrics json
    report_txt = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    (artifacts_dir / "metrics" / "classification_report.txt").write_text(report_txt)

    metrics = TrainResult(
        epochs=epochs,
        train_loss=float(train_losses[-1]) if train_losses else math.nan,
        val_loss=float(val_losses[-1]) if val_losses else math.nan,
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        confusion_matrix=cm,
    )
    save_json(asdict(metrics), artifacts_dir / "metrics" / "metrics.json")

    # MLflow logging (optional)
    try:
        import mlflow
        import mlflow.pytorch

        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "cats-dogs-classification"))
        with mlflow.start_run(run_name=os.getenv("RUN_NAME", "baseline-cnn")):
            mlflow.log_params(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
                    "model": "SimpleCNN",
                    "seed": seed,
                }
            )
            mlflow.log_metrics(
                {
                    "test_loss": float(test_loss),
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                }
            )
            mlflow.log_artifact(str(out_loss))
            mlflow.log_artifact(str(out_cm))
            mlflow.log_artifact(str(artifacts_dir / "metrics" / "classification_report.txt"))
            mlflow.pytorch.log_model(model, artifact_path="model")
    except Exception as e:
        log.warning("MLflow logging skipped (mlflow not configured/available): %s", e)

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline Cats vs Dogs CNN.")
    parser.add_argument("--processed", default="data/processed", help="Processed dataset directory.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-steps", type=int, default=0, help="Limit batches per epoch (0 = full). Useful for CI/tests.")
    args = parser.parse_args(argv)

    setup_logging()
    paths = Paths(project_root())

    try:
        res = train(
            processed_dir=paths.project_root / args.processed,
            models_dir=paths.models_dir,
            artifacts_dir=paths.artifacts_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            max_steps_per_epoch=args.max_steps,
        )
        log.info("Training complete: %s", res)
        return 0
    except Exception as e:
        log.exception("Training failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
