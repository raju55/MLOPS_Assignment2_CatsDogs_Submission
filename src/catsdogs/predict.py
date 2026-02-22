"""
catsdogs.predict

Utilities for loading a saved model and running inference on a single image.

Used by:
- FastAPI service (api/main.py)
- Unit tests
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image

from .config import CLASSES, IMAGE_SIZE
from .model import SimpleCNN

log = logging.getLogger(__name__)


def load_model(weights_path: Path, device: str | None = None) -> Tuple[SimpleCNN, torch.device]:
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(dev)
    ckpt = torch.load(weights_path, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, dev


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Preprocess PIL image into a NCHW float32 tensor in [0, 1].

    We avoid torch.from_numpy / Tensor.numpy() to keep inference robust on
    Windows where torch's NumPy bridge may fail when NumPy 2.x is installed.
    """
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE, resample=Image.BILINEAR)
    w, h = img.size
    raw = img.tobytes()
    if hasattr(torch, "frombuffer"):
        x = torch.frombuffer(raw, dtype=torch.uint8)
    else:
        x = torch.ByteTensor(torch.ByteStorage.from_buffer(raw))
    x = x.view(h, w, 3).permute(2, 0, 1).contiguous()  # CHW
    x = x.to(dtype=torch.float32).div_(255.0)
    return x.unsqueeze(0)  # NCHW


def predict_image(model: torch.nn.Module, device: torch.device, img: Image.Image) -> Dict[str, float | str]:
    x = preprocess_image(img).to(device)
    with torch.no_grad():
        logits = model(x)
        prob_dog = float(torch.sigmoid(logits).detach().cpu().view(-1)[0].item())
    prob_cat = 1.0 - prob_dog
    label = "Dog" if prob_dog >= 0.5 else "Cat"
    return {"label": label, "prob_cat": prob_cat, "prob_dog": prob_dog}
