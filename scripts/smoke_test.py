"""
scripts/smoke_test.py

Post-deploy smoke test required by Assignment-2 (M4):
- Calls /health (should return status=ok)
- Calls /predict with a small image upload (should return probabilities + label)

Usage:
  python scripts/smoke_test.py --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import io
import sys
import time

import requests
from PIL import Image


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--timeout-s", type=int, default=60)
    args = p.parse_args(argv)

    base = args.base_url.rstrip("/")

    # 1) health wait
    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(2)

    r = requests.get(f"{base}/health", timeout=5)
    if r.status_code != 200:
        print("Health check failed:", r.status_code, r.text)
        return 1

    # 2) predict
    img = Image.new("RGB", (224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("smoke.jpg", buf.getvalue(), "image/jpeg")}
    pr = requests.post(f"{base}/predict", files=files, timeout=20)
    if pr.status_code != 200:
        print("Predict failed:", pr.status_code, pr.text)
        return 1

    data = pr.json()
    for k in ["label", "prob_cat", "prob_dog", "inference_ms"]:
        if k not in data:
            print("Predict response missing key:", k, data)
            return 1

    print("Smoke test passed:", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
