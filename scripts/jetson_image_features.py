"""Measure cheap pre-inference image features for resolution selection.

All features come from raw image pixels. No detector output or annotation is
used by the router. The COCO JSON only supplies public image IDs and names.
"""

import json
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path("/tmp/yolo26-pilot-development")
NAMES = (
    "aspect_ratio",
    "gray_mean",
    "gray_std",
    "gray_p10",
    "gray_p90",
    "laplacian_mean",
    "laplacian_p90",
    "laplacian_high_density",
    "edge_density",
    "tile_std_mean",
    "tile_std_max",
    "entropy",
    "saturation_mean",
    "saturation_high_density",
)


def feature(bgr: np.ndarray) -> list[float]:
    height, width = bgr.shape[:2]
    small = cv2.resize(bgr, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edge = cv2.Canny(gray, 60, 120)
    tiles = gray.reshape(8, 20, 8, 20).transpose(0, 2, 1, 3)
    tile_std = tiles.std(axis=(2, 3))
    hist = np.histogram(gray, bins=16, range=(0, 256))[0].astype(np.float64)
    prob = hist[hist > 0] / hist.sum()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    values = [
        width / height,
        np.mean(gray),
        np.std(gray),
        np.percentile(gray, 10),
        np.percentile(gray, 90),
        np.mean(lap),
        np.percentile(lap, 90),
        np.mean(lap > 25),
        np.mean(edge > 0),
        np.mean(tile_std),
        np.max(tile_std),
        -np.sum(prob * np.log2(prob)),
        np.mean(saturation),
        np.mean(saturation > 100),
    ]
    return [float(item) for item in values]


def main() -> None:
    annotations = json.loads(Path("/tmp/instances_pilot_development.json").read_text())
    rows = {}
    times = []
    for image in annotations["images"]:
        bgr = cv2.imread(str(ROOT / "images" / image["file_name"]))
        if bgr is None:
            raise RuntimeError(f"Unreadable image {image['file_name']}")
        start = time.perf_counter()
        rows[str(image["id"])] = feature(bgr)
        times.append((time.perf_counter() - start) * 1000)
    if len(rows) != 1024:
        raise RuntimeError("Wrong development image count")
    summary = {
        "features": list(NAMES),
        "rows": rows,
        "median_ms_after_image_decode": float(np.median(times)),
        "p95_ms_after_image_decode": float(np.percentile(times, 95)),
        "no_ground_truth_or_detector_output": True,
    }
    path = ROOT / "preimage-features.json"
    path.write_text(json.dumps(summary, separators=(",", ":")))
    print(
        f"Measured {len(rows)} images, feature extraction median "
        f"{summary['median_ms_after_image_decode']:.3f} ms"
    )


if __name__ == "__main__":
    main()
