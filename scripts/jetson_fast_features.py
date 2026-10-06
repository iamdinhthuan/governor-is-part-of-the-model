"""Low-overhead image-only routing features, measured on Jetson."""

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
    "bright_fraction",
    "dark_fraction",
    "laplacian_mean",
    "laplacian_high_fraction",
    "edge_fraction",
    "tile_std_mean",
    "tile_std_max",
    "entropy",
    "saturation_mean",
)


def feature(bgr: np.ndarray) -> list[float]:
    height, width = bgr.shape[:2]
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edge = cv2.Canny(gray, 60, 120)
    tile = gray.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3)
    tile_std = tile.std(axis=(2, 3))
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).ravel()
    prob = hist[hist > 0] / hist.sum()
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    values = [
        width / height,
        gray.mean(),
        gray.std(),
        np.mean(gray > 200),
        np.mean(gray < 55),
        lap.mean(),
        np.mean(lap > 25),
        np.mean(edge > 0),
        tile_std.mean(),
        tile_std.max(),
        -np.sum(prob * np.log2(prob)),
        sat.mean(),
    ]
    return [float(value) for value in values]


def main() -> None:
    annotations = json.loads(Path("/tmp/instances_pilot_development.json").read_text())
    rows = {}
    durations = []
    for image in annotations["images"]:
        bgr = cv2.imread(str(ROOT / "images" / image["file_name"]))
        if bgr is None:
            raise RuntimeError(f"Unreadable {image['file_name']}")
        start = time.perf_counter()
        rows[str(image["id"])] = feature(bgr)
        durations.append((time.perf_counter() - start) * 1000)
    summary = {
        "features": list(NAMES),
        "rows": rows,
        "median_ms_after_image_decode": float(np.median(durations)),
        "p95_ms_after_image_decode": float(np.percentile(durations, 95)),
        "no_ground_truth_or_detector_output": True,
    }
    target = ROOT / "preimage-features-fast.json"
    target.write_text(json.dumps(summary, separators=(",", ":")))
    print(
        f"Measured {len(rows)} images; Jetson feature median "
        f"{summary['median_ms_after_image_decode']:.3f} ms"
    )


if __name__ == "__main__":
    main()
