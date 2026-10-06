"""Summarize deterministic ONNX Runtime vs Jetson TensorRT output drift."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def compare(key: str, prefix: str) -> dict:
    ort = np.asarray(
        json.loads((RESULTS / f"ort-{prefix}{key}-numeric.json").read_text())["values"],
        dtype=np.float64,
    ).reshape(300, 6)
    trt_values = json.loads((RESULTS / f"trt-{prefix}{key}-numeric.json").read_text())
    assert len(trt_values) == 1 and trt_values[0]["dimensions"] == "1x300x6"
    trt = np.asarray(trt_values[0]["values"], dtype=np.float64).reshape(300, 6)
    if not np.isfinite(ort).all() or not np.isfinite(trt).all():
        raise RuntimeError(f"Non-finite output for {key}")
    # TopK can reorder very low-scoring tied proposals. Inspect the ranked
    # first 20 separately from all 300 output rows.
    top = slice(0, 20)
    a, b = ort[:50], trt[:50]
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(
        a[:, 3] - a[:, 1], 0
    )
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(
        b[:, 3] - b[:, 1], 0
    )
    iou = inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)
    cost = (
        5 * (1 - iou)
        + np.abs(a[:, None, 4] - b[None, :, 4])
        + 10 * (a[:, None, 5] != b[None, :, 5])
    )
    row, col = linear_sum_assignment(cost)
    high = a[row, 4] >= 0.25
    same_class = a[row, 5] == b[col, 5]
    matched_iou = iou[row, col]
    return {
        "pair": key,
        "max_abs_all": float(np.max(np.abs(ort - trt))),
        "median_abs_all": float(np.median(np.abs(ort - trt))),
        "top20_class_agreement": float(np.mean(ort[top, 5] == trt[top, 5])),
        "top20_box_mae_px": float(np.mean(np.abs(ort[top, :4] - trt[top, :4]))),
        "top20_score_mae": float(np.mean(np.abs(ort[top, 4] - trt[top, 4]))),
        "ort_top_score": float(ort[0, 4]),
        "trt_top_score": float(trt[0, 4]),
        "top50_matched_class_agreement": float(np.mean(same_class)),
        "top50_matched_iou_mean": float(np.mean(matched_iou)),
        "ort_top50_above_025_count": int(np.sum(high)),
        "high_score_matched_class_and_iou50": (
            float(np.mean(same_class[high] & (matched_iou[high] >= 0.5)))
            if np.any(high)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="synthetic")
    args = parser.parse_args()
    prefix = "" if args.tag == "synthetic" else f"{args.tag}-"
    results = [compare(key, prefix) for key in ("float_8497", "qat_84172")]
    out = RESULTS / f"{prefix}numeric-agreement.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
