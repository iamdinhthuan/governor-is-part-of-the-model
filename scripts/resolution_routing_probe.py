"""Test if a low-resolution detector can *predict* when 640px is needed.

Use a disjoint 512-image fit / 512-image held-out development split.
GT-informed routing is diagnostic only; learned/heuristic routing uses
only 512px predictions at deployment. Report latency as an optimistic
two-pass cost estimate until an actual Jetson router is implemented.

Run: uv run --no-project --with pycocotools==2.0.10 \
    --with numpy==2.2.6 --with scikit-learn==1.7.2 \
    resolution_routing_probe.py
"""

import collections
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
NAMES = {"base640", "r576", "r512"}


def by_image(records: list[dict]) -> dict[int, list[dict]]:
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["image_id"]].append(record)
    return grouped


def load_predictions() -> dict[str, dict[int, list[dict]]]:
    result = {}
    with tarfile.open(ROOT / "artifacts/resolution-predictions.tgz", "r:gz") as tar:
        for name in NAMES:
            path = f"{name}-predictions.json"
            result[name] = by_image(json.load(tar.extractfile(path)))
    return result


def iou_box(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / max(
        1e-9, a[2] * a[3] + b[2] * b[3] - intersection
    )


def object_gain(
    ground_truth: list[dict], low: list[dict], high: list[dict]
) -> float:
    """Ground-truth-informed image score used as a diagnostic oracle ONLY."""

    def recall_value(rows: list[dict], annotation: dict) -> float:
        candidates = [
            row for row in rows
            if row["category_id"] == annotation["category_id"]
            and row["score"] >= 0.05
        ]
        best = max(
            (iou_box(annotation["bbox"], item["bbox"]) for item in candidates),
            default=0.0,
        )
        return float(best >= 0.5) + 0.5 * float(best >= 0.75)

    return sum(
        (2 if gt["area"] < 32 * 32 else 1)
        * (recall_value(high, gt) - recall_value(low, gt))
        for gt in ground_truth
        if not gt.get("iscrowd")
    )


def feature_vector(rows: list[dict], width: int, height: int) -> np.ndarray:
    """Cheap features available from one low-resolution inference only."""
    confident = [row for row in rows if row["score"] >= 0.05]
    scores = np.asarray([row["score"] for row in confident], dtype=np.float64)
    area = np.asarray(
        [row["bbox"][2] * row["bbox"][3] for row in confident],
        dtype=np.float64,
    )
    top = np.sort(scores)[-10:]
    return np.asarray(
        [
            math.log1p(len(confident)),
            sum(scores >= 0.25),
            sum(scores >= 0.5),
            sum((area < 32 * 32) & (scores >= 0.1)),
            sum((area < 48 * 48) & (scores >= 0.05)),
            sum((area < 32 * 32) & (scores >= 0.3)),
            sum(scores[(area < 32 * 32)]),
            float(np.mean(scores)) if len(scores) else 0,
            float(np.mean(top)) if len(top) else 0,
            float(np.std(scores)) if len(scores) else 0,
            float(np.median(area)) if len(area) else 0,
            float(width / height),
        ],
        dtype=np.float64,
    )


def official_ap(
    coco: COCO,
    records: dict[int, list[dict]],
    image_ids: list[int],
) -> dict[str, float]:
    predictions = [
        row for image_id in image_ids for row in records.get(image_id, [])
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = coco.loadRes(predictions)
        evaluator = COCOeval(coco, loaded, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "ap50_95": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap_small": float(evaluator.stats[3]),
    }


def main() -> None:
    coco = COCO(str(ROOT / "artifacts/instances_pilot_development.json"))
    all_ids = sorted(coco.getImgIds())
    ranked_ids = sorted(
        all_ids,
        key=lambda item: hashlib.sha256(f"router-seed-20261004:{item}".encode()).digest(),
    )
    fit_ids, test_ids = ranked_ids[:512], sorted(ranked_ids[512:])
    assert len(test_ids) == 512 and not set(fit_ids) & set(test_ids)
    predictions = load_predictions()
    for kind in NAMES:
        if not set(predictions[kind]) <= set(all_ids):
            raise RuntimeError(f"Prediction ID outside development split: {kind}")
    images = {item["id"]: item for item in coco.dataset["images"]}
    truth = by_image(coco.dataset["annotations"])
    gains = {
        image_id: object_gain(
            truth.get(image_id, []),
            predictions["r512"][image_id],
            predictions["base640"][image_id],
        )
        for image_id in all_ids
    }
    features = {
        image_id: feature_vector(
            predictions["r512"][image_id],
            images[image_id]["width"],
            images[image_id]["height"],
        )
        for image_id in all_ids
    }
    labels = np.asarray([gains[image_id] > 0 for image_id in fit_ids], dtype=int)
    if not 15 <= int(labels.sum()) <= len(labels) - 15:
        raise RuntimeError("Too few positive/negative training labels")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            random_state=20261004,
        ),
    )
    classifier.fit(
        np.stack([features[image_id] for image_id in fit_ids]), labels
    )
    probability = classifier.predict_proba(
        np.stack([features[image_id] for image_id in test_ids])
    )[:, 1]
    predicted = dict(zip(test_ids, probability.tolist()))
    small_box_proxy = {
        image_id: float(features[image_id][3]) for image_id in test_ids
    }
    random_score = {
        image_id: int.from_bytes(
            hashlib.sha256(f"random-control:{image_id}".encode()).digest()[:8],
            "big",
        )
        for image_id in test_ids
    }
    fixed = {
        name: official_ap(coco, predictions[name], test_ids)
        for name in ("r512", "r576", "base640")
    }
    latency = {
        name: json.loads(
            (ROOT / "results/resolution" / f"{name}-predictions.summary.json").read_text()
        )
        for name in ("r512", "r576", "base640")
    }
    results = {}
    for budget in (0.08, 0.16):
        n_selected = round(len(test_ids) * budget)
        options = {
            "gt_oracle_diagnostic": gains,
            "learned_low_resolution_only": predicted,
            "small_box_heuristic": small_box_proxy,
            "random_control": random_score,
        }
        for policy, score in options.items():
            # The test GT is read ONLY by gt_oracle_diagnostic. All other
            # rankings use low-resolution outputs or a fixed random hash.
            chosen = set(
                sorted(test_ids, key=lambda item: (-score[item], item))[:n_selected]
            )
            mixture = {
                image_id: (
                    predictions["base640"][image_id]
                    if image_id in chosen
                    else predictions["r512"][image_id]
                )
                for image_id in test_ids
            }
            ap = official_ap(coco, mixture, test_ids)
            actual_fraction = len(chosen) / len(test_ids)
            results[f"{policy}@{budget:.2f}"] = {
                **ap,
                "fallback_images": len(chosen),
                "fallback_fraction": actual_fraction,
                "estimated_host_inference_ms": (
                    latency["r512"]["host_input_inference_output_ms_median"]
                    + actual_fraction
                    * latency["base640"]["host_input_inference_output_ms_median"]
                ),
                "estimated_full_app_ms": (
                    latency["r512"]["decode_through_predictions_ms_median"]
                    + actual_fraction
                    * latency["base640"]["decode_through_predictions_ms_median"]
                ),
                "qualification": (
                    "uses held-out ground truth; not a deployable policy"
                    if policy == "gt_oracle_diagnostic"
                    else "low-resolution predictions only; optimistic cost estimate"
                ),
            }
    report = {
        "data": "COCO train2017 pilot development only",
        "fit_images": len(fit_ids),
        "heldout_images": len(test_ids),
        "fit_ids_sha256": hashlib.sha256(
            ",".join(map(str, sorted(fit_ids))).encode()
        ).hexdigest(),
        "heldout_ids_sha256": hashlib.sha256(
            ",".join(map(str, test_ids)).encode()
        ).hexdigest(),
        "fit_positive_images": int(labels.sum()),
        "fixed_heldout_ap": fixed,
        "fixed_full_split_timings_ms": latency,
        "policies": results,
        "limit": (
            "routing latency is an optimistic additive estimate; test-set GT "
            "is used only for an explicitly non-deployable diagnostic"
        ),
    }
    out = ROOT / "results/resolution-routing-probe.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Fit positives:", int(labels.sum()), "/", len(labels))
    print("Fixed held-out AP:", {name: round(value["ap50_95"], 5) for name, value in fixed.items()})
    for name, value in results.items():
        print(
            f"{name:39s} AP={value['ap50_95']:.5f} "
            f"small={value['ap_small']:.5f} "
            f"optimistic_app_ms={value['estimated_full_app_ms']:.2f}"
        )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
