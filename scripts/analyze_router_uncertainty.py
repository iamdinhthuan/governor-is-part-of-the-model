"""Check held-out routing gain across two halves and paired object recall.

Object recall is a *diagnostic*, not an AP confidence interval. Image-level
bootstrap resamples entire images; official AP is measured separately on
two pre-fixed halves of the held-out split.

Run: uv run --no-project --with pycocotools==2.0.10 \
    --with numpy==2.2.6 analyze_router_uncertainty.py
"""

import argparse
import collections
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "artifacts/router-heldout-predictions.tgz"
ANNOTATIONS = ROOT / "artifacts/instances_pilot_development.json"
IDS = ROOT / "artifacts/resolution-heldout-ids.json"


def iou_box(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / max(
        1e-9,
        a[2] * a[3] + b[2] * b[3] - intersection,
    )


def records(archive: tarfile.TarFile, name: str) -> dict[int, list[dict]]:
    grouped = collections.defaultdict(list)
    for row in json.load(archive.extractfile(f"{name}-heldout-predictions.json")):
        grouped[row["image_id"]].append(row)
    return grouped


def score_ap(coco: COCO, preds: dict, ids: list[int]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        result = coco.loadRes(
            [row for image_id in ids for row in preds.get(image_id, [])]
        )
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.imgIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "ap50_95": float(evaluator.stats[0]),
        "ap_small": float(evaluator.stats[3]),
    }


def recall_counts(
    objects: list[dict], detections: list[dict], small_only: bool
) -> tuple[int, int, int]:
    if small_only:
        objects = [item for item in objects if item["area"] < 32 * 32]
    objects = [item for item in objects if not item.get("iscrowd")]
    good50 = good75 = 0
    for item in objects:
        best = max(
            (
                iou_box(item["bbox"], pred["bbox"])
                for pred in detections
                if pred["category_id"] == item["category_id"]
                and pred["score"] >= 0.05
            ),
            default=0.0,
        )
        good50 += int(best >= 0.5)
        good75 += int(best >= 0.75)
    return good50, good75, len(objects)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-ids", type=Path, default=IDS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/router-uncertainty.json"
    )
    parser.add_argument("--expected-count", type=int, default=512)
    parser.add_argument("--label", default="pilot held-out development")
    args = parser.parse_args()
    coco = COCO(str(args.annotations))
    ids = json.loads(args.image_ids.read_text())
    if len(ids) != args.expected_count or len(set(ids)) != args.expected_count:
        raise RuntimeError("Wrong held-out image split")
    ranked = sorted(
        ids, key=lambda item: hashlib.sha256(f"router-uncertainty:{item}".encode()).digest()
    )
    half = len(ids) // 2
    halves = (sorted(ranked[:half]), sorted(ranked[half:]))
    with tarfile.open(args.archive, "r:gz") as archive:
        preds = {
            kind: records(archive, kind)
            for kind in ("fixed512", "fixed576", "fixed640", "router")
        }
    reference = "fixed576"
    halves_result = []
    for half in halves:
        summaries = {
            kind: score_ap(coco, preds[kind], half)
            for kind in (reference, "router", "fixed640")
        }
        halves_result.append(
            {
                "images": len(half),
                "image_ids_sha256": hashlib.sha256(
                    ",".join(map(str, half)).encode()
                ).hexdigest(),
                "ap": summaries,
                "router_minus_576_ap":
                    summaries["router"]["ap50_95"] - summaries[reference]["ap50_95"],
            }
        )
    truth = collections.defaultdict(list)
    for item in coco.dataset["annotations"]:
        truth[item["image_id"]].append(item)
    recall = {}
    rng = random.Random(20261004)
    for small_only in (False, True):
        per_image = {
            kind: np.asarray(
                [
                    recall_counts(
                        truth[image_id],
                        preds[kind][image_id],
                        small_only,
                    )
                    for image_id in ids
                ],
                dtype=np.int64,
            )
            for kind in (reference, "router")
        }
        if not np.array_equal(
            per_image[reference][:, 2], per_image["router"][:, 2]
        ):
            raise RuntimeError("Object denominators differ")
        weights = per_image[reference][:, 2]
        if not weights.sum():
            raise RuntimeError("No GT objects in bootstrap")
        metrics = {}
        for col, threshold in ((0, "recall50"), (1, "recall75")):
            raw_delta = (
                per_image["router"][:, col].sum()
                - per_image[reference][:, col].sum()
            ) / weights.sum()
            bootstrap = []
            for _ in range(5000):
                sample = np.asarray(
                    [rng.randrange(len(ids)) for _ in ids],
                    dtype=np.int64,
                )
                denominator = weights[sample].sum()
                if denominator:
                    bootstrap.append(
                        (
                            per_image["router"][sample, col].sum()
                            - per_image[reference][sample, col].sum()
                        ) / denominator
                    )
            metrics[threshold] = {
                "router_minus_576": float(raw_delta),
                "cluster_bootstrap_95pct": [
                    float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5)),
                ],
                "objects": int(weights.sum()),
            }
        recall["small" if small_only else "all"] = metrics
    report = {
        "scope": args.label,
        "official_ap_on_two_prefixed_halves": halves_result,
        "paired_object_recall_diagnostic_not_ap_confidence_interval": recall,
        "bootstrap_replicates": 5000,
    }
    out = args.output
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
