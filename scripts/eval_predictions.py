"""Evaluate Jetson predictions against a specified COCO-format split."""

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dataset", default="official COCO val2017")
    parser.add_argument("--image-ids-file", type=Path)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text())
    ground_truth = COCO(str(args.annotations))
    image_ids = set(ground_truth.getImgIds())
    category_ids = set(ground_truth.getCatIds())
    if not predictions:
        raise RuntimeError("Empty predictions")
    if not {p["image_id"] for p in predictions}.issubset(image_ids):
        raise RuntimeError("Prediction image ID outside the supplied COCO split")
    if not {p["category_id"] for p in predictions}.issubset(category_ids):
        raise RuntimeError("Prediction category ID outside official COCO")
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    if args.image_ids_file:
        subset_ids = json.loads(args.image_ids_file.read_text())
        if (
            not subset_ids
            or len(subset_ids) != len(set(subset_ids))
            or not set(subset_ids) <= image_ids
            or not {p["image_id"] for p in predictions} <= set(subset_ids)
        ):
            raise RuntimeError("Invalid COCO evaluation image ID subset")
        evaluator.params.imgIds = sorted(subset_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    summary = {
        "dataset": args.dataset,
        "evaluator": "pycocotools COCOeval bbox maxDets=100",
        "evaluated_images": len(evaluator.params.imgIds),
        "prediction_count": len(predictions),
        "ap50_95": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ap_small": float(evaluator.stats[3]),
        "ap_medium": float(evaluator.stats[4]),
        "ap_large": float(evaluator.stats[5]),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.summary}")


if __name__ == "__main__":
    main()
