"""Run six fixed TensorRT variants on the disjoint COCO development split."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/tmp/yolo26-pilot-development")
ANNOTATIONS = Path("/tmp/instances_pilot_development.json")
ENGINES = {
    "float": "pilot_float_fp16_opt0.engine",
    "qat": "pilot_qat_fp16_int8_opt0.engine",
    "l1": "pilot_l1_fp16_opt0.engine",
    "object": "pilot_object_fp16_opt0.engine",
    "l1_qat": "pilot_l1_qat_opt0.engine",
    "object_qat": "pilot_object_qat_opt0.engine",
}
ENGINE_SIZES = {}


def main() -> None:
    for kind, engine in ENGINES.items():
        predictions = ROOT / f"{kind}-predictions.json"
        summary = ROOT / f"{kind}-coco-ap.json"
        subprocess.run(
            [
                sys.executable,
                "/tmp/jetson_eval.py",
                "--engine", str(Path("/tmp") / engine),
                "--annotations", str(ANNOTATIONS),
                "--images-dir", str(ROOT / "images"),
                "--expected-count", "1024",
                "--limit", "1024",
                "--imgsz", str(ENGINE_SIZES.get(kind, 640)),
                "--output", str(predictions),
            ],
            env={**os.environ, "PYTHONPATH": "/tmp/yolo26-libs"},
            check=True,
            timeout=420,
        )
        subprocess.run(
            [
                sys.executable,
                "/tmp/eval_predictions.py",
                "--predictions", str(predictions),
                "--annotations", str(ANNOTATIONS),
                "--dataset", "COCO train2017 pilot development, 1024 images",
                "--summary", str(summary),
            ],
            env={**os.environ, "PYTHONPATH": "/tmp/yolo26-eval-libs"},
            check=True,
            timeout=120,
        )
        print(f"Completed {kind} development engine/AP", flush=True)


if __name__ == "__main__":
    main()
