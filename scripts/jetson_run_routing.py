"""Score matched 512-image Jetson controls and the pre-inference router."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/tmp/yolo26-pilot-development")
IDS = Path("/tmp/resolution-heldout-ids.json")
ANNOTATIONS = Path("/tmp/instances_pilot_development.json")


def main() -> None:
    for kind in ("fixed512", "fixed576", "fixed640", "router"):
        subprocess.run(
            [
                sys.executable,
                "/tmp/jetson_routed_eval.py",
                "--mode", kind,
                "--image-ids", str(IDS),
            ],
            env={**os.environ, "PYTHONPATH": "/tmp/yolo26-libs:/tmp"},
            timeout=360,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "/tmp/eval_predictions.py",
                "--predictions", str(ROOT / f"{kind}-heldout-predictions.json"),
                "--annotations", str(ANNOTATIONS),
                "--image-ids-file", str(IDS),
                "--dataset", "COCO train2017 router heldout development, 512 images",
                "--summary", str(ROOT / f"{kind}-heldout-coco-ap.json"),
            ],
            env={**os.environ, "PYTHONPATH": "/tmp/yolo26-eval-libs"},
            timeout=120,
            check=True,
        )
        print(f"Completed Jetson {kind} held-out evaluation", flush=True)


if __name__ == "__main__":
    main()
