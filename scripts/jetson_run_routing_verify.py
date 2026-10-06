"""Check frozen image router against fixed engines on verification images."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/tmp/yolo26-verification")
IDS = Path("/tmp/resolution-verification-ids.json")
ANNOTATIONS = Path("/tmp/instances_verification2017.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorized", action="store_true")
    args = parser.parse_args()
    tag = "_vec" if args.vectorized else ""
    for kind in ("fixed512", "fixed576", "fixed640", "router"):
        subprocess.run(
            [
                sys.executable,
                "/tmp/jetson_routed_eval.py",
                "--mode", kind,
                "--image-ids", str(IDS),
                "--annotations", str(ANNOTATIONS),
                "--images-dir", str(ROOT / "images"),
                "--output-dir", str(ROOT),
                "--expected-count", "1024",
                "--limit", "1024",
                "--tag", tag,
            ] + (["--vectorized"] if args.vectorized else []),
            env={**os.environ, "PYTHONPATH": "/tmp/yolo26-libs:/tmp"},
            timeout=750,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "/tmp/eval_predictions.py",
                "--predictions", str(ROOT / f"{kind}{tag}-heldout-predictions.json"),
                "--annotations", str(ANNOTATIONS),
                "--image-ids-file", str(IDS),
                "--dataset", "COCO train2017 unseen verification, 1024 images",
                "--summary", str(ROOT / f"{kind}{tag}-heldout-coco-ap.json"),
            ],
            env={**os.environ, "PYTHONPATH": "/tmp/yolo26-eval-libs"},
            timeout=180,
            check=True,
        )
        print(f"Verification completed for {kind}{tag}", flush=True)


if __name__ == "__main__":
    main()
