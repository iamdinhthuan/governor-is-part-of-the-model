"""Counterbalance router and fixed-576 timing without reselecting a policy."""

import os
from pathlib import Path
import subprocess
import sys

for mode, tag in (
    ("router", "_r1"),
    ("fixed576", "_r1"),
    ("fixed576", "_r2"),
    ("router", "_r2"),
):
    subprocess.run(
        [
            sys.executable,
            "/tmp/jetson_routed_eval.py",
            "--mode", mode,
            "--image-ids", "/tmp/resolution-heldout-ids.json",
            "--tag", tag,
        ],
        env={**os.environ, "PYTHONPATH": "/tmp/yolo26-libs:/tmp"},
        timeout=360,
        check=True,
    )
    print(f"Timed {mode} {tag}", flush=True)
