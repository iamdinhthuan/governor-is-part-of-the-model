"""Latency/throughput/energy matrix on the blind 1,024-image Jetson split.

Pipelines: our original application, the Ultralytics TensorRT predictor, and
the uint8 opt3 application sequential and with decode prefetch. Each runs
--repeats times in a rotated order with VDD_IN sampled in a separate process.
--clocks locked wraps the whole matrix in jetson_clocks and restores the
stored default afterwards, even on failure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path("/tmp/yolo26-energy")
ANN = "/tmp/instances_verification2017.json"
IMG = "/tmp/yolo26-verification/images"
IDS = "/tmp/resolution-verification-ids.json"
TRT_ENV = {"PYTHONPATH": "/tmp/yolo26-libs:/tmp"}


def torch_env() -> dict:
    env = {}
    for line in Path("/tmp/yolo26-torch-env.sh").read_text().split(";"):
        key, value = line.strip().removeprefix("export ").split("=", 1)
        env[key] = value
    env["PYTHONPATH"] += ":/tmp"
    env["YOLO_OFFLINE"] = "true"
    env["YOLO_AUTOINSTALL"] = "false"
    return env


def pipelines(run_dir: Path) -> dict:
    fast = [
        sys.executable, "/tmp/jetson_fast_eval.py",
        "--engine", "/tmp/yolo26n_84172_uint8_fp16_opt3.engine",
        "--annotations", ANN, "--images-dir", IMG, "--memory", "pinned",
    ]
    return {
        "original_app": ([
            sys.executable, "/tmp/jetson_routed_eval.py", "--mode", "fixed640",
            "--vectorized", "--image-ids", IDS, "--annotations", ANN,
            "--images-dir", IMG, "--output-dir", str(run_dir),
            "--expected-count", "1024", "--limit", "1024",
        ], TRT_ENV, run_dir / "fixed640-heldout-summary.json",
            run_dir / "fixed640-heldout-predictions.json"),
        "ultralytics": ([
            sys.executable, "/tmp/jetson_ultralytics_baseline.py",
            "--engine", "/tmp/ultra/yolo26n.engine", "--annotations", ANN,
            "--images-dir", IMG, "--predictions", str(run_dir / "predictions.json"),
            "--output", str(run_dir / "report.json"),
        ], torch_env(), run_dir / "report.json", run_dir / "predictions.json"),
        "uint8_sequential": (fast + [
            "--predictions", str(run_dir / "predictions.json"),
            "--output", str(run_dir / "report.json"),
        ], TRT_ENV, run_dir / "report.json", run_dir / "predictions.json"),
        "uint8_prefetch": (fast + [
            "--prefetch", "--workers", "1",
            "--predictions", str(run_dir / "predictions.json"),
            "--output", str(run_dir / "report.json"),
        ], TRT_ENV, run_dir / "report.json", run_dir / "predictions.json"),
    }


def run_matrix(clocks: str, repeats: int) -> None:
    names = list(pipelines(OUT).keys())
    for repeat in range(repeats):
        order = names[repeat % len(names):] + names[: repeat % len(names)]
        for name in order:
            run_dir = OUT / clocks / name / f"r{repeat}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command, extra_env, report, predictions = pipelines(run_dir)[name]
            samples = run_dir / "power.csv"
            sampler = subprocess.Popen([
                sys.executable, "/tmp/jetson_power.py", "sample", "--out", str(samples),
            ])
            try:
                subprocess.run(
                    command, env={**os.environ, **extra_env}, check=True, timeout=900
                )
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
            subprocess.run([
                sys.executable, "/tmp/jetson_power.py", "integrate",
                "--samples", str(samples), "--report", str(report),
            ], check=True)
            digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
            (run_dir / "predictions.sha256").write_text(digest + "\n")
            print(f"{clocks} {name} r{repeat} predictions {digest[:16]}", flush=True)
            time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clocks", choices=("default", "locked"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.clocks == "default":
        run_matrix("default", args.repeats)
        return
    store = Path("/tmp/jetson_clocks_default.conf")
    subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks", "--store", str(store)], check=True)
    try:
        subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks"], check=True)
        time.sleep(5)
        run_matrix("locked", args.repeats)
    finally:
        subprocess.run(
            ["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", str(store)], check=True
        )
        print("Restored default clocks from", store, flush=True)


if __name__ == "__main__":
    main()
