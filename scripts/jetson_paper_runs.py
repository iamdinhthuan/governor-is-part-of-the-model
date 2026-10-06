"""Final Jetson measurement campaign on COCO val2017 (5,000 images).

Phases (run sequentially, never concurrently with other GPU work):
  idle        board power with no workload
  dvfs        trtexec GPU compute time vs inter-inference idle gap
  ladder      cumulative pipeline ablation, YOLO26n 640
  sched       sync primitive x CUDA device scheduling flag
  stream      camera-rate arrival (15/30/60 fps): latency, power
  resolution  512/576/640 engines in a naive vs optimized pipeline
  generality  YOLO26s and YOLOv10n
  ap          COCOeval once per distinct full-val2017 prediction file

Every timed run samples VDD_IN/Tj/GPU/CPU clocks in a separate process.
"locked" wraps a phase in jetson_clocks and restores the stored default
afterwards. Finished runs (report with "power") are skipped, so the campaign
can be resumed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

OUT = Path("/tmp/paper")
ANN = "/tmp/instances_val2017.json"
IMG = "/tmp/yolo26-coco-val/val2017"
TRT_ENV = {"PYTHONPATH": "/tmp/yolo26-libs:/tmp"}
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

ENGINES = {
    "y26n_640_float": "/tmp/yolo26n_84172_fp16_opt3.engine",
    "y26n_640_uint8": "/tmp/yolo26n_84172_uint8_fp16_opt3.engine",
    "y26n_640_ultra": "/tmp/ultra/yolo26n.engine",
    "y26n_512_float": "/tmp/engines/y26n_512_float_opt3.engine",
    "y26n_512_uint8": "/tmp/engines/y26n_512_uint8_opt3.engine",
    "y26n_576_float": "/tmp/engines/y26n_576_float_opt3.engine",
    "y26n_576_uint8": "/tmp/engines/y26n_576_uint8_opt3.engine",
    "y26s_640_float": "/tmp/engines/y26s_640_float_opt3.engine",
    "y26s_640_uint8": "/tmp/engines/y26s_640_uint8_opt3.engine",
    "y26s_640_ultra": "/tmp/ultra_s/yolo26s.engine",
    "v10n_640_float": "/tmp/engines/v10n_640_float_opt3.engine",
    "v10n_640_uint8": "/tmp/engines/v10n_640_uint8_opt3.engine",
    "v10n_640_ultra": "/tmp/ultra_v10/yolov10n.engine",
}

# Pipeline rungs as jetson_fast_eval flags; "ultra" is the Ultralytics predictor.
RUNGS = {
    "float_seq_stream": ("float", ["--stream-sync"]),
    "uint8_seq_stream": ("uint8", ["--stream-sync"]),
    "uint8_seq_event": ("uint8", []),
    "uint8_prefetch_stream": ("uint8", ["--prefetch", "--stream-sync"]),
    "uint8_prefetch_event": ("uint8", ["--prefetch"]),
    "uint8_overlap_event": ("uint8", ["--prefetch", "--overlap"]),
    "uint8_overlap_event_w2": ("uint8", ["--prefetch", "--overlap", "--workers", "2"]),
    "float_overlap_event_w2": ("float", ["--prefetch", "--overlap", "--workers", "2"]),
    "uint8_adaptive_event_w2": ("uint8", ["--prefetch", "--overlap", "--adaptive",
                                          "--workers", "2"]),
    "uint8_overlap_event_w2_nogc": ("uint8", ["--prefetch", "--overlap", "--workers", "2",
                                              "--no-gc"]),
}


def torch_env() -> dict:
    env = {}
    for line in Path("/tmp/yolo26-torch-env.sh").read_text().split(";"):
        key, value = line.strip().removeprefix("export ").split("=", 1)
        env[key] = value
    env["PYTHONPATH"] += ":/tmp"
    env["YOLO_OFFLINE"] = "true"
    env["YOLO_AUTOINSTALL"] = "false"
    return env


def spec(model: str, rung: str, limit: int = 0, fps: float = 0.0, extra=()) -> dict:
    """Return {name, model, rung, limit, fps, extra}; built into a command per run."""
    return {"model": model, "rung": rung, "limit": limit, "fps": fps, "extra": list(extra)}


def command_for(item: dict, run_dir: Path):
    report, predictions = run_dir / "report.json", run_dir / "predictions.json"
    common = ["--annotations", ANN, "--images-dir", IMG,
              "--predictions", str(predictions), "--output", str(report)]
    if item["limit"]:
        common += ["--limit", str(item["limit"])]
    if item["fps"]:
        common += ["--fps", str(item["fps"])]
    if item["rung"] == "ultra":
        command = [sys.executable, "/tmp/jetson_ultralytics_baseline.py",
                   "--engine", ENGINES[f"{item['model']}_ultra"], *common]
        return command, torch_env(), report, predictions
    kind, flags = RUNGS[item["rung"]]
    command = [sys.executable, "/tmp/jetson_fast_eval.py",
               "--engine", ENGINES[f"{item['model']}_{kind}"], "--memory", "pinned",
               *flags, *item["extra"], *common]
    return command, TRT_ENV, report, predictions


class Sampler:
    def __init__(self, path: Path):
        self.process = subprocess.Popen(
            [sys.executable, "/tmp/jetson_power.py", "sample", "--out", str(path)])

    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=10)


def integrate(samples: Path, report: Path) -> None:
    subprocess.run([sys.executable, "/tmp/jetson_power.py", "integrate",
                    "--samples", str(samples), "--report", str(report)], check=True)


def done(report: Path) -> bool:
    return report.exists() and "power" in json.loads(report.read_text())


def run_one(phase: str, clocks: str, name: str, repeat: int, item: dict) -> None:
    run_dir = OUT / phase / clocks / name / f"r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command, extra_env, report, predictions = command_for(item, run_dir)
    if done(report):
        return
    (run_dir / "spec.json").write_text(json.dumps({**item, "command": command}, indent=2))
    sampler = Sampler(run_dir / "power.csv")
    try:
        result = subprocess.run(command, env={**os.environ, **extra_env},
                                timeout=3600, capture_output=True, text=True)
    finally:
        sampler.stop()
    (run_dir / "stdout.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{phase}/{clocks}/{name}/r{repeat} failed:\n{result.stderr[-3000:]}")
    integrate(run_dir / "power.csv", report)
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    (run_dir / "predictions.sha256").write_text(digest + "\n")
    data = json.loads(report.read_text())
    print(f"{time.strftime('%T')} {phase} {clocks} {name} r{repeat} "
          f"{data['throughput_images_per_s']:.1f} img/s "
          f"{data['power']['energy_per_image_mj']:.1f} mJ "
          f"{data['power']['mean_power_w']:.2f} W "
          f"GPU {data['power']['gpu_mhz_mean']:.0f} MHz "
          f"Tj {data['power']['tj_max_c']:.1f} C pred {digest[:12]}", flush=True)
    time.sleep(10)


def matrix(phase: str, clocks: str, items: dict, repeats: int) -> None:
    names = list(items)
    for repeat in range(repeats):
        shift = repeat % len(names)
        for name in names[shift:] + names[:shift]:
            run_one(phase, clocks, name, repeat, items[name])


def warm_cache() -> None:
    subprocess.run(f"cat {IMG}/*.jpg > /dev/null", shell=True, check=True)


def with_clocks(clocks: str, body) -> None:
    if clocks == "default":
        body()
        return
    # jetson_clocks --store refuses to overwrite an existing file.
    store = Path(f"/tmp/jetson_clocks_default_{time.strftime('%Y%m%d_%H%M%S')}.conf")
    subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks", "--store", str(store)], check=True)
    try:
        subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks"], check=True)
        time.sleep(5)
        body()
    finally:
        subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", str(store)],
                       check=True)
        print("Restored default clocks", flush=True)
        time.sleep(15)


def phase_idle(clocks: str, repeats: int) -> None:
    for repeat in range(repeats):
        run_dir = OUT / "idle" / clocks / f"r{repeat}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report = run_dir / "report.json"
        if done(report):
            continue
        sampler = Sampler(run_dir / "power.csv")
        start = time.time()
        time.sleep(70)
        sampler.stop()
        report.write_text(json.dumps({"window_unix": [start + 5, start + 65],
                                      "images_timed": 1}))
        integrate(run_dir / "power.csv", report)
        print(f"idle {clocks} r{repeat} done", flush=True)


STAT = re.compile(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, "
                  r"mean = ([\d.]+) ms, median = ([\d.]+) ms")
QPS = re.compile(r"Throughput: ([\d.]+) qps")


def phase_dvfs(clocks: str, repeats: int) -> None:
    for model in ("y26n_640_uint8", "y26s_640_uint8"):
        for repeat in range(repeats):
            for idle_ms in (0, 1, 2, 4, 8, 16, 33):
                run_dir = OUT / "dvfs" / clocks / model / f"idle{idle_ms}" / f"r{repeat}"
                run_dir.mkdir(parents=True, exist_ok=True)
                report = run_dir / "report.json"
                if done(report):
                    continue
                sampler = Sampler(run_dir / "power.csv")
                start = time.time()
                try:
                    result = subprocess.run(
                        [TRTEXEC, f"--loadEngine={ENGINES[model]}", "--warmUp=3000",
                         "--duration=15", f"--idleTime={idle_ms}", "--avgRuns=1"],
                        capture_output=True, text=True, timeout=300, check=True)
                finally:
                    end = time.time()
                    sampler.stop()
                (run_dir / "trtexec.log").write_text(result.stdout)
                stat, qps = STAT.search(result.stdout), QPS.search(result.stdout)
                minimum, maximum, mean, median = map(float, stat.groups())
                # Skip engine load/warm-up; trtexec's 15 s timed window ends the run.
                window = [end - 15.5, end - 0.5]
                report.write_text(json.dumps({
                    "model": model, "idle_ms": idle_ms, "clocks": clocks,
                    "gpu_compute_ms": {"min": minimum, "max": maximum,
                                       "mean": mean, "median": median},
                    "qps": float(qps.group(1)),
                    "images_timed": int(float(qps.group(1)) * 15),
                    "window_unix": window,
                }))
                integrate(run_dir / "power.csv", report)
                print(f"dvfs {clocks} {model} idle {idle_ms} r{repeat}: median "
                      f"{median} ms, {float(qps.group(1)):.1f} qps", flush=True)
                time.sleep(5)


LADDER = {name: spec("y26n_640", name) for name in RUNGS}
LADDER = {"ultralytics": spec("y26n_640", "ultra"), **LADDER}

SCHED = {
    f"{mode}_{sync}_{sched}": spec(
        "y26n_640", f"uint8_{mode}_{sync}", limit=2000, extra=["--sched", sched])
    for mode in ("seq", "prefetch") for sync in ("event", "stream")
    for sched in ("auto", "spin", "blocking")
}

STREAM_PIPES = ["ultra", "float_seq_stream", "uint8_seq_event",
                "uint8_prefetch_event", "uint8_overlap_event_w2", "uint8_adaptive_event_w2"]
STREAM = {
    f"{pipe}_fps{fps}": spec("y26n_640", pipe, limit=1000, fps=fps)
    for fps in (15, 30, 60) for pipe in STREAM_PIPES
}

RESOLUTION = {
    f"{size}_{pipe}": spec(f"y26n_{size}", pipe)
    for size in (512, 576, 640) for pipe in ("float_seq_stream", "uint8_overlap_event_w2")
}

GENERALITY = {
    f"{model}_{pipe}": spec(f"{model}_640", pipe)
    for model in ("y26s", "v10n") for pipe in ("ultra", "float_seq_stream", "uint8_overlap_event_w2")
}

MATRICES = {"ladder": LADDER, "sched": SCHED, "stream": STREAM,
            "resolution": RESOLUTION, "generality": GENERALITY}


def phase_ap() -> None:
    target = OUT / "ap"
    target.mkdir(parents=True, exist_ok=True)
    index_path = target / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for sha_file in sorted(OUT.glob("*/*/*/r*/predictions.sha256")):
        run_dir = sha_file.parent
        spec_data = json.loads((run_dir / "spec.json").read_text())
        if spec_data["limit"]:
            continue
        digest = sha_file.read_text().strip()
        index.setdefault(digest, {"runs": []})
        key = str(run_dir.relative_to(OUT))
        if key not in index[digest]["runs"]:
            index[digest]["runs"].append(key)
        summary = target / f"{digest[:16]}.json"
        if not summary.exists():
            subprocess.run([
                sys.executable, "/tmp/eval_predictions.py",
                "--predictions", str(run_dir / "predictions.json"),
                "--annotations", ANN, "--summary", str(summary),
            ], env={**os.environ, "PYTHONPATH": "/tmp/yolo26-eval-libs:/tmp/yolo26-libs:/tmp"},
                check=True, capture_output=True)
            print("AP", key, json.loads(summary.read_text())["ap50_95"], flush=True)
        index[digest]["summary"] = summary.name
        index_path.write_text(json.dumps(index, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", required=True)
    parser.add_argument("--clocks", nargs="+", default=["default", "locked"])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    for phase in args.phases:
        if phase == "ap":
            phase_ap()
            continue
        if phase in MATRICES:
            warm_cache()
        for clocks in args.clocks:
            print(f"=== {phase} {clocks} {time.strftime('%T')}", flush=True)
            if phase == "idle":
                with_clocks(clocks, lambda: phase_idle(clocks, args.repeats))
            elif phase == "dvfs":
                with_clocks(clocks, lambda: phase_dvfs(clocks, args.repeats))
            else:
                with_clocks(clocks, lambda: matrix(phase, clocks, MATRICES[phase],
                                                   args.repeats))
    print("CAMPAIGN_PHASES_DONE", " ".join(args.phases), flush=True)


if __name__ == "__main__":
    main()
