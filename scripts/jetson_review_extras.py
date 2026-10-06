"""Follow-up measurements requested in review.

  load        measured GPU load (devfreq load node) per ladder rung, 2000 images
  hysteresis  sequential pipeline started with clocks locked; governor restored after 10 s
  sampler     throughput with and without the power sampler running
  workers     full overlap pipeline with 1..5 decode workers, all 5000 images
  uclamp      sequential stream-sync pipeline pinned to CPU 0 with uclamp.min = 1024

Output: /tmp/paper/<phase>/default/<config>/r<k>/ (same layout as jetson_paper_runs).
Every run logs GPU load, GPU clock and the CPU 0-3 domain clock at 20 Hz.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/tmp")
import jetson_paper_runs as runs  # noqa: E402

GPU_LOAD = Path("/sys/devices/platform/bus@0/17000000.gpu/load")
GPU_FREQ = Path("/sys/class/devfreq/17000000.gpu/cur_freq")
POLICY0 = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")


def log_counters(path: Path, stop: threading.Event) -> None:
    with path.open("w") as out:
        while not stop.is_set():
            out.write(f"{time.time():.4f},{GPU_LOAD.read_text().strip()},"
                      f"{GPU_FREQ.read_text().strip()},{POLICY0.read_text().strip()}\n")
            time.sleep(0.05)


def eval_command(rung: str, run_dir: Path, limit: int, extra=()) -> list:
    kind, flags = runs.RUNGS[rung]
    return [sys.executable, "/tmp/jetson_fast_eval.py",
            "--engine", runs.ENGINES[f"y26n_640_{kind}"], "--memory", "pinned",
            *flags, *extra, *(["--limit", str(limit)] if limit else []),
            "--annotations", runs.ANN, "--images-dir", runs.IMG,
            "--predictions", str(run_dir / "predictions.json"),
            "--output", str(run_dir / "report.json")]


def run(phase: str, name: str, repeat: int, command: list, sample_power: bool = True,
        during=None) -> None:
    run_dir = runs.OUT / phase / "default" / name / f"r{repeat}"
    report = run_dir / "report.json"
    if report.exists() and (not sample_power or runs.done(report)):
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [c.replace("{run}", str(run_dir)) for c in command]
    (run_dir / "spec.json").write_text(json.dumps(
        {"phase": phase, "condition": name, "sample_power": sample_power,
         "command": command}, indent=2))
    stop = threading.Event()
    logger = threading.Thread(target=log_counters, args=(run_dir / "counters.csv", stop))
    logger.start()
    sampler = runs.Sampler(run_dir / "power.csv") if sample_power else None
    started = time.time()
    try:
        process = subprocess.Popen(command, env={**os.environ, **runs.TRT_ENV},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if during:
            during(started)
        output, _ = process.communicate(timeout=1800)
    finally:
        if sampler:
            sampler.stop()
        stop.set()
        logger.join()
    (run_dir / "stdout.log").write_text(output)
    if process.returncode:
        raise RuntimeError(f"{phase}/{name}/r{repeat} failed:\n{output[-3000:]}")
    if sample_power:
        runs.integrate(run_dir / "power.csv", report)
    data = json.loads(report.read_text())
    start, end = data["window_unix"]
    rows = [line.split(",") for line in (run_dir / "counters.csv").read_text().splitlines()]
    rows = [r for r in rows if start <= float(r[0]) <= end]
    data["gpu_load_mean"] = sum(int(r[1]) for r in rows) / len(rows) / 1000
    data["gpu_mhz_counter_mean"] = sum(int(r[2]) for r in rows) / len(rows) / 1e6
    data["policy0_mhz_mean"] = sum(int(r[3]) for r in rows) / len(rows) / 1000
    report.write_text(json.dumps(data, indent=2))
    digest = subprocess.run(["sha256sum", str(run_dir / "predictions.json")],
                            capture_output=True, text=True).stdout.split()[0]
    (run_dir / "predictions.sha256").write_text(digest + "\n")
    print(f"{time.strftime('%T')} {phase} {name} r{repeat} "
          f"{data['throughput_images_per_s']:.1f} img/s load {data['gpu_load_mean']:.2f} "
          f"GPU {data['gpu_mhz_counter_mean']:.0f} MHz cpu0 {data['policy0_mhz_mean']:.0f} MHz "
          f"pred {digest[:12]}", flush=True)
    time.sleep(10)


def phase_load() -> None:
    for rung in ("uint8_seq_stream", "uint8_seq_event", "uint8_prefetch_stream",
                 "uint8_prefetch_event", "uint8_overlap_event", "uint8_overlap_event_w2",
                 "float_seq_stream", "float_overlap_event_w2"):
        run("load", rung, 0, eval_command(rung, Path("{run}"), 2000))


def phase_hysteresis() -> None:
    store = f"/tmp/jetson_clocks_hyst_{int(time.time())}.conf"
    subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks", "--store", store], check=True)

    def release(started: float) -> None:
        time.sleep(max(0.0, started + 10.0 - time.time()))
        subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", store], check=True)

    for repeat in range(2):
        subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks"], check=True)
        time.sleep(2)
        try:
            run("hysteresis", "uint8_seq_stream_locked_then_released", repeat,
                eval_command("uint8_seq_stream", Path("{run}"), 0), during=release)
        finally:
            subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", store],
                           check=True)
    print("governor:", Path("/sys/class/devfreq/17000000.gpu/governor").read_text().strip(),
          "min", Path("/sys/class/devfreq/17000000.gpu/min_freq").read_text().strip(), flush=True)


def phase_sampler() -> None:
    for repeat in range(2):
        for rung in ("uint8_seq_stream", "uint8_overlap_event_w2"):
            for sampled in (True, False):
                name = f"{rung}_{'sampled' if sampled else 'unsampled'}"
                run("sampler", name, repeat, eval_command(rung, Path("{run}"), 2000),
                    sample_power=sampled)


def phase_workers() -> None:
    for repeat in range(2):
        for workers in (1, 2, 3, 4, 5):
            run("workers", f"uint8_overlap_event_w{workers}", repeat,
                eval_command("uint8_overlap_event", Path("{run}"), 0,
                             extra=["--workers", str(workers)]))


def phase_uclamp() -> None:
    for repeat in range(2):
        for name, prefix in (("stream", []),
                             ("stream_uclamp1024", [sys.executable, "/tmp/uclamp_exec.py",
                                                    "1024", "--"])):
            command = ["taskset", "-c", "0", *prefix,
                       *eval_command("uint8_seq_stream", Path("{run}"), 2000)]
            run("uclamp", name, repeat, command)


PHASES = {"load": phase_load, "hysteresis": phase_hysteresis, "sampler": phase_sampler,
          "workers": phase_workers, "uclamp": phase_uclamp}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phases", nargs="+", choices=list(PHASES))
    for phase in parser.parse_args().phases:
        PHASES[phase]()
        print("PHASE_DONE", phase, flush=True)
    print("EXTRAS_DONE", flush=True)
