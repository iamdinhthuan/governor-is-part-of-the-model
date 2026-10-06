"""Does the stream-sync slowdown come from the CPU clock dropping while the host sleeps?

Runs the sequential uint8 pipeline pinned to CPU 0, with stream or event
synchronization, with and without a busy-loop process pinned to CPU 1. CPUs 0-3
share one cpufreq policy, so the busy loop holds the cluster clock at its maximum
without touching the pipeline's core. Logs the policy-0 clock at 20 Hz.
Output: /tmp/paper/cpufreq/default/<condition>/r<k>/.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/tmp")
import jetson_paper_runs as runs  # noqa: E402

POLICY0 = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")
CONDITIONS = {
    "stream": ["--stream-sync"],
    "stream_clusterbusy": ["--stream-sync"],
    "event": [],
    "event_clusterbusy": [],
}


def log_policy0(path: Path, stop: threading.Event) -> None:
    with path.open("w") as out:
        while not stop.is_set():
            out.write(f"{time.time():.4f},{POLICY0.read_text().strip()}\n")
            time.sleep(0.05)


def run(condition: str, repeat: int) -> None:
    run_dir = runs.OUT / "cpufreq" / "default" / condition / f"r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report, predictions = run_dir / "report.json", run_dir / "predictions.json"
    if runs.done(report):
        return
    command = ["taskset", "-c", "0", sys.executable, "/tmp/jetson_fast_eval.py",
               "--engine", runs.ENGINES["y26n_640_uint8"], "--memory", "pinned",
               *CONDITIONS[condition], "--limit", "2000",
               "--annotations", runs.ANN, "--images-dir", runs.IMG,
               "--predictions", str(predictions), "--output", str(report)]
    (run_dir / "spec.json").write_text(json.dumps({"condition": condition,
                                                   "command": command}, indent=2))
    spinner = None
    if condition.endswith("clusterbusy"):
        spinner = subprocess.Popen(["taskset", "-c", "1", sys.executable, "-c",
                                    "while True: pass"])
        time.sleep(1.0)
    stop = threading.Event()
    logger = threading.Thread(target=log_policy0, args=(run_dir / "policy0.csv", stop))
    logger.start()
    sampler = runs.Sampler(run_dir / "power.csv")
    try:
        result = subprocess.run(command, env={**os.environ, **runs.TRT_ENV},
                                timeout=1800, capture_output=True, text=True)
    finally:
        sampler.stop()
        stop.set()
        logger.join()
        if spinner:
            spinner.terminate()
            spinner.wait(timeout=10)
    (run_dir / "stdout.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{condition}/r{repeat} failed:\n{result.stderr[-3000:]}")
    runs.integrate(run_dir / "power.csv", report)
    data = json.loads(report.read_text())
    start, end = data["window_unix"]
    freqs = [int(line.split(",")[1]) / 1000 for line in
             (run_dir / "policy0.csv").read_text().splitlines()
             if start <= float(line.split(",")[0]) <= end]
    data["policy0_mhz_mean"] = sum(freqs) / len(freqs)
    report.write_text(json.dumps(data, indent=2))
    print(f"{time.strftime('%T')} cpufreq {condition} r{repeat} "
          f"{data['throughput_images_per_s']:.1f} img/s "
          f"lat {data['latency_ms']['median']:.2f} ms "
          f"gpu {data['gpu_enqueue_to_done_ms']['median']:.2f} ms "
          f"cpu {data['process_cpu_ms_per_image']:.1f} ms "
          f"policy0 {data['policy0_mhz_mean']:.0f} MHz", flush=True)
    time.sleep(10)


def main() -> None:
    names = list(CONDITIONS)
    for repeat in range(2):
        shift = repeat % len(names)
        for name in names[shift:] + names[:shift]:
            run(name, repeat)
    print("CPUFREQ_DONE", flush=True)


if __name__ == "__main__":
    main()
