"""Pi 5 + Hailo-8 campaign driver (paper generality device).

Phases:
  idle     60 s of nothing, per governor (SoC idle power, clocks)
  bench    hailortcli benchmark of the HEF, with sampler attached
  ladder   cumulative pipeline rungs x governors x 2 repeats, 5000 val2017
  cpufreq  sequential pipeline pinned to CPU 0, with/without busy loop on CPU 1
           (the 4 cores share one frequency domain), ondemand only
  stream   1000 images at 15/30/60 fps, ondemand only, 2 repeats
Every run logs power.csv (pi_sampler.py) and integrates energy over the timed
window into report.json. Predictions are sha256-hashed.
"""

import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

BASE = Path("/home/urlab/paper")
VENV = BASE / "venv/bin/python"
EVAL = BASE / "pi_hailo_eval.py"
SAMPLER = BASE / "pi_sampler.py"
HEF = BASE / "models/yolov8n_h8.hef"
ANN = BASE / "data/instances_val2017.json"
IMG = BASE / "data/val2017"
OUT = BASE / "results"

GOVERNORS = {"ondemand": "ondemand", "performance": "performance"}

LADDER = {
    "seq": [],
    "prefetch": ["--prefetch", "--workers", "1"],
    "full_w1": ["--prefetch", "--workers", "1", "--overlap", "--inflight", "2"],
    "full_w2": ["--prefetch", "--workers", "2", "--overlap", "--inflight", "2"],
    "full_w2_spin": ["--prefetch", "--workers", "2", "--overlap", "--inflight", "2",
                     "--wait", "spin"],
    "full_w3": ["--prefetch", "--workers", "3", "--overlap", "--inflight", "2"],
}


def sh(command, **kw):
    return subprocess.run(command, shell=True, check=True, **kw)


def set_governor(name: str) -> None:
    sh(f"echo 123 | sudo -S sh -c 'echo {name} > "
       "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor' 2>/dev/null")
    time.sleep(1)
    got = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read_text().strip()
    if got != name:
        raise RuntimeError(f"governor is {got}, wanted {name}")


def warm_page_cache() -> None:
    sh(f"cat {IMG}/*.jpg > /dev/null")


def integrate(run_dir: Path, report: Path) -> None:
    data = json.loads(report.read_text())
    start, end = data["window_unix"]
    pts = []
    allpts = []
    for line in (run_dir / "power.csv").read_text().splitlines():
        f = line.split(",")
        row = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
        allpts.append(row)
        if start <= row[0] <= end:
            pts.append(row)
    if not pts:
        return
    # Interpolate the bracketing samples onto [start, end] so the partial
    # edge intervals are included (previously dropped, truncating up to one
    # sampling period at each side).
    before = [r for r in allpts if r[0] < start]
    after = [r for r in allpts if r[0] > end]
    if before:
        a, b = before[-1], pts[0]
        w = a[3] + (b[3] - a[3]) * (start - a[0]) / (b[0] - a[0])
        pts = [(start, a[1], a[2], w)] + pts
    if after:
        a, b = pts[-1], after[0]
        w = a[3] + (b[3] - a[3]) * (end - a[0]) / (b[0] - a[0])
        pts = pts + [(end, b[1], b[2], w)]
    energy = sum(0.5 * (a[3] + b[3]) * (b[0] - a[0]) for a, b in zip(pts, pts[1:]))
    data["power"] = {
        "sensor": "Pi5 PMIC rail sum (SoC side; excludes 5 V loads: Hailo, fan, USB)",
        "samples": len(pts),
        "mean_power_w": energy / (pts[-1][0] - pts[0][0]),
        "energy_j": energy,
        "duration_s": end - start,
        "energy_per_image_mj": 1000 * energy / data["images_timed"],
        "temp_start_c": pts[0][2], "temp_max_c": max(p[2] for p in pts),
        "cpu_mhz_mean": statistics.fmean(p[1] for p in pts),
    }
    report.write_text(json.dumps(data, indent=2) + "\n")


def run(phase: str, name: str, repeat: int, command: list, use_sampler=True) -> None:
    run_dir = OUT / phase / name / f"r{repeat}"
    report = run_dir / "report.json"
    if report.exists() and "power" in json.loads(report.read_text()):
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps({"phase": phase, "condition": name,
                                                   "command": command}, indent=2))
    # Device probe: HailoRT sometimes wedges process startup right after a
    # previous process exited; probe with an OS-level timeout first.
    probe = ["timeout", "25", str(VENV), "-c",
             "from hailo_platform import VDevice; d = VDevice(); d.release()"]
    if subprocess.run(probe, capture_output=True).returncode != 0:
        print(f"device probe failed for {phase}/{name}/r{repeat}; waiting 60 s",
              flush=True)
        time.sleep(60)
    sampler = None
    if use_sampler:
        sampler = subprocess.Popen([str(VENV), str(SAMPLER), "--out",
                                    str(run_dir / "power.csv")],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
    started = time.time()
    proc = None
    try:
        for attempt in (1, 2, 3, 4):
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=480)
            except subprocess.TimeoutExpired:
                proc = None
            if proc is not None and proc.returncode == 0:
                break
            tail = (proc.stdout + proc.stderr)[-400:] if proc else "timeout"
            print(f"RETRY {phase}/{name}/r{repeat} attempt {attempt}: {tail}",
                  flush=True)
            time.sleep(30 * attempt)  # let the device recover before the retry
        if proc is not None:
            (run_dir / "stdout.log").write_text(proc.stdout + proc.stderr)
        if proc is None or proc.returncode:
            raise RuntimeError(f"{phase}/{name}/r{repeat} failed 4 times")
    finally:
        if sampler:
            sampler.send_signal(signal.SIGTERM)
            sampler.wait()
    if report.exists():
        integrate(run_dir, report)
    pred = run_dir / "predictions.json"
    digest = (subprocess.run(["sha256sum", str(pred)], capture_output=True, text=True)
              .stdout.split()[0] if pred.exists() else "-")
    if pred.exists():
        (run_dir / "predictions.sha256").write_text(digest + "\n")
    data = json.loads(report.read_text()) if report.exists() else {
        "throughput_images_per_s": float("nan"), "power": {}}
    p = data.get("power", {})
    print(f"{time.strftime('%T')} {phase} {name} r{repeat} "
          f"{data['throughput_images_per_s']:.1f} img/s "
          f"{p.get('mean_power_w', float('nan')):.2f} W "
          f"{p.get('energy_per_image_mj', float('nan')):.1f} mJ "
          f"cpu {p.get('cpu_mhz_mean', float('nan')):.0f} MHz "
          f"pred {digest[:12]}", flush=True)
    time.sleep(10)


def eval_cmd(rung: str, run_dir: Path, limit=0, fps=0.0) -> list:
    cmd = [str(VENV), str(EVAL), "--hef", str(HEF), "--annotations", str(ANN),
           "--images-dir", str(IMG), *LADDER.get(rung, []),
           *(["--limit", str(limit)] if limit else []),
           *(["--fps", str(fps)] if fps else []),
           "--predictions", str(run_dir / "predictions.json"),
           "--output", str(run_dir / "report.json")]
    return cmd


def phase_idle() -> None:
    for gov in GOVERNORS:
        set_governor(gov)
        run_dir = OUT / "idle" / gov / "r0"
        report = run_dir / "report.json"
        if report.exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        sampler = subprocess.Popen([str(VENV), str(SAMPLER), "--out",
                                    str(run_dir / "power.csv")],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start, start_unix = time.perf_counter(), time.time()
        time.sleep(60)
        end_unix = time.time()
        sampler.send_signal(signal.SIGTERM)
        sampler.wait()
        report.write_text(json.dumps({"window_unix": [start_unix, end_unix],
                                      "images_timed": 1, "phase": "idle"}))
        # integrate manually: no images, energy only
        data = json.loads(report.read_text())
        pts = [l.split(",") for l in (run_dir / "power.csv").read_text().splitlines()]
        pts = [(float(t), float(m), float(tp), float(w))
               for t, m, tp, w in pts if start_unix <= float(t) <= end_unix]
        data["power"] = {"mean_power_w": statistics.fmean(p[3] for p in pts),
                         "cpu_mhz_mean": statistics.fmean(p[1] for p in pts),
                         "temp_max_c": max(p[2] for p in pts), "samples": len(pts)}
        report.write_text(json.dumps(data, indent=2))
        print(f"idle {gov}: {data['power']['mean_power_w']:.2f} W "
              f"cpu {data['power']['cpu_mhz_mean']:.0f} MHz", flush=True)


def phase_bench() -> None:
    for gov in GOVERNORS:
        set_governor(gov)
        run_dir = OUT / "bench" / f"hailortcli_{gov}" / "r0"
        if (run_dir / "bench.csv").exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        sampler = subprocess.Popen([str(VENV), str(SAMPLER), "--out",
                                    str(run_dir / "power.csv")],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        start_unix = time.time()
        proc = subprocess.run(["hailortcli", "benchmark", str(HEF), "-t", "20",
                               "--csv", str(run_dir / "bench.csv")],
                              capture_output=True, text=True)
        end_unix = time.time()
        sampler.send_signal(signal.SIGTERM)
        sampler.wait()
        (run_dir / "stdout.log").write_text(proc.stdout + proc.stderr)
        pts = [l.split(",") for l in (run_dir / "power.csv").read_text().splitlines()]
        pts = [(float(t), float(m), float(tp), float(w))
               for t, m, tp, w in pts if start_unix <= float(t) <= end_unix]
        (run_dir / "report.json").write_text(json.dumps({
            "window_unix": [start_unix, end_unix], "images_timed": 1,
            "power": {"mean_power_w": statistics.fmean(p[3] for p in pts),
                      "cpu_mhz_mean": statistics.fmean(p[1] for p in pts),
                      "temp_max_c": max(p[2] for p in pts), "samples": len(pts)},
            "note": "hailortcli benchmark -t 20; see bench.csv"}, indent=2))
        print(f"bench {gov}: {statistics.fmean(p[3] for p in pts):.2f} W", flush=True)
        time.sleep(10)


def phase_ladder() -> None:
    for rep in range(2):
        for gov in GOVERNORS:
            set_governor(gov)
            warm_page_cache()
            for rung in LADDER:
                run_dir = OUT / f"ladder_{gov}" / rung / f"r{rep}"
                run(f"ladder_{gov}", rung, rep, eval_cmd(rung, run_dir))


def phase_cpufreq() -> None:
    set_governor("ondemand")
    warm_page_cache()
    for rep in range(2):
        for name, busy in (("seq", False), ("seq_busy", True)):
            run_dir = OUT / "cpufreq" / name / f"r{rep}"
            cmd = ["taskset", "-c", "0",
                   *[c.replace("{run}", str(run_dir))
                     for c in eval_cmd("seq", run_dir, limit=2000)]]
            spinner = None
            if busy:
                spinner = subprocess.Popen(
                    ["taskset", "-c", "1", str(VENV), "-c",
                     "import time\nx=0\nwhile True: x+=1"])
            try:
                run("cpufreq", name, rep, cmd)
            finally:
                if spinner:
                    spinner.terminate()
                    spinner.wait()


def phase_stream() -> None:
    set_governor("ondemand")
    warm_page_cache()
    for rep in range(2):
        for rung in ("seq", "full_w2"):
            for fps in (15, 30, 60):
                name = f"{rung}_fps{fps}"
                run_dir = OUT / "stream" / name / f"r{rep}"
                run("stream", name, rep,
                    [c.replace("{run}", str(run_dir))
                     for c in eval_cmd(rung, run_dir, limit=1000, fps=fps)])


PHASES = {"idle": phase_idle, "bench": phase_bench, "ladder": phase_ladder,
          "cpufreq": phase_cpufreq, "stream": phase_stream}

if __name__ == "__main__":
    for phase in sys.argv[1:]:
        print(f"PHASE {phase}", flush=True)
        PHASES[phase]()
    set_governor("ondemand")
    print("CAMPAIGN_DONE governor restored:", flush=True)
    print(Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read_text())
