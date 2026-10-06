"""Rubik Pi 3 (QCM6490) campaign driver (paper generality device, CPU-only).

Phases:
  ladder   cumulative pipeline rungs x governors (schedutil, performance)
           x 2 repeats, 5000 val2017 images
  cpufreq  sequential single-threaded pipeline pinned to CPU 7 (prime core,
           own frequency domain) at fixed frequencies (userspace governor):
           direct engine-time versus clock measurement
  stream   1000 images at 15/30/60 fps, schedutil only, 2 repeats

This board has no trustworthy power sensor (qcom-battmgr values are
inconsistent), so runs log clocks/temperatures only (rubik_sampler.py).
Every run stores spec.json, stdout.log, samples.csv; predictions are
sha256-hashed. Governors are restored at the end.
"""

import json
from pathlib import Path
import signal
import statistics
import subprocess
import time

BASE = Path("/home/ubuntu/paper")
VENV = BASE / "venv/bin/python"
EVAL = BASE / "rubik_ort_eval.py"
SAMPLER = BASE / "rubik_sampler.py"
ONNX = BASE / "models/yolo26n.onnx"
ANN = BASE / "data/instances_val2017.json"
IMG = BASE / "data/val2017"
OUT = BASE / "results"
SUDO = "echo skyholic2026 | sudo -S"

GOVERNORS = ["schedutil", "performance"]
POLICIES = [0, 4, 7]

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


def set_governor(name: str, policies=POLICIES) -> None:
    for p in policies:
        sh(f"{SUDO} sh -c 'echo {name} > "
           f"/sys/devices/system/cpu/cpufreq/policy{p}/scaling_governor' 2>/dev/null")
    time.sleep(1)
    for p in policies:
        got = Path(f"/sys/devices/system/cpu/cpufreq/policy{p}/scaling_governor").read_text().strip()
        if got != name:
            raise RuntimeError(f"policy{p} governor is {got}, wanted {name}")


def set_fixed_freq(policy: int, khz: int) -> None:
    sh(f"{SUDO} sh -c 'echo {khz} > "
       f"/sys/devices/system/cpu/cpufreq/policy{policy}/scaling_setspeed' 2>/dev/null")
    time.sleep(0.5)


def warm_page_cache() -> None:
    sh(f"cat {IMG}/*.jpg > /dev/null")


def integrate(run_dir: Path, report: Path) -> None:
    data = json.loads(report.read_text())
    start, end = data["window_unix"]
    pts = []
    csv_path = run_dir / "samples.csv"
    if csv_path.exists():
        for line in csv_path.read_text().splitlines()[1:]:
            f = line.split(",")
            t = float(f[0])
            if start <= t <= end and len(f) >= 5:
                # columns: unix_t, cpu0_mhz, cpu4_mhz, cpu7_mhz, soc_temp_c
                pts.append((t, float(f[1]), float(f[3]), float(f[4])))
    if pts:
        data["clocks"] = {
            "samples": len(pts),
            "cpu0_mhz_mean": statistics.fmean(p[1] for p in pts),
            "cpu7_mhz_mean": statistics.fmean(p[2] for p in pts),
            "soc_temp_c_max": max(p[3] for p in pts),
        }
    report.write_text(json.dumps(data, indent=2) + "\n")


def run(phase: str, name: str, repeat: int, command: list) -> None:
    run_dir = OUT / phase / name / f"r{repeat}"
    report = run_dir / "report.json"
    if report.exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps({"phase": phase, "condition": name,
                                                   "command": command}, indent=2))
    sampler = subprocess.Popen([str(VENV), str(SAMPLER), "--out",
                                str(run_dir / "samples.csv")],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    proc = None
    try:
        for attempt in (1, 2):
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=3600)
            except subprocess.TimeoutExpired:
                proc = None
            if proc is not None and proc.returncode == 0:
                break
            tail = (proc.stdout + proc.stderr)[-400:] if proc else "timeout"
            print(f"RETRY {phase}/{name}/r{repeat} attempt {attempt}: {tail}",
                  flush=True)
            time.sleep(20)
        if proc is not None:
            (run_dir / "stdout.log").write_text(proc.stdout + proc.stderr)
        if proc is None or proc.returncode:
            raise RuntimeError(f"{phase}/{name}/r{repeat} failed twice")
    finally:
        sampler.send_signal(signal.SIGTERM)
        sampler.wait()
    if report.exists():
        integrate(run_dir, report)
    pred = run_dir / "predictions.json"
    digest = (subprocess.run(["sha256sum", str(pred)], capture_output=True, text=True)
              .stdout.split()[0] if pred.exists() else "-")
    if pred.exists():
        (run_dir / "predictions.sha256").write_text(digest + "\n")
    data = json.loads(report.read_text())
    clocks = data.get("clocks", {})
    print(f"{time.strftime('%T')} {phase} {name} r{repeat} "
          f"{data['images_per_s']:.1f} img/s "
          f"p50 {data['latency_median_ms']:.1f} ms "
          f"infer {data.get('infer_median_ms') or 0:.1f} ms "
          f"cpu7 {clocks.get('cpu7_mhz_mean', float('nan')):.0f} MHz "
          f"pred {digest[:12]}", flush=True)
    time.sleep(10)


def eval_cmd(rung: str, run_dir: Path, limit=0, fps=0.0, extra=None) -> list:
    cmd = [str(VENV), str(EVAL), "--onnx", str(ONNX), "--annotations", str(ANN),
           "--images-dir", str(IMG), "--warmup", "20",
           *LADDER.get(rung, []),
           *(["--limit", str(limit)] if limit else []),
           *(["--fps", str(fps)] if fps else []),
           *(extra or []),
           "--predictions", str(run_dir / "predictions.json"),
           "--output", str(run_dir / "report.json")]
    return cmd


def phase_ladder() -> None:
    for rep in range(2):
        for gov in GOVERNORS:
            set_governor(gov)
            warm_page_cache()
            for rung in LADDER:
                run_dir = OUT / f"ladder_{gov}" / rung / f"r{rep}"
                run(f"ladder_{gov}", rung, rep, eval_cmd(rung, run_dir, limit=1000))


def phase_accuracy() -> None:
    """All 5,000 images, sequential, once: the COCOeval reference for this board."""
    set_governor("schedutil")
    warm_page_cache()
    run_dir = OUT / "accuracy" / "seq" / "r0"
    run("accuracy", "seq", 0, eval_cmd("seq", run_dir))


def phase_cpufreq() -> None:
    avail = Path("/sys/devices/system/cpu/cpufreq/policy7/scaling_available_frequencies")
    freqs = sorted(int(x) for x in avail.read_text().split())
    picks = [freqs[0], freqs[len(freqs) // 2], freqs[-1]]
    set_governor("userspace", policies=[7])  # only the prime core
    warm_page_cache()
    try:
        for khz in picks:
            set_fixed_freq(7, khz)
            name = f"seq_cpu7_{khz // 1000}mhz"
            run_dir = OUT / "cpufreq" / name / "r0"
            cmd = ["taskset", "-c", "7",
                   *[c.replace("{run}", str(run_dir))
                     for c in eval_cmd("seq", run_dir, limit=200,
                                       extra=["--intra", "1"])]]
            run("cpufreq", name, 0, cmd)
    finally:
        set_governor("schedutil", policies=[7])


def phase_stream() -> None:
    set_governor("schedutil")
    warm_page_cache()
    for rep in range(2):
        for rung in ("seq", "full_w2"):
            for fps in (2, 5, 10):  # CPU inference sustains only a few fps
                name = f"{rung}_fps{fps}"
                run_dir = OUT / "stream" / name / f"r{rep}"
                run("stream", name, rep, eval_cmd(rung, run_dir, limit=300, fps=fps))


def main() -> None:
    phases = sys_phases()
    for phase in phases:
        print(f"PHASE {phase}", flush=True)
        {"ladder": phase_ladder, "cpufreq": phase_cpufreq,
         "stream": phase_stream, "accuracy": phase_accuracy}[phase]()
    set_governor("schedutil")
    print("DONE", flush=True)


def sys_phases() -> list:
    import sys
    return sys.argv[1:] or ["accuracy", "ladder", "cpufreq", "stream"]


if __name__ == "__main__":
    main()
