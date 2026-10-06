"""Rubik Pi 3 HTP/NPU campaign driver (QNN context on Hailo... on QCM6490 HTP).

Mirrors rubik_campaign.py but runs the user's QAIRT 2.49.40 context binary on
the HTP NPU via the qai-appbuilder 2.50.40 bundled runtime. Output goes to
results_htp/ so the CPU (onnxruntime) campaign stays untouched.

Phases: accuracy (5000 imgs, seq) / ladder (1000 imgs x schedutil+performance
x 2 reps x 6 rungs) / cpufreq (policy7 fixed freqs x seq, 200 imgs) /
stream (15/30/60 fps x seq/full_w2 x 2 reps, 500 imgs).
"""

import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

BASE = Path("/home/ubuntu/paper")
VENV = BASE / "venv_htp/bin/python"
EVAL = BASE / "rubik_htp_eval.py"
SAMPLER = BASE / "rubik_sampler.py"
CTX = Path(os.environ.get("HTP_CTX", str(BASE / "models/yolo26n_qnn_ctx.bin")))
LIBS = BASE / "venv_htp/lib/python3.12/site-packages/qai_appbuilder/libs"
ANN = BASE / "data/instances_val2017.json"
IMG = BASE / "data/val2017"
OUT = Path(os.environ.get("HTP_OUT", str(BASE / "results_htp")))
SUDO = "echo skyholic2026 | sudo -S"

ENV = dict(os.environ)
ENV["ADSP_LIBRARY_PATH"] = str(LIBS)
ENV["LD_LIBRARY_PATH"] = str(LIBS) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

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
        for attempt in (1, 2, 3):
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=900, env=ENV)
            except subprocess.TimeoutExpired:
                proc = None
            if proc is not None and proc.returncode == 0 and report.exists():
                break
            tail = (proc.stdout + proc.stderr)[-400:] if proc else "timeout"
            print(f"RETRY {phase}/{name}/r{repeat} attempt {attempt}: {tail}",
                  flush=True)
            time.sleep(20 * attempt)
        if proc is not None:
            (run_dir / "stdout.log").write_text(proc.stdout + proc.stderr)
        if proc is None or proc.returncode or not report.exists():
            raise RuntimeError(f"{phase}/{name}/r{repeat} failed 3 times")
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
    cmd = [str(VENV), str(EVAL), "--context", str(CTX), "--libs", str(LIBS),
           "--annotations", str(ANN), "--images-dir", str(IMG), "--warmup", "20",
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
    """All 5,000 images, sequential, once: the COCOeval reference for this stack."""
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
            cmd = ["taskset", "-c", "7", *eval_cmd("seq", run_dir, limit=200)]
            run("cpufreq", name, 0, cmd)
    finally:
        set_governor("schedutil", policies=[7])


def phase_stream() -> None:
    """rep 0 under schedutil, rep 1 under performance: the stream phase covers
    both governors without doubling its length."""
    for rep, gov in ((0, "schedutil"), (1, "performance")):
        set_governor(gov)
        warm_page_cache()
        for rung in ("seq", "full_w2"):
            for fps in (15, 30, 60):  # NPU sustains tens of fps
                name = f"{rung}_fps{fps}"
                run_dir = OUT / "stream" / name / f"r{rep}"
                run("stream", name, rep, eval_cmd(rung, run_dir, limit=500, fps=fps))


def main() -> None:
    import sys
    phases = sys.argv[1:] or ["accuracy", "ladder", "cpufreq", "stream"]
    for phase in phases:
        print(f"PHASE {phase}", flush=True)
        {"ladder": phase_ladder, "cpufreq": phase_cpufreq,
         "stream": phase_stream, "accuracy": phase_accuracy}[phase]()
    set_governor("schedutil")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
