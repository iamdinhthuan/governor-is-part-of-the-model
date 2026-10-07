"""Round-6 review experiments on the Jetson Orin Nano Super.

Round-7 reviews (Codex + Claude) found that the uclamp2 "full w2" cells and
the round-5 GIL control passed --overlap --workers 2 WITHOUT --prefetch, so
no decode pool was ever created (decode_workers: 0 in every report), and
that --stream-sync in the enqueue-ahead overlap loop waits for the frame
just enqueued, degenerating the pipeline to sequential. This script reruns
the full two-worker pipeline correctly (--prefetch --overlap --workers 2)
with a sleeping wait that has the SAME completion boundary as the spinning
arm: per-frame events created with cudaEventBlockingSync.

Phase A (default governors):
  evspin        - event sync (spins on this stack), reference
  evblock       - event sync with cudaEventBlockingSync (sleeps)
  evblock_ucl   - evblock + uclamp.min 1024
  streamblk     - stream sync + cudaDeviceScheduleBlockingSync (documents
                  the wrong-boundary pitfall WITH decode workers present)
Phase B (jetson_clocks locked, stored/restored):
  evspin_lock, evblock_lock, evblock_lock_si  (si = GIL switchinterval
                  0.5 ms via gil_exec.py), streamblk_lock

2 reps each, first 2,000 val2017 images, predictions hashed.
Output: /tmp/paper/round6/.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

UINT8_ENGINE = "/tmp/yolo26n_84172_uint8_fp16_opt3.engine"
TRT_ENV = {"PYTHONPATH": "/tmp/yolo26-libs:/tmp"}
SUDO_PW = Path("/tmp/.sudo_pw").read_text()

OUT = Path("/tmp/paper/round6")

FAST_EVAL = ["/usr/bin/python3", "/tmp/jetson_fast_eval.py", "--engine",
             UINT8_ENGINE, "--memory", "pinned", "--limit", "2000",
             "--annotations", "/tmp/instances_val2017.json",
             "--images-dir", "/tmp/yolo26-coco-val/val2017"]
BASE = ["--prefetch", "--overlap", "--workers", "2"]
UCLAMP = ["/usr/bin/python3", "/tmp/uclamp_exec.py", "1024", "--"]
GIL = ["/usr/bin/python3", "/tmp/gil_exec.py", "0.0005"]
STORE = "/tmp/jetson_clocks_round6.conf"


def sudo_sh(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-S", "sh", "-c", cmd], input=SUDO_PW,
                          text=True, capture_output=True, check=check)


def lock_clocks() -> None:
    sudo_sh(f"rm -f {STORE} && jetson_clocks --store {STORE}")
    sudo_sh("jetson_clocks")


def restore_clocks() -> None:
    # nvfancontrol restart failure makes --restore exit 1 even when the
    # clocks are restored; keep it non-fatal.
    subprocess.run(["sudo", "-S", "sh", "-c", f"jetson_clocks --restore {STORE}"],
                   input=SUDO_PW, text=True, capture_output=True)


def run(name: str, extra: list[str], wrapper: list[str] | None = None,
        reps=(0, 1)) -> None:
    for rep in reps:
        run_dir = OUT / name / f"r{rep}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report = run_dir / "report.json"
        if report.exists():
            continue
        cmd = list(FAST_EVAL) + extra + ["--predictions",
                                         str(run_dir / "predictions.json"),
                                         "--output", str(report)]
        if wrapper:
            # gil_exec.py takes the target SCRIPT path directly
            # (gil_exec.py <interval> <script> [args]); other wrappers
            # (uclamp_exec.py 1024 --) take the full command.
            if any("gil_exec" in w for w in wrapper):
                cmd = wrapper + cmd[1:]
            else:
                cmd = wrapper + cmd
        (run_dir / "spec.json").write_text(json.dumps(
            {"phase": "round6", "condition": name, "command": cmd}, indent=2))
        sampler = subprocess.Popen(
            [sys.executable, "/tmp/jetson_power.py", "sample",
             "--out", str(run_dir / "power.csv")])
        try:
            result = subprocess.run(cmd, env={**os.environ, **TRT_ENV},
                                    timeout=1200, capture_output=True,
                                    text=True)
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
        (run_dir / "stdout.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        subprocess.run([sys.executable, "/tmp/jetson_power.py", "integrate",
                        "--samples", str(run_dir / "power.csv"),
                        "--report", str(report)], check=True)
        digest = hashlib.sha256(
            (run_dir / "predictions.json").read_bytes()).hexdigest()
        (run_dir / "predictions.sha256").write_text(digest + "\n")
        data = json.loads(report.read_text())
        print(f"R6 {name} r{rep} {data['throughput_images_per_s']:.1f} img/s "
              f"cpu {data['process_cpu_ms_per_image']:.1f} ms "
              f"workers {data['decode_workers']} "
              f"pred {digest[:12]}", flush=True)
        time.sleep(10)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Phase A: default governors
    run("evspin", BASE)
    run("evblock", BASE + ["--event-blocking"])
    run("evblock_ucl", BASE + ["--event-blocking"], wrapper=UCLAMP)
    run("streamblk", BASE + ["--stream-sync", "--sched", "blocking"])
    print("PHASEA_DONE", flush=True)
    # Phase B: locked clocks
    lock_clocks()
    time.sleep(15)
    try:
        run("evspin_lock", BASE)
        run("evblock_lock", BASE + ["--event-blocking"])
        run("evblock_lock_si", BASE + ["--event-blocking"], wrapper=GIL)
        run("streamblk_lock", BASE + ["--stream-sync", "--sched", "blocking"])
    finally:
        summary = {p.parent.name + "/" + p.name: json.loads(p.read_text())
                   for p in sorted(OUT.glob("*/r*/report.json"))}
        (OUT / "summary.json").write_text(json.dumps(
            {k: {"img_s": v["throughput_images_per_s"],
                 "cpu_ms": v["process_cpu_ms_per_image"],
                 "decode_workers": v["decode_workers"],
                 "sync": v["sync"],
                 "event_blocking": v.get("event_blocking", False)}
             for k, v in summary.items()}, indent=2))
        restore_clocks()
    print("PHASEB_DONE", flush=True)
    out = sudo_sh("nvpmodel -q && cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                  check=False)
    print(out.stdout[-400:], flush=True)
    print("ROUND6_DONE", flush=True)


if __name__ == "__main__":
    main()
