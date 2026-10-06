"""Round-5 review experiments on the Jetson Orin Nano Super.

Phase 1 (EMC evidence under jetson_clocks): lock all clocks with
jetson_clocks, log the EMC readback at ~4 Hz for 10 s idle and during a
15 s trtexec gap-0 run, then restore. Documents whether jetson_clocks
actually holds the memory clock at its maximum.

Phase 2 (GIL convoy test for the full-w2 sleeping-wait rung):
  - full_w2 blocking-stream sync, default switch interval (same-session
    control),
  - same, with sys.setswitchinterval(0.0005) via a wrapper,
  - full_w2 event sync (same-session reference),
  - prefetch event sync (same-session reference for the uclamp2 table).
2 reps each, 2000 images, predictions hashed.

Output: /tmp/paper/round5/.
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

OUT = Path("/tmp/paper/round5")

FAST_EVAL = ["/usr/bin/python3", "/tmp/jetson_fast_eval.py", "--engine",
             UINT8_ENGINE, "--memory", "pinned", "--limit", "2000",
             "--annotations", "/tmp/instances_val2017.json",
             "--images-dir", "/tmp/yolo26-coco-val/val2017"]

GIL_WRAPPER = "/tmp/gil_exec.py"


def sudo_sh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-S", "sh", "-c", cmd], input=SUDO_PW,
                          text=True, capture_output=True, check=True)


def emc_readback() -> int:
    out = sudo_sh("cat /sys/kernel/debug/bpmp/debug/clk/emc/rate")
    return int(out.stdout.strip())


def phase1() -> None:
    d = OUT / "emc_jetson_clocks"
    d.mkdir(parents=True, exist_ok=True)
    store = "/tmp/jetson_clocks_round5.conf"
    sudo_sh(f"rm -f {store} && jetson_clocks --store {store}")
    sudo_sh("jetson_clocks")
    time.sleep(2)
    idle = [emc_readback() for _ in range(40)]
    time.sleep(0.25)
    proc = subprocess.Popen(
        ["/usr/src/tensorrt/bin/trtexec", f"--loadEngine={UINT8_ENGINE}",
         "--idleTime=0", "--duration=15", "--iterations=100000"],
        env={**os.environ, **TRT_ENV}, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    load = []
    while proc.poll() is None:
        load.append(emc_readback())
        time.sleep(0.25)
    proc.wait()
    (d / "report.json").write_text(json.dumps({
        "idle_emc_hz": idle, "load_emc_hz": load}, indent=2))
    print(f"EMC under jetson_clocks: idle {min(idle)}-{max(idle)}, "
          f"load {min(load)}-{max(load)}", flush=True)
    # nvfancontrol restart failure makes jetson_clocks --restore exit 1
    # even when the clocks are restored; keep it non-fatal.
    subprocess.run(["sudo", "-S", "sh", "-c",
                    f"jetson_clocks --restore {store}"],
                   input=SUDO_PW, text=True, capture_output=True)
    print("PHASE1_DONE", flush=True)


def run(name: str, extra: list[str], switchinterval: float | None) -> None:
    for rep in (0, 1):
        run_dir = OUT / "gil" / name / f"r{rep}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report = run_dir / "report.json"
        if report.exists() and "power" in json.loads(report.read_text()):
            continue
        cmd = list(FAST_EVAL) + extra + ["--predictions",
                                         str(run_dir / "predictions.json"),
                                         "--output", str(report)]
        if switchinterval is not None:
            cmd = ["/usr/bin/python3", GIL_WRAPPER,
                   str(switchinterval)] + cmd[1:]
        (run_dir / "spec.json").write_text(json.dumps(
            {"phase": "gil", "condition": name, "command": cmd}, indent=2))
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
        print(f"G {name} r{rep} {data['throughput_images_per_s']:.1f} img/s "
              f"cpu {data['process_cpu_ms_per_image']:.1f} ms "
              f"pred {digest[:12]}", flush=True)
        time.sleep(10)


def phase2() -> None:
    time.sleep(15)
    run("full_w2_stream_blocking", ["--overlap", "--workers", "2",
                                    "--stream-sync", "--sched", "blocking"],
        None)
    run("full_w2_stream_blocking_si0.5ms", ["--overlap", "--workers", "2",
                                            "--stream-sync", "--sched",
                                            "blocking"], 0.0005)
    run("full_w2_event", ["--overlap", "--workers", "2"], None)
    run("prefetch_event", ["--prefetch"], None)
    print("PHASE2_DONE", flush=True)


if __name__ == "__main__":
    phase1()
    phase2()
    print("ROUND5_DONE", flush=True)
