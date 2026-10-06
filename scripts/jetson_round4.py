"""Round-4 review experiments on the Jetson Orin Nano Super.

Phase A: fixed-GPU-frequency sweep with EMC pinned (mrq_rate_locked=1,
rate=3199000000) and EMC readback logged at ~4 Hz by a root sampler.
8 GPU devfreq steps x 3 reps, trtexec uint8 gap-0, 15 s each.
Plus 2 control runs (GPU 306 / 1020 MHz) with EMC left unlocked to document
what EMC does when free.

Phase B: uclamp.min=1024 on sleeping-wait optimized pipelines (prefetch and
full w2, stream sync, --sched blocking), with same-session no-clamp
references, 2 reps each, 2000 images, predictions hashed.

Restores EMC unlock + GPU range at the end. Output: /tmp/paper/eq1sweep_emc/
and /tmp/paper/uclamp2/.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
DEVFREQ = "/sys/class/devfreq/17000000.gpu"
EMC_DIR = "/sys/kernel/debug/bpmp/debug/clk/emc"
EMC_LOCK = f"{EMC_DIR}/mrq_rate_locked"
EMC_RATE = f"{EMC_DIR}/rate"
EMC_MAX = 3199000000
UINT8_ENGINE = "/tmp/yolo26n_84172_uint8_fp16_opt3.engine"
TRT_ENV = {"PYTHONPATH": "/tmp/yolo26-libs:/tmp"}
SUDO_PW = os.environ.get("JETSON_SUDO_PW", "")

STAT = re.compile(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, "
                  r"mean = ([\d.]+) ms, median = ([\d.]+) ms")
QPS = re.compile(r"Throughput: ([\d.]+) qps")

OUT_A = Path("/tmp/paper/eq1sweep_emc")
OUT_B = Path("/tmp/paper/uclamp2/default")


def sudo_sh(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-S", "sh", "-c", cmd], input=SUDO_PW,
                          text=True, capture_output=True, check=True, **kw)


def pin_gpu(freq: int) -> None:
    sudo_sh(f"echo 306000000 > {DEVFREQ}/min_freq; "
            f"echo 1020000000 > {DEVFREQ}/max_freq")
    sudo_sh(f"echo {freq} > {DEVFREQ}/min_freq; echo {freq} > {DEVFREQ}/max_freq")


def unpin_gpu() -> None:
    sudo_sh(f"echo 306000000 > {DEVFREQ}/min_freq; "
            f"echo 1020000000 > {DEVFREQ}/max_freq")


def lock_emc() -> None:
    sudo_sh(f"echo 1 > {EMC_LOCK}; echo {EMC_MAX} > {EMC_RATE}")
    time.sleep(1)
    rb = sudo_sh(f"cat {EMC_RATE}").stdout.strip()
    print(f"EMC locked, readback {int(rb) // 1000000} MHz", flush=True)


def unlock_emc() -> None:
    sudo_sh(f"echo 0 > {EMC_LOCK}")


class EmcSampler:
    """Root background sampler: 'epoch rate_hz' lines at ~4 Hz."""

    def __init__(self, path: Path):
        self.proc = subprocess.Popen(
            ["sudo", "-S", "sh", "-c",
             f"while :; do echo $(date +%s.%N) $(cat {EMC_RATE}); "
             f"sleep 0.25; done > {path}"],
            stdin=subprocess.PIPE, text=True)
        self.proc.stdin.write(SUDO_PW)
        self.proc.stdin.flush()

    def stop(self) -> None:
        # Bracket trick so pkill's own command line does not match.
        subprocess.run(["sudo", "-S", "sh", "-c",
                        "pkill -f 'bpmp/debug/clk/emc/rat[e]' || true"],
                       input=SUDO_PW, text=True, capture_output=True)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def trtexec_run(run_dir: Path, sampler_note: str) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_dir / "report.json"
    if report.exists() and "power" in json.loads(report.read_text()):
        return json.loads(report.read_text())
    sampler = subprocess.Popen(
        [sys.executable, "/tmp/jetson_power.py", "sample",
         "--out", str(run_dir / "power.csv")])
    emc = EmcSampler(run_dir / "emc.csv")
    try:
        result = subprocess.run(
            [TRTEXEC, f"--loadEngine={UINT8_ENGINE}", "--warmUp=3000",
             "--duration=15", "--idleTime=0", "--avgRuns=1"],
            capture_output=True, text=True, timeout=300, check=True)
    finally:
        end = time.time()
        emc.stop()
        sampler.terminate()
        sampler.wait(timeout=10)
    (run_dir / "trtexec.log").write_text(result.stdout)
    stat, qps = STAT.search(result.stdout), QPS.search(result.stdout)
    minimum, maximum, mean, median = map(float, stat.groups())
    emc_rates = [int(l.split()[1]) for l in
                 (run_dir / "emc.csv").read_text().splitlines() if l.strip()]
    data = {"model": "y26n_640_uint8", "idle_ms": 0, "note": sampler_note,
            "gpu_compute_ms": {"min": minimum, "max": maximum,
                               "mean": mean, "median": median},
            "qps": float(qps.group(1)),
            "images_timed": int(float(qps.group(1)) * 15),
            "window_unix": [end - 15.5, end - 0.5],
            "emc_mhz_min_max": [min(emc_rates) // 1000000,
                                max(emc_rates) // 1000000] if emc_rates else None}
    report.write_text(json.dumps(data, indent=2))
    subprocess.run([sys.executable, "/tmp/jetson_power.py", "integrate",
                    "--samples", str(run_dir / "power.csv"),
                    "--report", str(report)], check=True)
    return data


def phase_a() -> None:
    freqs = [306000000, 408000000, 510000000, 612000000, 714000000,
             816000000, 918000000, 1020000000]
    try:
        lock_emc()
        for freq in freqs:
            pin_gpu(freq)
            time.sleep(2)
            for rep in (0, 1, 2):
                d = trtexec_run(OUT_A / "pinned" / f"f{freq // 1000000}" / f"r{rep}",
                                "emc_pinned_3199")
                print(f"A pin {freq // 1000000} r{rep}: median "
                      f"{d['gpu_compute_ms']['median']:.3f} ms "
                      f"{d['qps']:.1f} qps emc {d['emc_mhz_min_max']}", flush=True)
                time.sleep(3)
        unlock_emc()
        time.sleep(3)
        for freq in (306000000, 1020000000):
            pin_gpu(freq)
            time.sleep(2)
            d = trtexec_run(OUT_A / "emc_free" / f"f{freq // 1000000}" / "r0",
                            "emc_unlocked_control")
            print(f"A ctrl {freq // 1000000}: median "
                  f"{d['gpu_compute_ms']['median']:.3f} ms "
                  f"{d['qps']:.1f} qps emc {d['emc_mhz_min_max']}", flush=True)
            time.sleep(3)
    finally:
        unpin_gpu()
        unlock_emc()
        print("GPU unpinned, EMC unlocked", flush=True)
    print("PHASE_A_DONE", flush=True)


FAST_EVAL = ["/usr/bin/python3", "/tmp/jetson_fast_eval.py", "--engine",
             UINT8_ENGINE, "--memory", "pinned", "--limit", "2000",
             "--annotations", "/tmp/instances_val2017.json",
             "--images-dir", "/tmp/yolo26-coco-val/val2017"]


def phase_b_run(name: str, extra: list[str], uclamp: bool) -> None:
    for rep in (0, 1):
        run_dir = OUT_B / name / f"r{rep}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report = run_dir / "report.json"
        if report.exists() and "power" in json.loads(report.read_text()):
            continue
        cmd = list(FAST_EVAL) + extra + ["--predictions",
                                         str(run_dir / "predictions.json"),
                                         "--output", str(report)]
        if uclamp:
            cmd = ["/usr/bin/python3", "/tmp/uclamp_exec.py", "1024", "--"] + cmd
        (run_dir / "spec.json").write_text(json.dumps(
            {"phase": "uclamp2", "condition": name, "command": cmd}, indent=2))
        sampler = subprocess.Popen(
            [sys.executable, "/tmp/jetson_power.py", "sample",
             "--out", str(run_dir / "power.csv")])
        try:
            result = subprocess.run(cmd, env={**os.environ, **TRT_ENV},
                                    timeout=1200, capture_output=True, text=True)
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
        print(f"B {name} r{rep} {data['throughput_images_per_s']:.1f} img/s "
              f"{data['power']['energy_per_image_mj']:.1f} mJ "
              f"pred {digest[:12]}", flush=True)
        time.sleep(10)


def phase_b() -> None:
    time.sleep(15)
    phase_b_run("prefetch_stream_blocking",
                ["--prefetch", "--stream-sync", "--sched", "blocking"], False)
    phase_b_run("prefetch_stream_blocking_uclamp1024",
                ["--prefetch", "--stream-sync", "--sched", "blocking"], True)
    phase_b_run("full_w2_stream_blocking",
                ["--overlap", "--workers", "2", "--stream-sync",
                 "--sched", "blocking"], False)
    phase_b_run("full_w2_stream_blocking_uclamp1024",
                ["--overlap", "--workers", "2", "--stream-sync",
                 "--sched", "blocking"], True)
    print("PHASE_B_DONE", flush=True)


if __name__ == "__main__":
    phase_a()
    phase_b()
    print("ROUND4_DONE", flush=True)
