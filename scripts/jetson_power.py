"""Sample Jetson board input power (INA3221 VDD_IN) and integrate over a window.

`sample` runs as a separate process until terminated, appending
"unix_time,millivolts,milliamps" lines. `integrate` computes mean power and
energy for the [start, end] window that an evaluator records.
"""

import argparse
import json
from pathlib import Path
import signal
import time


def find_rail(label: str = "VDD_IN") -> tuple[Path, Path]:
    for path in Path("/sys/bus/i2c/drivers/ina3221").glob("*/hwmon/hwmon*/in*_label"):
        if path.read_text().strip() == label:
            channel = path.name[2:-6]
            return path.parent / f"in{channel}_input", path.parent / f"curr{channel}_input"
    raise RuntimeError(f"Power rail {label} not found")


def find_thermal(name: str = "tj-thermal") -> Path:
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        if (zone / "type").read_text().strip() == name:
            return zone / "temp"
    raise RuntimeError(f"Thermal zone {name} not found")


def sample(out: Path, hz: float) -> None:
    volt, curr = find_rail()
    junction = find_thermal()
    gpu = Path("/sys/class/devfreq/17000000.gpu/cur_freq")
    cpus = [
        Path(f"/sys/devices/system/cpu/cpu{index}/cpufreq/scaling_cur_freq")
        for index in range(6)
    ]
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    period = 1.0 / hz
    with out.open("w") as handle:
        while running:
            now = time.time()
            handle.write(
                f"{now:.4f},{volt.read_text().strip()},{curr.read_text().strip()},"
                f"{junction.read_text().strip()},{gpu.read_text().strip()},"
                f"{max(int(path.read_text()) for path in cpus)}\n"
            )
            time.sleep(max(0.0, period - (time.time() - now)))


def integrate(samples: Path, start: float, end: float) -> dict:
    rows, temps, gpu_mhz, cpu_mhz = [], [], [], []
    for line in samples.read_text().splitlines():
        fields = line.split(",")
        t, mv, ma = map(float, fields[:3])
        rows.append((t, mv * ma / 1e6))
        if start <= t <= end and len(fields) > 3:
            temps.append(float(fields[3]) / 1000)
            if len(fields) > 5:
                gpu_mhz.append(float(fields[4]) / 1e6)
                cpu_mhz.append(float(fields[5]) / 1e3)
    inside = [(t, w) for t, w in rows if start <= t <= end]
    if len(inside) < 10:
        raise RuntimeError("Too few power samples inside timed window")
    # Include the partial intervals at the window edges by interpolating
    # the bracketing samples onto [start, end] (previously the edges were
    # dropped, which truncated up to one sampling period at each side).
    before = [r for r in rows if r[0] < start]
    after = [r for r in rows if r[0] > end]
    if before:
        (t0, w0), (t1, w1) = before[-1], inside[0]
        w_start = w0 + (w1 - w0) * (start - t0) / (t1 - t0)
        inside = [(start, w_start)] + inside
    if after:
        (t0, w0), (t1, w1) = inside[-1], after[0]
        w_end = w0 + (w1 - w0) * (end - t0) / (t1 - t0)
        inside = inside + [(end, w_end)]
    energy = sum(
        (t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(inside, inside[1:])
    )
    duration = inside[-1][0] - inside[0][0]
    return {
        "rail": "VDD_IN (module input)",
        "samples": len(inside),
        "mean_power_w": energy / duration,
        "energy_j": energy,
        "duration_s": duration,
        "tj_start_c": temps[0] if temps else None,
        "tj_max_c": max(temps) if temps else None,
        "gpu_mhz_mean": sum(gpu_mhz) / len(gpu_mhz) if gpu_mhz else None,
        "max_core_cpu_mhz_mean": sum(cpu_mhz) / len(cpu_mhz) if cpu_mhz else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--hz", type=float, default=20.0)
    i = sub.add_parser("integrate")
    i.add_argument("--samples", type=Path, required=True)
    i.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sample":
        sample(args.out, args.hz)
    else:
        report = json.loads(args.report.read_text())
        start, end = report["window_unix"]
        power = integrate(args.samples, start, end)
        power["energy_per_image_mj"] = 1000 * power["energy_j"] / report["images_timed"]
        report["power"] = power
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(power))


if __name__ == "__main__":
    main()
