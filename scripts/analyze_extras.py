"""Summaries of the review experiments (jetson_review_extras.py) -> results/paper/extras.json."""

import json
import statistics
from pathlib import Path

RAW = Path("results/paper/raw")
MODEL = json.loads(Path("results/paper/model.json").read_text())


def runs(phase: str, name: str):
    return sorted((RAW / phase / "default" / name).glob("r*"))


def counters(run: Path, start: float, end: float):
    rows = [line.split(",") for line in (run / "counters.csv").read_text().splitlines()]
    return [(float(r[0]), int(r[1]) / 1000, int(r[2]) / 1e6, int(r[3]) / 1000)
            for r in rows if start <= float(r[0]) <= end]


def cell(run: Path) -> dict:
    report = json.loads((run / "report.json").read_text())
    start, end = report["window_unix"]
    rows = counters(run, start, end)
    out = {"throughput": report["throughput_images_per_s"],
           "cpu_ms": report["process_cpu_ms_per_image"],
           "engine_ms": report["gpu_enqueue_to_done_ms"]["median"],
           "latency_median_ms": report["latency_ms"]["median"],
           "load": statistics.fmean(r[1] for r in rows),
           "gpu_mhz_harmonic": len(rows) / sum(1 / r[2] for r in rows),
           "policy0_mhz": statistics.fmean(r[3] for r in rows)}
    power = report.get("power")
    if power:
        out["energy_mj"] = power["energy_per_image_mj"]
        out["power_w"] = power["mean_power_w"]
    return out


def mean_cells(cells: list) -> dict:
    keys = set.intersection(*(set(c) for c in cells))
    return {k: {"mean": statistics.fmean(c[k] for c in cells),
                "min": min(c[k] for c in cells), "max": max(c[k] for c in cells)}
            for k in sorted(keys)} | {"n": len(cells)}


def load_phase() -> dict:
    estimate = {r["rung"]: r["utilization_model"] for r in MODEL["operating_points"]}
    out = {}
    for path in sorted((RAW / "load" / "default").iterdir()):
        c = cell(path / "r0")
        c["busy_fraction_model"] = estimate.get(path.name)
        out[path.name] = c
    return out


def hysteresis() -> list:
    """Clock trajectory of a sequential run whose first 10 s ran with locked clocks."""
    out = []
    for run in runs("hysteresis", "uint8_seq_stream_locked_then_released"):
        report = json.loads((run / "report.json").read_text())
        start, end = report["window_unix"]
        spec_start = min(float(l.split(",")[0]) for l in
                         (run / "counters.csv").read_text().splitlines())
        release = spec_start + 10.0
        rows = counters(run, spec_start, end)
        after = [r for r in rows if r[0] > release + 0.2]
        first_floor = next((r[0] - release for r in after if r[2] <= 306.5), None)
        settled = [r for r in after if r[0] > release + 5]
        out.append({"locked_mhz": statistics.fmean(r[2] for r in rows if r[0] < release - 0.5),
                    "locked_load": statistics.fmean(r[1] for r in rows if r[0] < release - 0.5),
                    "seconds_to_floor": first_floor,
                    "settled_mhz": statistics.fmean(r[2] for r in settled),
                    "settled_load": statistics.fmean(r[1] for r in settled),
                    "fraction_at_floor_after_5s":
                        sum(r[2] <= 306.5 for r in settled) / len(settled),
                    "trajectory": [(round(r[0] - release, 2), r[2], r[1]) for r in rows
                                   if -1 <= r[0] - release <= 4]})
    return out


def grouped(phase: str) -> dict:
    root = RAW / phase / "default"
    if not root.exists():
        return {}
    return {p.name: mean_cells([cell(r) for r in runs(phase, p.name)])
            for p in sorted(root.iterdir()) if runs(phase, p.name)}


if __name__ == "__main__":
    result = {"load": load_phase(), "uclamp": grouped("uclamp"), "sampler": grouped("sampler"),
              "workers": grouped("workers")}
    if (RAW / "hysteresis").exists():
        result["hysteresis"] = hysteresis()
    Path("results/paper/extras.json").write_text(json.dumps(result, indent=1))
    for rung, c in result["load"].items():
        print(f"load {rung:26s} X {c['throughput']:6.1f} load {c['load']:.3f} "
              f"model {c['busy_fraction_model']} f_h {c['gpu_mhz_harmonic']:6.0f} "
              f"cpu0 {c['policy0_mhz']:5.0f} engine {c['engine_ms']:.2f}")
    for phase in ("uclamp", "sampler", "workers"):
        for name, c in result[phase].items():
            print(f"{phase} {name:32s} " + " ".join(
                f"{k} {v['mean']:.2f}[{v['min']:.1f},{v['max']:.1f}]"
                for k, v in c.items() if k != "n"))
    for h in result.get("hysteresis", []):
        print("hysteresis", {k: v for k, v in h.items() if k != "trajectory"})
