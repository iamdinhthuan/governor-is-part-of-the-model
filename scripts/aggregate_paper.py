"""Aggregate the Jetson paper campaign (results/paper/raw) into summary JSON.

Each (phase, clocks, config) cell is summarized over its repeats: mean and
[min, max] of throughput, energy per image, mean power, clocks and latency,
plus the set of prediction hashes (must be a single hash per cell) and the
official COCOeval summary for that hash when it was evaluated.
"""

import json
from pathlib import Path
import statistics

RAW = Path("results/paper/raw")
OUT = Path("results/paper/summary.json")


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def ap_lookup():
    index_path = RAW / "ap" / "index.json"
    if not index_path.exists():
        return {}
    index = json.loads(index_path.read_text())
    return {digest: json.loads((RAW / "ap" / entry["summary"]).read_text())
            for digest, entry in index.items() if (RAW / "ap" / entry["summary"]).exists()}


def matrix_cells(phase, aps):
    cells = {}
    for clocks_dir in sorted((RAW / phase).glob("*")):
        for config_dir in sorted(clocks_dir.glob("*")):
            reports, hashes = [], set()
            for run in sorted(config_dir.glob("r*")):
                report = run / "report.json"
                if not report.exists():
                    continue
                data = json.loads(report.read_text())
                if "power" not in data:
                    continue
                reports.append(data)
                hashes.add((run / "predictions.sha256").read_text().strip())
            if not reports:
                continue
            spec = json.loads((config_dir / "r0" / "spec.json").read_text())
            power = [r["power"] for r in reports]
            cell = {
                "spec": {k: spec.get(k) for k in ("model", "rung", "limit", "fps", "extra",
                                                  "condition") if k in spec},
                "repeats": len(reports),
                "prediction_hashes": sorted(hashes),
                "throughput": stats([r["throughput_images_per_s"] for r in reports]),
                "energy_mj": stats([p["energy_per_image_mj"] for p in power]),
                "power_w": stats([p["mean_power_w"] for p in power]),
                "gpu_mhz": stats([p["gpu_mhz_mean"] for p in power]),
                "cpu_mhz": stats([p["max_core_cpu_mhz_mean"] for p in power]),
                "tj_max_c": stats([p["tj_max_c"] for p in power]),
                "latency_median_ms": stats([r["latency_ms"]["median"] for r in reports]),
                "latency_p95_ms": stats([r["latency_ms"]["p95"] for r in reports]),
                "latency_p99_ms": stats([r["latency_p99_ms"] for r in reports]),
                "cpu_ms_per_image": stats([r.get("process_cpu_ms_per_image") for r in reports]),
                "gpu_wall_median_ms": stats([
                    r["gpu_enqueue_to_done_ms"]["median"] for r in reports
                    if "gpu_enqueue_to_done_ms" in r]),
                "policy0_mhz": stats([r.get("policy0_mhz_mean") for r in reports]),
                "prediction_count": reports[0]["prediction_count"],
            }
            if len(hashes) == 1 and next(iter(hashes)) in aps:
                cell["coco"] = aps[next(iter(hashes))]
            cells.setdefault(clocks_dir.name, {})[config_dir.name] = cell
    return cells


def dvfs_cells():
    cells = {}
    for report in sorted((RAW / "dvfs").glob("*/*/idle*/r*/report.json")):
        data = json.loads(report.read_text())
        key = (data["clocks"], data["model"], data["idle_ms"])
        cells.setdefault(key, []).append(data)
    out = {}
    for (clocks, model, idle), runs in sorted(cells.items()):
        out.setdefault(clocks, {}).setdefault(model, {})[str(idle)] = {
            "repeats": len(runs),
            "gpu_compute_median_ms": stats([r["gpu_compute_ms"]["median"] for r in runs]),
            "gpu_compute_mean_ms": stats([r["gpu_compute_ms"]["mean"] for r in runs]),
            "qps": stats([r["qps"] for r in runs]),
            "power_w": stats([r["power"]["mean_power_w"] for r in runs]),
            "energy_mj": stats([r["power"]["energy_per_image_mj"] for r in runs]),
            "gpu_mhz": stats([r["power"]["gpu_mhz_mean"] for r in runs]),
        }
    return out


def idle_cells():
    out = {}
    for clocks_dir in sorted((RAW / "idle").glob("*")):
        runs = [json.loads(p.read_text()) for p in sorted(clocks_dir.glob("r*/report.json"))]
        out[clocks_dir.name] = {
            "repeats": len(runs),
            "power_w": stats([r["power"]["mean_power_w"] for r in runs]),
            "gpu_mhz": stats([r["power"]["gpu_mhz_mean"] for r in runs]),
            "cpu_mhz": stats([r["power"]["max_core_cpu_mhz_mean"] for r in runs]),
        }
    return out


def main() -> None:
    aps = ap_lookup()
    summary = {"ap_by_hash": aps}
    if (RAW / "idle").exists():
        summary["idle"] = idle_cells()
    if (RAW / "dvfs").exists():
        summary["dvfs"] = dvfs_cells()
    for phase in ("ladder", "sched", "stream", "resolution", "generality", "cpufreq"):
        if (RAW / phase).exists():
            summary[phase] = matrix_cells(phase, aps)
    # Same-session uclamp event-sync reference (raw/uclamp_event/uclamp/<clocks>/event/r*).
    ue = RAW / "uclamp_event" / "uclamp"
    if ue.exists():
        cells = {}
        for clocks_dir in sorted(ue.glob("*")):
            for cfg in sorted(clocks_dir.glob("*")):
                runs = [json.loads(p.read_text()) for p in sorted(cfg.glob("r*/report.json"))]
                runs = [r for r in runs if "power" in r]
                if runs:
                    cells.setdefault(clocks_dir.name, {})[cfg.name] = {
                        "repeats": len(runs),
                        "throughput": stats([r["throughput_images_per_s"] for r in runs]),
                        "energy_mj": stats([r["power"]["energy_per_image_mj"] for r in runs]),
                        "power_w": stats([r["power"]["mean_power_w"] for r in runs]),
                        "latency_median_ms": stats([r["latency_ms"]["median"] for r in runs]),
                        "cpu_ms_per_image": stats([r.get("process_cpu_ms_per_image") for r in runs]),
                        "policy0_mhz": stats([r.get("policy0_mhz_mean") for r in runs]),
                        "gpu_wall_median_ms": stats([
                            r["gpu_enqueue_to_done_ms"]["median"] for r in runs
                            if "gpu_enqueue_to_done_ms" in r]),
                    }
        summary["uclamp_event"] = cells
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
