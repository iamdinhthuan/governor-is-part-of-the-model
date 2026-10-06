#!/usr/bin/env python3
"""Aggregate Pi 5 + Hailo-8 campaign results into results/paper/pi.json.

Layout: results/pi/<phase>/<condition>/r<k>/report.json after rsync from
the Pi. The campaign driver has already integrated PMIC power into each
report's "power" dict (SoC rails only -- the Hailo-8 itself is on the
unmonitored 5 V input, disclosed in the paper).
"""
import json
import statistics
import sys
from pathlib import Path


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": statistics.fmean(vals), "std": statistics.stdev(vals),
            "n": len(vals)}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/pi")
    clusters = {}
    for rep in sorted(root.rglob("report.json")):
        rel = rep.relative_to(root).parts
        if len(rel) != 4:
            continue
        phase, condition = rel[0], rel[1]
        d = json.loads(rep.read_text())
        if d.get("images", 0) == 0 and phase not in ("idle", "bench"):
            continue
        key = f"{phase}|{condition}"
        entry = clusters.setdefault(key, {"phase": phase, "condition": condition,
                                          "runs": []})
        power = d.get("power") or {}
        lat = d.get("latency_ms") or {}
        npu = d.get("npu_submit_to_done_ms") or {}
        entry["runs"].append({
            "rep": rel[2],
            "images": d.get("images"),
            "images_per_s": d.get("images_per_s", d.get("throughput_images_per_s")),
            "latency_median_ms": d.get("latency_median_ms", lat.get("median")),
            "latency_p95_ms": d.get("latency_p95_ms", lat.get("p95")),
            "npu_median_ms": d.get("npu_median_ms", npu.get("median")),
            "cpu_ms_per_image": d.get("cpu_ms_per_image",
                                    d.get("process_cpu_ms_per_image")),
            "predictions": d.get("predictions", d.get("prediction_count")),
            "predictions_sha256": d.get("predictions_sha256"),
            "power_w": power.get("mean_power_w"),
            "mj_per_image_soc": power.get("energy_per_image_mj"),
            "cpu_mhz": power.get("cpu_mhz_mean"),
            "temp_c_max": power.get("temp_max_c"),
            "extra": {k: d[k] for k in ("hw_only_fps", "fps") if k in d},
        })
    out = {}
    for key, entry in clusters.items():
        runs = entry["runs"]
        agg = {
            "phase": entry["phase"], "condition": entry["condition"],
            "reps": len(runs),
            "images_per_s": mean_std([r["images_per_s"] for r in runs]),
            "latency_median_ms": mean_std([r["latency_median_ms"] for r in runs]),
            "latency_p95_ms": mean_std([r["latency_p95_ms"] for r in runs]),
            "npu_median_ms": mean_std([r["npu_median_ms"] for r in runs]),
            "cpu_ms_per_image": mean_std([r["cpu_ms_per_image"] for r in runs]),
            "power_w": mean_std([r["power_w"] for r in runs]),
            "mj_per_image_soc": mean_std([r["mj_per_image_soc"] for r in runs]),
            "cpu_mhz": mean_std([r["cpu_mhz"] for r in runs]),
            "temp_c_max": mean_std([r["temp_c_max"] for r in runs]),
            "runs": runs,
        }
        for extra_key in ("hw_only_fps", "fps"):
            vals = [r["extra"].get(extra_key) for r in runs]
            if any(v is not None for v in vals):
                agg[extra_key] = mean_std(vals)
        out[key] = agg
    dest = root.parent / "paper" / "pi.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} ({len(out)} conditions)")
    print(f"\n{'condition':36s} {'img/s':>13s} {'p50 ms':>8s} {'npu ms':>7s} "
          f"{'W':>6s} {'mJ/img':>7s} {'cpu MHz':>8s}")
    for key in sorted(out):
        if not key.startswith("ladder"):
            continue
        e = out[key]
        ips = e["images_per_s"]
        print(f"{key:36s} {ips['mean']:6.1f}±{ips['std']:<5.1f} "
              f"{e['latency_median_ms']['mean']:8.1f} "
              f"{(e['npu_median_ms'] or {'mean': 0})['mean']:7.2f} "
              f"{(e['power_w'] or {'mean': 0})['mean']:6.2f} "
              f"{(e['mj_per_image_soc'] or {'mean': 0})['mean']:7.1f} "
              f"{(e['cpu_mhz'] or {'mean': 0})['mean']:8.0f}")


if __name__ == "__main__":
    main()
