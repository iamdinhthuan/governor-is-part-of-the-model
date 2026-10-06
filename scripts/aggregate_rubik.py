#!/usr/bin/env python3
"""Aggregate Rubik Pi 3 campaign results into results/paper/rubik.json.

Reads results/rubik/<phase>/<condition>/r<k>/report.json after rsync from
the Rubik Pi 3. Clusters runs by phase+condition (reps become mean+std).
Includes clock and thermal stats from the rubik sampler (no power sensor
on this board -- disclosed in the paper).
"""
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results" / "rubik"


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": statistics.fmean(vals), "std": statistics.stdev(vals),
            "n": len(vals)}


def main() -> None:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = RESULTS
    clusters = {}
    for rep in sorted(root.rglob("report.json")):
        rel = rep.relative_to(root).parts  # (phase, condition, rN, report.json)
        if len(rel) != 4:
            continue
        phase, condition = rel[0], rel[1]
        d = json.loads(rep.read_text())
        if d.get("images", 0) == 0:
            continue
        key = f"{phase}|{condition}"
        entry = clusters.setdefault(key, {"phase": phase, "condition": condition,
                                          "runs": []})
        clocks = d.get("clocks") or {}
        entry["runs"].append({
            "rep": rel[2],
            "images": d.get("images"),
            "images_per_s": d.get("images_per_s"),
            "latency_median_ms": d.get("latency_median_ms"),
            "latency_p95_ms": d.get("latency_p95_ms"),
            "infer_median_ms": d.get("infer_median_ms"),
            "cpu_ms_per_image": d.get("cpu_ms_per_image"),
            "predictions": d.get("predictions"),
            "cpu0_mhz": clocks.get("cpu0_mhz_mean"),
            "cpu4_mhz": clocks.get("cpu4_mhz_mean"),
            "cpu7_mhz": clocks.get("cpu7_mhz_mean"),
            "soc_temp_c_max": clocks.get("soc_temp_c_max"),
        })
    # HTP stream runs assign one repeat per governor, so averaging them
    # would mix governors; emit one cell per run for the stream phase.
    if root.name.startswith("rubik_htp") and any(k.startswith("stream|") for k in clusters):
        for key in [k for k in clusters if k.startswith("stream|")]:
            entry = clusters.pop(key)
            for run in entry["runs"]:
                clusters[f"{key}|{run['rep']}"] = {**entry, "runs": [run]}
    out = {}
    for key, entry in clusters.items():
        runs = entry["runs"]
        out[key] = {
            "phase": entry["phase"], "condition": entry["condition"],
            "reps": len(runs),
            "images_per_s": mean_std([r["images_per_s"] for r in runs]),
            "latency_median_ms": mean_std([r["latency_median_ms"] for r in runs]),
            "latency_p95_ms": mean_std([r["latency_p95_ms"] for r in runs]),
            "infer_median_ms": mean_std([r["infer_median_ms"] for r in runs]),
            "cpu_ms_per_image": mean_std([r["cpu_ms_per_image"] for r in runs]),
            "cpu0_mhz": mean_std([r["cpu0_mhz"] for r in runs]),
            "cpu4_mhz": mean_std([r["cpu4_mhz"] for r in runs]),
            "cpu7_mhz": mean_std([r["cpu7_mhz"] for r in runs]),
            "soc_temp_c_max": mean_std([r["soc_temp_c_max"] for r in runs]),
            "runs": runs,
        }
    dest = root.parent / "paper" / f"{root.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} ({len(out)} conditions)")
    print(f"\n{'condition':38s} {'img/s':>14s} {'p50 ms':>8s} {'infer ms':>9s} "
          f"{'cpu7 MHz':>9s} {'Tmax C':>7s}")
    for key in sorted(out):
        if not key.startswith("ladder") and not key.startswith("accuracy"):
            continue
        e = out[key]
        ips = e["images_per_s"]
        p50 = e["latency_median_ms"]
        inf = e["infer_median_ms"]
        c7 = e["cpu7_mhz"]
        tmax = e["soc_temp_c_max"]
        print(f"{key:38s} {ips['mean']:6.2f}±{ips['std']:<5.2f} "
              f"{p50['mean']:8.1f} {inf['mean']:9.1f} "
              f"{c7['mean'] if c7 else float('nan'):9.0f} "
              f"{tmax['mean'] if tmax else float('nan'):7.1f}")


if __name__ == "__main__":
    main()
