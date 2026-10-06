"""Duty-cycle model checks from the raw campaign data (results/paper/raw).

1. Inverse-frequency engine-time model G(f) = G_max * f_max / f, using the
   time-weighted harmonic-mean GPU clock over each run's timed window.
   Checked on the trtexec idle-gap sweep and on non-overlapped application rungs.
2. GPU utilization at each operating point (busy time per wall time).
3. Energy decomposition E = P_idle / X + E_marg, and its prediction of
   fixed-rate streaming energy per frame.
Writes results/paper/model.json.
"""

import json
from pathlib import Path
import statistics

RAW = Path("results/paper/raw")
S = json.loads(Path("results/paper/summary.json").read_text())
F_MAX = 1020.0


def window_clock(run: Path):
    report = json.loads((run / "report.json").read_text())
    start, end = report["window_unix"]
    mhz = [float(line.split(",")[4]) / 1e6 for line in (run / "power.csv").read_text().splitlines()
           if start <= float(line.split(",")[0]) <= end]
    return statistics.harmonic_mean(mhz), statistics.fmean(mhz)


def g_max(model: str) -> float:
    cell = S["dvfs"]["locked"][model]
    return statistics.fmean(c["gpu_compute_median_ms"]["mean"] for c in cell.values())


def dvfs_check():
    rows = []
    for model in ("y26n_640_uint8", "y26s_640_uint8"):
        base = g_max(model)
        for run in sorted((RAW / "dvfs" / "default" / model).glob("idle*/r*")):
            report = json.loads((run / "report.json").read_text())
            hmean, mean = window_clock(run)
            measured = report["gpu_compute_ms"]["median"]
            predicted = base * F_MAX / hmean
            rows.append({"model": model, "idle_ms": report["idle_ms"], "run": run.name,
                         "f_harmonic_mhz": hmean, "f_mean_mhz": mean,
                         "measured_ms": measured, "predicted_ms": predicted,
                         "error_pct": 100 * (predicted - measured) / measured,
                         "utilization": report["qps"] * report["gpu_compute_ms"]["mean"] / 1000})
    return rows


def app_check():
    base = g_max("y26n_640_uint8")
    rows = []
    for rung in ("uint8_seq_stream", "uint8_seq_event", "uint8_prefetch_stream",
                 "uint8_prefetch_event", "float_seq_stream"):
        for run in sorted((RAW / "ladder" / "default" / rung).glob("r*")):
            report = json.loads((run / "report.json").read_text())
            hmean, mean = window_clock(run)
            measured = report["gpu_enqueue_to_done_ms"]["median"]
            predicted = base * F_MAX / hmean
            rows.append({"rung": rung, "run": run.name, "f_harmonic_mhz": hmean,
                         "f_mean_mhz": mean, "measured_ms": measured,
                         "predicted_ms": predicted,
                         "error_pct": 100 * (predicted - measured) / measured,
                         "utilization": report["throughput_images_per_s"] * measured / 1000})
    return rows


def operating_points():
    """Utilization for every default-governor ladder rung, from G(f_harmonic)."""
    base = g_max("y26n_640_uint8")
    rows = []
    for rung in S["ladder"]["default"]:
        if rung == "ultralytics":
            continue
        freqs = [window_clock(run)[0]
                 for run in sorted((RAW / "ladder" / "default" / rung).glob("r*"))]
        f = statistics.fmean(freqs)
        throughput = S["ladder"]["default"][rung]["throughput"]["mean"]
        rows.append({"rung": rung, "f_harmonic_mhz": f,
                     "f_mean_mhz": S["ladder"]["default"][rung]["gpu_mhz"]["mean"],
                     "utilization_model": throughput * base * F_MAX / f / 1000})
    return rows


def energy_model():
    idle = {k: v["power_w"]["mean"] for k, v in S["idle"].items()}
    ladder = {}
    for clocks, cells in S["ladder"].items():
        for name, cell in cells.items():
            x, p = cell["throughput"]["mean"], cell["power_w"]["mean"]
            ladder[f"{clocks}/{name}"] = {
                "throughput": x, "power_w": p, "energy_mj": cell["energy_mj"]["mean"],
                "idle_share_mj": 1000 * idle[clocks] / x,
                "marginal_mj": 1000 * (p - idle[clocks]) / x}
    stream = {}
    marg = {name: ladder[f"default/{name}"]["marginal_mj"] for name in S["ladder"]["default"]}
    alias = {"ultra": "ultralytics"}
    for name, cell in S.get("stream", {}).get("default", {}).items():
        pipe, fps = name.rsplit("_fps", 1)
        rate = cell["throughput"]["mean"]
        predicted = 1000 * idle["default"] / rate + marg[alias.get(pipe, pipe)]
        stream[name] = {"rate": rate, "measured_mj": cell["energy_mj"]["mean"],
                        "predicted_mj": predicted,
                        "error_pct": 100 * (predicted - cell["energy_mj"]["mean"])
                        / cell["energy_mj"]["mean"]}
    return {"idle_w": idle, "ladder": ladder, "stream": stream}


def power_dips(threshold_w: float = 2.0, edge_s: float = 1.0):
    """Power-dip events (entries below median - threshold) inside each run's timed window.

    The first and last second are excluded because board power ramps up at window start.
    """
    rows = []
    for clocks in ("default", "locked"):
        for rung in ("uint8_prefetch_event", "uint8_overlap_event_w2",
                     "uint8_adaptive_event_w2", "uint8_overlap_event_w2_nogc"):
            for run in sorted((RAW / "ladder" / clocks / rung).glob("r*")):
                report = json.loads((run / "report.json").read_text())
                start, end = report["window_unix"]
                pts = []
                for line in (run / "power.csv").read_text().splitlines():
                    f = line.split(",")
                    t = float(f[0]) - start
                    if edge_s <= t <= end - start - edge_s:
                        pts.append((t, float(f[1]) * float(f[2]) / 1e6))
                watts = [w for _, w in pts]
                floor = statistics.median(watts) - threshold_w
                onsets = [round(t, 2) for (t, w), (_, w0) in zip(pts[1:], pts)
                          if w < floor <= w0]
                rows.append({"clocks": clocks, "rung": rung, "run": run.name,
                             "power_sd_w": statistics.pstdev(watts),
                             "dip_onsets_s": onsets, "dips": len(onsets)})
    return rows


def main():
    out = {"g_max_ms": {m: g_max(m) for m in ("y26n_640_uint8", "y26s_640_uint8")},
           "dvfs": dvfs_check(), "app": app_check(),
           "operating_points": operating_points(), "energy": energy_model(),
           "power_dips": power_dips()}
    errors = [abs(r["error_pct"]) for r in out["dvfs"] if r["f_harmonic_mhz"] < 1019]
    out["dvfs_abs_error_pct_downclocked"] = {"mean": statistics.fmean(errors),
                                             "max": max(errors), "n": len(errors)}
    Path("results/paper/model.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["g_max_ms"]))
    print("dvfs |err| (down-clocked points):", out["dvfs_abs_error_pct_downclocked"])
    for r in out["dvfs"]:
        print(f"  {r['model']} gap {r['idle_ms']:2d} {r['run']} f_h {r['f_harmonic_mhz']:6.0f} "
              f"meas {r['measured_ms']:6.2f} pred {r['predicted_ms']:6.2f} "
              f"err {r['error_pct']:+5.1f}% util {r['utilization']:.2f}")
    for r in out["app"]:
        print(f"  app {r['rung']:22s} {r['run']} f_h {r['f_harmonic_mhz']:6.0f} "
              f"meas {r['measured_ms']:6.2f} pred {r['predicted_ms']:6.2f} "
              f"err {r['error_pct']:+5.1f}% util {r['utilization']:.2f}")
    for r in out["operating_points"]:
        print(f"  op {r['rung']:24s} f_h {r['f_harmonic_mhz']:6.0f} u {r['utilization_model']:.2f}")
    for r in out["power_dips"]:
        print(f"  dips {r['clocks']:7s} {r['rung']:28s} {r['run']} sd {r['power_sd_w']:.2f} W "
              f"dips {r['dips']} at {r['dip_onsets_s']}")
    for k, v in out["energy"]["stream"].items():
        print(f"  stream {k:32s} meas {v['measured_mj']:6.1f} pred {v['predicted_mj']:6.1f} "
              f"err {v['error_pct']:+5.1f}%")


if __name__ == "__main__":
    main()
