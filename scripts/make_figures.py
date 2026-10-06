"""Figures for the paper from results/paper/summary.json (run via uv with matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = json.loads(Path("results/paper/summary.json").read_text())
FIG = Path("paper/figs")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "legend.frameon": False})
W1, W2 = 3.4, 7.0
COLORS = {"default": "#1f77b4", "locked": "#d62728"}

LADDER_ORDER = [
    ("ultralytics", "Ultralytics"),
    ("float_seq_stream", "float, seq"),
    ("uint8_seq_stream", "+uint8 in-graph"),
    ("uint8_seq_event", "+event sync"),
    ("uint8_prefetch_event", "+prefetch"),
    ("uint8_overlap_event", "+overlap"),
    ("uint8_overlap_event_w2", "+2 decoders"),
    ("float_overlap_event_w2", "float, full sched."),
]


def err(cell, key):
    s = cell[key]
    return s["mean"], [[s["mean"] - s["min"]], [s["max"] - s["mean"]]]


def ladder():
    cells = S["ladder"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.1))
    width = 0.38
    for k, clocks in enumerate(("default", "locked")):
        for j, (metric, label) in enumerate((("throughput", "images/s"),
                                             ("energy_mj", "mJ / image"),
                                             ("power_w", "input W"))):
            ax = axes[j]
            for i, (name, _) in enumerate(LADDER_ORDER):
                if name not in cells.get(clocks, {}):
                    continue
                mean, e = err(cells[clocks][name], metric)
                ax.bar(i + (k - 0.5) * width, mean, width, yerr=e, capsize=1.2,
                       color=COLORS[clocks], label=clocks if i == 0 else None,
                       error_kw={"linewidth": 0.5})
            ax.set_xticks(range(len(LADDER_ORDER)))
            ax.set_xticklabels([l for _, l in LADDER_ORDER], rotation=55, ha="right")
            ax.set_ylabel(label)
            ax.grid(axis="y", linewidth=0.3, alpha=0.5)
            ax.set_axisbelow(True)
    for clocks in ("default", "locked"):
        if clocks in S.get("idle", {}):
            axes[2].axhline(S["idle"][clocks]["power_w"]["mean"], color=COLORS[clocks],
                            linestyle="--", linewidth=0.7)
    axes[2].text(len(LADDER_ORDER) - 0.5, S["idle"]["default"]["power_w"]["mean"] - 1.2,
                 "idle", fontsize=6, ha="right")
    axes[0].legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "ladder.pdf")


def dvfs():
    cells = S["dvfs"]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.15))
    for model, marker, label in (("y26n_640_uint8", "o", "YOLO26n"),
                                 ("y26s_640_uint8", "s", "YOLO26s")):
        for clocks in ("default", "locked"):
            series = cells.get(clocks, {}).get(model)
            if not series:
                continue
            idle = sorted(series, key=int)
            x = [int(v) for v in idle]
            meds = [series[v]["gpu_compute_median_ms"] for v in idle]
            axes[0].errorbar(x, [m["mean"] for m in meds],
                             yerr=[[m["mean"] - m["min"] for m in meds],
                                   [m["max"] - m["mean"] for m in meds]],
                             marker=marker, markersize=2.5, linewidth=0.8, color=COLORS[clocks],
                             linestyle="-" if model.startswith("y26n") else "--",
                             capsize=1.2, elinewidth=0.5,
                             label=f"{label}, {clocks}")
            axes[1].plot(x, [series[v]["gpu_mhz"]["mean"] for v in idle], marker=marker,
                         markersize=2.5, linewidth=0.8, color=COLORS[clocks],
                         linestyle="-" if model.startswith("y26n") else "--")
    axes[0].set_xlabel("requested host idle time (ms)")
    axes[0].set_ylabel("compute median (ms)")
    axes[1].set_xlabel("requested host idle time (ms)")
    axes[1].set_ylabel("GPU clock (MHz)")
    for ax in axes:
        ax.grid(linewidth=0.3, alpha=0.5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=6, loc="upper center", ncol=2, columnspacing=0.8,
               handlelength=1.6, handletextpad=0.3)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.8))
    fig.savefig(FIG / "dvfs.pdf")


STREAM_PIPES = [("ultra", "Ultralytics"), ("float_seq_stream", "float, seq"),
                ("uint8_seq_event", "uint8, seq, event"),
                ("uint8_prefetch_event", "uint8, prefetch"),
                ("uint8_overlap_event_w2", "uint8, full"),
                ("uint8_adaptive_event_w2", "uint8, arrival-aware")]


def stream():
    cells = S["stream"]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.8))
    overload_marked = False
    for pipe, label in STREAM_PIPES:
        for clocks, style in (("default", "-"), ("locked", ":")):
            group = cells.get(clocks, {})
            fps = [f for f in (15, 30, 60) if f"{pipe}_fps{f}" in group]
            if not fps:
                continue
            p95 = [group[f"{pipe}_fps{f}"]["latency_p95_ms"]["mean"] for f in fps]
            power = [group[f"{pipe}_fps{f}"]["energy_mj"]["mean"] for f in fps]
            # A cell is overloaded when the processed rate falls short of arrivals;
            # its latency then reflects a growing queue, not a steady state.
            over = [group[f"{pipe}_fps{f}"]["throughput"]["mean"] < 0.98 * f for f in fps]
            line, = axes[0].plot(fps, p95, style, marker="o", markersize=2.5, linewidth=0.8,
                                 label=label if clocks == "default" else None)
            axes[1].plot(fps, power, style, marker="o", markersize=2.5, linewidth=0.8,
                         color=line.get_color())
            for x, y, ov in zip(fps, p95, over):
                if ov:
                    axes[0].plot(x, y, marker="o", markersize=5.5, markerfacecolor="none",
                                 markeredgecolor=line.get_color(), markeredgewidth=0.8,
                                 linestyle="none",
                                 label="overloaded (finite run, queue grows)"
                                 if not overload_marked else None)
                    overload_marked = True
    idle_w = S["idle"]["default"]["power_w"]["mean"]
    rates = [15, 30, 60]
    axes[1].plot(rates, [idle_w / r * 1000 for r in rates], "--", color="0.5", linewidth=0.8,
                 label="idle reference")
    axes[1].legend(fontsize=5.5, loc="upper right")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("arrival rate (fps)")
    axes[0].set_ylabel("p95 latency (ms)")
    axes[1].set_xlabel("arrival rate (fps)")
    axes[1].set_ylabel("mJ / processed frame")
    for ax in axes:
        ax.set_xticks([15, 30, 60])
        ax.grid(linewidth=0.3, alpha=0.5)
    axes[0].legend(fontsize=5.5, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "stream.pdf")


def resolution():
    cells = S["resolution"]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.8))
    for pipe, label, marker in (("float_seq_stream", "naive (float, seq)", "o"),
                                ("uint8_overlap_event_w2", "optimized", "s"),
                                ("uint8_overlap_event_w2_nogc", "optimized, GC off", "^")):
        for clocks, style in (("default", "-"), ("locked", ":")):
            group = cells.get(clocks, {})
            sizes = [s for s in (512, 576, 640) if f"{s}_{pipe}" in group]
            if not sizes:
                continue
            ap = [group[f"{s}_{pipe}"]["coco"]["ap50_95"] for s in sizes
                  if "coco" in group[f"{s}_{pipe}"]]
            if len(ap) != len(sizes):
                continue
            thr_s = [group[f"{s}_{pipe}"]["throughput"] for s in sizes]
            mj_s = [group[f"{s}_{pipe}"]["energy_mj"] for s in sizes]
            thr = [t["mean"] for t in thr_s]
            mj = [m["mean"] for m in mj_s]
            color = ("#7f7f7f" if pipe.startswith("float")
                     else "#2ca02c" if pipe.endswith("nogc") else "#1f77b4")
            axes[0].errorbar(thr, ap,
                             xerr=[[t["mean"] - t["min"] for t in thr_s],
                                   [t["max"] - t["mean"] for t in thr_s]],
                             fmt=style, marker=marker, markersize=3, linewidth=0.8,
                             color=color, capsize=1.2, elinewidth=0.5,
                             label=f"{label}, {clocks}")
            axes[1].errorbar(mj, ap,
                             xerr=[[m["mean"] - m["min"] for m in mj_s],
                                   [m["max"] - m["mean"] for m in mj_s]],
                             fmt=style, marker=marker, markersize=3, linewidth=0.8,
                             color=color, capsize=1.2, elinewidth=0.5)
            for s, x, y in zip(sizes, thr, ap):
                if pipe.endswith("nogc"):
                    ha, off = "left", (4, 3)
                elif pipe.startswith("uint8"):
                    ha, off = "right", (-4, -2)
                else:
                    ha, off = "left", (4, -2)
                axes[0].annotate(str(s), (x, y), fontsize=5, ha=ha,
                                 xytext=off, textcoords="offset points")
            for s, x, y in zip(sizes, mj, ap):
                if pipe.endswith("nogc"):
                    continue  # sizes are identified in the left panel
                axes[1].annotate(str(s), (x, y), fontsize=5, ha="left",
                                 xytext=(4, -2), textcoords="offset points")
    axes[0].set_xlabel("images/s")
    axes[0].set_ylabel("COCO AP50-95")
    axes[1].set_xlabel("mJ / image")
    for ax in axes:
        ax.grid(linewidth=0.3, alpha=0.5)
    axes[0].set_xlim(0, 250)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="upper center", ncol=2)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "resolution.pdf")


if __name__ == "__main__":
    for name, fn in (("ladder", ladder), ("dvfs", dvfs), ("stream", stream),
                     ("resolution", resolution)):
        if name in S:
            fn()
            print("figure", name)


RAW = Path("results/paper/raw")


def load_trace(run: Path, seconds: float = 12.0):
    report = json.loads((run / "report.json").read_text())
    start, end = report["window_unix"]
    t, w, mhz = [], [], []
    for line in (run / "power.csv").read_text().splitlines():
        f = line.split(",")
        ts = float(f[0])
        if start <= ts <= min(end, start + seconds):
            t.append(ts - start)
            w.append(float(f[1]) * float(f[2]) / 1e6)
            mhz.append(float(f[4]) / 1e6)
    return t, w, mhz


def traces():
    runs = [("uint8_seq_stream", "sequential, stream sync", "#7f7f7f"),
            ("uint8_seq_event", "sequential, event sync", "#2ca02c"),
            ("uint8_prefetch_event", "prefetch", "#ff7f0e"),
            ("uint8_overlap_event_w2", "full schedule", "#1f77b4")]
    # Third panel: the lock-then-release experiment (extras.json hysteresis).
    # Each entry carries [t_offset_s, gpu_mhz, gpu_load] samples; t=0 is the
    # moment jetson_clocks was released.
    try:
        hyst = json.loads(Path("results/paper/extras.json").read_text())["hysteresis"]
    except Exception:
        hyst = None
    nrows = 3 if hyst else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(W1, 2.6 + 1.1 * (nrows - 2)),
                             sharex=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.9][:nrows]})
    for rung, label, color in runs:
        t, w, mhz = load_trace(RAW / "ladder" / "default" / rung / "r0")
        axes[0].plot(t, mhz, color=color, linewidth=0.7, label=label)
        axes[1].plot(t, w, color=color, linewidth=0.5)
    axes[1].axhline(S["idle"]["default"]["power_w"]["mean"], color="k", linestyle="--",
                    linewidth=0.6)
    axes[1].text(11.8, S["idle"]["default"]["power_w"]["mean"] + 0.3, "idle", fontsize=6,
                 ha="right")
    axes[1].axhline(S["idle"]["locked"]["power_w"]["mean"], color="#d62728", linestyle=":",
                    linewidth=0.6)
    axes[1].text(0.2, S["idle"]["locked"]["power_w"]["mean"] + 0.3, "idle, locked",
                 fontsize=6, ha="left", color="#d62728")
    axes[0].set_ylabel("GPU clock (MHz)")
    axes[1].set_ylabel("module input power (W)")
    axes[1].set_xlabel("time in timed window (s)")
    axes[0].legend(fontsize=5.5, loc="lower right", bbox_to_anchor=(1.0, 0.06), ncol=2,
                   columnspacing=0.8, handlelength=1.2)
    if hyst:
        ax3 = axes[2]
        ax3b = ax3.twinx()
        for k, run in enumerate(hyst):
            tr = run["trajectory"]
            t = [p[0] for p in tr]
            mhz = [p[1] for p in tr]
            load = [p[2] for p in tr]
            color = "#1f77b4" if k == 0 else "#7f7f7f"
            ax3.plot(t, mhz, color=color, linewidth=0.8,
                     label="GPU clock" if k == 0 else None)
            ax3b.plot(t, load, color=color, linewidth=0.6, linestyle="--",
                      label="GPU load" if k == 0 else None)
        ax3b.set_ylim(0, 1.05)
        ax3b.set_ylabel("driver GPU load", fontsize=6)
        ax3b.tick_params(labelsize=5.5)
        ax3.axvline(0, color="k", linewidth=0.6, linestyle=":")
        ax3.text(0.1, 1050, "lock released", fontsize=5.5, va="top")
        ax3.set_xlabel("time from release (s)")
        ax3.set_ylabel("GPU clock (MHz)")
        ax3.set_ylim(250, 1100)
        ax3.grid(linewidth=0.3, alpha=0.5)
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax3b.get_legend_handles_labels()
        ax3.legend(h1 + h2, l1 + l2, fontsize=5.5, loc="center right")
    for ax in list(axes):
        ax.grid(linewidth=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "traces.pdf")


def measured_load(rung: str):
    """Mean driver GPU load and harmonic-mean GPU clock in the timed window of the load phase."""
    run = RAW / "load" / "default" / rung / "r0"
    if not (run / "counters.csv").exists():
        return None
    start, end = json.loads((run / "report.json").read_text())["window_unix"]
    rows = [line.split(",") for line in (run / "counters.csv").read_text().splitlines()]
    rows = [r for r in rows if start <= float(r[0]) <= end]
    load = sum(int(r[1]) for r in rows) / len(rows) / 1000
    harmonic = len(rows) / sum(1e6 / int(r[2]) for r in rows)
    return load, harmonic


def operating():
    model = json.loads(Path("results/paper/model.json").read_text())
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for name, marker, label in (("y26n_640_uint8", "o", "trtexec sweep, YOLO26n (est.)"),
                                ("y26s_640_uint8", "s", "trtexec sweep, YOLO26s (est.)")):
        pts = [r for r in model["dvfs"] if r["model"] == name]
        ax.scatter([r["utilization"] for r in pts], [r["f_harmonic_mhz"] for r in pts],
                   marker=marker, s=10, facecolors="none", edgecolors="#555555",
                   linewidths=0.6, label=label)
    labels = {"uint8_seq_stream": "seq/stream", "uint8_seq_event": "seq/event",
              "float_seq_stream": "float seq", "uint8_prefetch_stream": "prefetch/stream",
              "uint8_prefetch_event": "prefetch/event", "uint8_overlap_event": "overlap",
              "uint8_overlap_event_w2": "full", "float_overlap_event_w2": "float full"}
    pts = {rung: measured_load(rung) for rung in labels}
    pts = {k: v for k, v in pts.items() if v}
    ax.scatter([v[0] for v in pts.values()], [v[1] for v in pts.values()],
               marker="D", s=12, color="#d62728", label="application rungs (measured load)",
               zorder=3)
    offsets = {"seq/event": (-6, -11), "seq/stream": (-6, -11), "float seq": (-30, -11),
               "prefetch/stream": (-52, -2), "prefetch/event": (-50, 0), "overlap": (-30, -2),
               "full": (-12, -9), "float full": (4, -3)}
    for rung, (load, mhz) in pts.items():
        text = labels[rung]
        ax.annotate(text, (load, mhz), fontsize=5, xytext=offsets.get(text, (3, 3)),
                    textcoords="offset points")
    ax.axvspan(0.53, 0.74, color="#1f77b4", alpha=0.08, linewidth=0)
    ax.annotate("observed operating-load range\n(not identified thresholds)",
                xy=(0.74, 380), xytext=(0.80, 240), fontsize=5, color="#1f77b4",
                ha="center", arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                             color="#1f77b4"))
    ax.set_xlabel("GPU busy fraction / driver GPU load")
    ax.set_ylabel("GPU clock (MHz)")
    ax.set_xlim(0.25, 1.03)
    ax.set_ylim(200, 1080)
    ax.grid(linewidth=0.3, alpha=0.5)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, fontsize=5.5, loc="upper center", ncol=2,
               columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.86))
    fig.savefig(FIG / "operating.pdf")


def timeline():
    """Schematic schedules; stage durations are the measured medians of Sec. 4."""
    dec, lb, rows = 7.5, 3.9, 1.7
    g_slow, g_fast = 13.2, 4.4
    fig, axes = plt.subplots(3, 1, figsize=(W2, 1.9), sharex=True)
    colors = {"decode": "#9ecae1", "letterbox": "#c6dbef", "GPU": "#fd8d3c", "rows": "#a1d99b"}

    def bar(ax, y, start, length, kind, text=None):
        ax.broken_barh([(start, length)], (y - 0.35, 0.7), facecolors=colors[kind],
                       edgecolor="k", linewidth=0.3)
        if text and length > 3:
            ax.text(start + length / 2, y, text, ha="center", va="center", fontsize=5,
                    clip_on=True)

    def ready(ax, x, y):
        ax.plot(x, y, marker="v", color="k", markersize=2.6, clip_on=False, zorder=5)

    ax = axes[0]
    t = 0.0
    for i in range(3):
        bar(ax, 1, t, dec, "decode", f"decode {i}"); t += dec
        bar(ax, 1, t, lb, "letterbox"); t += lb
        bar(ax, 0, t, g_slow, "GPU", f"GPU {i} @306 MHz"); t += g_slow
        bar(ax, 1, t, rows, "rows"); t += rows
        ready(ax, t, 1.42)
    ax.set_title("(a) sequential: GPU idle during host work, governor holds 306 MHz",
                 fontsize=6.5, loc="left")
    ax = axes[1]
    host = dec + lb
    for i in range(6):
        bar(ax, 2, i * host, host, "decode", f"decode {i}")
    t = host
    for i in range(5):
        start = max(t, (i + 1) * host)
        bar(ax, 0, start, g_fast, "GPU", f"GPU {i}")
        bar(ax, 1, start + g_fast, rows, "rows")
        t = start + g_fast + rows
        ready(ax, t, 1.42)
    ax.set_title("(b) prefetch: decode of $i{+}1$ overlaps GPU of $i$; clock rises (911 MHz)",
                 fontsize=6.5, loc="left")
    ax = axes[2]
    for w in range(2):
        for i in range(8):
            bar(ax, 2 + w, w * host / 2 + i * host, host, "decode", f"dec {2 * i + w}")
    for i in range(14):
        start = host + i * host / 2
        bar(ax, 0, start, g_fast, "GPU")
        bar(ax, 1, start + g_fast, rows, "rows")
        ready(ax, start + g_fast + rows, 1.42)
    ax.set_title("(c) full: two decoders, rows of $i{-}1$ overlap GPU of $i$; GPU near 1020 MHz",
                 fontsize=6.5, loc="left")
    for ax, labels in zip(axes, (["GPU", "CPU"], ["GPU", "rows", "worker"],
                                 ["GPU", "rows", "worker 1", "worker 2"])):
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.tick_params(axis="x", labelsize=6)
    axes[2].set_xlim(0, 95)
    axes[2].set_xlabel("time (ms), illustrative from measured stage medians", fontsize=6.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "timeline.pdf")


EXTRA = {"traces": traces, "operating": operating, "timeline": timeline}
if __name__ == "__main__":
    for name, fn in EXTRA.items():
        fn()
        print("figure", name)
