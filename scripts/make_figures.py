"""Figures for the paper from results/paper/*.json and raw traces (run via uv with matplotlib).

Style follows the scientific-figure-making skill (figures4papers): Helvetica/Arial,
top/right spines off, frameless legends, no background grid, the skill's semantic
palette (blue = deployment default / key result, red = locked-clock contrast,
greens = improvements, neutrals = baselines), black-edged bars with in-place
values and hatching for print-safe separation.

The skill's demos are drawn on large canvases (font 16-24 pt) and scaled down by
LaTeX. Here every figure is drawn at its final printed width instead (cas-dc
column 3.30 in, text 6.84 in) with the equivalent printed sizes, and saved
without bbox trimming, so the sizes set here are the sizes in the PDF.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

S = json.loads(Path("results/paper/summary.json").read_text())
RAW = Path("results/paper/raw")
FIG = Path("paper/figs")
FIG.mkdir(parents=True, exist_ok=True)
PREVIEW = Path("paper/figs_preview")

W1, W2 = 3.30, 6.84
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.8,
    "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
    "axes.linewidth": 0.9, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "axes.labelpad": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "lines.linewidth": 1.3, "lines.markersize": 4.2, "lines.solid_capstyle": "round",
    "errorbar.capsize": 1.5, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
})

# Semantic palette of the scientific-figure-making skill.
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
    "grey": "#767676", "dark": "#4D4D4D", "black": "#272727",
}
BLUE, BLUE2 = PALETTE["blue_main"], PALETTE["blue_secondary"]
RED, RED2, RED1 = PALETTE["red_strong"], PALETTE["red_2"], PALETTE["red_1"]
GREEN, GREEN2, GREEN1 = PALETTE["green_3"], PALETTE["green_2"], PALETTE["green_1"]
TEAL, VIOLET, NEUTRAL = PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"]
GREY, DARK, BLACK = PALETTE["grey"], PALETTE["dark"], PALETTE["black"]
CLOCK = {"default": BLUE, "locked": RED}
CLOCK_LABEL = {"default": "default governors", "locked": "locked (jetson_clocks)"}


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold", pad=4)


def finalize(fig, name):
    """Vector PDF for the paper plus a 300 dpi PNG preview."""
    fig.savefig(FIG / f"{name}.pdf")
    PREVIEW.mkdir(exist_ok=True)
    fig.savefig(PREVIEW / f"{name}.png", dpi=300)
    plt.close(fig)


def err(cell, key):
    s = cell[key]
    return s["mean"], [[s["mean"] - s["min"]], [s["max"] - s["mean"]]]


# --------------------------------------------------------------------------
# Fig. dvfs: trtexec idle-gap sweep

def floor_lines(ax, labels=True, x=0.99):
    for mhz, text in ((306, "306 MHz floor"), (1020, "1020 MHz max")):
        ax.axhline(mhz, color=BLACK, linestyle="--", linewidth=0.7, alpha=0.3, zorder=0)
        if labels:
            ax.text(x, mhz + 18, text, transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=6, color=DARK)


def dvfs():
    cells = S["dvfs"]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.35))
    for model, marker, ls in (("y26n_640_uint8", "o", "-"), ("y26s_640_uint8", "s", "--")):
        for clocks in ("default", "locked"):
            series = cells.get(clocks, {}).get(model)
            if not series:
                continue
            idle = sorted(series, key=int)
            x = [int(v) for v in idle]
            meds = [series[v]["gpu_compute_median_ms"] for v in idle]
            c = CLOCK[clocks]
            kw = dict(marker=marker, linestyle=ls, color=c, markeredgecolor=c,
                      markeredgewidth=0.9, markerfacecolor="white" if clocks == "locked" else c)
            axes[0].errorbar(x, [m["mean"] for m in meds],
                             yerr=[[m["mean"] - m["min"] for m in meds],
                                   [m["max"] - m["mean"] for m in meds]],
                             elinewidth=0.7, **kw)
            axes[1].plot(x, [series[v]["gpu_mhz"]["mean"] for v in idle], **kw)
    panel(axes[0], "a", "Engine compute time")
    panel(axes[1], "b", "Mean GPU clock")
    axes[0].set_ylabel("median compute time (ms)")
    axes[1].set_ylabel("GPU clock (MHz)")
    axes[1].set_ylim(200, 1120)
    axes[1].set_yticks([306, 500, 750, 1020])
    floor_lines(axes[1], labels=False)
    for ax in axes:
        ax.set_xticks([0, 4, 8, 16, 33])
        ax.set_xlabel("host idle gap (ms)")
    handles = [Line2D([], [], color=BLUE, marker="o", label="YOLO26n, default"),
               Line2D([], [], color=BLUE, marker="s", ls="--", label="YOLO26s, default"),
               Line2D([], [], color=RED, marker="o", mfc="white", mew=0.9, label="YOLO26n, locked"),
               Line2D([], [], color=RED, marker="s", mfc="white", mew=0.9, ls="--",
                      label="YOLO26s, locked")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, handlelength=2.4)
    finalize(fig, "dvfs")


# --------------------------------------------------------------------------
# Fig. operating: governor operating points

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
    fig, ax = plt.subplots(figsize=(W1, 2.75))
    ax.axvspan(0.53, 0.74, color=BLUE2, alpha=0.13, linewidth=0)
    ax.text(0.635, 1.01, "observed load band\n0.53\u20130.74", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6.5, color=BLUE)
    floor_lines(ax, labels=False)
    for name, marker, label in (("y26n_640_uint8", "o", "trtexec sweep, YOLO26n"),
                                ("y26s_640_uint8", "s", "trtexec sweep, YOLO26s")):
        pts = [r for r in model["dvfs"] if r["model"] == name]
        ax.scatter([r["utilization"] for r in pts], [r["f_harmonic_mhz"] for r in pts],
                   marker=marker, s=18, facecolors="white", edgecolors=GREY,
                   linewidths=0.9, label=label, zorder=2)
    labels = {"float_seq_stream": "float, seq.", "uint8_seq_stream": "seq., stream",
              "uint8_seq_event": "seq., event", "uint8_prefetch_stream": "prefetch, stream",
              "uint8_prefetch_event": "prefetch, event", "uint8_overlap_event": "overlap",
              "uint8_overlap_event_w2": "full", "float_overlap_event_w2": "float, full"}
    pts = {rung: measured_load(rung) for rung in labels}
    pts = {k: v for k, v in pts.items() if v}
    ax.scatter([v[0] for v in pts.values()], [v[1] for v in pts.values()],
               marker="D", s=24, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3,
               label="application pipelines (driver load)")
    # (text position in data units, horizontal alignment)
    place = {"float, seq.": ((0.43, 400), "center"), "seq., stream": ((0.505, 225), "center"),
             "seq., event": ((0.70, 225), "center"), "prefetch, stream": ((0.80, 640), "left"),
             "prefetch, event": ((0.80, 800), "left"), "overlap": ((0.70, 1065), "right"),
             "full": ((0.905, 930), "center"), "float, full": ((0.40, 600), "left")}
    for rung, (load, mhz) in pts.items():
        text = labels[rung]
        (tx, ty), ha = place[text]
        ax.annotate(text, (load, mhz), xytext=(tx, ty), textcoords="data", fontsize=6.5,
                    ha=ha, va="center", color=BLACK,
                    arrowprops=dict(arrowstyle="-", linewidth=0.5, color=GREY,
                                    shrinkA=1.5, shrinkB=2.5))
    ax.set_xlabel("GPU busy fraction (sweep) or driver GPU load (pipelines)")
    ax.set_ylabel("harmonic-mean GPU clock (MHz)")
    ax.set_xlim(0.28, 1.03)
    ax.set_ylim(180, 1110)
    ax.set_yticks([306, 500, 750, 1020])
    fig.legend(loc="outside lower center", ncol=2, handletextpad=0.3)
    finalize(fig, "operating")


# --------------------------------------------------------------------------
# Fig. traces: clock and power traces + lock-then-release

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
    runs = [("uint8_seq_stream", "sequential, stream sync", GREY, (0, (4, 1.5))),
            ("uint8_seq_event", "sequential, event sync", VIOLET, "-"),
            ("uint8_prefetch_event", "prefetch", TEAL, "-"),
            ("uint8_overlap_event_w2", "full schedule", BLUE, "-")]
    hyst = json.loads(Path("results/paper/extras.json").read_text())["hysteresis"]
    fig, axes = plt.subplots(3, 1, figsize=(W1, 4.35),
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    for rung, label, color, ls in runs:
        t, w, mhz = load_trace(RAW / "ladder" / "default" / rung / "r0")
        axes[0].plot(t, mhz, color=color, linestyle=ls, linewidth=1.2, label=label)
        axes[1].plot(t, w, color=color, linestyle=ls, linewidth=1.0)
    idle_d = S["idle"]["default"]["power_w"]["mean"]
    idle_l = S["idle"]["locked"]["power_w"]["mean"]
    axes[1].axhline(idle_d, color=BLACK, linestyle=":", linewidth=1.0)
    axes[1].axhline(idle_l, color=RED, linestyle=":", linewidth=1.0)
    panel(axes[0], "a", "GPU clock, first 12 s of each run")
    panel(axes[1], "b", "Module input power")
    axes[0].set_ylabel("GPU clock (MHz)")
    axes[0].set_ylim(200, 1120)
    axes[0].set_yticks([306, 500, 750, 1020])
    floor_lines(axes[0], labels=False)
    axes[1].set_ylabel("input power (W)")
    axes[1].set_ylim(3, 18.5)
    for ax in axes[:2]:
        ax.set_xlim(0, 12)
        ax.set_xlabel("time in timed window (s)")
    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.2, label=lab)
               for _, lab, c, ls in runs]
    handles += [Line2D([], [], color=BLACK, linestyle=":", linewidth=1.0, label="idle power, default"),
                Line2D([], [], color=RED, linestyle=":", linewidth=1.0, label="idle power, locked")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, handlelength=2.2,
               fontsize=6.6)

    ax3 = axes[2]
    ax3b = ax3.twinx()
    ax3b.spines["right"].set_visible(True)
    for k, run in enumerate(hyst):
        tr = run["trajectory"]
        t = [p[0] for p in tr]
        color = BLUE if k == 0 else TEAL
        ax3b.plot(t, [p[2] for p in tr], color=color, linewidth=0.6, alpha=0.45)
        ax3.plot(t, [p[1] for p in tr], color=color, linewidth=1.5, zorder=3)
    ax3.set_zorder(ax3b.get_zorder() + 1)
    ax3.patch.set_visible(False)
    ax3.axvline(0, color=BLACK, linewidth=0.8, linestyle=":")
    ax3.text(0.06, 215, "lock released", fontsize=6.5, va="bottom", ha="left", color=DARK)
    ax3.set_ylim(200, 1120)
    ax3.set_yticks([306, 500, 750, 1020])
    ax3b.set_ylim(-1.0, 1.05)
    ax3b.set_yticks([0, 0.5, 1.0])
    ax3b.spines["right"].set_bounds(0, 1.0)
    ax3.set_xlabel("time from release (s)")
    ax3.set_ylabel("GPU clock (MHz)")
    ax3b.set_ylabel("GPU load")
    ax3b.yaxis.set_label_coords(1.145, 1.5 / 2.05)  # centre of the bounded 0-1 spine
    panel(ax3, "c", "Sequential pipeline released from 1020 MHz")
    handles = [Line2D([], [], color=BLUE, linewidth=1.5, label="release, repeat 1"),
               Line2D([], [], color=TEAL, linewidth=1.5, label="release, repeat 2"),
               Line2D([], [], color=GREY, linewidth=0.6, alpha=0.7,
                      label="load (right axis)")]
    fig.legend(handles=handles, loc="outside lower center", fontsize=6.5, ncol=3,
               handlelength=1.6)
    finalize(fig, "traces")


# --------------------------------------------------------------------------
# Fig. timeline: schematic schedules

STAGE = {"decode": "#C3D6EC", "letterbox": "#E4EDF7", "GPU": PALETTE["red_2"],
         "rows": PALETTE["green_2"]}


def timeline():
    """Schematic schedules; stage durations are the measured medians of Sec. 5."""
    dec, lb, rows = 7.5, 3.9, 1.7
    g_slow, g_fast = 13.2, 4.4
    XMAX = 80
    colors = STAGE
    fig, axes = plt.subplots(3, 1, figsize=(W2, 2.95), sharex=True,
                             gridspec_kw={"height_ratios": [2, 3, 4]})

    def bar(ax, y, start, length, kind, text=None):
        ax.broken_barh([(start, length)], (y - 0.36, 0.72), facecolors=colors[kind],
                       edgecolor=DARK, linewidth=0.5)
        if text:
            ax.text(start + length / 2, y, text, ha="center", va="center", fontsize=6.5,
                    color=BLACK, clip_on=True)

    def ready(ax, x, y):
        ax.plot(x, y, marker="v", color=BLACK, markersize=4.0, clip_on=False, zorder=5,
                linestyle="none")

    ax = axes[0]
    t = 0.0
    for i in range(3):
        bar(ax, 1, t, dec, "decode", f"decode {i}"); t += dec
        bar(ax, 1, t, lb, "letterbox"); t += lb
        bar(ax, 0, t, g_slow, "GPU", f"engine {i} at 306 MHz"); t += g_slow
        bar(ax, 1, t, rows, "rows"); t += rows
        ready(ax, t, 1.55)
    ax.set_title("(a) Sequential: every host stage is GPU idle time; the governor holds "
                 "306 MHz", loc="left", fontweight="bold", pad=6)
    ax = axes[1]
    host = dec + lb
    for i in range(6):
        bar(ax, 2, i * host, host, "decode", f"decode {i}")
    t = host
    for i in range(5):
        start = max(t, (i + 1) * host)
        bar(ax, 0, start, g_fast, "GPU", f"{i}")
        bar(ax, 1, start + g_fast, rows, "rows")
        t = start + g_fast + rows
        ready(ax, t, 1.55)
    ax.set_title("(b) Prefetch: decode of frame i+1 overlaps the engine of frame i; "
                 "the clock rises (911 MHz)", loc="left", fontweight="bold", pad=6)
    ax = axes[2]
    for w in range(2):
        for i in range(8):
            start = w * host / 2 + i * host
            if start + host <= XMAX:
                bar(ax, 2 + w, start, host, "decode", f"decode {2 * i + w}")
    for i in range(14):
        start = host + i * host / 2
        if start + g_fast + rows > XMAX:
            break
        bar(ax, 0, start, g_fast, "GPU")
        bar(ax, 1, start + g_fast, rows, "rows")
        ready(ax, start + g_fast + rows, 1.55)
    ax.set_title("(c) Full: two decode workers; rows of frame i\u22121 overlap the engine "
                 "of frame i; GPU near 1020 MHz", loc="left", fontweight="bold", pad=6)
    for ax, labels in zip(axes, (["GPU", "CPU"], ["GPU", "rows", "worker"],
                                 ["GPU", "rows", "worker 1", "worker 2"])):
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[2].set_xlim(0, XMAX)
    axes[2].set_xlabel("time (ms); schematic, stage lengths from measured medians")
    handles = [Patch(facecolor=colors[k], edgecolor=DARK, linewidth=0.5, label=lab)
               for k, lab in (("decode", "decode"), ("letterbox", "letterbox"),
                              ("GPU", "engine (GPU)"), ("rows", "detection rows"))]
    handles.append(Line2D([], [], marker="v", color=BLACK, linestyle="none", markersize=4.0,
                          label="detections ready"))
    fig.legend(handles=handles, loc="outside upper right", ncol=5, handlelength=1.2)
    finalize(fig, "timeline")


# --------------------------------------------------------------------------
# Fig. ladder: throughput / energy / power along the ladder

LADDER_ORDER = [
    ("ultralytics", "Ultralytics predictor"),
    ("float_seq_stream", "float input, sequential"),
    ("uint8_seq_stream", "+ uint8 in-graph input"),
    ("uint8_seq_event", "+ event synchronization"),
    ("uint8_prefetch_event", "+ prefetch"),
    ("uint8_overlap_event", "+ overlap"),
    ("uint8_overlap_event_w2", "+ second decode worker"),
    ("float_overlap_event_w2", "float input, full schedule"),
]
BAR_STYLE = {"default": dict(color=BLUE, edgecolor=BLACK, linewidth=0.5),
             "locked": dict(color="white", edgecolor=RED, hatch="//////", linewidth=0.6)}


def ladder():
    cells = S["ladder"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.55), sharey=True)
    h = 0.40
    n = len(LADDER_ORDER)
    fmt = {"throughput": "{:.0f}", "energy_mj": "{:.0f}", "power_w": "{:.1f}"}
    for k, clocks in enumerate(("default", "locked")):
        for j, metric in enumerate(("throughput", "energy_mj", "power_w")):
            ax = axes[j]
            for i, (name, _) in enumerate(LADDER_ORDER):
                if name not in cells.get(clocks, {}):
                    continue
                mean, e = err(cells[clocks][name], metric)
                y = i + (k - 0.5) * h
                ax.barh(y, mean, h, xerr=e, label=CLOCK_LABEL[clocks] if i == 0 else None,
                        error_kw={"linewidth": 0.6, "capsize": 1.2, "ecolor": BLACK},
                        **BAR_STYLE[clocks])
                xt = (mean + e[1][0]) * (1.06 if metric == "throughput" else 1.0)
                pad = 0 if metric == "throughput" else (2.5 if metric == "energy_mj" else 0.25)
                ax.text(xt + pad, y, fmt[metric].format(mean), va="center", ha="left",
                        fontsize=5.6, color=CLOCK[clocks], zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.25))
    panel(axes[0], "a", "Throughput")
    panel(axes[1], "b", "Energy per image")
    panel(axes[2], "c", "Mean input power")
    axes[0].set_xlabel("images/s (log scale)")
    axes[1].set_xlabel("mJ / image")
    axes[2].set_xlabel("module input power (W)")
    axes[0].set_xscale("log")
    axes[0].set_xlim(20, 520)
    axes[0].set_xticks([25, 50, 100, 200])
    axes[0].set_xticklabels(["25", "50", "100", "200"])
    axes[0].minorticks_off()
    axes[1].set_xlim(0, 235)
    axes[2].set_xlim(0, 21.5)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([lab for _, lab in LADDER_ORDER])
    axes[0].set_ylim(n - 0.45, -0.55)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    for clocks in ("default", "locked"):
        p = S["idle"][clocks]["power_w"]["mean"]
        axes[2].axvline(p, color=CLOCK[clocks], linestyle=":", linewidth=1.0, zorder=0.5)
    handles, names = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=BLUE, linestyle=":", linewidth=1.0),
                Line2D([], [], color=RED, linestyle=":", linewidth=1.0)]
    names += ["idle power, default", "idle power, locked"]
    fig.legend(handles, names, loc="outside upper right", ncol=4)
    finalize(fig, "ladder")


# --------------------------------------------------------------------------
# Fig. stream: latency and energy at camera rates

STREAM_PIPES = [("ultra", "Ultralytics", GREY, "o"),
                ("float_seq_stream", "float, sequential", BLACK, "s"),
                ("uint8_seq_event", "sequential, event", RED, "^"),
                ("uint8_prefetch_event", "prefetch", TEAL, "v"),
                ("uint8_overlap_event_w2", "full", BLUE, "D"),
                ("uint8_adaptive_event_w2", "arrival-aware", VIOLET, "P")]


def stream():
    cells = S["stream"]["default"]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.75))
    handles = []
    for pipe, label, color, marker in STREAM_PIPES:
        fps = [f for f in (15, 30, 60) if f"{pipe}_fps{f}" in cells]
        if not fps:
            continue
        c = [cells[f"{pipe}_fps{f}"] for f in fps]
        p95 = [x["latency_p95_ms"]["mean"] for x in c]
        mj = [x["energy_mj"]["mean"] for x in c]
        # Overloaded: processed rate short of arrivals; latency reflects a growing queue.
        over = [x["throughput"]["mean"] < 0.98 * f for x, f in zip(c, fps)]
        for ax, y in ((axes[0], p95), (axes[1], mj)):
            ax.plot(fps, y, color=color, linewidth=1.2, alpha=0.85)
            for xf, yv, ov in zip(fps, y, over):
                ax.plot(xf, yv, marker=marker, color=color, markersize=4.4,
                        markeredgecolor=color, markerfacecolor="white" if ov else color,
                        markeredgewidth=0.9)
        handles.append(Line2D([], [], color=color, marker=marker, markersize=4.4, label=label))
    idle_w = S["idle"]["default"]["power_w"]["mean"]
    rates = [15, 30, 60]
    axes[1].plot(rates, [idle_w / r * 1000 for r in rates], ":", color=BLACK, linewidth=1.0)
    handles.append(Line2D([], [], color=BLACK, linestyle=":", linewidth=1.0,
                          label="idle power / rate"))
    handles.append(Line2D([], [], color=GREY, marker="o", markerfacecolor="white",
                          markeredgewidth=0.9, linestyle="none", markersize=4.4,
                          label="overloaded cell"))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("p95 latency (ms)")
    axes[1].set_ylabel("mJ / processed frame")
    panel(axes[0], "a", "Tail latency")
    panel(axes[1], "b", "Energy per frame")
    for ax in axes:
        ax.set_xticks(rates)
        ax.set_xlabel("arrival rate (fps)")
        ax.set_xlim(10, 65)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, handlelength=2.0)
    finalize(fig, "stream")


# --------------------------------------------------------------------------
# Fig. resolution: AP versus throughput and energy

def resolution():
    cells = S["resolution"]["default"]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.55), sharey=True)
    series = (("float_seq_stream", "naive", GREY, "o"),
              ("uint8_overlap_event_w2", "optimized", BLUE, "s"),
              ("uint8_overlap_event_w2_nogc", "optimized, GC off", TEAL, "^"))
    for pipe, label, color, marker in series:
        sizes = [s for s in (512, 576, 640) if f"{s}_{pipe}" in cells]
        c = [cells[f"{s}_{pipe}"] for s in sizes]
        if not c or any("coco" not in x for x in c):
            continue
        ap = [x["coco"]["ap50_95"] for x in c]
        for ax, key in ((axes[0], "throughput"), (axes[1], "energy_mj")):
            st = [x[key] for x in c]
            ax.errorbar([v["mean"] for v in st], ap,
                        xerr=[[v["mean"] - v["min"] for v in st],
                              [v["max"] - v["mean"] for v in st]],
                        marker=marker, color=color, markeredgecolor=color, markersize=4.4,
                        linewidth=1.2, elinewidth=0.7, label=label if ax is axes[0] else None)
            for s, x, y in zip(sizes, [v["mean"] for v in st], ap):
                if ax is axes[0] and pipe.endswith("w2"):
                    ax.annotate(str(s), (x, y), xytext=(-5, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=6.3, color=DARK)
                elif ax is axes[0] and pipe.startswith("float"):
                    ax.annotate(str(s), (x, y), xytext=(5, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=6.3, color=DARK)
                elif ax is axes[1] and pipe.startswith("float"):
                    ax.annotate(str(s), (x, y), xytext=(-5, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=6.3, color=DARK)
    panel(axes[0], "a", "Throughput")
    panel(axes[1], "b", "Energy")
    axes[0].set_ylabel("COCO AP (0.50:0.95)")
    axes[0].set_xlabel("images/s (log scale)")
    axes[1].set_xlabel("mJ / image")
    axes[0].set_xscale("log")
    axes[0].set_xlim(22, 400)
    axes[0].set_xticks([25, 50, 100, 200])
    axes[0].set_xticklabels(["25", "50", "100", "200"])
    axes[0].minorticks_off()
    axes[1].set_xlim(40, 200)
    fig.legend(loc="outside lower center", ncol=3)
    finalize(fig, "resolution")


# --------------------------------------------------------------------------
# Fig. framework: overview of the study

def framework():
    H = 2.75
    fig = plt.figure(figsize=(W2, H), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W2)
    ax.set_ylim(0, H)
    ax.axis("off")
    host_c, gpu_c, gov_c, meas_c = "#E4EDF7", PALETTE["red_1"], PALETTE["green_1"], "#F2F2F2"
    STYLE = "round,pad=0.0,rounding_size=0.04"

    def rect(x, y, w, h, fc, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=STYLE, fc=fc, ec=DARK, lw=lw))

    def box(x, y, w, h, title, sub=None, fc="white", fs=6.8, sfs=6.0):
        rect(x, y, w, h, fc)
        if sub is None:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                    linespacing=1.2)
        else:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs,
                    fontweight="bold")
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=sfs,
                    color=DARK, linespacing=1.1)

    def arrow(p, q, color=DARK, lw=0.9, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color,
                                     lw=lw, connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=0.5, shrinkB=0.5))

    def header(x, text):
        ax.text(x, H - 0.08, text, ha="left", va="top", fontsize=7.6, fontweight="bold")

    # (a) pipeline and schedules ------------------------------------------
    header(0.06, "(a) Detection pipeline and schedules")
    xs, w, h, gap = 0.08, 1.12, 0.33, 0.13
    stages = [("JPEG frame", "backlog or camera stream", "white"),
              ("decode + letterbox", "CPU, stage D", host_c),
              ("enqueue + wait", "CPU, stage M", host_c),
              ("TensorRT engine", "GPU, stage G(f)", gpu_c),
              ("detection rows", "CPU, stage R; no NMS", host_c)]
    y = H - 0.30 - h
    ys = []
    for title, sub, fc in stages:
        box(xs, y, w, h, title, sub, fc=fc)
        ys.append(y)
        y -= h + gap
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow((xs + w / 2, y0), (xs + w / 2, y1 + h))
    xs2, w2, h2 = 1.30, 1.06, 0.40
    sched = [("sequential", "stages in series; GPU idle\nduring all host work"),
             ("+ prefetch", "decode of frame i+1 runs\nduring the engine of frame i"),
             ("+ overlap", "rows of frame i\u22121 run\nduring the engine of frame i"),
             ("+ second decoder", "decode leaves the\ncritical path")]
    gap2 = (5 * h + 4 * gap - 4 * h2) / 3
    y = H - 0.30 - h2
    prev = None
    for name, desc in sched:
        box(xs2, y, w2, h2, name, desc, fc="white", fs=6.6, sfs=5.9)
        if prev is not None:
            arrow((xs2 + w2 / 2, prev), (xs2 + w2 / 2, y + h2))
        prev = y
        y -= h2 + gap2
    ax.text(xs2 + w2, 0.07, "identical predictions on every step", ha="right",
            va="bottom", fontsize=5.9, color="0.3", style="italic")

    # (b) feedback loop -----------------------------------------------------
    x0 = 2.58
    header(x0, "(b) Two governors close the loop")
    top = H - 0.30
    box(x0 + 0.10, top - 0.26, 2.05, 0.26, "host schedule sets the period T", fc=host_c)
    bw, bh = 0.92, 0.40
    lx, rx = x0 + 0.10, x0 + 1.23
    uy = top - 0.26 - 0.16 - bh
    ly = uy - 0.30 - bh
    box(lx, uy, bw, bh, "GPU duty cycle", "u = G(f) / T", fc=gpu_c)
    box(rx, uy, bw, bh, "GPU governor", "load band, 25 ms poll", fc=gov_c)
    box(rx, ly, bw, bh, "GPU clock f", "306\u20131020 MHz", fc=gpu_c)
    box(lx, ly, bw, bh, "engine time", r"$G(f) = G_{\max}\,f_{\max}/f$", fc=gpu_c)
    arrow((lx + bw / 2, top - 0.26), (lx + bw / 2, uy + bh))
    arrow((lx + bw, uy + bh / 2), (rx, uy + bh / 2))
    arrow((rx + bw / 2, uy), (rx + bw / 2, ly + bh))
    arrow((rx, ly + bh / 2), (lx + bw, ly + bh / 2))
    arrow((lx + bw / 2, ly + bh), (lx + bw / 2, uy))
    ax.text((lx + bw + rx) / 2, (uy + ly + bh) / 2, "u below the band: clock falls",
            ha="center", va="center", fontsize=5.9, color=DARK, linespacing=1.1)
    cy = 0.10
    ch = ly - 0.16 - cy
    rect(x0 + 0.10, cy, 2.05, ch, gov_c)
    ax.text(x0 + 0.10 + 1.025, cy + ch - 0.11, "CPU governor (schedutil)", ha="center",
            va="center", fontsize=6.8, fontweight="bold")
    ax.text(x0 + 0.10 + 1.025, cy + (ch - 0.2) / 2,
            "a sleeping CUDA wait looks idle, so the host\n"
            "clock falls; host stages and submission slow,\n"
            "T grows and u falls with it", ha="center", va="center", fontsize=6.0,
            color=DARK, linespacing=1.15)
    arrow((lx + 0.20, cy + ch), (lx + 0.20, ly), color=RED, lw=1.0)
    ax.text(lx + 0.26, (cy + ch + ly) / 2, "slower host", ha="left", va="center",
            fontsize=5.9, color=RED)

    # (c) measurement and platforms ---------------------------------------
    x1 = 4.92
    header(x1, "(c) Measurement and replication")
    wc = 1.86
    items = [("Power", "INA3221 VDD_IN at 20 Hz; energy per image"),
             ("Clocks and load", "GPU devfreq, CPU cpufreq, driver GPU load"),
             ("Accuracy", "COCOeval on 5,000 val2017 images;\nsha256 of every prediction file"),
             ("Latency", "frame release to finished rows; p50, p95")]
    y = H - 0.30
    for name, desc in items:
        hh = 0.37 if "\n" in desc else 0.28
        y -= hh
        rect(x1, y, wc, hh, meas_c)
        ax.text(x1 + 0.07, y + hh - 0.085, name, ha="left", va="center", fontsize=6.6,
                fontweight="bold")
        ax.text(x1 + 0.07, y + (hh - 0.16) / 2 + 0.01, desc, ha="left", va="center",
                fontsize=5.9, color=DARK, linespacing=1.1)
        y -= 0.055
    boards = [("Jetson Orin Nano Super", "GPU, TensorRT; main study", gpu_c),
              ("Raspberry Pi 5 + Hailo-8", "NPU without user clock control", host_c),
              ("Rubik Pi 3 (QCM6490)", "CPU and Hexagon NPU", gov_c)]
    bh3 = (y - 0.10 - 2 * 0.05) / 3
    for name, desc, fc in boards:
        y -= bh3
        rect(x1, y, wc, bh3, fc)
        ax.text(x1 + 0.07, y + bh3 * 0.66, name, ha="left", va="center", fontsize=6.4,
                fontweight="bold")
        ax.text(x1 + 0.07, y + bh3 * 0.28, desc, ha="left", va="center", fontsize=5.9,
                color=DARK)
        y -= 0.05
    for xv in (2.48, 4.83):
        ax.plot([xv, xv], [0.10, H - 0.10], color="0.82", linewidth=0.6)
    finalize(fig, "framework")


ALL = {"framework": framework, "dvfs": dvfs, "operating": operating, "traces": traces,
       "timeline": timeline, "ladder": ladder, "stream": stream, "resolution": resolution}

if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
        print("figure", name)
