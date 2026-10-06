"""LaTeX tables for the paper, generated from results/paper/summary.json."""

import json
from pathlib import Path

S = json.loads(Path("results/paper/summary.json").read_text())
TAB = Path("paper/tables")
TAB.mkdir(parents=True, exist_ok=True)

LADDER = [
    ("ultralytics", "Ultralytics \\texttt{predict}", "--"),
    ("float_seq_stream", "Ours: float input, sequential", "stream"),
    ("uint8_seq_stream", "+ in-graph uint8 preprocessing", "stream"),
    ("uint8_seq_event", "+ event synchronization", "event"),
    ("uint8_prefetch_stream", "+ decode prefetch (stream sync)", "stream"),
    ("uint8_prefetch_event", "+ decode prefetch", "event"),
    ("uint8_overlap_event", "+ post-processing overlap", "event"),
    ("uint8_overlap_event_w2", "+ second decode worker", "event"),
    ("float_overlap_event_w2", "Full scheduling, float input", "event"),
    ("uint8_adaptive_event_w2", "Full, arrival-aware overlap", "event"),
    ("uint8_overlap_event_w2_nogc", "Full, GC paused in timed loop", "event"),
]
MIDRULE_AFTER = ("ultralytics", "uint8_overlap_event_w2", "float_overlap_event_w2")


def m(cell, key, fmt="{:.1f}"):
    return fmt.format(cell[key]["mean"]) if cell.get(key) else "--"


def ap_of(cell):
    coco = cell.get("coco")
    return (f"{coco['ap50_95']:.4f}", f"{coco['ap_small']:.4f}") if coco else ("--", "--")


def ladder():
    d, l = S["ladder"]["default"], S["ladder"]["locked"]
    rows = []
    for name, label, _ in LADDER:
        a, b = d[name], l[name]
        ap, aps = ap_of(a)
        rows.append(" & ".join([
            label, ap, aps,
            m(a, "throughput"), m(a, "energy_mj"), m(a, "power_w", "{:.2f}"),
            m(a, "gpu_mhz", "{:.0f}"), m(a, "cpu_ms_per_image"),
            m(b, "throughput"), m(b, "energy_mj"), m(b, "power_w", "{:.2f}"),
        ]) + r" \\")
        if name in MIDRULE_AFTER:
            rows[-1] += r" \midrule"
    body = "\n".join(rows)
    bound, wide = 2.5, []
    for name, label, _ in LADDER:
        for clocks in ("default", "locked"):
            t = S["ladder"][clocks][name]["throughput"]
            spread = (t["max"] - t["min"]) / t["mean"] * 100
            if spread >= bound:
                short = {"float_seq_stream": "the float sequential pipeline",
                         "uint8_prefetch_stream": "prefetch with stream synchronization"}.get(name, name)
                wide.append(f"{short} under {clocks} clocks "
                            f"({t['min']:.1f}--{t['max']:.1f} images/s)")
    exceptions = (" except " + " and ".join(wide)) if wide else ""
    text = rf"""\begin{{table*}}[t]
\centering
\caption{{Cumulative pipeline ablation for YOLO26n at 640 on all 5,000 COCO val2017
images (batch 1, FP16 TensorRT, \texttt{{MAXN\_SUPER}}). Each cell is the mean of three
rotated runs; run-to-run throughput ranges are below {bound}\% of the mean for every
cell{exceptions}. The last two rows are variants of the full uint8 pipeline.
AP is official COCOeval. Each row's predictions were byte-identical across
all six runs (both clock settings), and rows with the same input type share one
prediction file. GPU MHz is the mean sampled \texttt{{devfreq}} clock under the default
governor (always 1020 when locked). CPU ms is process CPU time per image.}}
\label{{tab:ladder}}
\small
\setlength{{\tabcolsep}}{{3.4pt}}
\begin{{tabular}}{{lcc rrrrr rrr}}
\toprule
& & & \multicolumn{{5}}{{c}}{{Default governors}} & \multicolumn{{3}}{{c}}{{\texttt{{jetson\_clocks}}}} \\
\cmidrule(lr){{4-8}} \cmidrule(lr){{9-11}}
Pipeline & AP & AP$_S$ & img/s & mJ/img & W & GPU MHz & CPU ms & img/s & mJ/img & W \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (TAB / "ladder.tex").write_text(text)


def generality():
    if "generality" not in S:
        return
    d, l = S["generality"].get("default", {}), S["generality"].get("locked", {})
    names = {"y26s": "YOLO26s", "v10n": "YOLOv10n"}
    pipes = [("ultra", "Ultralytics"), ("float_seq_stream", "Ours, naive"),
             ("uint8_overlap_event_w2", "Ours, full")]
    rows = []
    for model, model_label in names.items():
        for pipe, pipe_label in pipes:
            key = f"{model}_{pipe}"
            if key not in d:
                continue
            a, b = d[key], l.get(key, {})
            ap, aps = ap_of(a)
            rows.append(" & ".join([
                model_label if pipe == "ultra" else "", pipe_label, ap, aps,
                m(a, "throughput"), m(a, "energy_mj"), m(a, "power_w", "{:.2f}"),
                m(a, "gpu_mhz", "{:.0f}"),
                m(b, "throughput"), m(b, "energy_mj"),
            ]) + r" \\")
        rows[-1] += r" \midrule"
    rows[-1] = rows[-1].replace(r" \midrule", "")
    body = "\n".join(rows)
    text = rf"""\begin{{table*}}[t]
\centering
\caption{{Other NMS-free detectors on all 5,000 val2017 images, 640, batch 1
(default: mean of two runs; locked: one run). ``Naive'' is float input with sequential execution and
stream synchronization; ``full'' is uint8 input, two prefetch workers, overlap
and event synchronization.}}
\label{{tab:generality}}
\small
\setlength{{\tabcolsep}}{{2.6pt}}
\begin{{tabular}}{{llcc rrrr rr}}
\toprule
& & & & \multicolumn{{4}}{{c}}{{Default}} & \multicolumn{{2}}{{c}}{{Locked}} \\
\cmidrule(lr){{5-8}} \cmidrule(lr){{9-10}}
Model & Pipeline & AP & AP$_S$ & img/s & mJ & W & MHz & img/s & mJ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (TAB / "generality.tex").write_text(text)


def sched():
    if "sched" not in S:
        return
    d = S["sched"]["default"]
    rows = []
    for mode, mode_label in (("seq", "sequential"), ("prefetch", "prefetch")):
        for sync in ("stream", "event"):
            for flag in ("auto", "spin", "blocking"):
                key = f"{mode}_{sync}_{flag}"
                if key not in d:
                    continue
                c = d[key]
                rows.append(" & ".join([
                    mode_label if sync == "stream" and flag == "auto" else "",
                    sync if flag == "auto" else "", flag,
                    m(c, "throughput"), m(c, "energy_mj"), m(c, "power_w", "{:.2f}"),
                    m(c, "gpu_mhz", "{:.0f}"), m(c, "cpu_mhz", "{:.0f}"),
                    m(c, "cpu_ms_per_image"), m(c, "gpu_wall_median_ms", "{:.2f}"),
                ]) + r" \\")
        rows[-1] += r" \midrule"
    rows[-1] = rows[-1].replace(r" \midrule", "")
    body = "\n".join(rows)
    repeats = {c.get("repeats") for c in d.values()}
    runs = {2: "two", 3: "three"}.get(max(repeats), str(max(repeats)))
    text = rf"""\begin{{table*}}[t]
\centering
\caption{{Synchronization primitive $\times$ CUDA device scheduling flag (first
2,000 val2017 images, default governors, mean of {runs} runs, uint8 engine).
``GPU ms'' is the median enqueue-to-completion wall time of one inference; CPU MHz is
the mean of the fastest core's sampled clock.}}
\label{{tab:sched}}
\small
\setlength{{\tabcolsep}}{{2.4pt}}
\begin{{tabular}}{{lll rrrrrrr}}
\toprule
Mode & Sync & Flag & img/s & mJ & W & GPU MHz & CPU MHz & CPU ms & GPU ms \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (TAB / "sched.tex").write_text(text)


def cpufreq():
    if "cpufreq" not in S:
        return
    d = S["cpufreq"]["default"]
    labels = [("stream", "stream", "--"), ("stream_clusterbusy", "stream", "busy loop"),
              ("event", "event", "--"), ("event_clusterbusy", "event", "busy loop")]
    rows = []
    for key, sync, spinner in labels:
        c = d[key]
        rows.append(" & ".join([
            sync, spinner, m(c, "throughput"), m(c, "latency_median_ms"),
            m(c, "cpu_ms_per_image"), m(c, "policy0_mhz", "{:.0f}"),
            m(c, "gpu_wall_median_ms", "{:.2f}"), m(c, "power_w", "{:.2f}"),
        ]) + r" \\")
    extras = Path("results/paper/extras.json")
    if extras.exists():
        u = json.loads(extras.read_text()).get("uclamp", {})
        rows[-1] += r" \midrule"
        for key, remedy in (("stream", "--"), ("stream_uclamp1024", "uclamp")):
            if key in u:
                c = u[key]
                rows.append(" & ".join([
                    "stream", remedy, m(c, "throughput"), m(c, "latency_median_ms"),
                    m(c, "cpu_ms"), m(c, "policy0_mhz", "{:.0f}"),
                    m(c, "engine_ms", "{:.2f}"), m(c, "power_w", "{:.2f}"),
                ]) + r" \\")
    body = "\n".join(rows)
    text = rf"""\begin{{table}}[t]
\centering
\caption{{CPU-clock test. Sequential uint8 pipeline pinned to CPU 0 (first 2,000 val2017
images, default governors, mean of two runs). ``Busy loop'' marks a separate process
spinning on CPU 1, which shares CPU 0's frequency domain; ``uclamp'' sets the pipeline's
\texttt{{uclamp.min}} to 1024 (bottom block, a separate session with its own stream-sync
reference). Cluster MHz is the sampled clock of that domain; CPU ms counts only the pipeline
process. All runs produced identical predictions.}}
\label{{tab:cpufreq}}
\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{ll rrrrrr}}
\toprule
Sync & Remedy & img/s & p50 ms & CPU ms & Cluster MHz & GPU ms & W \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (TAB / "cpufreq.tex").write_text(text)


def dvfs():
    model = json.loads(Path("results/paper/model.json").read_text())
    locked = S["dvfs"]["locked"]
    rows = []
    for gap in (0, 1, 2, 4, 8, 16, 33):
        cells = []
        for name in ("y26n_640_uint8", "y26s_640_uint8"):
            runs = [r for r in model["dvfs"] if r["model"] == name and r["idle_ms"] == gap]
            mean = lambda k: sum(r[k] for r in runs) / len(runs)
            cells += [f"{mean('measured_ms'):.2f}", f"{mean('f_harmonic_mhz'):.0f}",
                      f"{mean('predicted_ms'):.2f}",
                      f"{locked[name][str(gap)]['gpu_compute_median_ms']['mean']:.2f}"]
        rows.append(" & ".join([str(gap)] + cells) + r" \\")
    body = "\n".join(rows)
    text = rf"""\begin{{table}}[t]
\centering
\caption{{Idle-gap sweep (\texttt{{trtexec}}, uint8 FP16 engines, mean of two
15\,s runs). Default governors: measured median compute time, harmonic-mean
GPU clock $\bar f$, and the prediction $G_{{\max}} f_{{\max}}/\bar f$ of
\cref{{eq:gf}}. ``Lock'' is the measured median with \texttt{{jetson\_clocks}}.}}
\label{{tab:dvfs}}
\small
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{r rrrr rrrr}}
\toprule
& \multicolumn{{4}}{{c}}{{YOLO26n}} & \multicolumn{{4}}{{c}}{{YOLO26s}} \\
\cmidrule(lr){{2-5}} \cmidrule(lr){{6-9}}
Gap (ms) & ms & $\bar f$ & pred. & lock & ms & $\bar f$ & pred. & lock \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (TAB / "dvfs.tex").write_text(text)


def stream():
    if "stream" not in S:
        return
    d = S["stream"]["default"]
    pipes = [("ultra", "Ultralytics"), ("float_seq_stream", "Float, sequential"),
             ("uint8_seq_event", "uint8, seq., event"),
             ("uint8_prefetch_event", "+ prefetch"),
             ("uint8_overlap_event_w2", "+ overlap, 2 workers"),
             ("uint8_adaptive_event_w2", "+ arrival-aware overlap")]
    rows = []
    for pipe, label in pipes:
        cells = [label]
        for fps in (15, 30, 60):
            c = d.get(f"{pipe}_fps{fps}")
            if c is None:
                cells += ["--", "--", "--"]
                continue
            rate = c["throughput"]["mean"]
            if rate < 0.98 * fps:
                cells += [rf"\multicolumn{{2}}{{c}}{{sat.\ ({rate:.0f}/s)}}",
                          f"{c['energy_mj']['mean']:.0f}"]
            else:
                cells += [f"{c['latency_median_ms']['mean']:.1f}",
                          f"{c['latency_p95_ms']['mean']:.1f}",
                          f"{c['energy_mj']['mean']:.0f}"]
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    text = rf"""\begin{{table*}}[t]
\centering
\caption{{Streaming at fixed arrival rates (first 1,000 val2017 images, YOLO26n 640,
default governors, mean of two runs). Latency runs from frame arrival to finished
detection rows (ms); energy is board energy per frame (mJ). ``sat.'' marks a pipeline
that cannot sustain the arrival rate; its queue grows without bound, and the
achieved rate is given instead of latency. Idle board power alone amounts to
312, 156 and 78\,mJ per frame at 15, 30 and 60\,fps.}}
\label{{tab:stream}}
\small
\begin{{tabular}}{{l rrr rrr rrr}}
\toprule
& \multicolumn{{3}}{{c}}{{15 fps}} & \multicolumn{{3}}{{c}}{{30 fps}} & \multicolumn{{3}}{{c}}{{60 fps}} \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}} \cmidrule(lr){{8-10}}
Pipeline & p50 & p95 & mJ & p50 & p95 & mJ & p50 & p95 & mJ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (TAB / "stream.tex").write_text(text)


if __name__ == "__main__":
    stream()
    dvfs()
    ladder()
    generality()
    sched()
    cpufreq()
    print("tables written")
