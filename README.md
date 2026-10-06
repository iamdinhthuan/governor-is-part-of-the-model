# Artifact: "The governor is part of the model"

Measurement code, raw traces, and paper source for the manuscript
*The governor is part of the model: duty-cycle-aware deployment of object
detectors on edge accelerators* (submitted to the Journal of Systems
Architecture).

## Boards and software

| Board | Accelerator | Stack |
|---|---|---|
| NVIDIA Jetson Orin Nano Super (8 GB) | Ampere GPU, 1020 MHz max | JetPack 6.2, TensorRT 10.3, CUDA 12.6, Python 3.10 |
| Raspberry Pi 5 (8 GB) | Hailo-8 (26 TOPS) | HailoRT 4.20, Raspberry Pi OS 64-bit |
| Rubik Pi 3 (QCM6490) | Hexagon HTP (V68) / CPU | QNN via `qai-appbuilder` 2.50.40 / ONNX Runtime |

## Layout

- `paper/` — LaTeX source (Elsevier `cas-dc`), figures (`figs/`), tables
  (`tables/`), bibliography (`refs.bib`). Build with any TeX Live or
  `tectonic main.tex` (run twice).
- `scripts/` — measurement and analysis code:
  - `jetson_paper_runs.py` — main Jetson campaign (ladder, energy, locked
    clocks), `jetson_power.py` — 20 Hz power/clock/load sampler,
    `jetson_ultralytics_baseline.py` — stock-predictor baseline,
    `jetson_review_extras.py` — follow-ups (driver GPU-load logging,
    lock-then-release hysteresis, GC pauses, uclamp, stage breakdown),
    `jetson_round4.py` — EMC-logged fixed-frequency sweep (8 clocks x 3
    reps) and uclamp on the optimized pipelines,
    `jetson_run_resolution.py`, `jetson_routed_eval.py`,
    `jetson_run_routing_verify.py` — resolution sweep and the router,
    `jetson_cpufreq_test.py`, `uclamp_exec.py` — CPU-clock experiments,
  - `pi_campaign.py`, `pi_hailo_eval.py`, `pi_sampler.py` — Pi 5 + Hailo-8,
  - `rubik_campaign.py`, `rubik_htp_campaign.py`, `rubik_htp_eval.py`,
    `rubik_ort_eval.py`, `rubik_sampler.py` — Rubik Pi 3 (CPU and HTP),
  - `modal_*.py` — compression screening (pruning / QAT / distillation) on
    a cloud GPU (Appendix A),
  - `aggregate_paper.py`, `aggregate_pi.py`, `aggregate_rubik.py` —
    per-run reports -> `results/paper/*.json` summaries,
    `make_tables.py` — summaries -> `paper/tables/*.tex`,
    `make_figures.py` — raw traces -> `paper/figs/*.pdf`,
    `eval_predictions.py`, `compare_numeric.py`, `analyze_extras.py`,
    `analyze_model.py` — hash/accuracy verification and misc analysis.
- `results/paper/` — aggregated summaries (`summary.json`, `pi.json`,
  `rubik.json`, `rubik_htp.json`, `extras.json`, ...) and `raw/`: one
  directory per run (`<experiment>/<configuration>/<repeat>/`) containing
  `report.json` (per-run metrics), `power.csv` / `counters.csv` /
  `emc.csv` traces and `predictions.sha256`. The prediction files
  themselves (1.6 GB of COCO-format JSON) are not in this repository;
  their SHA-256 hashes are, and the full set is archived on Zenodo
  (see the data statement in the paper).

## Table/figure map

| Paper item | Source data | Generator |
|---|---|---|
| Table 1 (platforms) | manual | `paper/tables/` |
| Table 2 (idle-gap sweep) | `raw/dvfs/` | `make_tables.py` |
| Table 3 (fixed-frequency sweep) | `raw/eq1sweep_emc/` | prose numbers recomputed from `report.json` |
| Table 4 (measurement boundaries) | `results/paper/summary.json` | `make_tables.py` |
| Table 5 (ladder), Fig. 4 | `raw/ladder/` | `make_tables.py` |
| Table 6 (streaming) | `raw/stream/` | `make_tables.py` |
| Table 7 (CPU-clock test) | `raw/cpufreq/`, `raw/uclamp_event/` | `make_tables.py` |
| Table 8 (uclamp on optimized pipelines) | `raw/uclamp2/` | mean of the two `report.json` files per cell |
| Tables 9-12 (resolution, generality, GC) | `raw/resolution/` (incl. GC-off twins), `raw/generality/`, `extras.json` | `make_tables.py` |
| Tables 13-17 (Pi, Rubik CPU, Rubik HTP) | `results/paper/pi.json`, `rubik*.json` | `aggregate_pi.py`, `aggregate_rubik.py` |
| Table A.1 (compression screening) | `modal_*.py` runs | manual (see Appendix A) |
| Fig. 1 (dvfs) | `raw/dvfs/` | `make_figures.py` |
| Fig. 2 (operating points) | `raw/load/`, `extras.json` | `make_figures.py` |
| Fig. 3 (traces + lock-release) | `raw/ladder/`, `extras.json` | `make_figures.py` |
| Figs. 5-7 (stream, resolution) | `raw/stream/`, `raw/resolution/` | `make_figures.py` |

## Verification

Every run stores the SHA-256 of its prediction file
(`predictions.sha256`); all schedule comparisons in the paper are between
runs with byte-identical hashes. `eval_predictions.py` re-checks hashes
and COCOeval summaries against the aggregated tables.

## Notes

- Hardware-dependent scripts require the respective boards; the Jetson
  scripts assume JetPack 6.2 with `tegrastats`, `jetson_clocks` and
  TensorRT 10.3, and read `JETSON_SUDO_PW` from the environment for the
  few clock-pinning steps.
- COCO val2017 images and the YOLO26 weights are not redistributed here;
  the scripts expect a standard `val2017` layout and the Ultralytics
  checkpoints named in the paper.
