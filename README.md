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
    `jetson_round5.py`, `gil_exec.py` — EMC readback under
    `jetson_clocks` and the GIL switch-interval control on the full
    two-worker pipeline,
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
| Table 1 (related-work positioning) | manual | `sections/related.tex` |
| Table 2 (platform and software) | manual | `sections/method.tex` |
| Table 3 (measurement boundaries) | manual | `sections/method.tex` |
| Table 4 (idle-gap sweep) | `raw/dvfs/` | `make_tables.py` |
| Table 5 (fixed-frequency sweep) | `raw/eq1sweep_emc/` | prose numbers recomputed from `report.json` |
| Table 6 (ladder) | `raw/ladder/` | `make_tables.py` |
| Table 7 (sync x CUDA device flags) | `raw/sched/` | `make_tables.py` |
| Table 8 (CPU-clock test) | `raw/cpufreq/`, `raw/uclamp_event/` | `make_tables.py` |
| Table 9 (uclamp on optimized pipelines) | `raw/uclamp2/` | mean of the two `report.json` files per cell |
| Table 10 (streaming) | `raw/stream/` | `make_tables.py` |
| Table 11 (other NMS-free detectors) | `raw/generality/` | `make_tables.py` |
| Table 12 (replication platforms) | manual | `sections/crossplatform.tex` |
| Tables 13-16 (Pi ladder, Rubik CPU ladder, Rubik HTP ladder, HTP head comparison) | `results/paper/pi.json`, `rubik*.json` | `aggregate_pi.py`, `aggregate_rubik.py` |
| Table 17 (streaming on the NPU boards) | `results/paper/pi.json`, `rubik*.json` | `aggregate_pi.py`, `aggregate_rubik.py` |
| Table A.1 (compression screening) | `modal_*.py` runs | manual (see Appendix A) |
| Round-5 follow-ups (EMC readback under `jetson_clocks`; GIL switch-interval control; locked-clock references quoted in Sec. 6.2 and the Table 9 caption) | `raw/round5/` | `jetson_round5.py`, `gil_exec.py` |
| Fig. 1 (dvfs) | `raw/dvfs/` | `make_figures.py` |
| Fig. 2 (operating points) | `raw/load/`, `extras.json` | `make_figures.py` |
| Fig. 3 (traces + lock-release) | `raw/ladder/`, `extras.json` | `make_figures.py` |
| Fig. 4 (schedule timeline) | schematic; measured stage medians (Table 3) | `make_figures.py` |
| Fig. 5 (ladder bars) | `raw/ladder/`, `results/paper/summary.json` | `make_figures.py` |
| Fig. 6 (stream) | `raw/stream/` | `make_figures.py` |
| Fig. 7 (resolution) | `raw/resolution/` | `make_figures.py` |

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
