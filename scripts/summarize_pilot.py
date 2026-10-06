"""Combine verified development-only training and Jetson results.

Run: python3 summarize_pilot.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
VARIANTS = {
    "float": "float",
    "qat": "qat",
    "l1": "l1",
    "object": "object",
    "l1_qat": "l1-qat",
    "object_qat": "object-qat",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    layers = read(ROOT / "results/pilot-jetson-layer-report.json")
    rows = {}
    for engine_name, training_name in VARIANTS.items():
        train = read(ROOT / "results" / f"pilot-{training_name}.json")
        ap = read(ROOT / "results/pilot-jetson" / f"{engine_name}-coco-ap.json")
        timing = read(
            ROOT / "results/pilot-jetson"
            / f"{engine_name}-predictions.summary.json"
        )
        if (
            train["seed"] != 20261004
            or train["epochs"] != 5
            or ap["dataset"] != "COCO train2017 pilot development, 1024 images"
            or timing["images_evaluated"] != 1024
            or timing["nms"] is not False
        ):
            raise RuntimeError(f"Mismatched experimental contract: {engine_name}")
        if train["kind"] != training_name:
            raise RuntimeError(f"Wrong training checkpoint: {engine_name}")
        rows[engine_name] = {
            "pytorch_ultralytics_dev_map50_95": train["development_map50_95"],
            "jetson_official_dev_ap50_95": ap["ap50_95"],
            "jetson_official_dev_small_ap": ap["ap_small"],
            "jetson_host_input_inference_output_ms_median":
                timing["host_input_inference_output_ms_median"],
            "jetson_host_input_inference_output_ms_p95":
                timing["host_input_inference_output_ms_p95"],
            "onnx_qdq_pairs": train["onnx_q_nodes"],
            "engine_bytes": layers[engine_name]["engine_bytes"],
            "engine_sha256": layers[engine_name]["engine_sha256"],
            "engine_convolution_output_precision":
                layers[engine_name]["convolution_output_precision"],
            "checkpoint_sha256": train["checkpoint_sha256"],
            "onnx_sha256": train["onnx_sha256"],
            "first_input_sha256": timing["first_input_sha256"],
        }
    if len({row["first_input_sha256"] for row in rows.values()}) != 1:
        raise RuntimeError("Engines did not run identical first input tensors")
    if rows["l1"]["engine_bytes"] <= rows["l1_qat"]["engine_bytes"]:
        raise RuntimeError("Unexpected mixed-precision engine size")
    compiler_checks = {}
    for kind in ("opt3_float", "opt3_qat", "gated_opt0"):
        ap = read(ROOT / "results/pilot-jetson" / f"{kind}-coco-ap.json")
        timing = read(
            ROOT / "results/pilot-jetson"
            / f"{kind}-predictions.summary.json"
        )
        if (
            ap["dataset"] != "COCO train2017 pilot development, 1024 images"
            or timing["images_evaluated"] != 1024
            or timing["first_input_sha256"] != rows["float"]["first_input_sha256"]
        ):
            raise RuntimeError(f"Compiler comparison changed input: {kind}")
        compiler_checks[kind] = {
            "official_development_ap50_95": ap["ap50_95"],
            "host_input_inference_output_ms_median":
                timing["host_input_inference_output_ms_median"],
            "engine_bytes": layers[kind]["engine_bytes"],
            "engine_sha256": layers[kind]["engine_sha256"],
            "convolution_output_precision":
                layers[kind]["convolution_output_precision"],
            "builder_optimization_level": 0 if kind == "gated_opt0" else 3,
        }
    result = {
        "scope": "pilot only; COCO train2017 held-out development, not COCO val2017",
        "seed": 20261004,
        "train_images": 1024,
        "development_images": 1024,
        "epochs": 5,
        "builder": "TensorRT 10.3, optimization level 0, static 640, batch 1",
        "object_and_l1_physical_pruning_fraction": 0.05557715334255986,
        "latency_caveat": "Python host H2D+inference+D2H; no image decode, no clock/power normalization",
        "engine_official_ap_caveat": "pycocotools, not Ultralytics mAP",
        "models": rows,
        "compiler_checks": compiler_checks,
    }
    destination = ROOT / "results/pilot-comparison.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, row in rows.items():
        print(
            f"{name:12s} AP={row['jetson_official_dev_ap50_95']:.5f} "
            f"small={row['jetson_official_dev_small_ap']:.5f} "
            f"host_ms={row['jetson_host_input_inference_output_ms_median']:.3f}"
        )
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
