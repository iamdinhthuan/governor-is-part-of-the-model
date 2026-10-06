"""Disable two compiler-expensive QAT activation boundaries, then re-export.

This is a precision-map pilot motivated by the *measured* TensorRT
per-layer profile, not a claim of full W8A8 or a pruned detector.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("yolo26n-precision-boundary-pilot")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.172",
        "nvidia-modelopt==0.47.0",
        "huggingface_hub==0.36.0",
        "onnx==1.18.0",
    )
    .env({"YOLO_CONFIG_DIR": "/tmp", "WANDB_MODE": "disabled"})
)
data_volume = modal.Volume.from_name("yolo26-coco-train2017")
result_volume = modal.Volume.from_name("yolo26n-pilot-results")


@app.function(
    image=image,
    gpu="L40S",
    timeout=2400,
    volumes={"/data": data_volume, "/output": result_volume},
)
def precision_gate() -> tuple[dict, bytes]:
    import shutil

    import onnx
    from ultralytics import YOLO

    saved = Path("/output/qat-e5-seed20261004/weights/best.pt")
    separate = Path("/tmp/qat-gated-pilot.pt")
    shutil.copyfile(saved, separate)
    model = YOLO(str(separate))
    quantizers = {
        name: module
        for name, module in model.model.named_modules()
        if "input_quantizer" in name
    }
    chosen = (
        "model.2.m.0.cv1.conv.input_quantizer",
        "model.2.cv2.conv.input_quantizer",
    )
    for name in chosen:
        if name not in quantizers:
            raise RuntimeError(
                f"Missing quantizer {name}; nearby: "
                f"{[item for item in quantizers if item.startswith('model.2')][:30]}"
            )
        quantizers[name].disable()
    metrics = model.val(
        data="/data/coco/pilot.yaml",
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        nms=False,
        rect=False,
        plots=False,
    )
    exported = Path(
        model.export(
            format="onnx", imgsz=640, dynamic=False, simplify=False, device=0, nms=False
        )
    )
    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph)
    q_count = sum(node.op_type == "QuantizeLinear" for node in graph.graph.node)
    dq_count = sum(node.op_type == "DequantizeLinear" for node in graph.graph.node)
    if not (0 < q_count == dq_count < 195):
        raise RuntimeError(f"Precision gating did not reduce Q/DQ: {q_count}/{dq_count}")
    blob = exported.read_bytes()
    target = Path("/output/qat-gated-probe")
    target.mkdir(parents=True, exist_ok=True)
    (target / "pilot_qat_gated.onnx").write_bytes(blob)
    result = {
        "source_checkpoint_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
        "disabled_activation_quantizers": list(chosen),
        "qat_onnx_q_nodes_before": 195,
        "qat_onnx_q_nodes_after": q_count,
        "qat_onnx_dq_nodes_after": dq_count,
        "development_map50_95_ultralytics": float(metrics.box.map),
        "onnx_sha256": hashlib.sha256(blob).hexdigest(),
        "qualifier": "precision boundary diagnostic, no pruning, development-only",
    }
    (target / "precision-gate-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    result_volume.commit()
    return result, blob


@app.local_entrypoint()
def main():
    summary, blob = precision_gate.remote()
    base = Path(__file__).parent
    (base / "artifacts" / "pilot_qat_gated.onnx").write_bytes(blob)
    (base / "results" / "pilot-precision-gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("Saved precision boundary pilot")
