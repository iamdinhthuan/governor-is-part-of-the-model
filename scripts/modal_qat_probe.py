"""Probe Ultralytics YOLO26n QAT export on one L40S.

COCO8 and one epoch only check the graph and training path. They do not
calibrate a publishable model or establish any accuracy result.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("yolo26n-qat-graph-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.172",
        "onnx==1.18.0",
        "nvidia-modelopt==0.47.0",
        "huggingface_hub==0.36.0",
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics", "WANDB_MODE": "disabled"})
)


@app.function(image=image, gpu="L40S", timeout=1200)
def qat_probe() -> tuple[bytes, dict]:
    import importlib.metadata

    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        quantize=8,
        project="/tmp/yolo26n-qat-probe",
        name="one-epoch",
        exist_ok=True,
        val=False,
        plots=False,
        save=True,
    )
    quantizer_count = sum(
        "TensorQuantizer" in type(module).__name__
        for module in model.model.modules()
    )
    if quantizer_count == 0:
        raise RuntimeError("quantize=8 did not insert any tensor quantizers")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            simplify=False,
            device=0,
            nms=False,
        )
    )
    data = exported.read_bytes()
    if len(data) > 50_000_000:
        raise RuntimeError(f"QAT ONNX exceeded 50 MB: {len(data)}")
    graph = onnx.load(str(exported), load_external_data=False)
    output_shapes = [
        [d.dim_value for d in o.type.tensor_type.shape.dim]
        for o in graph.graph.output
    ]
    counts = {
        op: sum(n.op_type == op for n in graph.graph.node)
        for op in ("QuantizeLinear", "DequantizeLinear", "Conv", "Softmax", "TopK")
    }
    if counts["QuantizeLinear"] == 0 or counts["DequantizeLinear"] == 0:
        raise RuntimeError(f"QAT ONNX contains no Q/DQ nodes: {counts}")
    return data, json.loads(
        json.dumps(
            {
                "ultralytics": ultralytics.__version__,
                "torch": str(torch.__version__),
                "modelopt": importlib.metadata.version("nvidia-modelopt"),
                "dataset": "COCO8 smoke only",
                "qat_epochs": 1,
                "tensor_quantizer_modules": quantizer_count,
                "output_shapes": output_shapes,
                "op_counts": counts,
                "onnx_sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "w8a8_backend_verified": False,
            }
        )
    )


@app.local_entrypoint()
def main():
    data, metadata = qat_probe.remote()
    out = Path(__file__).parent / "artifacts" / "yolo26n_qat_coco8_probe.onnx"
    out.write_bytes(data)
    if hashlib.sha256(out.read_bytes()).hexdigest() != metadata["onnx_sha256"]:
        raise RuntimeError("Transferred QAT ONNX SHA256 mismatch")
    summary = Path(__file__).parent / "results" / "modal-qat-probe.json"
    summary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Saved QAT graph probe to {out} and metadata to {summary}")
