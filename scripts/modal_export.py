"""Export the untouched NMS-free YOLO26n checkpoint for a Jetson parser probe."""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("yolo26n-onnx-feasibility")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.172", "onnx==1.18.0")
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics", "WANDB_MODE": "disabled"})
)


@app.function(image=image, gpu="L40S", timeout=900)
def export_onnx() -> tuple[bytes, dict]:
    import onnx
    import ultralytics
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    if not hasattr(model.model.model[-1], "one2one_cv2"):
        raise RuntimeError("The expected one-to-one end-to-end head is absent")
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
    graph = onnx.load(str(exported), load_external_data=False)
    outputs = [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in graph.graph.output]
    if outputs != [[1, 300, 6]]:
        raise RuntimeError(f"Unexpected one-to-one graph outputs: {outputs}")
    data = exported.read_bytes()
    if len(data) > 30_000_000:
        raise RuntimeError(f"Unexpectedly large export: {len(data)} bytes")
    return data, {
        "checkpoint": "yolo26n.pt",
        "checkpoint_sha256": hashlib.sha256(Path("yolo26n.pt").read_bytes()).hexdigest(),
        "ultralytics": ultralytics.__version__,
        "image_size": 640,
        "nms": False,
        "output_shapes": outputs,
        "onnx_sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


@app.local_entrypoint()
def main():
    data, metadata = export_onnx.remote()
    artifact_dir = Path(__file__).parent / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    out = artifact_dir / "yolo26n_84172_640_one2one.onnx"
    out.write_bytes(data)
    if hashlib.sha256(out.read_bytes()).hexdigest() != metadata["onnx_sha256"]:
        raise RuntimeError("Transferred ONNX SHA256 mismatch")
    (artifact_dir / "onnx_84172_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved {out} ({metadata['bytes']} bytes)")
