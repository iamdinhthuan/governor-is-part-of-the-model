"""Export matched static-resolution NMS-free graphs for Jetson screening.

Run: modal run modal_resolution_export.py
Exports the same pretrained checkpoint at 512 and 576, without retraining
or changing precision. The existing 640 graph is the matched control.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("yolo26n-resolution-screen")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.172", "onnx==1.18.0")
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics"})
)


@app.function(image=image, gpu="L40S", timeout=1500)
def export() -> list[tuple[int, dict, bytes]]:
    import onnx
    from ultralytics import YOLO

    results = []
    for size in (512, 576):
        model = YOLO("yolo26n.pt")
        checkpoint_sha = hashlib.sha256(Path("yolo26n.pt").read_bytes()).hexdigest()
        if checkpoint_sha != "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef":
            raise RuntimeError("Checkpoint hash differs from 640 baseline")
        output = Path(
            model.export(
                format="onnx",
                imgsz=size,
                dynamic=False,
                simplify=False,
                device=0,
                nms=False,
            )
        )
        graph = onnx.load(str(output))
        onnx.checker.check_model(graph)
        dimensions = [
            d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim
        ]
        input_dims = [
            d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim
        ]
        if dimensions != [1, 300, 6] or input_dims != [1, 3, size, size]:
            raise RuntimeError(
                f"Unexpected output or input contract: {dimensions}, {input_dims}"
            )
        blob = output.read_bytes()
        results.append(
            (
                size,
                {
                    "image_size": size,
                    "checkpoint_sha256": checkpoint_sha,
                    "onnx_sha256": hashlib.sha256(blob).hexdigest(),
                    "output_shape": dimensions,
                    "input_shape": input_dims,
                    "nms": False,
                    "ultralytics": "8.4.172",
                },
                blob,
            )
        )
    return results


@app.local_entrypoint()
def main():
    root = Path(__file__).parent
    for size, metadata, blob in export.remote():
        path = root / "artifacts" / f"resolution_{size}.onnx"
        path.write_bytes(blob)
        (root / "results" / f"resolution_{size}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(f"Saved {size} one-to-one ONNX: {metadata['onnx_sha256']}")
