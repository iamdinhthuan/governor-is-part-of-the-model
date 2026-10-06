"""Bounded YOLO26n smoke test on one Modal L40S, not a research benchmark.

Run from this directory: modal run modal_smoke.py
The remote function has a 30-minute hard timeout and uses only one GPU.
"""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-feasibility-smoke")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.97", "onnx==1.18.0")
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics", "WANDB_MODE": "disabled"})
)


@app.function(image=image, gpu="L40S", timeout=1800)
def smoke() -> dict:
    import platform

    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.manual_seed(20261004)
    model = YOLO("yolo26n.pt")
    head = model.model.model[-1]
    info = {
        "ultralytics": ultralytics.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "head_class": type(head).__name__,
        "head_end2end": getattr(head, "end2end", None),
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "dataset": "coco8.yaml (smoke only, not a benchmark)",
    }

    metrics = model.val(
        data="coco8.yaml",
        imgsz=640,
        batch=8,
        device=0,
        plots=False,
        verbose=False,
    )
    info["pretrained_coco8_map50_95"] = float(metrics.box.map)
    info["pretrained_coco8_map50"] = float(metrics.box.map50)

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
    info["onnx_outputs"] = [
        {
            "name": output.name,
            "shape": [
                dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                for dim in output.type.tensor_type.shape.dim
            ],
        }
        for output in graph.graph.output
    ]
    info["onnx_op_types"] = sorted({node.op_type for node in graph.graph.node})

    train_model = YOLO("yolo26n.pt")
    train_model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project="/tmp/yolo26n-smoke",
        name="one-epoch",
        exist_ok=True,
        val=False,
        plots=False,
        save=False,
    )
    info["train_one_epoch_completed"] = True
    # Converting through JSON prevents Modal from serializing library-specific
    # string subclasses (for example torch.torch_version.TorchVersion).
    return json.loads(json.dumps(info))


@app.local_entrypoint()
def main():
    result = smoke.remote()
    out = Path(__file__).parent / "results" / "modal-smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Smoke-test summary saved to {out}")
