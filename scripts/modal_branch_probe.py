"""Check YOLO26n branch selection under Ultralytics 8.4.172 on one image."""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-branch-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.172", "pycocotools==2.0.10", "numpy<3")
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics", "WANDB_MODE": "disabled"})
)


@app.function(image=image, timeout=300)
def inspect_branch() -> dict:
    import numpy as np
    import torch
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    before = bool(model.model.model[-1].end2end)
    model.predict(
        source=np.zeros((640, 640, 3), dtype=np.uint8),
        imgsz=640,
        device="cpu",
        nms=False,
        verbose=False,
    )
    backend = model.predictor.model
    backend_model = backend.model
    head = backend_model.model[-1]
    with torch.inference_mode():
        raw = backend_model(torch.zeros(1, 3, 640, 640))
    first = raw[0] if isinstance(raw, (tuple, list)) else raw
    return {
        "loaded_head_end2end": before,
        "predict_nms_arg": model.predictor.args.nms,
        "backend_end2end": getattr(backend, "end2end", None),
        "backend_head_end2end": getattr(head, "end2end", None),
        "backend_raw_output_shape": list(first.shape),
    }


@app.local_entrypoint()
def main():
    result = inspect_branch.remote()
    out = Path(__file__).parent / "results" / "branch-selection-84172.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved branch selection probe to {out}")
