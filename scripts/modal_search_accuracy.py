"""Screen unrecovered, structurally pruned detectors on disjoint development.

This is an accuracy *gate*, not final AP. Run no more than three kinds at
once: `modal run modal_search_accuracy.py --kinds baseline,p34,ffn`.
"""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-compiler-candidate-accuracy")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.172")
    .env({"YOLO_CONFIG_DIR": "/tmp"})
)
data_volume = modal.Volume.from_name("yolo26-coco-train2017")
result_volume = modal.Volume.from_name("yolo26n-pilot-results")


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    volumes={"/data": data_volume, "/output": result_volume},
)
def measure(kind: str) -> dict:
    import hashlib

    from ultralytics import YOLO

    if kind not in {"baseline", "p3", "p34", "ffn"}:
        raise ValueError(kind)
    checkpoint = (
        Path("yolo26n.pt")
        if kind == "baseline"
        else Path("/output/compiler-search") / kind / "pruned.pt"
    )
    model = YOLO(str(checkpoint))
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
    return {
        "candidate": kind,
        "scope": "COCO train2017 1024 held-out development; no recovery",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "ultralytics_dev_map50_95": float(metrics.box.map),
        "ultralytics_dev_map50": float(metrics.box.map50),
        "no_recovery": True,
    }


@app.local_entrypoint()
def main(kinds: str = "baseline,p34,ffn"):
    names = kinds.split(",")
    if not names or len(names) > 3 or len(set(names)) != len(names):
        raise ValueError("At most three distinct accuracy jobs may run")
    calls = {name: measure.spawn(name) for name in names}
    for name, call in calls.items():
        result = call.get()
        out = Path(__file__).parent / "results" / f"search_accuracy_{name}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"Saved raw development AP for {name}: {result['ultralytics_dev_map50_95']:.5f}")
