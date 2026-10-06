"""Matched five-epoch YOLO26n pilot on disjoint COCO train2017 splits.

Run: modal run modal_pilot_train.py
By default launches exactly two L40S jobs concurrently: float and QAT.
Run `modal run modal_pilot_train.py --kinds l1` to recover the physical
L1 pruning baseline. These are development experiments, not paper results.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("yolo26n-pilot-float-versus-qat")
data_volume = modal.Volume.from_name("yolo26-coco-train2017")
result_volume = modal.Volume.from_name("yolo26n-pilot-results", create_if_missing=True)
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


@app.function(
    image=image,
    gpu="L40S",
    timeout=7200,
    volumes={"/data": data_volume, "/output": result_volume},
)
def train(kind: str) -> dict:
    import copy

    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class KeepPrunedArchitecture(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if not isinstance(weights, torch.nn.Module):
                raise RuntimeError("Cannot initialize the physical pruned architecture")
            if weights.model[-1].nc != self.data["nc"]:
                raise RuntimeError("Pruned checkpoint class count differs from the data")
            return copy.deepcopy(weights)

    if kind not in {
        "float", "qat", "l1", "object", "l1-qat", "object-qat",
        "l1-kd", "object-kd",
    }:
        raise ValueError(kind)
    pruned = kind.startswith(("l1", "object"))
    quantized = "qat" in kind
    distillation = kind.endswith("-kd")
    dataset = Path("/data/coco/pilot.yaml")
    if not Path("/data/coco/PILOT_READY.json").exists():
        raise RuntimeError("Official train2017 pilot split is not ready")
    seed = 20261004
    torch.manual_seed(seed)
    starting_checkpoint = {
        "l1": "/output/prune-budget/yolo26n_l1_budget.pt",
        "object": "/output/prune-object/yolo26n_l1_object.pt",
    }.get(kind.split("-")[0], "yolo26n.pt")
    model = YOLO(starting_checkpoint)
    starting_params = sum(parameter.numel() for parameter in model.model.parameters())
    channel_sizes = (
        [model.model.model[index].conv.out_channels for index in (1, 3, 5, 7)]
        if pruned
        else None
    )
    checkpoint_sha = hashlib.sha256(Path(starting_checkpoint).read_bytes()).hexdigest()
    name = f"{kind}-e5-seed{seed}"
    params = {
        "data": str(dataset),
        "epochs": 5,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 4,
        "seed": seed,
        "deterministic": True,
        "nms": False,
        "plots": False,
        "save": True,
        "project": "/output",
        "name": name,
        "exist_ok": True,
    }
    if quantized:
        params["quantize"] = 8
    teacher = Path("/output/float-e5-seed20261004/weights/best.pt")
    if distillation:
        if not teacher.exists():
            raise RuntimeError("Frozen pilot float teacher is missing")
        params["distill_model"] = str(teacher)
    if pruned:
        model.train(trainer=KeepPrunedArchitecture, **params)
    else:
        model.train(**params)
    trained_params = sum(parameter.numel() for parameter in model.model.parameters())
    if pruned and [
        model.model.model[index].conv.out_channels for index in (1, 3, 5, 7)
    ] != channel_sizes:
        raise RuntimeError("Training restored original backbone channel widths")
    if pruned and not quantized and trained_params != starting_params:
        raise RuntimeError(
            f"Trainer rebuilt the dense architecture: {starting_params} -> "
            f"{trained_params}"
        )
    metrics = model.val(
        data=str(dataset),
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        nms=False,
        rect=False,
        plots=False,
    )
    quantizers = sum(
        "TensorQuantizer" in type(module).__name__ for module in model.model.modules()
    )
    if quantized and quantizers == 0:
        raise RuntimeError("QAT training did not retain tensor quantizers")
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
    q_count = sum(node.op_type == "QuantizeLinear" for node in graph.graph.node)
    dq_count = sum(node.op_type == "DequantizeLinear" for node in graph.graph.node)
    if quantized and not (q_count == dq_count > 0):
        raise RuntimeError("QAT export did not preserve Q/DQ nodes")
    result_dir = Path("/output") / name
    copied = result_dir / f"yolo26n_{kind}_pilot.onnx"
    if exported != copied:
        copied.write_bytes(exported.read_bytes())
    metadata = {
        "kind": kind,
        "epochs": 5,
        "seed": seed,
        "data": "official train2017, 1024 train / 1024 development",
        "ultralytics": ultralytics.__version__,
        "torch": str(torch.__version__),
        "checkpoint_sha256": checkpoint_sha,
        "starting_checkpoint": starting_checkpoint,
        "starting_parameters": starting_params,
        "trained_parameters": trained_params,
        "pruned_stage_channels": channel_sizes,
        "distillation_teacher_sha256": (
            hashlib.sha256(teacher.read_bytes()).hexdigest() if distillation else None
        ),
        "development_map50_95": float(metrics.box.map),
        "development_map50": float(metrics.box.map50),
        "tensor_quantizers": quantizers,
        "onnx_q_nodes": q_count,
        "onnx_dq_nodes": dq_count,
        "onnx_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
        "onnx_bytes": copied.stat().st_size,
        "save_dir": str(result_dir),
    }
    (result_dir / "pilot-summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    result_volume.commit()
    return json.loads(json.dumps(metadata))


@app.local_entrypoint()
def main(kinds: str = "float,qat"):
    chosen = kinds.split(",")
    if not chosen or len(chosen) > 3 or len(set(chosen)) != len(chosen):
        raise ValueError("Specify up to three distinct training variants")
    calls = {kind: train.spawn(kind) for kind in chosen}
    out = Path(__file__).parent / "results"
    for kind, call in calls.items():
        result = call.get()
        target = out / f"pilot-{kind}.json"
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"Saved {kind} pilot to {target}")
