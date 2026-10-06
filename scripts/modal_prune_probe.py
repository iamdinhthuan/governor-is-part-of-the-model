"""Test dependency-safe physical YOLO26 channel removal and ONNX export.

Run: modal run modal_prune_probe.py --mode budget
The single-layer case is only a graph probe. The multi-layer L1 variant is
an actual baseline candidate, but needs matched retraining and AP evaluation.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("yolo26n-structured-pruning-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.172",
        "torch-pruning==1.6.1",
        "onnx==1.18.0",
    )
)
output_volume = modal.Volume.from_name("yolo26n-pilot-results", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=1800, volumes={"/output": output_volume})
def probe(mode: str) -> tuple[dict, bytes]:
    import onnx
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO

    yolo = YOLO("yolo26n.pt")
    net = yolo.model.cuda().eval()
    head = net.model[-1]
    head.end2end = True
    head.export = True
    example = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        before = net(example)
    if tuple(before.shape) != (1, 300, 6):
        raise RuntimeError(f"Unexpected pre-prune output: {before.shape}")
    before_params = sum(parameter.numel() for parameter in net.parameters())
    # Ultralytics inference checkpoints may have every parameter frozen;
    # dependency tracing requires an autograd graph.
    net.requires_grad_(True)

    if mode not in {"single", "budget", "object"}:
        raise ValueError(mode)
    target_layers = [1] if mode == "single" else [1, 3, 5, 7]
    risk = (
        json.loads(Path("/output/object-risk/scores.json").read_text())["channel_scores"]
        if mode == "object"
        else None
    )
    # Torch-Pruning calls .eval() during tracing. Override only the head's
    # training flag *inside* forward to return differentiable raw predictions,
    # while keeping every BN layer and the backbone in evaluation mode.
    class RawHead(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            self.wrapped.model[-1].training = True
            try:
                result = self.wrapped(x)
                return (
                    result["one2many"]["boxes"],
                    result["one2many"]["scores"],
                    result["one2one"]["boxes"],
                    result["one2one"]["scores"],
                )
            finally:
                self.wrapped.model[-1].training = False

    layer_changes = []
    for layer_id in target_layers:
        target = net.model[layer_id].conv
        old_channels = target.out_channels
        if old_channels < 16:
            raise RuntimeError(f"Unexpectedly narrow conv {layer_id}: {old_channels}")
        remove = max(1, old_channels // 4)
        scores = (
            torch.tensor(risk[str(layer_id)], device="cuda")
            if risk is not None
            else target.weight.detach().abs().sum(dim=(1, 2, 3))
        )
        if scores.numel() != old_channels or not torch.isfinite(scores).all():
            raise RuntimeError(f"Invalid pruning scores at layer {layer_id}")
        indices = scores.argsort()[:remove].tolist()
        trace_model = RawHead(net).eval()
        with torch.enable_grad():
            graph = tp.DependencyGraph().build_dependency(
                trace_model,
                example_inputs=example,
            )
        if target not in graph.module2node:
            raise RuntimeError(f"Conv {layer_id} is absent from dependency graph")
        group = graph.get_pruning_group(
            target, tp.prune_conv_out_channels, idxs=indices
        )
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Dependency group for conv {layer_id} was rejected")
        group.prune()
        layer_changes.append(
            {
                "layer": f"model.{layer_id}.conv",
                "before": old_channels,
                "after": target.out_channels,
            }
        )
        print(f"Physically pruned {layer_changes[-1]}", flush=True)
    with torch.no_grad():
        after = net(example)
    if tuple(after.shape) != (1, 300, 6) or not torch.isfinite(after).all():
        raise RuntimeError("Pruned model changed the end-to-end output contract")
    after_params = sum(parameter.numel() for parameter in net.parameters())
    if not after_params < before_params:
        raise RuntimeError("Pruning did not physically reduce parameters")

    head.export = False
    result_dir = Path("/output") / f"prune-{mode}"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / f"yolo26n_l1_{mode}.pt"
    yolo.save(str(checkpoint))
    # Verify that the saved *physical* architecture reloads before export.
    loaded = YOLO(str(checkpoint))
    loaded_head = loaded.model.model[-1]
    loaded_head.end2end = True
    loaded_head.export = True
    loaded.model.cuda().eval()
    with torch.no_grad():
        reloaded = loaded.model(example)
    if tuple(reloaded.shape) != (1, 300, 6):
        raise RuntimeError("Pruned checkpoint did not reload correctly")
    reloaded_params = sum(parameter.numel() for parameter in loaded.model.parameters())
    if reloaded_params != after_params:
        raise RuntimeError("Checkpoint reverted to unpruned architecture")

    exported = Path(
        loaded.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            simplify=False,
            device=0,
            nms=False,
        )
    )
    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph)
    if [dim.dim_value for dim in graph.graph.output[0].type.tensor_type.shape.dim] != [
        1,
        300,
        6,
    ]:
        raise RuntimeError("Pruned ONNX is not a one-to-one detector")
    blob = exported.read_bytes()
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(blob).hexdigest(),
        "target_layers": layer_changes,
        "criterion": (
            "smallest foreground object saliency with fake-INT8 error proxy"
            if mode == "object"
            else "smallest L1 weight norm"
        ),
        "parameters_before": before_params,
        "parameters_after": after_params,
        "parameters_removed_fraction": 1 - after_params / before_params,
        "output_shape": [1, 300, 6],
        "qualification": "physical pruning, no matched retraining/AP yet",
    }
    (result_dir / "probe-summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    output_volume.commit()
    return metadata, blob


@app.local_entrypoint()
def main(mode: str = "single"):
    metadata, blob = probe.remote(mode)
    base = Path(__file__).parent
    (base / "artifacts" / f"yolo26n_l1_prune_{mode}.onnx").write_bytes(blob)
    (base / "results" / f"prune-{mode}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print("Saved physical pruning probe and one-to-one ONNX")
