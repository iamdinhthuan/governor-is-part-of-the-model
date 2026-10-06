"""Compile-driven, structurally valid pruning candidates in timed YOLO stages.

Run `modal run modal_compiler_search.py --candidates p3` first. Once the
contract passes, `--candidates p34,ffn` runs at most two further L40S
exports. No accuracy claim is made from *unrecovered* checkpoints.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("yolo26n-compiler-guided-search")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.172",
        "torch-pruning==1.6.1",
        "onnx==1.18.0",
    )
    .env({"YOLO_CONFIG_DIR": "/tmp"})
)
output_volume = modal.Volume.from_name("yolo26n-pilot-results")

# Both class-tower targets feed an internal depthwise layer. The final
# 80-class output layer is *not* pruned. FFN0 is internal to C2PSA.
TARGETS = {
    "p3": ("model.23.one2one_cv3.0.0.1.conv",),
    "p34": (
        "model.23.one2one_cv3.0.0.1.conv",
        "model.23.one2one_cv3.1.0.1.conv",
    ),
    "ffn": ("model.22.m.0.1.ffn.0.conv",),
}


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    volumes={"/output": output_volume},
)
def prune(candidate: str) -> tuple[dict, bytes]:
    import onnx
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO

    if candidate not in TARGETS:
        raise ValueError(candidate)
    torch.manual_seed(20261004)
    yolo = YOLO("yolo26n.pt")
    checkpoint_sha = hashlib.sha256(Path("yolo26n.pt").read_bytes()).hexdigest()
    if checkpoint_sha != "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef":
        raise RuntimeError("Pretrained checkpoint does not match pilot baseline")
    net = yolo.model.cuda().eval().requires_grad_(True)
    head = net.model[-1]
    head.end2end = True
    head.export = True
    example = torch.zeros(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        baseline = net(example)
    if tuple(baseline.shape) != (1, 300, 6):
        raise RuntimeError("Pretrained network is not one-to-one")
    before_params = sum(p.numel() for p in net.parameters())

    class RawHead(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            self.wrapped.model[-1].training = True
            try:
                raw = self.wrapped(x)
                return (
                    raw["one2many"]["boxes"],
                    raw["one2many"]["scores"],
                    raw["one2one"]["boxes"],
                    raw["one2one"]["scores"],
                )
            finally:
                self.wrapped.model[-1].training = False

    changes = []
    for name in TARGETS[candidate]:
        module = dict(net.named_modules()).get(name)
        if not isinstance(module, torch.nn.Conv2d):
            raise RuntimeError(f"Not a prunable convolution: {name}")
        old_channels = module.out_channels
        # Round to backend-friendly multiples of 16. Keep a measurable
        # quarter of the original pointwise/FFN channels.
        remove = old_channels // 4
        if remove % 16:
            remove = 16 * max(1, round(remove / 16))
        scores = module.weight.detach().abs().sum((1, 2, 3))
        indices = scores.argsort()[:remove].tolist()
        with torch.enable_grad():
            graph = tp.DependencyGraph().build_dependency(
                RawHead(net).eval(), example_inputs=example
            )
        if module not in graph.module2node:
            raise RuntimeError(f"Missing dependency graph node: {name}")
        group = graph.get_pruning_group(
            module, tp.prune_conv_out_channels, idxs=indices
        )
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Dependency group is invalid: {name}")
        group.prune()
        with torch.no_grad():
            output = net(example)
        if tuple(output.shape) != (1, 300, 6) or not torch.isfinite(output).all():
            raise RuntimeError(f"Pruned graph broke one-to-one output: {name}")
        changes.append(
            {"layer": name, "before": old_channels, "after": module.out_channels}
        )
        print(f"Validated physical channel group: {changes[-1]}", flush=True)

    after_params = sum(p.numel() for p in net.parameters())
    if not 0 < after_params < before_params:
        raise RuntimeError("No physical parameter reduction")
    head.export = False
    result_dir = Path("/output/compiler-search") / candidate
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / "pruned.pt"
    yolo.save(str(checkpoint))
    reloaded = YOLO(str(checkpoint))
    reloaded.model.cuda().eval()
    reloaded.model.model[-1].end2end = True
    reloaded.model.model[-1].export = True
    with torch.no_grad():
        verified = reloaded.model(example)
    if (
        tuple(verified.shape) != (1, 300, 6)
        or sum(p.numel() for p in reloaded.model.parameters()) != after_params
    ):
        raise RuntimeError("Pruned physical checkpoint did not reload")

    exported = Path(
        reloaded.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            simplify=False,
            device=0,
            nms=False,
        )
    )
    onnx.checker.check_model(onnx.load(str(exported)))
    blob = exported.read_bytes()
    graph = onnx.load_from_string(blob)
    dims = [
        axis.dim_value for axis in graph.graph.output[0].type.tensor_type.shape.dim
    ]
    if dims != [1, 300, 6]:
        raise RuntimeError(f"Export changed one-to-one output: {dims}")
    copied = result_dir / "pruned.onnx"
    copied.write_bytes(blob)
    summary = {
        "candidate": candidate,
        "target_layers": changes,
        "criterion": "smallest L1 weight norm, hardware-aligned width",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "pretrained_sha256": checkpoint_sha,
        "onnx_sha256": hashlib.sha256(blob).hexdigest(),
        "onnx_bytes": len(blob),
        "parameters_before": before_params,
        "parameters_after": after_params,
        "parameters_removed_fraction": 1 - after_params / before_params,
        "output_shape": dims,
        "status": "unrecovered screening graph; no AP inference yet",
    }
    (result_dir / "search-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    output_volume.commit()
    return summary, blob


@app.local_entrypoint()
def main(candidates: str = "p3"):
    names = candidates.split(",")
    if not names or len(names) > 3 or len(set(names)) != len(names):
        raise ValueError("At most three distinct candidates are allowed")
    if not set(names) <= TARGETS.keys():
        raise ValueError(f"Unknown search candidates: {names}")
    calls = {name: prune.spawn(name) for name in names}
    base = Path(__file__).parent
    for name, call in calls.items():
        summary, blob = call.get()
        (base / "artifacts" / f"search_{name}.onnx").write_bytes(blob)
        (base / "results" / f"search_{name}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"Saved physical search candidate {name}")
