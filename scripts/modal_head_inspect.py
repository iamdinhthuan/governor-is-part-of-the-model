"""Read YOLO26 detection-head control flow before structural pruning."""

import inspect

import modal

app = modal.App("yolo26n-pruning-head-inspection")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.172")
)


@app.function(image=image, timeout=600)
def inspect_head() -> None:
    from ultralytics import YOLO

    head = YOLO("yolo26n.pt").model.model[-1]
    print(f"Head class: {type(head)!r}")
    print(f"end2end: {head.end2end}")
    for name in ("forward", "forward_end2end", "forward_head"):
        if hasattr(head, name):
            print(f"\n{name}:\n{inspect.getsource(getattr(type(head), name))}")
    from ultralytics import YOLO

    network = YOLO("yolo26n.pt").model
    for index, block in enumerate(network.model):
        widths = {}
        for name in ("conv", "cv1", "cv2", "cv3"):
            submodule = getattr(block, name, None)
            convolution = getattr(submodule, "conv", None)
            if convolution is not None:
                widths[name] = (
                    convolution.in_channels,
                    convolution.out_channels,
                )
        print(
            f"stage={index} from={getattr(block, 'f', None)} "
            f"type={type(block).__name__} "
            f"parameters={sum(p.numel() for p in block.parameters())} "
            f"widths={widths} "
            f"blocks={len(block.m) if hasattr(block, 'm') and hasattr(block.m, '__len__') else 0}"
        )
    import torch

    for stage in (2, 16, 19, 22, 23):
        block = network.model[stage]
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                print(
                    f"conv_stage={stage} name={name} in={module.in_channels} "
                    f"out={module.out_channels} kernel={module.kernel_size} "
                    f"groups={module.groups}"
                )


@app.local_entrypoint()
def main():
    inspect_head.remote()
