"""Pilot object-conditioned channel risk on COCO train2017, never val2017.

For each correctly matched one-to-one detection, measure the channel's
foreground |activation * confidence gradient| plus a local fake-INT8
perturbation proxy. The proxy is *not* a TensorRT precision assignment.
"""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-object-risk-pilot")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.172")
)
data_volume = modal.Volume.from_name("yolo26-coco-train2017")
out_volume = modal.Volume.from_name("yolo26n-pilot-results")


@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
    volumes={"/data": data_volume, "/output": out_volume},
)
def score_channels() -> dict:
    import hashlib

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    torch.manual_seed(20261004)
    net = YOLO("yolo26n.pt").model.cuda().eval().requires_grad_(True)
    net.model[-1].end2end = True
    net.model[-1].export = True
    indices = [1, 3, 5, 7]
    features = {}
    handles = []
    for index in indices:
        def hook(_, __, output, layer=index):
            output.retain_grad()
            features[layer] = output

        handles.append(net.model[index].conv.register_forward_hook(hook))
    paths = [
        Path(line)
        for line in Path("/data/coco/pilot_train.txt").read_text().splitlines()[:128]
    ]
    saliency = {i: torch.zeros(net.model[i].conv.out_channels, device="cuda") for i in indices}
    quant_error = {i: torch.zeros_like(saliency[i]) for i in indices}
    matched = 0
    small_matches = 0
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Unreadable official image {path}")
        height, width = bgr.shape[:2]
        ratio = min(640 / height, 640 / width)
        resized = cv2.resize(
            bgr, (round(width * ratio), round(height * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        dw, dh = (640 - resized.shape[1]) / 2, (640 - resized.shape[0]) / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        tensor = torch.from_numpy(
            np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
        ).cuda().float()[None] / 255.0
        detections = net(tensor)[0]
        loss_terms = []
        boxes = []
        label = Path("/data/coco/labels/train2017") / (path.stem + ".txt")
        for line in label.read_text().splitlines() if label.exists() else []:
            cls, cx, cy, w, h = map(float, line.split())
            raw_x1, raw_y1 = (cx - w / 2) * width, (cy - h / 2) * height
            raw_x2, raw_y2 = (cx + w / 2) * width, (cy + h / 2) * height
            box = torch.tensor(
                [
                    raw_x1 * ratio + left,
                    raw_y1 * ratio + top,
                    raw_x2 * ratio + left,
                    raw_y2 * ratio + top,
                ],
                device="cuda",
            )
            inter = (
                torch.minimum(detections[:, 2:4], box[2:])
                - torch.maximum(detections[:, :2], box[:2])
            ).clamp(min=0).prod(dim=1)
            area = (detections[:, 2:4] - detections[:, :2]).clamp(min=0).prod(dim=1)
            truth_area = (box[2:] - box[:2]).clamp(min=0).prod()
            iou = inter / (area + truth_area - inter + 1e-8)
            iou = iou * (detections[:, 5].round() == int(cls))
            best = int(iou.argmax())
            if float(iou[best]) < 0.3:
                continue
            is_small = w * width * h * height < 32**2
            weight = 2.0 if is_small else 1.0
            loss_terms.append(detections[best, 4] * weight)
            boxes.append((box.detach(), weight))
            matched += 1
            small_matches += int(is_small)
        if not loss_terms:
            continue
        net.zero_grad(set_to_none=True)
        torch.stack(loss_terms).sum().backward()
        with torch.no_grad():
            for index in indices:
                feature = features[index][0]
                gradient = features[index].grad[0]
                _, grid_h, grid_w = feature.shape
                step = feature.abs().amax().clamp(min=1e-8) / 127
                perturbation = feature - (feature / step).round().clamp(-127, 127) * step
                for box, weight in boxes:
                    x1 = max(0, min(grid_w - 1, int(box[0] * grid_w / 640)))
                    x2 = max(x1 + 1, min(grid_w, int(torch.ceil(box[2] * grid_w / 640))))
                    y1 = max(0, min(grid_h - 1, int(box[1] * grid_h / 640)))
                    y2 = max(y1 + 1, min(grid_h, int(torch.ceil(box[3] * grid_h / 640))))
                    region = (slice(None), slice(y1, y2), slice(x1, x2))
                    saliency[index] += weight * (
                        feature[region] * gradient[region]
                    ).abs().mean(dim=(1, 2))
                    quant_error[index] += weight * (
                        perturbation[region] * gradient[region]
                    ).abs().mean(dim=(1, 2))
        features.clear()
    for handle in handles:
        handle.remove()
    if matched < 100 or small_matches < 10:
        raise RuntimeError(f"Too few matched objects: {matched}, small: {small_matches}")
    scores = {
        str(index): (saliency[index] + 10 * quant_error[index]).cpu().tolist()
        for index in indices
    }
    summary = {
        "source": "official train2017 pilot train split, first 128 images",
        "image_names_sha256": hashlib.sha256(
            ",".join(path.name for path in paths).encode()
        ).hexdigest(),
        "matched_objects": matched,
        "small_matches": small_matches,
        "channel_scores": scores,
        "criterion": "foreground |a*grad(object-score)| + 10*|grad*(a-Q8(a))|",
        "caveat": "fake-INT8 proxy is not the compiled TensorRT layer precision map",
    }
    target = Path("/output/object-risk")
    target.mkdir(parents=True, exist_ok=True)
    (target / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    out_volume.commit()
    return summary


@app.local_entrypoint()
def main():
    summary = score_channels.remote()
    out = Path(__file__).parent / "results" / "object-risk-pilot.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved pilot object saliency scores: {out}")
