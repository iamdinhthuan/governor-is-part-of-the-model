"""Prepare official COCO val2017 and evaluate untouched YOLO26n.

Run: modal run modal_coco_val.py
The data preparation runs without a GPU. Evaluation uses one L40S.
Do not use val2017 for pruning sensitivity or hyperparameter selection.
"""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-coco-val2017")
volume = modal.Volume.from_name("yolo26-coco-val2017", create_if_missing=True)
prep_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests==2.32.5", "pyyaml==6.0.3"
)
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics==8.4.172",
        "pycocotools==2.0.10",
        "numpy<3",
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics", "WANDB_MODE": "disabled"})
)


@app.function(image=prep_image, volumes={"/data": volume}, timeout=2400)
def prepare() -> dict:
    import collections
    import shutil
    import zipfile

    import requests
    import yaml

    root = Path("/data/coco")
    images = root / "images" / "val2017"
    labels = root / "labels" / "val2017"
    annotation = root / "annotations" / "instances_val2017.json"
    ready = root / "READY.json"
    if ready.exists():
        return json.loads(ready.read_text())

    root.mkdir(parents=True, exist_ok=True)

    def download(url: str, target: Path, expected_bytes: int) -> None:
        # The COCO bucket's path-style S3 endpoint has a valid HTTPS
        # certificate. Its vanity hostname currently has a TLS name mismatch.
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if int(response.headers.get("Content-Length", "-1")) != expected_bytes:
                raise RuntimeError(f"Unexpected COCO archive size at {url}")
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if target.stat().st_size != expected_bytes:
            raise RuntimeError(f"Incomplete COCO archive at {target}")

    archive = root / "val2017.zip"
    download(
        "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
        archive,
        expected_bytes=815585330,
    )
    images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = [
            info
            for info in zipped.infolist()
            if info.filename.startswith("val2017/")
            and info.filename.endswith(".jpg")
            and Path(info.filename).name == info.filename.split("/")[-1]
        ]
        if len(members) != 5000:
            raise RuntimeError(f"Expected 5000 COCO validation images, got {len(members)}")
        for info in members:
            with zipped.open(info) as source, (images / Path(info.filename).name).open(
                "wb"
            ) as destination:
                shutil.copyfileobj(source, destination)
    archive.unlink()  # This task created the archive; retain the extracted data.

    archive = root / "annotations_trainval2017.zip"
    download(
        "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        archive,
        expected_bytes=252907541,
    )
    annotation.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open("annotations/instances_val2017.json") as source, annotation.open(
            "wb"
        ) as destination:
            shutil.copyfileobj(source, destination)
    archive.unlink()

    coco = json.loads(annotation.read_text())
    category_ids = sorted(c["id"] for c in coco["categories"])
    class_from_id = {category: index for index, category in enumerate(category_ids)}
    names_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    image_by_id = {image["id"]: image for image in coco["images"]}
    if len(category_ids) != 80 or len(image_by_id) != 5000:
        raise RuntimeError("Unexpected COCO val2017 category or image count")

    rows = collections.defaultdict(list)
    for item in coco["annotations"]:
        if item.get("iscrowd"):
            continue
        image = image_by_id[item["image_id"]]
        x, y, width, height = item["bbox"]
        if width <= 0 or height <= 0:
            continue
        w, h = image["width"], image["height"]
        center_x, center_y = x + width / 2, y + height / 2
        rows[Path(image["file_name"]).stem].append(
            f"{class_from_id[item['category_id']]} "
            f"{center_x / w:.8f} {center_y / h:.8f} "
            f"{width / w:.8f} {height / h:.8f}\n"
        )
    labels.mkdir(parents=True, exist_ok=True)
    for stem, lines in rows.items():
        (labels / f"{stem}.txt").write_text("".join(lines))
    dataset = {
        "path": str(root),
        "train": "images/val2017",
        "val": "images/val2017",
        "names": [names_by_id[c] for c in category_ids],
    }
    (root / "coco_val.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    summary = {
        "image_count": len(image_by_id),
        "category_count": len(category_ids),
        "label_file_count": len(rows),
        "label_count": sum(map(len, rows.values())),
        "annotations": str(annotation),
        "dataset_yaml": str(root / "coco_val.yaml"),
    }
    ready.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    volume.commit()
    return summary


@app.function(
    image=eval_image, gpu="L40S", volumes={"/data": volume}, timeout=1800
)
def evaluate() -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import ultralytics
    from ultralytics import YOLO

    root = Path("/data/coco")
    info = json.loads((root / "READY.json").read_text())
    gt = COCO(info["annotations"])
    category_ids = sorted(gt.getCatIds())
    model = YOLO("yolo26n.pt")
    head = model.model.model[-1]
    if not hasattr(head, "one2one_cv2"):
        raise RuntimeError("YOLO26n has no one-to-one detection branch")
    metrics = model.val(
        data=info["dataset_yaml"],
        imgsz=640,
        batch=32,
        rect=False,  # Match the static 640x640 TensorRT engine.
        device=0,
        workers=4,
        plots=False,
        save_json=True,
        nms=False,  # 8.4.142+ selects the one-to-one branch explicitly.
        project="/tmp/yolo26n-coco-baseline",
        name="float-square",
        exist_ok=True,
    )
    predictions_path = Path(metrics.save_dir) / "predictions.json"
    predictions = json.loads(predictions_path.read_text())
    if not predictions:
        raise RuntimeError("No COCO predictions were saved")
    # A custom dataset YAML makes Ultralytics use one-based contiguous
    # category IDs rather than the sparse official COCO IDs.
    predicted_ids = {int(p["category_id"]) for p in predictions}
    if not predicted_ids.issubset(set(range(1, 81))):
        raise RuntimeError(f"Unexpected prediction category IDs: {sorted(predicted_ids)}")
    for prediction in predictions:
        prediction["category_id"] = category_ids[int(prediction["category_id"]) - 1]
    result = gt.loadRes(predictions)
    evaluator = COCOeval(gt, result, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "dataset": "official COCO val2017, 5000 images",
        "ultralytics": ultralytics.__version__,
        "checkpoint": "yolo26n.pt",
        "image_size": 640,
        "rect": False,
        "precision": "PyTorch float baseline; no engine/AP comparison yet",
        "nms_free_requested": True,
        "one_to_one_runtime_verified": False,
        "head_end2end_after_validation": bool(head.end2end),
        "predictions": len(predictions),
        "ultralytics_map50_95": float(metrics.box.map),
        "ultralytics_map50": float(metrics.box.map50),
        "official_cocoeval_ap50_95": float(evaluator.stats[0]),
        "official_cocoeval_ap50": float(evaluator.stats[1]),
        "official_cocoeval_ap75": float(evaluator.stats[2]),
        "official_cocoeval_small": float(evaluator.stats[3]),
        "official_cocoeval_medium": float(evaluator.stats[4]),
        "official_cocoeval_large": float(evaluator.stats[5]),
    }


@app.local_entrypoint()
def main():
    info = prepare.remote()
    print(f"Prepared {info['image_count']} COCO validation images")
    result = evaluate.remote()
    out = Path(__file__).parent / "results" / "coco-val2017-float-square.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved COCO val2017 baseline to {out}")
