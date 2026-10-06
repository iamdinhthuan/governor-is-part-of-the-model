"""Prepare disjoint COCO train2017 pilot splits without using validation data.

Run: modal run modal_train_data.py
Downloads the official train archive to a persistent Modal volume. Extracts
only 2,048 deterministically selected images (1,024 calibration/training and
1,024 development), leaving val2017 untouched. The ZIP is kept for later
full-dataset training. This CPU-only task has a bounded four-hour timeout.
"""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-coco-train2017-prep")
volume = modal.Volume.from_name("yolo26-coco-train2017", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests==2.32.5", "pyyaml==6.0.3"
)


@app.function(image=image, volumes={"/data": volume}, timeout=14400, memory=8192)
def prepare() -> dict:
    import collections
    import hashlib
    import random
    import shutil
    import zipfile

    import requests
    import yaml

    root = Path("/data/coco")
    ready = root / "PILOT_READY.json"
    if ready.exists():
        return json.loads(ready.read_text())
    root.mkdir(parents=True, exist_ok=True)

    def download(url: str, path: Path, expected_bytes: int) -> None:
        if path.exists() and path.stat().st_size == expected_bytes:
            return
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            length = int(response.headers.get("Content-Length", "-1"))
            if length != expected_bytes:
                raise RuntimeError(f"Unexpected archive length {length} at {url}")
            with path.open("wb") as destination:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        destination.write(chunk)
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Incomplete archive {path}")

    annotations_zip = root / "annotations_trainval2017.zip"
    download(
        "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        annotations_zip,
        252907541,
    )
    annotations_path = root / "instances_train2017.json"
    with zipfile.ZipFile(annotations_zip) as archive:
        with archive.open("annotations/instances_train2017.json") as source, (
            annotations_path.open("wb")
        ) as destination:
            shutil.copyfileobj(source, destination)
    annotations_zip.unlink()  # Created by this task.
    data = json.loads(annotations_path.read_text())
    images = data["images"]
    categories = sorted(data["categories"], key=lambda item: item["id"])
    if len(images) != 118287 or len(categories) != 80:
        raise RuntimeError("Unexpected official COCO train2017 metadata")

    shuffled = sorted(images, key=lambda item: item["id"])
    random.Random(20261004).shuffle(shuffled)
    selected = shuffled[:2048]
    train, development = selected[:1024], selected[1024:]
    selected_by_id = {item["id"]: item for item in selected}
    class_by_id = {item["id"]: index for index, item in enumerate(categories)}
    labels_by_stem = collections.defaultdict(list)
    for item in data["annotations"]:
        image_info = selected_by_id.get(item["image_id"])
        if image_info is None or item.get("iscrowd"):
            continue
        x, y, width, height = item["bbox"]
        if width <= 0 or height <= 0:
            continue
        w, h = image_info["width"], image_info["height"]
        labels_by_stem[Path(image_info["file_name"]).stem].append(
            f"{class_by_id[item['category_id']]} "
            f"{(x + width / 2) / w:.8f} {(y + height / 2) / h:.8f} "
            f"{width / w:.8f} {height / h:.8f}\n"
        )
    del data

    archive_path = root / "train2017.zip"
    download(
        "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip",
        archive_path,
        19336861798,
    )
    volume.commit()  # Retain the costly verified download before extraction.

    image_dir = root / "images" / "train2017"
    label_dir = root / "labels" / "train2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for index, item in enumerate(selected):
            destination = image_dir / item["file_name"]
            if not destination.exists():
                with archive.open("train2017/" + item["file_name"]) as source, (
                    destination.open("wb")
                ) as target:
                    shutil.copyfileobj(source, target)
            lines = labels_by_stem.get(destination.stem)
            if lines:
                (label_dir / f"{destination.stem}.txt").write_text("".join(lines))
            if (index + 1) % 256 == 0:
                print(f"Extracted {index + 1}/2048 train images", flush=True)

    def write_split(name: str, subset: list[dict]) -> str:
        path = root / f"{name}.txt"
        path.write_text(
            "".join(str(image_dir / item["file_name"]) + "\n" for item in subset)
        )
        return path.name

    train_file = write_split("pilot_train", train)
    dev_file = write_split("pilot_development", development)
    dataset = {
        "path": str(root),
        "train": train_file,
        "val": dev_file,
        "names": [category["name"] for category in categories],
    }
    dataset_path = root / "pilot.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False))
    split_fingerprint = hashlib.sha256(
        ",".join(str(item["id"]) for item in selected).encode()
    ).hexdigest()
    summary = {
        "source": "official COCO train2017",
        "seed": 20261004,
        "train_images": len(train),
        "development_images": len(development),
        "selected_ids_sha256": split_fingerprint,
        "labels": sum(map(len, labels_by_stem.values())),
        "dataset_yaml": str(dataset_path),
        "train_archive_bytes": archive_path.stat().st_size,
        "separate_from_val2017": True,
    }
    ready.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    summary = prepare.remote()
    out = Path(__file__).parent / "results" / "coco-train-pilot.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Prepared train/development split: {out}")
