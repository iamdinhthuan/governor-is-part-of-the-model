"""Prepare a fresh train2017-only verification split for the frozen router.

Images occupy shuffled positions [2048:3072], disjoint from the pilot
train/development 2048 images and never COCO val2017. CPU-only.
"""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-resolution-external-verification")
data_volume = modal.Volume.from_name("yolo26-coco-train2017")
result_volume = modal.Volume.from_name("yolo26n-pilot-results")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    volumes={"/data": data_volume, "/output": result_volume},
    timeout=3600,
    memory=4096,
)
def prepare() -> dict:
    import hashlib
    import random
    import zipfile

    root = Path("/data/coco")
    data = json.loads((root / "instances_train2017.json").read_text())
    shuffled = sorted(data["images"], key=lambda item: item["id"])
    random.Random(20261004).shuffle(shuffled)
    pilot = {item["id"] for item in shuffled[:2048]}
    independent = shuffled[2048:3072]
    ids = {item["id"] for item in independent}
    if len(ids) != 1024 or ids & pilot:
        raise RuntimeError("Verification split overlaps pilot")
    subset = {
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": data["categories"],
        "images": independent,
        "annotations": [
            annotation
            for annotation in data["annotations"]
            if annotation["image_id"] in ids
        ],
    }
    out = Path("/output/resolution-verification")
    out.mkdir(parents=True, exist_ok=True)
    (out / "instances_verification2017.json").write_text(
        json.dumps(subset, separators=(",", ":"))
    )
    checksums = {}
    with zipfile.ZipFile(root / "train2017.zip") as archive:
        for index, item in enumerate(independent):
            name = item["file_name"]
            checksums[name] = hashlib.sha256(
                archive.read("train2017/" + name)
            ).hexdigest()
            if (index + 1) % 256 == 0:
                print(f"Verified official hashes: {index+1}/1024", flush=True)
    (out / "image-hashes.json").write_text(
        json.dumps(checksums, sort_keys=True) + "\n"
    )
    summary = {
        "source": "official COCO train2017",
        "selection": "seed 20261004, shuffled positions [2048:3072]",
        "images": len(ids),
        "annotations": len(subset["annotations"]),
        "categories": len(subset["categories"]),
        "image_ids_sha256": hashlib.sha256(
            ",".join(map(str, sorted(ids))).encode()
        ).hexdigest(),
        "pilot_overlap": 0,
        "never_seen_by_frozen_router": True,
    }
    (out / "verification-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    result_volume.commit()
    return summary


@app.local_entrypoint()
def main():
    metadata = prepare.remote()
    path = Path(__file__).parent / "results/resolution-verification-split.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Prepared independent verification split: {path}")
