"""Bundle the disjoint COCO train2017 development split for Jetson AP."""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-coco-dev-bundle")
data_volume = modal.Volume.from_name("yolo26-coco-train2017")
result_volume = modal.Volume.from_name("yolo26n-pilot-results")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/data": data_volume, "/output": result_volume},
    timeout=1800,
    memory=4096,
)
def bundle() -> dict:
    import hashlib
    import zipfile

    root = Path("/data/coco")
    dev_paths = [Path(line) for line in (root / "pilot_development.txt").read_text().splitlines()]
    if len(dev_paths) != 1024 or len({item.name for item in dev_paths}) != 1024:
        raise RuntimeError("Unexpected development split")
    names = {item.name for item in dev_paths}
    data = json.loads((root / "instances_train2017.json").read_text())
    selected_images = [item for item in data["images"] if item["file_name"] in names]
    ids = {item["id"] for item in selected_images}
    if len(ids) != 1024:
        raise RuntimeError("Split and official annotations do not match")
    annotation = {
        "info": data["info"],
        "licenses": data["licenses"],
        "images": selected_images,
        "annotations": [item for item in data["annotations"] if item["image_id"] in ids],
        "categories": data["categories"],
    }
    target = Path("/output/development_bundle")
    target.mkdir(parents=True, exist_ok=True)
    (target / "instances_development2017.json").write_text(
        json.dumps(annotation, separators=(",", ":"))
    )
    zip_path = target / "images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for item in sorted(dev_paths):
            archive.write(item, arcname=item.name)
    summary = {
        "split": "COCO train2017 held-out development (not val2017)",
        "images": len(selected_images),
        "annotations": len(annotation["annotations"]),
        "archive_bytes": zip_path.stat().st_size,
        "archive_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "image_ids_sha256": hashlib.sha256(
            ",".join(map(str, sorted(ids))).encode()
        ).hexdigest(),
    }
    (target / "bundle-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    result_volume.commit()
    return summary


@app.local_entrypoint()
def main():
    summary = bundle.remote()
    out = Path(__file__).parent / "results" / "pilot-dev-bundle.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Created Jetson development bundle: {out}")
