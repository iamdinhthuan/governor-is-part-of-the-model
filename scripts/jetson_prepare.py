"""Download COCO val2017 images to a temporary Jetson-only directory."""

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/tmp/yolo26-coco-val")
IMAGES = ROOT / "val2017"
READY = ROOT / "READY.json"
URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
EXPECTED_BYTES = 815585330


def main() -> None:
    if READY.exists() and len(list(IMAGES.glob("*.jpg"))) == 5000:
        print(READY.read_text())
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "val2017.zip"
    with urllib.request.urlopen(URL, timeout=120) as source, archive.open("wb") as dest:
        length = int(source.headers.get("Content-Length", "-1"))
        if length != EXPECTED_BYTES:
            raise RuntimeError(f"Unexpected COCO archive size: {length}")
        shutil.copyfileobj(source, dest)
    if archive.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Incomplete COCO archive")
    with zipfile.ZipFile(archive) as zipped:
        images = [
            member
            for member in zipped.infolist()
            if member.filename.startswith("val2017/")
            and member.filename.endswith(".jpg")
        ]
        if len(images) != 5000:
            raise RuntimeError(f"Expected 5000 images, found {len(images)}")
        for member in images:
            name = Path(member.filename).name
            with zipped.open(member) as source, (IMAGES / name).open("wb") as dest:
                shutil.copyfileobj(source, dest)
    archive.unlink()  # This script created the verified archive.
    metadata = {"images": len(images), "source": URL, "root": str(IMAGES)}
    READY.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
