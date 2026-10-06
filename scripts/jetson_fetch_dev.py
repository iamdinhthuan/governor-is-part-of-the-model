"""Download the fixed train2017 development image IDs directly on Jetson.

This avoids relaying 163 MB of public COCO images through a slow SSH link.
The manifest hashes come from the official archive, independently extracted
on Modal. Run with the matching annotation/manifest files in /tmp.
"""

import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

def download(item: tuple[str, str], root: Path) -> None:
    name, digest = item
    target = root / name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        return
    url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/" + name
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                blob = response.read()
            if hashlib.sha256(blob).hexdigest() != digest:
                raise RuntimeError(f"Official image SHA256 mismatch: {name}")
            temporary = target.with_suffix(".download")
            temporary.write_bytes(blob)
            temporary.replace(target)
            return
        except (TimeoutError, OSError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed to download {name}") from exc
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("/tmp/pilot-dev-image-hashes.json")
    )
    parser.add_argument(
        "--root", type=Path, default=Path("/tmp/yolo26-pilot-development/images")
    )
    parser.add_argument("--expected-count", type=int, default=1024)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if len(manifest) != args.expected_count:
        raise RuntimeError("Unexpected development image count")
    args.root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda item: download(item, args.root), sorted(manifest.items())))
    if len(list(args.root.glob("*.jpg"))) != args.expected_count:
        raise RuntimeError("Development image extraction is incomplete")
    print(f"Downloaded and verified {len(manifest)} COCO train2017 images")


if __name__ == "__main__":
    main()
