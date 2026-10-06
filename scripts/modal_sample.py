"""Export a deterministic preprocessed COCO val image for numeric checks."""

import json
from pathlib import Path

import modal

app = modal.App("yolo26n-coco-numeric-sample")
volume = modal.Volume.from_name("yolo26-coco-val2017")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.172", "numpy<3")
)


@app.function(image=image, volumes={"/data": volume}, timeout=120)
def sample() -> tuple[bytes, dict]:
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    path = sorted(Path("/data/coco/images/val2017").glob("*.jpg"))[0]
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Cannot read {path.name}")
    padded = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=bgr)
    rgb_chw = np.ascontiguousarray(
        padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32
    )
    tensor = (rgb_chw / 255.0)[None]
    return tensor.tobytes(order="C"), {
        "image": path.name,
        "original_hw": list(bgr.shape[:2]),
        "input_shape": list(tensor.shape),
        "letterbox": "Ultralytics LetterBox 640 auto=False stride=32",
        "normalization": "RGB float32 [0,1]",
    }


@app.local_entrypoint()
def main():
    data, metadata = sample.remote()
    out = Path(__file__).parent / "artifacts" / "numeric_coco_input_fp32.bin"
    out.write_bytes(data)
    (Path(__file__).parent / "artifacts" / "numeric_coco_input.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved {metadata['image']} input tensor to {out}")
