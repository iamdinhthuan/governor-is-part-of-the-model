#!/usr/bin/env python3
"""Generate QNN INT8 calibration raws for the yolo26n e2e ONNX.

Each raw file is the model input tensor in the ONNX's own layout:
(1,3,640,640) float32 NCHW, RGB, [0,1], with the same letterbox geometry
as the measurement pipelines (round(d - 0.1)).

Usage: make_calib.py <val2017_dir> <out_dir> [n=500]
Writes <out_dir>/<stem>.raw plus <out_dir>/calib_list.txt (one path/line,
the format qnn-onnx-converter --input_list expects).
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    images_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(images_dir.glob("*.jpg"))[:n]
    lines = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        r = min(640 / h, 640 / w)
        nw, nh = round(w * r - 0.1), round(h * r - 0.1)  # match eval letterbox
        dw, dh = (640 - nw) / 2, (640 - nh) / 2
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, np.uint8)
        top, left = round(dh - 0.1), round(dw - 0.1)
        canvas[top:top + nh, left:left + nw] = resized
        # [0,1] float: the ONNX takes normalized input and the eval feeds
        # /255 floats, so calibration must use the same scale
        x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        raw = out_dir / (f.stem + ".raw")
        x.tofile(raw)
        lines.append(str(raw))
    (out_dir / "calib_list.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} raws -> {out_dir}")


if __name__ == "__main__":
    main()
