"""Standard Ultralytics TensorRT predictor baseline on the blind Jetson split.

Uses Ultralytics' own engine export and model.predict() per image file
(its decode, letterbox, inference and postprocess), one-to-one head with
nms=False, conf=0.001, max_det=300. Writes COCO predictions and timings.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.annotations.read_text())
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if args.limit:
        images = images[: args.limit]
    category_ids = sorted(item["id"] for item in data["categories"])
    model = YOLO(str(args.engine), task="detect")
    kwargs = dict(imgsz=640, conf=0.001, max_det=300, nms=False, verbose=False)

    predictions, wall, speed = [], [], {"preprocess": [], "inference": [], "postprocess": []}
    begin = None
    t0 = time.perf_counter() + 0.5
    for index, image in enumerate(images):
        if index == args.warmup:
            begin = time.perf_counter()
            begin_unix = time.time()
            begin_cpu = time.process_time()
        if args.fps > 0:
            arrival = t0 + index / args.fps
            delay = arrival - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        start = time.perf_counter()
        origin = arrival if args.fps > 0 else start
        result = model.predict(str(args.images_dir / image["file_name"]), **kwargs)[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls_index in zip(xyxy, scores, classes):
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append({
                "image_id": image["id"],
                "category_id": category_ids[int(cls_index)],
                "bbox": [
                    round(float(x1), 3), round(float(y1), 3),
                    round(float(x2 - x1), 3), round(float(y2 - y1), 3),
                ],
                "score": round(float(score), 6),
            })
        if index >= args.warmup:
            wall.append((time.perf_counter() - origin) * 1000)
            for key in speed:
                speed[key].append(result.speed[key])
    elapsed = time.perf_counter() - begin
    cpu_seconds = time.process_time() - begin_cpu
    end_unix = time.time()

    args.predictions.write_text(json.dumps(predictions))
    stat = lambda v: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                      "p95": float(np.percentile(v, 95))}
    report = {
        "engine": str(args.engine),
        "images": len(images),
        "images_timed": len(wall),
        "prediction_count": len(predictions),
        "arrival_fps": args.fps,
        "latency_ms": stat(wall),
        "latency_p99_ms": float(np.percentile(wall, 99)),
        "ultralytics_speed_ms": {key: stat(value) for key, value in speed.items()},
        "throughput_images_per_s": len(wall) / elapsed,
        "process_cpu_ms_per_image": 1000 * cpu_seconds / len(wall),
        "window_unix": [begin_unix, end_unix],
        "note": "Per-image span is the predict() call from file path through "
                "COCO rows; Ultralytics does decode, letterbox and postprocess.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fps": args.fps,
                      "latency_median_ms": round(report["latency_ms"]["median"], 3),
                      "latency_p95_ms": round(report["latency_ms"]["p95"], 3),
                      "images_per_s": round(report["throughput_images_per_s"], 2),
                      "predictions": len(predictions)}))


if __name__ == "__main__":
    main()
