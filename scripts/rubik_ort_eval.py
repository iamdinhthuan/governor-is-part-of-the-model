#!/usr/bin/env python3
"""Rubik Pi 3 (QCM6490) YOLO26n CPU pipeline mirroring jetson_fast_eval.py.

ONNX Runtime CPU execution of the end-to-end (1,300,6) export: OpenCV decode,
letterbox to 640x640 (value 114), float32 [0,1] NCHW on the host, COCO rows.
Options mirror the other platforms:
  --prefetch / --workers K  decode+letterbox of upcoming images overlaps inference.
  --overlap  / --inflight N N in-flight session.run calls on an executor; the
             completion callback builds the COCO rows of a finished frame.
  --wait sleep|spin  how the main thread waits for a free inference slot.
  --intra N          onnxruntime intra-op threads (default 0 = all cores).
  --fps F  camera-like arrivals; latency from arrival to finished rows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import sys
from pathlib import Path
import statistics
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

SCORE_MIN = 0.001


def wait_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def load(path: Path, size: int, arrival: float):
    """Decode and resize; geometry identical to the other pipelines.

    ``arrival`` is the frame's availability time (perf_counter epoch), or
    0.0 in backlog mode. Latency starts at availability, so queueing behind
    a slow pipeline is included (Jetson semantics); in backlog mode it
    starts at decode time instead.
    """
    wait_until(arrival)
    started = arrival if arrival > 0.0 else time.perf_counter()
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Unreadable {path}")
    height, width = bgr.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    dw, dh = (size - new_width) / 2, (size - new_height) / 2
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return started, resized, ratio, round(dh - 0.1), round(dw - 0.1)


def rows_for(output, image, ratio, left, top, category_ids, predictions) -> None:
    """Identical conversion to jetson_fast_eval.py: (1,300,6) xyxy rows."""
    rows = output[0][output[0, :, 4] >= SCORE_MIN]
    classes = np.rint(rows[:, 5]).astype(np.int32)
    if np.any((classes < 0) | (classes >= 80)):
        raise RuntimeError("Invalid class index")
    boxes = rows[:, :4].astype(np.float64)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / ratio, 0, image["width"])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / ratio, 0, image["height"])
    for (x1, y1, x2, y2), score, cls_index in zip(boxes, rows[:, 4], classes):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--inflight", type=int, default=2)
    parser.add_argument("--wait", choices=("sleep", "spin"), default="sleep")
    parser.add_argument("--intra", type=int, default=0)
    parser.add_argument("--no-gc", action="store_true")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.overlap and not args.prefetch:
        raise RuntimeError("--overlap needs --prefetch (decode must not block the submitter)")

    data = json.loads(args.annotations.read_text())
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if args.limit:
        images = images[: args.limit]
    category_ids = sorted(item["id"] for item in data["categories"])
    paths = [args.images_dir / item["file_name"] for item in images]
    arrivals = np.arange(len(paths)) / args.fps if args.fps > 0 else np.zeros(len(paths))

    options = ort.SessionOptions()
    options.log_severity_level = 3
    if args.intra > 0:
        options.intra_op_num_threads = args.intra
        options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    in_shape = session.get_inputs()[0].shape  # (1,3,640,640)
    size = int(in_shape[-1])

    slots = args.inflight if args.overlap else 1
    canvases = [np.full((size, size, 3), 114, dtype=np.uint8) for _ in range(slots)]
    inputs = [np.empty((1, 3, size, size), dtype=np.float32) for _ in range(slots)]

    def fill(slot: int, resized: np.ndarray, top: int, left: int) -> None:
        canvas = canvases[slot]
        canvas[:] = 114
        h, w = resized.shape[:2]
        canvas[top:top + h, left:left + w] = resized
        # host preprocessing: BGR->RGB, /255, HWC->NCHW (the "float" host cost)
        np.divide(canvas[:, :, ::-1], 255.0, out=inputs[slot][0].transpose(1, 2, 0))

    pool = ThreadPoolExecutor(max_workers=args.workers) if args.prefetch else None
    infer_pool = (ThreadPoolExecutor(max_workers=slots) if args.overlap else None)
    epoch = time.perf_counter()  # arrival times are relative to this instant

    def arrival_at(i: int) -> float:
        return epoch + arrivals[i] if args.fps > 0 else 0.0

    pending = [pool.submit(load, paths[i], size, arrival_at(i))
               for i in range(min(args.workers, len(paths)))] if pool else []

    predictions: list = []
    starts = [0.0] * len(images)
    latencies: list[float] = []
    infer_wall: list[float] = []
    count_lock = threading.Lock()
    slot_sem = threading.Semaphore(slots)

    begin = begin_unix = begin_cpu = None

    def finish_rows(index: int, output, ratio, left, top) -> None:
        rows_for(output, images[index], ratio, left, top, category_ids, predictions)
        if index >= args.warmup:
            latencies.append((time.perf_counter() - starts[index]) * 1000)

    def infer_cb(index, slot, ratio, left, top, submitted):
        output = session.run(None, {in_name: inputs[slot]})[0]
        if index >= args.warmup:
            infer_wall.append((time.perf_counter() - submitted) * 1000)
        finish_rows(index, output, ratio, left, top)
        slot_sem.release()

    for index in range(len(images)):
        if index == args.warmup:
            if args.no_gc:
                gc.collect(); gc.disable()
            begin, begin_unix, begin_cpu = time.perf_counter(), time.time(), time.process_time()
        started, resized, ratio, top, left = (pending.pop(0).result() if pool
                                              else load(paths[index], size, arrival_at(index)))
        if pool:
            ahead = index + args.workers
            if ahead < len(images):
                pending.append(pool.submit(load, paths[ahead], size, arrival_at(ahead)))
        starts[index] = started
        if args.overlap:
            if args.wait == "spin":
                while not slot_sem.acquire(blocking=False):
                    time.sleep(0)  # poll without a blocking futex wait
            else:
                slot_sem.acquire()
            slot = index % slots  # free because at most `slots` outstanding
            fill(slot, resized, top, left)
            submitted = time.perf_counter()
            infer_pool.submit(infer_cb, index, slot, ratio, left, top, submitted)
        else:
            fill(0, resized, top, left)
            submitted = time.perf_counter()
            output = session.run(None, {in_name: inputs[0]})[0]
            if index >= args.warmup:
                infer_wall.append((time.perf_counter() - submitted) * 1000)
            finish_rows(index, output, ratio, left, top)

    if args.overlap:
        for _ in range(slots):
            slot_sem.acquire()
    if pool:
        pool.shutdown()
    if infer_pool:
        infer_pool.shutdown()

    if begin is None:
        raise RuntimeError("No timed images")
    elapsed = time.perf_counter() - begin
    cpu_seconds = time.process_time() - begin_cpu
    timed = len(images) - args.warmup

    if args.predictions:
        args.predictions.write_text(json.dumps(predictions))

    latency_sorted = sorted(latencies)
    report = {
        "size": size,
        "prefetch": args.prefetch,
        "overlap": args.overlap,
        "intra": args.intra,
        "fps": args.fps,
        "images": timed,
        "window_s": elapsed,
        "window_unix": [begin_unix, begin_unix + elapsed],
        "images_per_s": timed / elapsed,
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": latency_sorted[max(0, int(0.95 * len(latency_sorted)) - 1)],
        "infer_median_ms": statistics.median(infer_wall) if infer_wall else None,
        "cpu_ms_per_image": cpu_seconds / timed * 1000,
        "predictions": len(predictions),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("size", "prefetch", "overlap", "fps",
                                             "latency_median_ms", "latency_p95_ms",
                                             "infer_median_ms", "images_per_s",
                                             "cpu_ms_per_image", "predictions")}))
    sys.stdout.flush()
    os._exit(0)  # avoid slow interpreter teardown with big threads alive


if __name__ == "__main__":
    main()
