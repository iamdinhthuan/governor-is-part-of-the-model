"""YOLOv8n application on a Raspberry Pi 5 + Hailo-8 (paper generality device).

Mirrors jetson_fast_eval.py: OpenCV decode, letterbox to 640x640 (value 114),
uint8 NHWC RGB into the HEF (normalization and NMS run on the chip), COCO rows.
Options:
  --prefetch / --workers K  decode+letterbox of upcoming images overlaps NPU work.
  --overlap  / --inflight N asynchronous inference, N frames in flight; the
             completion callback builds the COCO rows of a finished frame while
             the NPU works on the next one.
  --wait sleep|spin  how the main thread waits for a free async slot.
  --fps F  camera-like arrivals; latency from arrival to finished rows.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import sys
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from hailo_platform import VDevice

SCORE_MIN = 0.001


def wait_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def load(path: Path, size: int, arrival: float):
    """Decode and resize; geometry identical to the Jetson pipeline."""
    wait_until(arrival)
    started = time.perf_counter()
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Unreadable {path}")
    height, width = bgr.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    dw, dh = (size - new_width) / 2, (size - new_height) / 2
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # Hailo models are trained on RGB; the flip happens on the host here.
    return started, np.ascontiguousarray(resized[:, :, ::-1]), ratio, round(dh - 0.1), round(dw - 0.1)


def rows_for(out, image, ratio, left, top, size, category_ids, predictions) -> None:
    """out: flat packed NMS-by-class buffer (verified empirically against the
    InferVStreams structured API): 80 variable-length segments laid out
    contiguously, each = count float followed by count rows of
    [y1, x1, y2, x2, score] with normalized coordinates. Max size is
    80 + 80*100*5 = 40080 floats (the HEF's declared "maximum frame size")."""
    offset = 0
    for cls_index in range(80):
        if offset >= out.size:
            break
        n = min(int(out[offset]), 100)
        offset += 1
        dets = out[offset:offset + n * 5].reshape(n, 5)[:n]
        offset += n * 5
        dets = dets[dets[:, 4] >= SCORE_MIN]
        if len(dets) == 0:
            continue
        boxes = dets[:, :4].astype(np.float64) * size
        x1 = np.clip((boxes[:, 1] - left) / ratio, 0, image["width"])
        y1 = np.clip((boxes[:, 0] - top) / ratio, 0, image["height"])
        x2 = np.clip((boxes[:, 3] - left) / ratio, 0, image["width"])
        y2 = np.clip((boxes[:, 2] - top) / ratio, 0, image["height"])
        for j in range(len(dets)):
            if x2[j] <= x1[j] or y2[j] <= y1[j]:
                continue
            predictions.append({
                "image_id": image["id"],
                "category_id": category_ids[cls_index],
                "bbox": [
                    round(float(x1[j]), 3), round(float(y1[j]), 3),
                    round(float(x2[j] - x1[j]), 3), round(float(y2[j] - y1[j]), 3),
                ],
                "score": round(float(dets[j, 4]), 6),
            })


def summary(values):
    arr = np.array(values, dtype=np.float64)
    return {"median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean())} if len(arr) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--inflight", type=int, default=2)
    parser.add_argument("--wait", choices=("sleep", "spin"), default="sleep")
    parser.add_argument("--no-gc", action="store_true")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.overlap and not args.prefetch:
        raise RuntimeError("--overlap needs --prefetch (decode must not block the submitter)")

    # Watchdog: HailoRT 4.23 occasionally hangs (device acquisition, lost async
    # completion). If no image finishes for 120 s after startup grace, exit
    # with a distinctive code so the driver can retry the run.
    progress = [time.perf_counter()]

    def watchdog():
        time.sleep(180)  # startup grace: device, annotations, warm-up
        while True:
            if time.perf_counter() - progress[0] > 120:
                print("watchdog: no completion for 120 s, exiting",
                      file=sys.stderr, flush=True)
                os._exit(3)
            time.sleep(5)

    threading.Thread(target=watchdog, daemon=True).start()

    data = json.loads(args.annotations.read_text())
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if args.limit:
        images = images[: args.limit]
    category_ids = sorted(item["id"] for item in data["categories"])

    device = VDevice()
    model = device.create_infer_model(str(args.hef))
    model.set_batch_size(1)
    in_info = model.inputs[0]
    out_info = model.outputs[0]
    if tuple(in_info.shape) not in ((640, 640, 3), (1, 640, 640, 3)):
        raise RuntimeError(f"Unexpected input shape {in_info.shape}")
    size = in_info.shape[-3] if len(in_info.shape) == 3 else in_info.shape[1]
    out_shape = tuple(out_info.shape)
    out_squeezed = out_shape[1:] if len(out_shape) == 4 and out_shape[0] == 1 else out_shape
    if not (len(out_squeezed) == 1 and out_squeezed[0] % 80 == 0
            and (out_squeezed[0] // 80 - 1) % 5 == 0):
        raise RuntimeError(f"Expected flat NMS-by-class output, got {out_shape}")
    configured = model.configure()
    slots = args.inflight if args.overlap else 1
    canvases, out_bufs, bindings = [], [], []
    for _ in range(slots):
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        out = np.zeros(out_squeezed, dtype=np.float32)
        b = configured.create_bindings()
        b.input(in_info.name).set_buffer(canvas)
        b.output(out_info.name).set_buffer(out)
        canvases.append(canvas)
        out_bufs.append(out)
        bindings.append(b)

    paths = [args.images_dir / image["file_name"] for image in images]
    t0 = time.perf_counter() + 0.5
    arrivals = [t0 + i / args.fps if args.fps > 0 else 0.0 for i in range(len(images))]
    pool = ThreadPoolExecutor(max_workers=args.workers) if args.prefetch else None
    pending = (
        [pool.submit(load, paths[i], size, arrivals[i])
         for i in range(min(args.workers, len(images)))] if pool else None)

    predictions, latency, infer_wall = [], [], []
    starts = {}
    begin = begin_unix = begin_cpu = None
    count_lock = threading.Lock()

    def fill(slot, resized, top, left):
        canvas = canvases[slot]
        canvas[...] = 114
        canvas[top: top + resized.shape[0], left: left + resized.shape[1]] = resized

    def finish_rows(index, slot, ratio, left, top):
        rows_for(out_bufs[slot], images[index], ratio, left, top, size,
                 category_ids, predictions)
        if index >= args.warmup:
            origin = arrivals[index] if args.fps > 0 else starts[index]
            latency.append((time.perf_counter() - origin) * 1000)

    configured.activate()
    progress[0] = time.perf_counter()
    try:
        if args.overlap:
            free = list(range(slots))
            errors = []

            def callback(info, index=None, slot=None, ratio=None, left=None, top=None,
                         submitted=None):
                if info.exception is not None:
                    errors.append(info.exception)
                    with count_lock:
                        free.append(slot)
                    return
                if index >= args.warmup:
                    infer_wall.append((time.perf_counter() - submitted) * 1000)
                finish_rows(index, slot, ratio, left, top)
                progress[0] = time.perf_counter()
                with count_lock:
                    free.append(slot)

            for index in range(len(images)):
                if index == args.warmup:
                    if args.no_gc:
                        gc.collect(); gc.disable()
                    begin, begin_unix, begin_cpu = time.perf_counter(), time.time(), time.process_time()
                started, resized, ratio, top, left = (pending.pop(0).result() if pool
                                                      else load(paths[index], size, arrivals[index]))
                if pool:
                    ahead = index + args.workers
                    if ahead < len(images):
                        pending.append(pool.submit(load, paths[ahead], size, arrivals[ahead]))
                starts[index] = started
                while not free:
                    if args.wait == "spin":
                        time.sleep(0)  # poll without a blocking wait
                    else:
                        configured.wait_for_async_ready(1000)
                slot = free.pop(0)
                fill(slot, resized, top, left)
                submitted = time.perf_counter()
                configured.run_async(
                    [bindings[slot]],
                    lambda completion_info, i=index, s=slot, r=ratio, l=left, t=top,
                           sub=submitted:
                        callback(completion_info, i, s, r, l, t, sub))
            while len(free) < slots:
                if args.wait == "spin":
                    time.sleep(0)
                else:
                    configured.wait_for_async_ready(1000)
            if errors:
                raise RuntimeError(f"async inference failed: {errors[0]}")
        else:
            for index in range(len(images)):
                if index == args.warmup:
                    if args.no_gc:
                        gc.collect(); gc.disable()
                    begin, begin_unix, begin_cpu = time.perf_counter(), time.time(), time.process_time()
                started, resized, ratio, top, left = (pending.pop(0).result() if pool
                                                      else load(paths[index], size, arrivals[index]))
                if pool:
                    ahead = index + args.workers
                    if ahead < len(images):
                        pending.append(pool.submit(load, paths[ahead], size, arrivals[ahead]))
                starts[index] = started
                fill(0, resized, top, left)
                submitted = time.perf_counter()
                job = configured.run_async([bindings[0]], lambda completion_info: None)
                job.wait(1000)
                if index >= args.warmup:
                    infer_wall.append((time.perf_counter() - submitted) * 1000)
                finish_rows(index, 0, ratio, left, top)
                progress[0] = time.perf_counter()
    finally:
        configured.deactivate()
        configured.shutdown()
        device.release()
    elapsed = time.perf_counter() - begin
    cpu_seconds = time.process_time() - begin_cpu
    end_unix = time.time()
    if pool:
        pool.shutdown()

    if args.predictions:
        args.predictions.write_text(json.dumps(predictions))
    report = {
        "hef": str(args.hef),
        "size": size,
        "input": "uint8 NHWC RGB (in-HEF normalization), NMS on-chip",
        "prefetch": args.prefetch,
        "overlap": args.overlap,
        "inflight": args.inflight if args.overlap else 1,
        "wait": args.wait,
        "gc_disabled": args.no_gc,
        "decode_workers": args.workers if args.prefetch else 0,
        "arrival_fps": args.fps,
        "images": len(images),
        "images_timed": len(latency),
        "prediction_count": len(predictions),
        "latency_ms": summary(latency),
        "latency_p99_ms": float(np.percentile(latency, 99)) if latency else None,
        "npu_submit_to_done_ms": summary(infer_wall),
        "throughput_images_per_s": len(latency) / elapsed,
        "process_cpu_ms_per_image": 1000 * cpu_seconds / len(latency),
        "window_unix": [begin_unix, end_unix],
        "latency_definition": (
            "arrival to finished COCO rows" if args.fps > 0
            else "start of this image's JPEG decode to finished COCO rows"),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "size": size, "prefetch": args.prefetch, "overlap": args.overlap,
        "fps": args.fps,
        "latency_median_ms": round(report["latency_ms"]["median"], 3),
        "latency_p95_ms": round(report["latency_ms"]["p95"], 3),
        "npu_median_ms": round(report["npu_submit_to_done_ms"]["median"], 3),
        "images_per_s": round(report["throughput_images_per_s"], 2),
        "cpu_ms_per_image": round(report["process_cpu_ms_per_image"], 2),
        "predictions": len(predictions),
    }))
    # HailoRT 4.23 segfaults in C++ destructors at interpreter shutdown even
    # after a clean deactivate/shutdown/release; skip them deliberately.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
