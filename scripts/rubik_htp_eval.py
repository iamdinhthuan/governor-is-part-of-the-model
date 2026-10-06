#!/usr/bin/env python3
"""Rubik Pi 3 (QCM6490) YOLO26n HTP/NPU pipeline mirroring rubik_ort_eval.py.

Runs the user's QNN context binary (built with QAIRT 2.49.40, standard
(1,84,8400) head, INT8) on the HTP NPU via the qai-appbuilder 2.50.40
bundled runtime (its libQnnHtp.so + V68 skel; ADSP_LIBRARY_PATH must point
at the wheel's libs dir). Host NMS (conf 0.001, IoU 0.7, max 300 dets) with
cv2.dnn.NMSBoxes replaces the e2e head -- same settings as Ultralytics val.

The qai-appbuilder/QNN stack is NOT safe under concurrent Inference calls
(~0.3% of calls wedge the DSP session irreversibly), so this pipeline keeps
exactly one QNNContext and issues all native calls from one dedicated
inference thread -- the pattern that ran the full 5,000-image accuracy pass
without a single failure. Overlap comes from pipelining host stages around
the serialized NPU calls:

  decode+letterbox (worker pool) -> input fill (main) -> NPU (serial thread)
      -> host NMS + COCO rows (post pool, overlaps the next NPU call)

Options mirror the other platforms:
  --prefetch / --workers K  decode+letterbox of upcoming images overlaps inference.
  --overlap  / --inflight N depth of the input-buffer queue ahead of the NPU
             (N buffers let the NPU stay busy while the host fills the next).
  --wait sleep|spin  how the main thread waits for a free input buffer.
  --fps F  camera-like arrivals; latency from arrival to finished rows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import queue
import statistics
import sys
import threading
import time

import cv2
import numpy as np

SCORE_MIN = 0.001
NMS_IOU = 0.7
MAX_DET = 300
TOPK_PRE_NMS = 30000  # Ultralytics val behaviour


def wait_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def load(path: Path, size: int, arrival: float):
    """Decode and resize; geometry identical to the other pipelines."""
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


_3out_idx: dict = {}  # lazily resolved output-tensor roles


def rows_for_3out(outputs, image, ratio, left, top, category_ids, predictions) -> None:
    """Split e2e head: three output tensors boxes(1,300,4) scores(1,300,1)
    classes(1,300,1) -- same semantics as the (1,300,6) head but with
    per-tensor quantization scales (the packed (1,300,6) tensor would force
    scores onto the boxes' scale and zero them).

    The context does NOT emit outputs in the ONNX graph-output order (it
    follows the converter's tensor creation order), so roles are resolved
    once at first use: boxes is the (1,300,4) tensor; of the two (1,300,1)
    tensors the integral-valued one (Mod->Cast of indices) is classes and
    the other is scores.
    """
    if not _3out_idx:
        boxes_i = next(i for i, o in enumerate(outputs) if np.asarray(o).shape[-1] == 4)
        rest = [i for i in range(len(outputs)) if i != boxes_i]
        a, b = rest[0], rest[1]
        va = np.asarray(outputs[a])[0, :, 0]
        classes_i, scores_i = (a, b) if np.all(va == np.round(va)) else (b, a)
        _3out_idx.update(boxes=boxes_i, scores=scores_i, classes=classes_i)
        print(f"3out mapping: boxes=out{boxes_i} scores=out{scores_i} "
              f"classes=out{classes_i}", flush=True)
    boxes_raw = np.asarray(outputs[_3out_idx["boxes"]])[0]
    scores = np.asarray(outputs[_3out_idx["scores"]])[0, :, 0]
    cls_f = np.asarray(outputs[_3out_idx["classes"]])[0, :, 0]
    keep = scores >= SCORE_MIN
    if not np.any(keep):
        return
    rows_for_xyxy(boxes_raw[keep], scores[keep], cls_f[keep],
                  image, ratio, left, top, category_ids, predictions)


def rows_for_xyxy(rows4, scores, cls_f, image, ratio, left, top,
                  category_ids, predictions) -> None:
    classes = np.rint(cls_f).astype(np.int32)
    if np.any((classes < 0) | (classes >= 80)):
        raise RuntimeError("Invalid class index")
    boxes = rows4.astype(np.float64)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / ratio, 0, image["width"])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / ratio, 0, image["height"])
    for (x1, y1, x2, y2), score, cls_index in zip(boxes, scores, classes):
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append({
            "image_id": image["id"],
            "category_id": category_ids[int(cls_index)],
            "bbox": [round(float(x1), 3), round(float(y1), 3),
                     round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
            "score": round(float(score), 6),
        })


def rows_for_e2e(outputs, image, ratio, left, top, category_ids, predictions) -> None:
    """End-to-end (1,300,6) head: xyxy rows, no NMS (baked into the graph).

    Identical conversion to rubik_ort_eval.py / jetson_fast_eval.py.
    """
    output = outputs[0]
    keep = output[0, :, 4] >= SCORE_MIN
    if not np.any(keep):
        return
    rows = output[0][keep]
    rows_for_xyxy(rows[:, :4], rows[:, 4], rows[:, 5],
                  image, ratio, left, top, category_ids, predictions)


def rows_for_nms(outputs, image, ratio, left, top, category_ids, predictions) -> None:
    """Standard (1,84,8400) head: cx,cy,w,h in 640-space + 80 class scores.

    Host NMS mirrors Ultralytics val: conf >= 0.001, top-30000, IoU 0.7,
    max 300 detections; then conversion to original-image coordinates
    identical to the other pipelines.
    """
    out = np.asarray(outputs[0]).reshape(84, -1)
    cls_scores = out[4:]
    scores = cls_scores.max(axis=0)
    classes = cls_scores.argmax(axis=0)
    keep = scores >= SCORE_MIN
    if not np.any(keep):
        return
    boxes = out[:4, keep].T.astype(np.float64)  # cx,cy,w,h in 640 space
    scores = scores[keep]
    classes = classes[keep]
    if len(scores) > TOPK_PRE_NMS:
        idx = np.argpartition(scores, -TOPK_PRE_NMS)[-TOPK_PRE_NMS:]
        boxes, scores, classes = boxes[idx], scores[idx], classes[idx]
    xywh = boxes.copy()
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    picked = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), SCORE_MIN, NMS_IOU)
    if len(picked) == 0:
        return
    picked = np.asarray(picked).reshape(-1)
    if len(picked) > MAX_DET:
        picked = picked[np.argsort(scores[picked])[::-1][:MAX_DET]]
    for i in picked:
        cx, cy, w, h = boxes[i]
        x1 = min(max((cx - w / 2 - left) / ratio, 0), image["width"])
        y1 = min(max((cy - h / 2 - top) / ratio, 0), image["height"])
        x2 = min(max((cx + w / 2 - left) / ratio, 0), image["width"])
        y2 = min(max((cy + h / 2 - top) / ratio, 0), image["height"])
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append({
            "image_id": image["id"],
            "category_id": category_ids[int(classes[i])],
            "bbox": [round(float(x1), 3), round(float(y1), 3),
                     round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
            "score": round(float(scores[i]), 6),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True,
                        help="QNN context .bin extracted from the EPContext ONNX")
    parser.add_argument("--libs", type=Path, required=True,
                        help="dir with libQnnHtp.so/libQnnSystem.so (appbuilder wheel)")
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

    os.environ.setdefault("ADSP_LIBRARY_PATH", str(args.libs))
    from qai_appbuilder.qnncontext import QNNContext

    data = json.loads(args.annotations.read_text())
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if args.limit:
        images = images[: args.limit]
    category_ids = sorted(item["id"] for item in data["categories"])
    paths = [args.images_dir / item["file_name"] for item in images]
    arrivals = np.arange(len(paths)) / args.fps if args.fps > 0 else np.zeros(len(paths))

    size = 640
    slots = args.inflight if args.overlap else 1
    context = QNNContext("yolo26n_htp", str(args.context),
                         str(args.libs / "libQnnHtp.so"),
                         str(args.libs / "libQnnSystem.so"))
    in_shape = [int(d) for d in context.getInputShapes()[0]]
    out_shapes = [[int(d) for d in s] for s in context.getOutputShapes()]
    nhwc = in_shape[1] == size  # (1,640,640,3) vs (1,3,640,640)
    if len(out_shapes) == 3:
        mode = "e2e-split(3out)"  # boxes/scores/classes: per-tensor scales
        rows_for = rows_for_3out
    elif len(out_shapes) == 1 and out_shapes[0][1:] == [300, 6]:
        mode = "e2e"  # NMS-free head vs standard (1,84,8400)
        rows_for = rows_for_e2e
    else:
        mode = "standard+hostNMS"
        rows_for = rows_for_nms
    print(f"context in={in_shape} ({'NHWC' if nhwc else 'NCHW'}) "
          f"out={out_shapes} ({mode})", flush=True)
    canvases = [np.full((size, size, 3), 114, dtype=np.uint8) for _ in range(slots)]
    inputs = [np.empty(in_shape, dtype=np.float32) for _ in range(slots)]

    def fill(slot: int, resized: np.ndarray, top: int, left: int) -> None:
        canvas = canvases[slot]
        canvas[:] = 114
        h, w = resized.shape[:2]
        canvas[top:top + h, left:left + w] = resized
        if nhwc:  # BGR->RGB + /255, NHWC float32 (no transpose needed)
            np.divide(canvas[:, :, ::-1], 255.0, out=inputs[slot][0])
        else:  # NCHW: strided write into (3,640,640)
            np.divide(canvas[:, :, ::-1].transpose(2, 0, 1), 255.0,
                      out=inputs[slot][0])

    pool = ThreadPoolExecutor(max_workers=args.workers) if args.prefetch else None
    epoch = time.perf_counter()  # arrival times are relative to this instant

    def arrival_at(i: int) -> float:
        return epoch + arrivals[i] if args.fps > 0 else 0.0

    pending = [pool.submit(load, paths[i], size, arrival_at(i))
               for i in range(min(args.workers, len(paths)))] if pool else []

    pred_rows: list[list] = [[] for _ in images]  # per-image rows: order-stable
    starts = [0.0] * len(images)
    latencies: list[float] = []
    infer_wall: list[float] = []
    worker_errors: list = []
    slot_sem = threading.Semaphore(slots)

    begin = begin_unix = begin_cpu = None

    def finish_rows(index: int, output, ratio, left, top) -> None:
        rows_for(output, images[index], ratio, left, top, category_ids,
                 pred_rows[index])
        if index >= args.warmup:
            latencies.append((time.perf_counter() - starts[index]) * 1000)

    def finish_cb(index: int, output, ratio, left, top) -> None:
        try:
            finish_rows(index, output, ratio, left, top)
        except BaseException as exc:
            worker_errors.append((index, repr(exc)))
            import traceback
            traceback.print_exc()

    dsp_retries = 0

    def infer_loop(infer_q: "queue.Queue", post_pool: ThreadPoolExecutor) -> None:
        """The ONLY thread that touches the QNN context (native calls serialized)."""
        nonlocal dsp_retries
        while True:
            item = infer_q.get()
            if item is None:
                return
            index, slot, ratio, left, top, submitted = item
            try:
                outputs = context.Inference([inputs[slot]])
                for attempt in range(4):  # rare empty-output hiccup: backoff retry
                    if outputs:
                        break
                    dsp_retries += 1
                    time.sleep(0.05 * (attempt + 1))
                    outputs = context.Inference([inputs[slot]])
                if not outputs:
                    worker_errors.append((index, "empty DSP output after retries"))
                    continue
                if index >= args.warmup:
                    infer_wall.append((time.perf_counter() - submitted) * 1000)
                post_pool.submit(finish_cb, index, outputs, ratio, left, top)
            except BaseException as exc:
                worker_errors.append((index, repr(exc)))
                import traceback
                traceback.print_exc()
            finally:
                slot_sem.release()  # input buffer consumed

    infer_q = None
    infer_thread = None
    post_pool = None
    if args.overlap:
        infer_q = queue.Queue()
        post_pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
        infer_thread = threading.Thread(target=infer_loop,
                                        args=(infer_q, post_pool), daemon=True)
        infer_thread.start()

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
            slot = index % slots
            fill(slot, resized, top, left)
            infer_q.put((index, slot, ratio, left, top, time.perf_counter()))
        else:
            fill(0, resized, top, left)
            submitted = time.perf_counter()
            outputs = context.Inference([inputs[0]])
            if index >= args.warmup:
                infer_wall.append((time.perf_counter() - submitted) * 1000)
            finish_rows(index, outputs, ratio, left, top)

    if args.overlap:
        for _ in range(slots):
            slot_sem.acquire()  # all submitted inferences consumed by the NPU thread
        infer_q.put(None)
        infer_thread.join()
        post_pool.shutdown()  # waits for the last NMS tasks
    if dsp_retries:
        print(f"dsp_retries={dsp_retries}", flush=True)
    if worker_errors:
        print(f"WORKER_ERRORS {worker_errors[:3]}", flush=True)
        sys.exit(2)
    if pool:
        pool.shutdown()

    if begin is None:
        raise RuntimeError("No timed images")
    elapsed = time.perf_counter() - begin
    cpu_seconds = time.process_time() - begin_cpu
    timed = len(images) - args.warmup

    predictions = [row for rows in pred_rows for row in rows]
    if args.predictions:
        args.predictions.write_text(json.dumps(predictions))

    latency_sorted = sorted(latencies)
    report = {
        "size": size,
        "model": f"yolo26n QNN context {args.context.name}, {mode}, INT8",
        "runtime": "qai-appbuilder 2.50.40 bundled libQnnHtp (V68)",
        "postprocess": ("none (NMS-free head, score>=0.001)" if mode.startswith("e2e") else
                        "host NMS conf>=0.001 IoU 0.7 max 300 (cv2.dnn.NMSBoxes)"),
        "prefetch": args.prefetch,
        "overlap": args.overlap,
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
    os._exit(0)  # skip AppBuilder/DSP teardown, which is not hang-safe


if __name__ == "__main__":
    main()
