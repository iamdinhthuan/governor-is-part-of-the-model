"""Optimized NMS-free YOLO26 Jetson application on a uint8-input TensorRT engine.

The host writes the letterboxed uint8 BGR image straight into a pinned (or
mapped zero-copy) buffer; channel flip, transpose and /255 run inside the
engine. Options:
  --prefetch  decode/letterbox of upcoming images overlaps GPU work.
  --overlap   double-buffered I/O: COCO-row construction for image i-1 runs
              while image i is on the GPU.
  --fps F     images "arrive" at a fixed rate (camera-like); latency is
              measured from arrival to finished COCO rows. With F=0 they are
              available immediately and latency starts at decode start.
Writes COCO predictions with the same row construction for COCOeval.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import tensorrt as trt

from jetson_breakdown import check, cuda_runtime, summary

MAPPED = 2  # cudaHostAllocMapped
SCHED = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}  # cudaDeviceSchedule*


def wait_until(deadline: float) -> None:
    delay = deadline - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


def load(path: Path, size: int, arrival: float):
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
    return started, resized, ratio, round(dh - 0.1), round(dw - 0.1)


def rows_for(output, image, ratio, left, top, category_ids, predictions) -> None:
    rows = output[0][output[0, :, 4] >= 0.001]
    classes = np.rint(rows[:, 5]).astype(np.int32)
    if np.any((classes < 0) | (classes >= 80)):
        raise RuntimeError("Invalid class index")
    boxes = rows[:, :4].astype(np.float64)
    # Subtract the integer pixel offset actually used for placement, not the
    # fractional padding: the latter shifts boxes by 0.5 px when padding is odd.
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
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--memory", choices=("pinned", "mapped"), default="pinned")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-gc", action="store_true",
                        help="disable Python's cyclic GC during the timed loop")
    parser.add_argument("--adaptive", action="store_true",
                        help="with --overlap: finish in-flight work when the next frame is not ready")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--stream-sync", action="store_true")
    parser.add_argument("--sched", choices=("default", *SCHED), default="default")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.overlap and args.memory != "pinned":
        raise RuntimeError("--overlap uses pinned double buffers")
    if args.adaptive and not (args.overlap and args.prefetch):
        raise RuntimeError("--adaptive needs --overlap and --prefetch")

    data = json.loads(args.annotations.read_text())
    images = sorted(data["images"], key=lambda item: item["file_name"])
    if args.limit:
        images = images[: args.limit]
    category_ids = sorted(item["id"] for item in data["categories"])
    cuda = cuda_runtime()
    if args.sched != "default":
        # Must precede primary-context creation (engine deserialization).
        check(cuda.cudaSetDeviceFlags(ctypes.c_uint(SCHED[args.sched])), "cudaSetDeviceFlags")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    shape = tuple(engine.get_tensor_shape("images"))
    float_input = engine.get_tensor_dtype("images") == trt.float32
    if float_input:
        valid = len(shape) == 4 and shape[:2] == (1, 3) and shape[2] == shape[3]
        size = shape[2]
    else:
        valid = (
            len(shape) == 4 and shape[0] == 1 and shape[3] == 3 and shape[1] == shape[2]
            and engine.get_tensor_dtype("images") == trt.uint8
        )
        size = shape[1]
    if not valid or tuple(engine.get_tensor_shape("output0")) != (1, 300, 6):
        raise RuntimeError(
            "Expected uint8 (1,S,S,3) or float32 (1,3,S,S) input and (1,300,6) output"
        )
    context = engine.create_execution_context()
    canvas = np.empty((size, size, 3), dtype=np.uint8)

    flags_now = ctypes.c_uint()
    check(cuda.cudaGetDeviceFlags(ctypes.byref(flags_now)), "cudaGetDeviceFlags")
    cuda.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint,
    ]
    cuda.cudaHostGetDevicePointer.restype = ctypes.c_int
    nbytes, obytes = size * size * 3 * (4 if float_input else 1), 300 * 6 * 4
    slots = 2 if args.overlap else 1
    flags = MAPPED if args.memory == "mapped" else 0
    stream = ctypes.c_void_p()
    check(cuda.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    h_in, h_out, d_in, d_out, events, hosts, outputs = [], [], [], [], [], [], []
    for _ in range(slots):
        hi, ho, di, do, ev = (ctypes.c_void_p() for _ in range(5))
        check(cuda.cudaHostAlloc(ctypes.byref(hi), nbytes, flags), "cudaHostAlloc")
        check(cuda.cudaHostAlloc(ctypes.byref(ho), obytes, 0), "cudaHostAlloc")
        check(cuda.cudaMalloc(ctypes.byref(do), obytes), "cudaMalloc")
        if args.memory == "mapped":
            check(cuda.cudaHostGetDevicePointer(ctypes.byref(di), hi, 0), "devptr")
        else:
            check(cuda.cudaMalloc(ctypes.byref(di), nbytes), "cudaMalloc")
        check(cuda.cudaEventCreate(ctypes.byref(ev)), "cudaEventCreate")
        h_in.append(hi); h_out.append(ho); d_in.append(di); d_out.append(do)
        events.append(ev)
        hosts.append(
            np.ctypeslib.as_array(
                ctypes.cast(hi, ctypes.POINTER(ctypes.c_float)), (1, 3, size, size))
            if float_input else
            np.ctypeslib.as_array(
                ctypes.cast(hi, ctypes.POINTER(ctypes.c_uint8)), (size, size, 3))
        )
        outputs.append(np.ctypeslib.as_array(
            ctypes.cast(ho, ctypes.POINTER(ctypes.c_float)), (1, 300, 6)))

    paths = [args.images_dir / image["file_name"] for image in images]
    t0 = time.perf_counter() + 0.5
    arrivals = [
        t0 + index / args.fps if args.fps > 0 else 0.0 for index in range(len(images))
    ]
    pool = ThreadPoolExecutor(max_workers=args.workers) if args.prefetch else None
    pending = (
        [pool.submit(load, paths[i], size, arrivals[i])
         for i in range(min(args.workers, len(images)))]
        if pool else None
    )

    predictions, latency, gpu_wall = [], [], []
    starts = {}
    inflight = None
    begin = begin_unix = None

    def finish(item) -> None:
        index, slot, ratio, left, top, enqueued = item
        if args.stream_sync:
            check(cuda.cudaStreamSynchronize(stream), "stream sync")
        else:
            check(cuda.cudaEventSynchronize(events[slot]), "event sync")
        if index >= args.warmup:
            gpu_wall.append((time.perf_counter() - enqueued) * 1000)
        rows_for(outputs[slot], images[index], ratio, left, top, category_ids, predictions)
        if index >= args.warmup:
            origin = arrivals[index] if args.fps > 0 else starts[index]
            latency.append((time.perf_counter() - origin) * 1000)

    for index, image in enumerate(images):
        if index == args.warmup:
            if args.no_gc:
                gc.collect()
                gc.disable()
            begin, begin_unix = time.perf_counter(), time.time()
            begin_cpu = time.process_time()
        if pool:
            if args.adaptive and inflight is not None and not pending[0].done():
                # Next frame not ready: finish the in-flight one now instead of
                # holding its result for a whole inter-arrival period.
                finish(inflight)
                inflight = None
            started, resized, ratio, top, left = pending.pop(0).result()
            ahead = index + args.workers
            if ahead < len(images):
                pending.append(pool.submit(load, paths[ahead], size, arrivals[ahead]))
        else:
            started, resized, ratio, top, left = load(paths[index], size, arrivals[index])
        starts[index] = started
        slot = index % slots
        host = hosts[slot]
        target = canvas if float_input else host
        target[...] = 114
        target[top: top + resized.shape[0], left: left + resized.shape[1]] = resized
        if float_input:
            # Same op order as jetson_eval.letterbox, so the float tensor is bit-identical.
            host[0] = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        enqueued = time.perf_counter()
        if args.memory == "pinned":
            check(cuda.cudaMemcpyAsync(d_in[slot], h_in[slot], nbytes, 1, stream), "H2D")
        context.set_tensor_address("images", d_in[slot].value)
        context.set_tensor_address("output0", d_out[slot].value)
        if not context.execute_async_v3(stream.value):
            raise RuntimeError("TensorRT execution failed")
        check(cuda.cudaMemcpyAsync(h_out[slot], d_out[slot], obytes, 2, stream), "D2H")
        check(cuda.cudaEventRecord(events[slot], stream), "event record")
        current = (index, slot, ratio, left, top, enqueued)
        if args.overlap:
            if inflight is not None:
                finish(inflight)
            inflight = current
        else:
            finish(current)
    if inflight is not None:
        finish(inflight)
    elapsed = time.perf_counter() - begin
    cpu_seconds = time.process_time() - begin_cpu
    end_unix = time.time()
    if pool:
        pool.shutdown()

    if args.predictions:
        args.predictions.write_text(json.dumps(predictions))
    report = {
        "engine": str(args.engine),
        "size": size,
        "input": "float32 host-preprocessed" if float_input else "uint8 in-graph preprocessing",
        "memory": args.memory,
        "prefetch": args.prefetch,
        "overlap": args.overlap,
        "adaptive_overlap": args.adaptive,
        "gc_disabled": args.no_gc,
        "decode_workers": args.workers if args.prefetch else 0,
        "arrival_fps": args.fps,
        "images": len(images),
        "images_timed": len(latency),
        "prediction_count": len(predictions),
        "latency_ms": summary(latency),
        "latency_p99_ms": float(np.percentile(latency, 99)),
        "gpu_enqueue_to_done_ms": summary(gpu_wall),
        "throughput_images_per_s": len(latency) / elapsed,
        "process_cpu_ms_per_image": 1000 * cpu_seconds / len(latency),
        "sync": "stream" if args.stream_sync else "event",
        "sched_request": args.sched,
        "cuda_device_flags": flags_now.value,
        "window_unix": [begin_unix, end_unix],
        "latency_definition": (
            "arrival to finished COCO rows" if args.fps > 0
            else "start of this image's JPEG decode to finished COCO rows"
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "size": size, "prefetch": args.prefetch, "overlap": args.overlap,
        "fps": args.fps,
        "latency_median_ms": round(report["latency_ms"]["median"], 3),
        "latency_p95_ms": round(report["latency_ms"]["p95"], 3),
        "images_per_s": round(report["throughput_images_per_s"], 2),
        "cpu_ms_per_image": round(report["process_cpu_ms_per_image"], 2),
        "predictions": len(predictions),
    }))


if __name__ == "__main__":
    main()
