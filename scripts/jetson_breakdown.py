"""Per-stage latency attribution for the static NMS-free YOLO26 Jetson pipeline.

CPU stages use perf_counter; transfers and TensorRT compute use CUDA events on
one stream, so GPU work is not hidden by Python dispatch. The pipeline is the
same one used for the COCOeval runs (cv2 decode, letterbox, float32 NCHW
input, one-to-one output decode without NMS).
"""

import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
import tensorrt as trt

from jetson_eval import letterbox


def cuda_runtime():
    cuda = ctypes.CDLL(ctypes.util.find_library("cudart"))
    p, s, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    signatures = {
        "cudaMalloc": [ctypes.POINTER(p), s],
        "cudaHostAlloc": [ctypes.POINTER(p), s, ctypes.c_uint],
        "cudaFree": [p],
        "cudaFreeHost": [p],
        "cudaMemcpyAsync": [p, p, s, i, p],
        "cudaStreamCreate": [ctypes.POINTER(p)],
        "cudaStreamSynchronize": [p],
        "cudaEventCreate": [ctypes.POINTER(p)],
        "cudaEventRecord": [p, p],
        "cudaEventSynchronize": [p],
        "cudaEventElapsedTime": [ctypes.POINTER(ctypes.c_float), p, p],
    }
    for name, args in signatures.items():
        getattr(cuda, name).argtypes = args
        getattr(cuda, name).restype = ctypes.c_int
    return cuda


def check(code: int, label: str) -> None:
    if code:
        raise RuntimeError(f"{label} failed with CUDA status {code}")


def summary(values: list[float]) -> dict:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(
        json.loads(args.annotations.read_text())["images"],
        key=lambda item: item["file_name"],
    )[: args.limit]
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if tuple(engine.get_tensor_shape("images")) != (1, 3, args.size, args.size):
        raise RuntimeError("Engine input shape does not match --size")
    context = engine.create_execution_context()

    cuda = cuda_runtime()
    nbytes = 3 * args.size * args.size * 4
    d_in, d_out, stream = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    check(cuda.cudaMalloc(ctypes.byref(d_in), nbytes), "cudaMalloc")
    check(cuda.cudaMalloc(ctypes.byref(d_out), 300 * 6 * 4), "cudaMalloc")
    check(cuda.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    context.set_tensor_address("images", d_in.value)
    context.set_tensor_address("output0", d_out.value)
    events = [ctypes.c_void_p() for _ in range(4)]
    for event in events:
        check(cuda.cudaEventCreate(ctypes.byref(event)), "cudaEventCreate")
    if args.pinned:
        h_in, h_out = ctypes.c_void_p(), ctypes.c_void_p()
        check(cuda.cudaHostAlloc(ctypes.byref(h_in), nbytes, 0), "cudaHostAlloc")
        check(cuda.cudaHostAlloc(ctypes.byref(h_out), 300 * 6 * 4, 0), "cudaHostAlloc")
        pinned_in = np.ctypeslib.as_array(
            ctypes.cast(h_in, ctypes.POINTER(ctypes.c_float)), (1, 3, args.size, args.size)
        )
        output = np.ctypeslib.as_array(
            ctypes.cast(h_out, ctypes.POINTER(ctypes.c_float)), (1, 300, 6)
        )
    else:
        output = np.empty((1, 300, 6), dtype=np.float32)

    stages = {key: [] for key in (
        "imread", "letterbox_to_float", "pinned_stage_copy", "h2d", "compute",
        "d2h", "gpu_wall_including_dispatch", "output_decode", "total",
    )}
    elapsed = ctypes.c_float()

    def gpu_ms(start, end) -> float:
        check(cuda.cudaEventElapsedTime(ctypes.byref(elapsed), start, end), "elapsed")
        return float(elapsed.value)

    for index, image in enumerate(images):
        t0 = time.perf_counter()
        bgr = cv2.imread(str(args.images_dir / image["file_name"]))
        t1 = time.perf_counter()
        tensor, ratio, dw, dh = letterbox(bgr, args.size)
        t2 = time.perf_counter()
        if args.pinned:
            pinned_in[...] = tensor
            source = h_in
        else:
            source = ctypes.c_void_p(tensor.ctypes.data)
        t3 = time.perf_counter()
        cuda.cudaEventRecord(events[0], stream)
        check(cuda.cudaMemcpyAsync(d_in, source, nbytes, 1, stream), "H2D")
        cuda.cudaEventRecord(events[1], stream)
        if not context.execute_async_v3(stream.value):
            raise RuntimeError("TensorRT execution failed")
        cuda.cudaEventRecord(events[2], stream)
        destination = h_out if args.pinned else ctypes.c_void_p(output.ctypes.data)
        check(cuda.cudaMemcpyAsync(destination, d_out, output.nbytes, 2, stream), "D2H")
        cuda.cudaEventRecord(events[3], stream)
        check(cuda.cudaStreamSynchronize(stream), "sync")
        t4 = time.perf_counter()
        rows = output[0][output[0, :, 4] >= 0.001]
        boxes = rows[:, :4].astype(np.float64)
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / ratio, 0, image["width"])
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / ratio, 0, image["height"])
        t5 = time.perf_counter()
        if index < args.warmup:
            continue
        stages["imread"].append((t1 - t0) * 1000)
        stages["letterbox_to_float"].append((t2 - t1) * 1000)
        stages["pinned_stage_copy"].append((t3 - t2) * 1000)
        stages["h2d"].append(gpu_ms(events[0], events[1]))
        stages["compute"].append(gpu_ms(events[1], events[2]))
        stages["d2h"].append(gpu_ms(events[2], events[3]))
        stages["gpu_wall_including_dispatch"].append((t4 - t3) * 1000)
        stages["output_decode"].append((t5 - t4) * 1000)
        stages["total"].append((t5 - t0) * 1000)

    report = {
        "engine": str(args.engine),
        "size": args.size,
        "pinned_host_buffers": args.pinned,
        "images_timed": len(stages["total"]),
        "warmup_images": args.warmup,
        "stages_ms": {key: summary(value) for key, value in stages.items()},
        "note": "Output decode here is vectorized coordinate mapping only; "
                "COCO dict construction is excluded.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: round(v["median"], 3) for k, v in report["stages_ms"].items()}))


if __name__ == "__main__":
    main()
