"""Evaluate a static-shape YOLO26 TensorRT engine on COCO val2017.

The application performs letterbox and one-to-one output decode on the CPU,
without NMS. It writes COCO-format predictions for independent COCOeval.
All downloads and CUDA allocations are local to the Jetson; no training.
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

ROOT = Path("/tmp/yolo26-coco-val")


def cuda_runtime():
    library = ctypes.util.find_library("cudart")
    if not library:
        raise RuntimeError("CUDA runtime not found")
    cuda = ctypes.CDLL(library)
    cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    cuda.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cuda.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cuda.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    for name in (
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaStreamCreate",
        "cudaStreamSynchronize",
        "cudaStreamDestroy",
    ):
        getattr(cuda, name).restype = ctypes.c_int
    return cuda


def check(code: int, label: str) -> None:
    if code:
        raise RuntimeError(f"{label} failed with CUDA status {code}")


def letterbox(bgr: np.ndarray, size: int = 640):
    height, width = bgr.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    dw, dh = (size - new_width) / 2, (size - new_height) / 2
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        round(dh - 0.1),
        round(dh + 0.1),
        round(dw - 0.1),
        round(dw + 0.1),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    tensor = (tensor.astype(np.float32) / 255.0)[None]
    if tensor.shape != (1, 3, size, size):
        raise RuntimeError(f"Unexpected letterbox tensor {tensor.shape}")
    return tensor, ratio, dw, dh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "val2017")
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--verify-input-sha256")
    parser.add_argument("--confidence", type=float, default=0.001)
    args = parser.parse_args()

    if args.images_dir == ROOT / "val2017" and not (ROOT / "READY.json").exists():
        raise RuntimeError("COCO images have not been prepared on Jetson")
    if not args.images_dir.is_dir():
        raise RuntimeError(f"Image directory does not exist: {args.images_dir}")
    annotations = json.loads(args.annotations.read_text())
    categories = sorted(c["id"] for c in annotations["categories"])
    images = sorted(annotations["images"], key=lambda x: x["file_name"])
    if len(images) != args.expected_count or len(categories) != 80:
        raise RuntimeError("Unexpected COCO image count or category map")
    if not 1 <= args.limit <= args.expected_count:
        raise ValueError("--limit must be in [1, --expected-count]")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"Cannot load TensorRT engine: {args.engine}")
    context = engine.create_execution_context()
    input_name, output_name = "images", "output0"
    if tuple(engine.get_tensor_shape(input_name)) != (
        1, 3, args.imgsz, args.imgsz
    ):
        raise RuntimeError("Unexpected TensorRT input binding")
    if tuple(engine.get_tensor_shape(output_name)) != (1, 300, 6):
        raise RuntimeError("Unexpected TensorRT one-to-one output binding")
    if engine.get_tensor_dtype(input_name) != trt.float32:
        raise RuntimeError("Expected float32 image input")
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    if output_dtype != np.float32:
        raise RuntimeError(f"Unexpected TensorRT output type {output_dtype}")

    cuda = cuda_runtime()
    device_input, device_output, stream = (
        ctypes.c_void_p(),
        ctypes.c_void_p(),
        ctypes.c_void_p(),
    )
    input_bytes = 1 * 3 * args.imgsz * args.imgsz * np.dtype(np.float32).itemsize
    output = np.empty((1, 300, 6), dtype=np.float32)
    check(cuda.cudaMalloc(ctypes.byref(device_input), input_bytes), "cudaMalloc input")
    check(cuda.cudaMalloc(ctypes.byref(device_output), output.nbytes), "cudaMalloc output")
    check(cuda.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    context.set_tensor_address(input_name, device_input.value)
    context.set_tensor_address(output_name, device_output.value)

    predictions = []
    timings = []
    preprocessing_timings = []
    total_timings = []
    first_digest = None
    try:
        for index, image in enumerate(images[: args.limit]):
            start_image = time.perf_counter()
            path = args.images_dir / image["file_name"]
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"Cannot read {path}")
            tensor, ratio, dw, dh = letterbox(bgr, args.imgsz)
            preprocessing_timings.append((time.perf_counter() - start_image) * 1000)
            if index == 0:
                first_digest = hashlib.sha256(tensor.tobytes()).hexdigest()
                if (
                    args.verify_input_sha256
                    and first_digest != args.verify_input_sha256
                ):
                    raise RuntimeError(
                        "Jetson preprocessing differs from the saved Modal tensor"
                    )
            start = time.perf_counter()
            check(
                cuda.cudaMemcpy(
                    device_input,
                    ctypes.c_void_p(tensor.ctypes.data),
                    tensor.nbytes,
                    1,
                ),
                "cudaMemcpy host to device",
            )
            if not context.execute_async_v3(stream.value):
                raise RuntimeError(f"TensorRT execution failed for {path.name}")
            check(cuda.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            check(
                cuda.cudaMemcpy(
                    ctypes.c_void_p(output.ctypes.data),
                    device_output,
                    output.nbytes,
                    2,
                ),
                "cudaMemcpy device to host",
            )
            timings.append((time.perf_counter() - start) * 1000)
            if not np.isfinite(output).all():
                raise RuntimeError(f"Non-finite engine output for {path.name}")
            for x1, y1, x2, y2, score, cls in output[0]:
                if score < args.confidence:
                    continue
                class_index = int(round(float(cls)))
                if not 0 <= class_index < 80:
                    raise RuntimeError(f"Invalid class index {cls}")
                x1, x2 = np.clip((x1 - dw) / ratio, 0, image["width"]), np.clip(
                    (x2 - dw) / ratio, 0, image["width"]
                )
                y1, y2 = np.clip((y1 - dh) / ratio, 0, image["height"]), np.clip(
                    (y2 - dh) / ratio, 0, image["height"]
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append(
                    {
                        "image_id": image["id"],
                        "category_id": categories[class_index],
                        "bbox": [
                            round(float(x1), 3),
                            round(float(y1), 3),
                            round(float(x2 - x1), 3),
                            round(float(y2 - y1), 3),
                        ],
                        "score": round(float(score), 6),
                    }
                )
            total_timings.append((time.perf_counter() - start_image) * 1000)
            if (index + 1) % 500 == 0:
                print(f"Processed {index + 1}/{args.limit}", flush=True)
    finally:
        check(cuda.cudaStreamDestroy(stream), "cudaStreamDestroy")
        check(cuda.cudaFree(device_input), "cudaFree input")
        check(cuda.cudaFree(device_output), "cudaFree output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions))
    summary = {
        "engine": args.engine.name,
        "images_evaluated": args.limit,
        "nms": False,
        "preprocessing": f"OpenCV letterbox {args.imgsz} RGB float32 [0,1]",
        "first_input_sha256": first_digest,
        "confidence": args.confidence,
        "predictions": len(predictions),
        "host_input_inference_output_ms_median": float(np.median(timings)),
        "host_input_inference_output_ms_p95": float(np.percentile(timings, 95)),
        "image_decode_letterbox_ms_median": float(
            np.median(preprocessing_timings)
        ),
        "decode_through_predictions_ms_median": float(np.median(total_timings)),
        "note": (
            "Host H2D+infer+D2H excludes CPU decode/letterbox. "
            "Decode-through-predictions includes CPU decode, letterbox and "
            "COCO row construction, but excludes JSON serialization and COCOeval."
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
