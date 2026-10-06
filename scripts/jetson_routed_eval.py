"""Benchmark per-image resolution routing and fixed controls on Jetson.

The decision is computed from raw image pixels before any model inference.
Each image runs exactly one TensorRT engine. Both engines remain resident;
one shared CUDA stream and preallocated buffers prevent model-loading cost
from contaminating per-image timings.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import tensorrt as trt

from jetson_eval import check, cuda_runtime, letterbox
from jetson_fast_features import NAMES, feature

ENGINES = {
    512: Path("/tmp/resolution_512_opt0.engine"),
    576: Path("/tmp/resolution_576_opt0.engine"),
    640: Path("/tmp/yolo26n_84172_fp16_opt0.engine"),
}
ROOT = Path("/tmp/yolo26-pilot-development")
ANNOTATIONS = Path("/tmp/instances_pilot_development.json")


def quantile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("router", "fixed512", "fixed576", "fixed640"), required=True)
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "images")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--expected-count", type=int, default=512)
    parser.add_argument("--policy", type=Path, default=Path("/tmp/preimage-router-policy.json"))
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--tag", default="")
    parser.add_argument("--vectorized", action="store_true")
    args = parser.parse_args()
    if args.tag and (
        len(args.tag) > 32 or not args.tag.replace("_", "").isalnum()
    ):
        raise RuntimeError("Invalid rerun tag")

    data = json.loads(args.annotations.read_text())
    image_ids = json.loads(args.image_ids.read_text())
    if len(image_ids) != args.expected_count or len(set(image_ids)) != args.expected_count:
        raise RuntimeError(f"Expected exactly {args.expected_count} split IDs")
    images = sorted(
        [item for item in data["images"] if item["id"] in set(image_ids)],
        key=lambda item: item["file_name"],
    )
    if len(images) != args.expected_count or len(data["categories"]) != 80:
        raise RuntimeError("Invalid COCO split or category count")
    if not 1 <= args.limit <= args.expected_count:
        raise RuntimeError("Invalid held-out image limit")
    images = images[: args.limit]
    image_ids_digest = hashlib.sha256(
        ",".join(map(str, sorted(image_ids))).encode()
    ).hexdigest()

    policy = None
    weights = None
    if args.mode == "router":
        policy = json.loads(args.policy.read_text())
        if (
            policy["feature_names"] != list(NAMES)
            or not policy["no_ground_truth_or_detector_input_at_inference"]
        ):
            raise RuntimeError("Router features do not match frozen fit policy")
        weights = np.asarray(policy["weights"], dtype=np.float64)
        if weights.shape != (len(NAMES),):
            raise RuntimeError("Wrong frozen router coefficient count")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    needed = (512, 640) if args.mode == "router" else (int(args.mode[5:]),)
    engines = {}
    contexts = {}
    for size in needed:
        engine = runtime.deserialize_cuda_engine(ENGINES[size].read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT could not read {ENGINES[size]}")
        if (
            tuple(engine.get_tensor_shape("images")) != (1, 3, size, size)
            or tuple(engine.get_tensor_shape("output0")) != (1, 300, 6)
            or engine.get_tensor_dtype("images") != trt.float32
            or engine.get_tensor_dtype("output0") != trt.float32
        ):
            raise RuntimeError(f"Unexpected TensorRT bindings at {size}")
        engines[size] = engine
        contexts[size] = engine.create_execution_context()

    cuda = cuda_runtime()
    input_ptr, output_ptr, stream = (
        ctypes.c_void_p(),
        ctypes.c_void_p(),
        ctypes.c_void_p(),
    )
    output = np.empty((1, 300, 6), dtype=np.float32)
    check(
        cuda.cudaMalloc(ctypes.byref(input_ptr), 1 * 3 * 640 * 640 * 4),
        "cudaMalloc input",
    )
    check(cuda.cudaMalloc(ctypes.byref(output_ptr), output.nbytes), "cudaMalloc output")
    check(cuda.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    for context in contexts.values():
        context.set_tensor_address("images", input_ptr.value)
        context.set_tensor_address("output0", output_ptr.value)

    category_ids = sorted(item["id"] for item in data["categories"])
    predictions = []
    decisions = {}
    total_times, gpu_times, feature_times, decoding_times = [], [], [], []
    size_counts = {size: 0 for size in needed}
    begin_unix = time.time()
    try:
        for index, image in enumerate(images):
            start_image = time.perf_counter()
            path = args.images_dir / image["file_name"]
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"Unreadable {path}")
            if args.mode == "router":
                start_features = time.perf_counter()
                values = np.asarray(feature(bgr), dtype=np.float64)
                logit = float(np.dot(weights, values) + policy["bias"])
                size = 640 if logit >= policy["threshold_logit"] else 512
                feature_times.append((time.perf_counter() - start_features) * 1000)
            else:
                size = needed[0]
            size_counts[size] += 1
            decisions[image["id"]] = size
            tensor, ratio, dw, dh = letterbox(bgr, size)
            start_gpu = time.perf_counter()
            check(
                cuda.cudaMemcpy(
                    input_ptr,
                    ctypes.c_void_p(tensor.ctypes.data),
                    tensor.nbytes,
                    1,
                ),
                "cudaMemcpy host to device",
            )
            if not contexts[size].execute_async_v3(stream.value):
                raise RuntimeError(f"TensorRT failed on image {image['id']}")
            check(cuda.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            check(
                cuda.cudaMemcpy(
                    ctypes.c_void_p(output.ctypes.data),
                    output_ptr,
                    output.nbytes,
                    2,
                ),
                "cudaMemcpy device to host",
            )
            gpu_times.append((time.perf_counter() - start_gpu) * 1000)
            if not np.isfinite(output).all():
                raise RuntimeError(f"Non-finite engine output for {image['id']}")
            start_decode = time.perf_counter()
            if args.vectorized:
                candidate = output[0][output[0, :, 4] >= 0.001]
                if len(candidate):
                    classes = np.rint(candidate[:, 5]).astype(np.int32)
                    if np.any((classes < 0) | (classes >= 80)):
                        raise RuntimeError("Invalid class index")
                    coords = candidate[:, :4].astype(np.float64)
                    coords[:, [0, 2]] = np.clip(
                        (coords[:, [0, 2]] - dw) / ratio,
                        0, image["width"],
                    )
                    coords[:, [1, 3]] = np.clip(
                        (coords[:, [1, 3]] - dh) / ratio,
                        0, image["height"],
                    )
                    for xyxy, score, cls_index in zip(
                        coords, candidate[:, 4], classes
                    ):
                        x1, y1, x2, y2 = xyxy
                        if x2 <= x1 or y2 <= y1:
                            continue
                        predictions.append(
                            {
                                "image_id": image["id"],
                                "category_id": category_ids[int(cls_index)],
                                "bbox": [
                                    round(float(x1), 3),
                                    round(float(y1), 3),
                                    round(float(x2 - x1), 3),
                                    round(float(y2 - y1), 3),
                                ],
                                "score": round(float(score), 6),
                            }
                        )
            else:
                for x1, y1, x2, y2, score, cls in output[0]:
                    if score < 0.001:
                        continue
                    cls_index = int(round(float(cls)))
                    if not 0 <= cls_index < 80:
                        raise RuntimeError(f"Invalid class index {cls}")
                    x1 = np.clip((x1 - dw) / ratio, 0, image["width"])
                    x2 = np.clip((x2 - dw) / ratio, 0, image["width"])
                    y1 = np.clip((y1 - dh) / ratio, 0, image["height"])
                    y2 = np.clip((y2 - dh) / ratio, 0, image["height"])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    predictions.append(
                        {
                            "image_id": image["id"],
                            "category_id": category_ids[cls_index],
                            "bbox": [
                                round(float(x1), 3),
                                round(float(y1), 3),
                                round(float(x2 - x1), 3),
                                round(float(y2 - y1), 3),
                            ],
                            "score": round(float(score), 6),
                        }
                    )
            decoding_times.append((time.perf_counter() - start_decode) * 1000)
            total_times.append((time.perf_counter() - start_image) * 1000)
            if (index + 1) % 128 == 0:
                print(f"{args.mode}: processed {index + 1}/{len(images)}", flush=True)
    finally:
        end_unix = time.time()
        check(cuda.cudaStreamDestroy(stream), "cudaStreamDestroy")
        check(cuda.cudaFree(input_ptr), "cudaFree input")
        check(cuda.cudaFree(output_ptr), "cudaFree output")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.mode}{args.tag}-heldout-predictions.json"
    out.write_text(json.dumps(predictions))
    summary = {
        "mode": args.mode,
        "images": len(images),
        "images_timed": len(images),
        "window_unix": [begin_unix, end_unix],
        "throughput_images_per_s": len(images) / (end_unix - begin_unix),
        "heldout_ids_sha256": image_ids_digest,
        "no_nms": True,
        "size_counts": size_counts,
        "prediction_count": len(predictions),
        "mean_decode_through_predictions_ms": float(np.mean(total_times)),
        "median_decode_through_predictions_ms": float(np.median(total_times)),
        "p95_decode_through_predictions_ms": quantile(total_times, 95),
        "median_h2d_inference_d2h_ms": float(np.median(gpu_times)),
        "median_detection_decode_ms": float(np.median(decoding_times)),
        "decoding_mode": "vectorized" if args.vectorized else "scalar",
        "median_router_feature_ms": (
            float(np.median(feature_times)) if feature_times else 0.0
        ),
        "policy_sha256": (
            hashlib.sha256(args.policy.read_bytes()).hexdigest()
            if policy else None
        ),
        "route_decisions": decisions if policy else None,
        "note": (
            "Application timing includes one CPU image decode, routing "
            "features if enabled, letterbox, TensorRT and COCO row construction. "
            "It excludes JSON serialization and COCOeval."
        ),
    }
    (args.output_dir / f"{args.mode}{args.tag}-heldout-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "route_decisions"}))


if __name__ == "__main__":
    main()
