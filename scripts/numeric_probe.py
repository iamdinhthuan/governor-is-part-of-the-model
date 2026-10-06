"""Compare ONNX Runtime output against the same input fed to Jetson TensorRT.

Run with:
  uv run --with numpy==2.2.6 --with onnxruntime==1.22.1 numeric_probe.py

The generated input is synthetic and is only a numerical graph probe.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--tag", default="synthetic")
    args = parser.parse_args()
    if args.input:
        image = np.fromfile(args.input, dtype=np.float32).reshape(1, 3, 640, 640)
        input_path = args.input
    else:
        rng = np.random.default_rng(20261004)
        image = rng.random((1, 3, 640, 640), dtype=np.float32)
        input_path = ARTIFACTS / "numeric_probe_input_fp32.bin"
        input_path.write_bytes(image.tobytes(order="C"))
    prefix = "" if args.tag == "synthetic" else f"{args.tag}-"

    for key, graph in (
        ("float_8497", "yolo26n_640_one2one.onnx"),
        ("qat_84172", "yolo26n_qat_coco8_probe.onnx"),
    ):
        session = ort.InferenceSession(
            str(ARTIFACTS / graph), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: image})[0]
        if output.shape != (1, 300, 6) or not np.isfinite(output).all():
            raise RuntimeError(f"Unexpected {key} output: {output.shape}")
        out = RESULTS / f"ort-{prefix}{key}-numeric.json"
        out.write_text(
            json.dumps(
                {
                    "engine": "onnxruntime-cpu",
                    "onnxruntime": ort.__version__,
                    "input": str(input_path.name),
                    "graph": graph,
                    "shape": list(output.shape),
                    "values": output.ravel().astype(float).tolist(),
                }
            )
            + "\n"
        )
        print(f"{key}: wrote {out}, range [{output.min():.3g}, {output.max():.3g}]")


if __name__ == "__main__":
    main()
