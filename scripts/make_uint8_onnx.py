"""Prepend uint8 NHWC-BGR input preprocessing to a static YOLO26 one-to-one ONNX.

The host then transfers the letterboxed uint8 image (4x fewer bytes than
float32 NCHW) and skips the CPU channel flip, transpose and normalization.
Usage: uv run --no-project --with onnx==1.17.0 make_uint8_onnx.py IN OUT
"""

import hashlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    source, target = sys.argv[1], sys.argv[2]
    model = onnx.load(source)
    graph = model.graph
    if [item.name for item in graph.input] != ["images"]:
        raise RuntimeError("Expected one graph input named images")
    dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[:2] != [1, 3] or dims[2] != dims[3] or dims[2] <= 0:
        raise RuntimeError(f"Expected static (1,3,S,S) input, got {dims}")
    size = dims[2]

    for node in graph.node:
        node.input[:] = ["images_float" if name == "images" else name for name in node.input]
    graph.initializer.extend([
        numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "bgr_to_rgb"),
        numpy_helper.from_array(np.array(255.0, dtype=np.float32), "pixel_scale"),
    ])
    prefix = [
        helper.make_node("Cast", ["images"], ["images_cast"], to=TensorProto.FLOAT),
        helper.make_node("Gather", ["images_cast", "bgr_to_rgb"], ["images_rgb"], axis=3),
        helper.make_node("Transpose", ["images_rgb"], ["images_nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Div", ["images_nchw", "pixel_scale"], ["images_float"]),
    ]
    nodes = prefix + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.input[:]
    graph.input.append(
        helper.make_tensor_value_info("images", TensorProto.UINT8, [1, size, size, 3])
    )
    onnx.checker.check_model(model)
    onnx.save(model, target)
    digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
    print(f"Wrote {target}: uint8 (1,{size},{size},3) BGR input, sha256 {digest}")


if __name__ == "__main__":
    main()
