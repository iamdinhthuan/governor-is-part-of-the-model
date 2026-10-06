"""Summarize physical TensorRT layer precision for six pilot engines."""

import collections
import hashlib
import json
from pathlib import Path

SPECS = {
    "float": ("pilot_float_layers.json", "pilot_float_fp16_opt0.engine"),
    "qat": ("pilot_qat_layers.json", "pilot_qat_fp16_int8_opt0.engine"),
    "l1": ("pilot_l1_layers.json", "pilot_l1_fp16_opt0.engine"),
    "object": ("pilot_object_layers.json", "pilot_object_fp16_opt0.engine"),
    "l1_qat": ("pilot_l1_qat_layers.json", "pilot_l1_qat_opt0.engine"),
    "object_qat": ("pilot_object_qat_layers.json", "pilot_object_qat_opt0.engine"),
    "opt3_float": ("pilot_float_opt3_layers.json", "pilot_float_fp16_opt3.engine"),
    "opt3_qat": ("pilot_qat_opt3_layers.json", "pilot_qat_mixed_opt3.engine"),
    "gated_opt0": ("pilot_qat_gated_layers.json", "pilot_qat_gated_opt0.engine"),
}


def precision(format_description: str) -> str:
    for label in ("Int8", "FP16", "FP32"):
        if label in format_description:
            return label
    return format_description


report = {}
for kind, (layers_file, engine_file) in SPECS.items():
    layers = json.loads((Path("/tmp") / layers_file).read_text())["Layers"]
    convs = [
        layer for layer in layers
        if "Convolution" in layer.get("LayerType", "")
    ]
    conv_precision = collections.Counter(
        precision(output["Format/Datatype"])
        for layer in convs
        for output in layer["Outputs"]
    )
    all_precision = collections.Counter(
        precision(output["Format/Datatype"])
        for layer in layers
        for output in layer["Outputs"]
    )
    engine = Path("/tmp") / engine_file
    report[kind] = {
        "layer_count": len(layers),
        "convolution_count": len(convs),
        "convolution_output_precision": dict(conv_precision),
        "all_layer_output_precision": dict(all_precision),
        "engine_bytes": engine.stat().st_size,
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
    }
print(json.dumps(report, sort_keys=True, indent=2))
