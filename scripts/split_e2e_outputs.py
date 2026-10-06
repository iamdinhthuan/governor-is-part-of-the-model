#!/usr/bin/env python3
"""Split the yolo26n e2e ONNX's final Concat into three graph outputs.

The (1,300,6) output packs boxes (range 0-640), scores (0-1) and class ids
(0-79) into ONE tensor; under INT8 per-tensor quantization that single
tensor gets one scale covering the boxes, which destroys the score column.
Splitting the tail into three output tensors gives each its own scale.

In:  yolo26n_e2e.onnx   (output0 = Concat[boxes, scores, classes])
Out: yolo26n_e2e_3out.onnx with outputs boxes(1,300,4), scores(1,300,1),
     classes(1,300,1)
"""
import sys

import onnx
from onnx import helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph

concat = [n for n in g.node if n.output and n.output[0] == "output0"]
assert len(concat) == 1, "expected exactly one node producing output0"
concat = concat[0]
assert concat.op_type == "Concat" and len(concat.input) == 3
boxes_t, scores_t, classes_t = list(concat.input)

# shapes from the concat semantics: boxes (1,300,4), scores/classes (1,300,1)
del g.output[:]
g.output.extend([
    helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 300, 4]),
    helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 300, 1]),
    helper.make_tensor_value_info("classes", TensorProto.FLOAT, [1, 300, 1]),
])
g.node.remove(concat)
# rename the three tail tensors to the new output names
renames = {boxes_t: "boxes", scores_t: "scores", classes_t: "classes"}
for n in g.node:
    for i, name in enumerate(n.output):
        if name in renames:
            n.output[i] = renames[name]
onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst, "outputs:", [o.name for o in m.graph.output])
