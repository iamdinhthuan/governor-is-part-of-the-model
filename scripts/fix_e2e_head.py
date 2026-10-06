#!/usr/bin/env python3
"""Fix the yolo26n e2e ONNX head for INT8 per-tensor quantization.

Two problems in the exported graph, both fatal to the score channel under
per-tensor uint8 quantization:

1. Concat_3(Mul_2 boxes[1,64,8400], Sigmoid scores[1,80,8400]) -> Transpose
   -> Split puts boxes (range ~[-50,800]) and scores ([0,1]) in ONE tensor,
   so the shared encoding crushes scores to 0 downstream.
   Fix: bypass Concat_3/Transpose/Split; transpose the two branches
   separately so each keeps its own encoding.

2. The final Concat packs boxes/scores/classes into one (1,300,6) output,
   again one shared encoding.  Fix: expose boxes (1,300,4), scores
   (1,300,1), classes (1,300,1) as three separate graph outputs.

Usage: fix_e2e_head.py in.onnx out.onnx
"""
import sys

import onnx
from onnx import helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
by_out = {}
for n in g.node:
    for o in n.output:
        by_out[o] = n

concat3 = by_out["/model.23/Concat_3_output_0"]
transpose = by_out["/model.23/Transpose_output_0"]
split = by_out["/model.23/Split_output_0"]
boxes_src, scores_src = list(concat3.input)  # Mul_2_output_0, Sigmoid_output_0
perm = list(transpose.attribute[0].ints)

t_boxes = helper.make_node("Transpose", [boxes_src], ["_tb"], perm=perm,
                           name="/fix/Transpose_boxes")
t_scores = helper.make_node("Transpose", [scores_src], ["_ts"], perm=perm,
                            name="/fix/Transpose_scores")

# rewire consumers of the split outputs
for n in g.node:
    for i, name in enumerate(n.input):
        if name == "/model.23/Split_output_0":
            n.input[i] = "_tb"
        elif name == "/model.23/Split_output_1":
            n.input[i] = "_ts"

# replace the bypassed nodes in place to keep the graph topologically sorted
idx = list(g.node).index(concat3)
g.node.remove(concat3)
g.node.remove(transpose)
g.node.remove(split)
g.node.insert(idx, t_boxes)
g.node.insert(idx + 1, t_scores)

# split the final concat into three outputs
final = by_out["output0"]
assert final.op_type == "Concat" and len(final.input) == 3
boxes_t, scores_t, classes_t = list(final.input)
del g.output[:]
g.output.extend([
    helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 300, 4]),
    helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 300, 1]),
    helper.make_tensor_value_info("classes", TensorProto.FLOAT, [1, 300, 1]),
])
g.node.remove(final)
renames = {boxes_t: "boxes", scores_t: "scores", classes_t: "classes"}
for n in g.node:
    for i, name in enumerate(n.output):
        if name in renames:
            n.output[i] = renames[name]

onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst, "outputs:", [o.name for o in m.graph.output])
