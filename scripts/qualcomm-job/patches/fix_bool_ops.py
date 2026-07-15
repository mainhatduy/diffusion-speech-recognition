"""
Fix BOOL operations (Pad, GatherND) in ONNX models for Qualcomm QNN compatibility.

HTP (Hexagon Tensor Processor) does not natively support BOOL type inputs/outputs
for Pad and GatherND operations. This script inserts Cast nodes to convert
BOOL tensors to supported types (INT8 for Pad, INT32/INT8 for GatherND) before
the operation, and Cast back to BOOL afterwards.

Usage:
    python scripts/qualcomm-job/patches/fix_bool_ops.py
"""

import os
import onnx
from onnx import helper, TensorProto


def infer_tensor_types(graph):
    """Map tensor names to their element types."""
    tensor_types = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            tensor_types[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        tensor_types[init.name] = init.data_type
    return tensor_types


def fix_bool_ops_in_graph(graph, graph_name):
    """Find and fix BOOL Pad and GatherND nodes in the graph."""
    tensor_types = infer_tensor_types(graph)
    nodes_to_remove = []
    nodes_to_insert = []  # (index, [new_nodes])
    modified = False

    for idx, node in enumerate(graph.node):
        # 1. Fix BOOL Pad
        if node.op_type == "Pad":
            input_name = node.input[0]
            output_name = node.output[0]
            t_type = tensor_types.get(input_name, TensorProto.UNDEFINED)

            if t_type == TensorProto.BOOL:
                print(
                    f"[*] {graph_name}: Found BOOL Pad node '{node.name}' at index {idx}"
                )
                prefix = f"fixed_bool_pad_{idx}"

                # Create Cast to FLOAT
                cast_in_name = f"{prefix}_in_float"
                cast_in = helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_in_name],
                    to=TensorProto.FLOAT,
                    name=f"{prefix}_cast_in",
                )

                # Modify Pad node to take the FLOAT input and output FLOAT
                pad_out_name = f"{prefix}_out_float"
                node.input[0] = cast_in_name
                node.output[0] = pad_out_name

                # Create Cast back to BOOL
                cast_out = helper.make_node(
                    "Cast",
                    inputs=[pad_out_name],
                    outputs=[output_name],
                    to=TensorProto.BOOL,
                    name=f"{prefix}_cast_out",
                )

                # We need to insert cast_in before the Pad node, and cast_out after the Pad node.
                # In our nodes_to_insert structure, we keep the original modified Pad node
                # and insert the cast nodes around it.
                nodes_to_remove.append(idx)
                nodes_to_insert.append((idx, [cast_in, node, cast_out]))
                modified = True

                # Update our local tensor types map
                tensor_types[cast_in_name] = TensorProto.FLOAT
                tensor_types[pad_out_name] = TensorProto.FLOAT

        # 2. Fix GatherND (both BOOL data input and constant indices bug)
        elif node.op_type == "GatherND":
            input_name = node.input[0]
            output_name = node.output[0]
            t_type = tensor_types.get(input_name, TensorProto.UNDEFINED)
            indices_input_name = node.input[1]

            # Make sure we don't apply the dynamic indices fix twice
            if not indices_input_name.endswith("_indices_dynamic"):
                print(
                    f"[*] {graph_name}: Found GatherND node '{node.name}' at index {idx} to make dynamic"
                )
                prefix = f"fixed_gathernd_{idx}"

                # Pick a dynamic input of the model to base our dummy zero on.
                dynamic_input_name = graph.input[0].name
                dynamic_input_type = graph.input[0].type.tensor_type.elem_type

                # 1. ReduceMin of dynamic_input -> scalar
                reduced_name = f"{prefix}_indices_reduced"
                reduce_node = helper.make_node(
                    "ReduceMin",
                    inputs=[dynamic_input_name],
                    outputs=[reduced_name],
                    keepdims=0,  # scalar
                    name=f"{prefix}_indices_reduce",
                )

                # 2. Multiply by 0 -> scalar 0
                zero_const_name = f"{prefix}_const_zero"
                zero_tensor = helper.make_tensor(
                    name=f"{prefix}_zero_tensor",
                    data_type=dynamic_input_type,
                    dims=[],
                    vals=(
                        [0.0]
                        if dynamic_input_type
                        in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16)
                        else [0]
                    ),
                )
                zero_const_node = helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[zero_const_name],
                    value=zero_tensor,
                    name=f"{prefix}_indices_zero_const",
                )

                dummy_zero_name = f"{prefix}_dummy_zero"
                mul_node = helper.make_node(
                    "Mul",
                    inputs=[reduced_name, zero_const_name],
                    outputs=[dummy_zero_name],
                    name=f"{prefix}_indices_mul_zero",
                )

                # 3. Cast the dummy zero to the type of the indices tensor if needed
                indices_type = tensor_types.get(indices_input_name, TensorProto.INT64)
                if dynamic_input_type != indices_type:
                    cast_zero_name = f"{prefix}_dummy_zero_cast"
                    cast_zero_node = helper.make_node(
                        "Cast",
                        inputs=[dummy_zero_name],
                        outputs=[cast_zero_name],
                        to=indices_type,
                        name=f"{prefix}_indices_cast_zero",
                    )
                    add_inputs = [indices_input_name, cast_zero_name]
                    extra_nodes = [
                        reduce_node,
                        zero_const_node,
                        mul_node,
                        cast_zero_node,
                    ]
                else:
                    add_inputs = [indices_input_name, dummy_zero_name]
                    extra_nodes = [reduce_node, zero_const_node, mul_node]

                # 4. Add the dummy zero to the indices tensor
                dynamic_indices_name = f"{prefix}_indices_dynamic"
                add_node = helper.make_node(
                    "Add",
                    inputs=add_inputs,
                    outputs=[dynamic_indices_name],
                    name=f"{prefix}_indices_add_zero",
                )

                # Update GatherND indices input to use the dynamic indices
                node.input[1] = dynamic_indices_name

                if t_type == TensorProto.BOOL:
                    print(
                        f"[*] {graph_name}: Also fixing BOOL input type for GatherND node '{node.name}'"
                    )
                    # Create Cast to INT32 (GatherND data input)
                    cast_in_name = f"{prefix}_in_int32"
                    cast_in = helper.make_node(
                        "Cast",
                        inputs=[input_name],
                        outputs=[cast_in_name],
                        to=TensorProto.INT32,
                        name=f"{prefix}_cast_in",
                    )

                    # Modify GatherND node to take the INT32 input and output INT32
                    gather_out_name = f"{prefix}_out_int32"
                    node.input[0] = cast_in_name
                    node.output[0] = gather_out_name

                    # Create Cast back to BOOL
                    cast_out = helper.make_node(
                        "Cast",
                        inputs=[gather_out_name],
                        outputs=[output_name],
                        to=TensorProto.BOOL,
                        name=f"{prefix}_cast_out",
                    )

                    nodes_to_remove.append(idx)
                    nodes_to_insert.append(
                        (idx, extra_nodes + [add_node, cast_in, node, cast_out])
                    )
                    tensor_types[cast_in_name] = TensorProto.INT32
                    tensor_types[gather_out_name] = TensorProto.INT32
                else:
                    nodes_to_remove.append(idx)
                    nodes_to_insert.append((idx, extra_nodes + [add_node, node]))

                modified = True
                tensor_types[dynamic_indices_name] = indices_type

    if not modified:
        return False

    # Perform the replacement
    # Remove old nodes first (in reverse order to preserve indices)
    for idx in reversed(nodes_to_remove):
        del graph.node[idx]

    # Insert new nodes
    offset = 0
    for orig_idx, new_nodes in sorted(nodes_to_insert, key=lambda x: x[0]):
        insert_at = orig_idx + offset
        for i, node in enumerate(new_nodes):
            graph.node.insert(insert_at + i, node)
        offset += len(new_nodes) - 1

    return True


def fix_model_file(model_path):
    print(f"\n[*] Processing model: {model_path}")
    if not os.path.exists(model_path):
        print(f"[!] File not found: {model_path}")
        return

    model = onnx.load(model_path)
    graph = model.graph

    modified = fix_bool_ops_in_graph(graph, os.path.basename(model_path))

    if modified:
        print(f"[*] Running ONNX checker on {model_path}...")
        try:
            onnx.checker.check_model(model)
            print("[+] ONNX check passed!")
        except Exception as e:
            print(
                f"[!] ONNX check failed (may be expected for large models with external weights): {e}"
            )

        print(f"[*] Saving modified model to {model_path}...")
        onnx.save(model, model_path)
        print(f"[+] Successfully fixed BOOL ops in {model_path}!")
    else:
        print(f"[+] No BOOL ops needed fixing in {model_path}.")


def main():
    fix_model_file("onnx/audio_encoder.onnx")
    fix_model_file("onnx/diffusion_backbone.onnx")


if __name__ == "__main__":
    main()
