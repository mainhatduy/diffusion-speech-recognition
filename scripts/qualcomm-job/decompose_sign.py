"""
Decompose the ONNX `Sign` operator into supported primitives for Qualcomm NPU.

The QNN HTP compiler does not support the `Sign` operator.
We decompose it mathematically:
    Sign(x) = Where(x > 0, 1, Where(x < 0, -1, 0))

This uses only: Greater, Less, Where, Cast, Constant — all supported by QNN HTP.

The script auto-detects the input tensor's data type and generates constants with matching types.

Usage:
    python scripts/qualcomm-job/decompose_sign.py
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


# Map ONNX elem_type to numpy dtype
ELEM_TYPE_TO_NP = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.INT16: np.int16,
    TensorProto.INT8: np.int8,
    TensorProto.UINT8: np.uint8,
}


def infer_input_type(graph, input_name: str) -> int:
    """Infer the ONNX TensorProto element type of a named tensor in the graph."""
    # Check graph inputs
    for vi in graph.input:
        if vi.name == input_name:
            return vi.type.tensor_type.elem_type
    # Check value_info (intermediate tensors)
    for vi in graph.value_info:
        if vi.name == input_name:
            return vi.type.tensor_type.elem_type
    # Check initializers
    for init in graph.initializer:
        if init.name == input_name:
            return init.data_type
    # Default to float32
    print(f"  [!] Could not infer type for '{input_name}', defaulting to float32")
    return TensorProto.FLOAT


def decompose_sign_nodes(graph):
    """Find and decompose all Sign nodes in the graph.
    
    For integer types (like INT64), we need a special approach since
    Greater/Less comparisons require matching types:
    
    For integer input x:
        1. Cast x to float32
        2. Create float32 constants (0.0, 1.0, -1.0)
        3. Apply Greater/Less/Where on float32
        4. Cast result back to original integer type
    
    For float input x:
        Directly use matching-type constants.
    """
    nodes_to_remove = []
    nodes_to_insert = []  # (index, [new_nodes])

    for idx, node in enumerate(graph.node):
        if node.op_type != "Sign":
            continue

        input_var = node.input[0]
        output_var = node.output[0]
        prefix = f"decomposed_sign_{idx}"
        
        elem_type = infer_input_type(graph, input_var)
        np_dtype = ELEM_TYPE_TO_NP.get(elem_type, np.float32)
        is_integer = elem_type in (TensorProto.INT64, TensorProto.INT32, TensorProto.INT16, TensorProto.INT8, TensorProto.UINT8)
        
        print(f"[*] Found Sign node: {node.name} at index {idx}")
        print(f"    Input: {input_var} (elem_type={elem_type}, dtype={np_dtype.__name__}, is_int={is_integer})")
        print(f"    Output: {output_var}")

        new_nodes = []
        
        if is_integer:
            # For integer types: cast to float32, do comparison, cast back
            cast_input_name = f"{prefix}_input_f32"
            cast_to_float = helper.make_node(
                "Cast", [input_var], [cast_input_name],
                to=TensorProto.FLOAT,
                name=f"{prefix}_cast_to_float"
            )
            new_nodes.append(cast_to_float)
            comparison_input = cast_input_name
            const_onnx_type = TensorProto.FLOAT
            const_np_type = np.float32
        else:
            comparison_input = input_var
            const_onnx_type = elem_type
            const_np_type = np_dtype
        
        # Constants in the comparison type
        zero_tensor = numpy_helper.from_array(
            np.array(0.0, dtype=const_np_type), name=f"{prefix}_zero_val"
        )
        one_tensor = numpy_helper.from_array(
            np.array(1.0, dtype=const_np_type), name=f"{prefix}_one_val"
        )
        neg_one_tensor = numpy_helper.from_array(
            np.array(-1.0, dtype=const_np_type), name=f"{prefix}_neg_one_val"
        )

        const_zero = helper.make_node(
            "Constant", [], [f"{prefix}_zero"], value=zero_tensor, name=f"{prefix}_const_zero"
        )
        const_one = helper.make_node(
            "Constant", [], [f"{prefix}_one"], value=one_tensor, name=f"{prefix}_const_one"
        )
        const_neg_one = helper.make_node(
            "Constant", [], [f"{prefix}_neg_one"], value=neg_one_tensor, name=f"{prefix}_const_neg_one"
        )
        new_nodes.extend([const_zero, const_one, const_neg_one])

        # x > 0
        greater_node = helper.make_node(
            "Greater",
            inputs=[comparison_input, f"{prefix}_zero"],
            outputs=[f"{prefix}_is_positive"],
            name=f"{prefix}_greater",
        )
        new_nodes.append(greater_node)

        # x < 0
        less_node = helper.make_node(
            "Less",
            inputs=[comparison_input, f"{prefix}_zero"],
            outputs=[f"{prefix}_is_negative"],
            name=f"{prefix}_less",
        )
        new_nodes.append(less_node)

        # inner_where = Where(x < 0, -1, 0)
        inner_where = helper.make_node(
            "Where",
            inputs=[f"{prefix}_is_negative", f"{prefix}_neg_one", f"{prefix}_zero"],
            outputs=[f"{prefix}_neg_or_zero"],
            name=f"{prefix}_inner_where",
        )
        new_nodes.append(inner_where)

        if is_integer:
            # Sign(x) = Where(x > 0, 1, inner_where) in float32
            float_result_name = f"{prefix}_result_f32"
            outer_where = helper.make_node(
                "Where",
                inputs=[f"{prefix}_is_positive", f"{prefix}_one", f"{prefix}_neg_or_zero"],
                outputs=[float_result_name],
                name=f"{prefix}_outer_where",
            )
            new_nodes.append(outer_where)
            
            # Cast back to original integer type
            cast_back = helper.make_node(
                "Cast", [float_result_name], [output_var],
                to=elem_type,
                name=f"{prefix}_cast_to_int"
            )
            new_nodes.append(cast_back)
        else:
            # Direct output for float types
            outer_where = helper.make_node(
                "Where",
                inputs=[f"{prefix}_is_positive", f"{prefix}_one", f"{prefix}_neg_or_zero"],
                outputs=[output_var],
                name=f"{prefix}_outer_where",
            )
            new_nodes.append(outer_where)

        nodes_to_remove.append(idx)
        nodes_to_insert.append((idx, new_nodes))

    if not nodes_to_remove:
        print("[!] No Sign nodes found in the graph.")
        return False

    # Remove old nodes and insert new ones (process in reverse to maintain indices)
    for idx in reversed(nodes_to_remove):
        del graph.node[idx]

    offset = 0
    for orig_idx, new_nodes in sorted(nodes_to_insert, key=lambda x: x[0]):
        insert_at = orig_idx + offset
        for i, node in enumerate(new_nodes):
            graph.node.insert(insert_at + i, node)
        offset += len(new_nodes) - 1  # -1 because we already removed the original

    return True


def main():
    model_path = "onnx/audio_encoder.onnx"
    print(f"[*] Loading ONNX model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    # Count Sign nodes
    sign_count = sum(1 for n in graph.node if n.op_type == "Sign")
    print(f"[*] Found {sign_count} Sign node(s) in the graph.")

    if sign_count == 0:
        print("[+] No Sign nodes to decompose. Model is already compatible.")
        return

    success = decompose_sign_nodes(graph)
    if not success:
        return

    # Validate
    print("[*] Running ONNX checker...")
    try:
        onnx.checker.check_model(model)
        print("[+] ONNX check passed!")
    except Exception as e:
        print(f"[!] ONNX check failed (may be expected for large models): {e}")

    # Save
    output_path = model_path  # overwrite
    print(f"[*] Saving modified model to: {output_path}")
    onnx.save(model, output_path)
    print("[+] Sign decomposition complete!")


if __name__ == "__main__":
    main()
