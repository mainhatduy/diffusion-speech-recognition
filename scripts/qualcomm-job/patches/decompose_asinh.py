import onnx
from onnx import helper, TensorProto

def main():
    print("[*] Loading ONNX model onnx/audio_encoder.onnx...")
    model = onnx.load("onnx/audio_encoder.onnx")
    graph = model.graph

    # Find the node index
    asinh_idx = None
    asinh_node = None
    for idx, node in enumerate(graph.node):
        if node.op_type == "Asinh":
            asinh_idx = idx
            asinh_node = node
            break

    if asinh_node is None:
        print("[!] No Asinh node found in the model graph. It might already be decomposed.")
        return

    print(f"[*] Found Asinh node: {asinh_node.name} at index {asinh_idx} with inputs {asinh_node.input} and outputs {asinh_node.output}")
    input_var = asinh_node.input[0]
    output_var = asinh_node.output[0]

    # Create new nodes
    # 1. Constant one
    one_tensor = helper.make_tensor(
        name="const_one_tensor",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[1.0]
    )
    const_one_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_one"],
        value=one_tensor,
        name="const_one_node"
    )

    # 2. x^2
    mul_node = helper.make_node(
        "Mul",
        inputs=[input_var, input_var],
        outputs=["x_squared"],
        name="decomposed_asinh_mul"
    )

    # 3. x^2 + 1
    add_one_node = helper.make_node(
        "Add",
        inputs=["x_squared", "const_one"],
        outputs=["x_squared_plus_one"],
        name="decomposed_asinh_add_one"
    )

    # 4. sqrt(x^2 + 1)
    sqrt_node = helper.make_node(
        "Sqrt",
        inputs=["x_squared_plus_one"],
        outputs=["sqrt_val"],
        name="decomposed_asinh_sqrt"
    )

    # 5. x + sqrt(x^2 + 1)
    add_sqrt_node = helper.make_node(
        "Add",
        inputs=[input_var, "sqrt_val"],
        outputs=["sum_val"],
        name="decomposed_asinh_add_sqrt"
    )

    # 6. log(x + sqrt(x^2 + 1))
    log_node = helper.make_node(
        "Log",
        inputs=["sum_val"],
        outputs=[output_var],
        name="decomposed_asinh_log"
    )

    # Remove old node and insert new ones at the exact index to preserve topological sort
    print("[*] Replacing Asinh node with decomposed math nodes in-place...")
    del graph.node[asinh_idx]
    
    new_nodes = [const_one_node, mul_node, add_one_node, sqrt_node, add_sqrt_node, log_node]
    for i, node in enumerate(new_nodes):
        graph.node.insert(asinh_idx + i, node)

    # Check and save
    print("[*] Running ONNX checker to verify graph consistency...")
    try:
        onnx.checker.check_model(model)
        print("[+] ONNX check passed!")
    except Exception as e:
        print(f"[!] ONNX check failed: {e}")
        return
    
    print("[*] Saving modified model back to onnx/audio_encoder.onnx...")
    onnx.save(model, "onnx/audio_encoder.onnx")
    print("[+] Decomposition complete!")

if __name__ == "__main__":
    main()
