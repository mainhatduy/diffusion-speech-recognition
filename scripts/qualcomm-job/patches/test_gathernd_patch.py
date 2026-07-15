import onnx
from onnx import helper, TensorProto
import os


def test_patch():
    model_path = "onnx/diffusion_backbone.onnx"
    if not os.path.exists(model_path):
        print("Model file not found")
        return

    model = onnx.load(model_path)
    graph = model.graph

    # Let's find GatherND nodes
    for idx, node in enumerate(graph.node):
        if node.op_type == "GatherND":
            print(f"Found GatherND node: {node.name} at index {idx}")
            print(f"Inputs: {list(node.input)}")
            print(f"Outputs: {list(node.output)}")

            # Let's test building the patch nodes
            prefix = f"test_patch_gathernd_{idx}"
            indices_input_name = node.input[1]
            dynamic_input = graph.input[0]
            dynamic_input_name = dynamic_input.name
            dynamic_input_type = dynamic_input.type.tensor_type.elem_type

            reduce_node = helper.make_node(
                "ReduceMin",
                inputs=[dynamic_input_name],
                outputs=[f"{prefix}_reduced"],
                keepdims=0,
                name=f"{prefix}_reduce",
            )

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
                outputs=[f"{prefix}_const_zero"],
                value=zero_tensor,
                name=f"{prefix}_zero_const",
            )

            mul_node = helper.make_node(
                "Mul",
                inputs=[f"{prefix}_reduced", f"{prefix}_const_zero"],
                outputs=[f"{prefix}_dummy_zero"],
                name=f"{prefix}_mul_zero",
            )

            # For testing, assume indices_type is INT64
            cast_zero_node = helper.make_node(
                "Cast",
                inputs=[f"{prefix}_dummy_zero"],
                outputs=[f"{prefix}_dummy_zero_cast"],
                to=TensorProto.INT64,
                name=f"{prefix}_cast_zero",
            )

            add_node = helper.make_node(
                "Add",
                inputs=[indices_input_name, f"{prefix}_dummy_zero_cast"],
                outputs=[f"{prefix}_indices_dynamic"],
                name=f"{prefix}_add_zero",
            )

            print("Successfully constructed test patch nodes!")
            return


if __name__ == "__main__":
    test_patch()
