"""
run_all_patches.py
==================
Orchestrator script to apply all Qualcomm HTP compatibility patches
to the Speech Translation models.

Steps:
  1. Restore diffusion_backbone.onnx from diffusion_backbone_clean.onnx
  2. Fix high-rank GatherND (Rank 5 -> 4) in diffusion_backbone.onnx
  3. Decompose Asinh operator in audio_encoder.onnx
  4. Decompose Sign operator in audio_encoder.onnx
  5. Fix BOOL Pad/GatherND ops in both models
  6. Repackage both models to their respective .pkg.onnx directories
"""

import os
import sys
import subprocess
import onnx
from onnx import helper, TensorProto


def run_command(cmd, desc):
    print(f"\n[*] Running: {desc}...")
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Error: {desc} failed!")
        print(f"    stdout: {result.stdout}")
        print(f"    stderr: {result.stderr}")
        sys.exit(1)
    print(f"[+] Success: {desc} completed.")
    if result.stdout.strip():
        print("    " + result.stdout.strip().replace("\n", "\n    "))


def restore_backbone():
    print("\n[*] Restoring diffusion_backbone.onnx to clean state...")
    clean_path = "onnx/diffusion_backbone_clean.onnx"
    target_path = "onnx/diffusion_backbone.onnx"
    data_name = "diffusion_backbone.onnx.data"

    if not os.path.exists(clean_path):
        print(f"[!] Error: Clean model not found at {clean_path}")
        sys.exit(1)

    model = onnx.load(clean_path)
    onnx.save(
        model,
        target_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    print(f"[+] Restored {target_path} using external weights {data_name}")


def patch_gathernd_backbone():
    print(
        "\n[*] Applying GatherND Rank 5 -> Rank 4 reduction on diffusion_backbone.onnx..."
    )
    model_path = "onnx/diffusion_backbone.onnx"
    model = onnx.load(model_path)
    graph = model.graph

    # Build shape lookup
    shapes = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.HasField("dim_value") else "?")
            shapes[vi.name] = dims
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)

    # Find GatherND node
    gnd_node = None
    gnd_idx = None
    for idx, node in enumerate(graph.node):
        if node.op_type == "GatherND":
            idx_shape = shapes.get(node.input[1], [])
            if len(idx_shape) >= 5 and idx_shape[-1] == 2:
                gnd_node = node
                gnd_idx = idx
                print(
                    f"    Found Rank-{len(idx_shape)} GatherND node '{node.name}' at index {idx}"
                )
                break

    if not gnd_node:
        print(
            "[!] No High-rank GatherND found in diffusion_backbone.onnx (already patched?)"
        )
        return

    orig_data = gnd_node.input[0]
    orig_indices = gnd_node.input[1]
    final_out = gnd_node.output[0]

    p = f"gnd_fix_{gnd_idx}"

    # Subgraph replacement nodes
    c0 = helper.make_node(
        "Constant",
        [],
        [f"{p}_c0"],
        value=helper.make_tensor(f"{p}_c0", TensorProto.INT64, [], [0]),
        name=f"{p}_const0",
    )
    c1 = helper.make_node(
        "Constant",
        [],
        [f"{p}_c1"],
        value=helper.make_tensor(f"{p}_c1", TensorProto.INT64, [], [1]),
        name=f"{p}_const1",
    )
    row = helper.make_node(
        "Gather", [orig_indices, f"{p}_c0"], [f"{p}_row"], axis=-1, name=f"{p}_row"
    )
    col = helper.make_node(
        "Gather", [orig_indices, f"{p}_c1"], [f"{p}_col"], axis=-1, name=f"{p}_col"
    )
    dshape = helper.make_node("Shape", [orig_data], [f"{p}_dshape"], name=f"{p}_dshape")
    T = helper.make_node(
        "Gather", [f"{p}_dshape", f"{p}_c1"], [f"{p}_T"], axis=0, name=f"{p}_T"
    )
    mul = helper.make_node("Mul", [f"{p}_row", f"{p}_T"], [f"{p}_mul"], name=f"{p}_mul")
    lin = helper.make_node(
        "Add", [f"{p}_mul", f"{p}_col"], [f"{p}_lin"], name=f"{p}_lin"
    )
    cast_data = helper.make_node(
        "Cast", [orig_data], [f"{p}_dataf"], to=TensorProto.FLOAT, name=f"{p}_cast_data"
    )
    flat_shape = helper.make_node(
        "Constant",
        [],
        [f"{p}_fs"],
        value=helper.make_tensor(f"{p}_fs", TensorProto.INT64, [1], [-1]),
        name=f"{p}_fs",
    )
    flat = helper.make_node(
        "Reshape", [f"{p}_dataf", f"{p}_fs"], [f"{p}_flat"], name=f"{p}_flat"
    )
    gather = helper.make_node(
        "Gather", [f"{p}_flat", f"{p}_lin"], [f"{p}_outf"], axis=0, name=f"{p}_gather"
    )
    cast_back = helper.make_node(
        "Cast", [f"{p}_outf"], [final_out], to=TensorProto.BOOL, name=f"{p}_cast_back"
    )

    new_nodes = [
        c0,
        c1,
        row,
        col,
        dshape,
        T,
        mul,
        lin,
        cast_data,
        flat_shape,
        flat,
        gather,
        cast_back,
    ]

    # Remove old GatherND node and insert new subgraph
    del graph.node[gnd_idx]
    for i, nn in enumerate(new_nodes):
        graph.node.insert(gnd_idx + i, nn)

    # Save the patched model with external weights
    onnx.save(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="diffusion_backbone.onnx.data",
    )
    print(f"[+] Saved patched model to {model_path} with external weights")


def main():
    # Step 1: Restore backbone
    restore_backbone()

    # Step 2: Apply custom GatherND patch
    patch_gathernd_backbone()

    # Step 3: Decompose Asinh in audio_encoder
    run_command(
        ["python", "scripts/qualcomm-job/patches/decompose_asinh.py"],
        "Decomposing Asinh in audio_encoder.onnx",
    )

    # Step 4: Decompose Sign in audio_encoder
    run_command(
        ["python", "scripts/qualcomm-job/patches/decompose_sign.py"],
        "Decomposing Sign in audio_encoder.onnx",
    )

    # Step 5: Fix BOOL operations
    run_command(
        ["python", "scripts/qualcomm-job/patches/fix_bool_ops.py"],
        "Fixing BOOL Pad and GatherND operations",
    )

    # Step 6: Repackage models
    run_command(
        ["python", "scripts/qualcomm-job/patches/repackage_models.py"],
        "Repackaging models to .pkg.onnx",
    )

    print("\n" + "=" * 50)
    print("  ALL COMPATIBILITY PATCHES SUCCESSFULLY APPLIED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
