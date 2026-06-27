"""
Fix high-rank GatherND for Qualcomm HTP compatibility.

Problem:
    The audio encoder (Moonshine) contains a GatherND node where the indices
    tensor has rank 5 ([B, H, T, T, 2]), which exceeds the HTP limit.

    The node does: output[...] = data[indices[..., 0], indices[..., 1]]
    where data=[T, T] (2D boolean causal mask) and indices=[B, H, T, T, 2].

    Previously, fix_bool_ops.py wrapped this GatherND with Cast nodes and
    added dynamic helpers, but QNN HTP still can't handle it because the
    indices rank is 5.

Solution:
    Replace GatherND(data[T,T], indices[B,H,T,T,2]) with:
      row_idx   = Gather(indices, 0, axis=-1)  [B,H,T,T]
      col_idx   = Gather(indices, 1, axis=-1)  [B,H,T,T]
      T_val     = Shape(data)[1]               scalar
      linear    = row_idx * T_val + col_idx    [B,H,T,T]
      data_float= Cast(data, FLOAT)            [T,T]
      data_flat = Reshape(data_float, [-1])    [T*T]
      out_float = Gather(data_flat, linear, axis=0)  [B,H,T,T]
      output    = Cast(out_float, BOOL)        [B,H,T,T]

    This removes GatherND entirely and only uses Gather (rank <= 4).

Note:
    This script operates on onnx/audio_encoder.onnx which must already have
    the fix_bool_ops.py patch applied (Cast wrappers around GatherND).
    After running, always run repackage_models.py.

Usage:
    python scripts/qualcomm-job/fix_gathernd_rank.py
"""

import os
import onnx
from onnx import helper, TensorProto
from onnx import shape_inference
import onnxruntime as ort
import numpy as np


def build_output_map(graph):
    """Return dict: output_name -> node."""
    out_map = {}
    for node in graph.node:
        for o in node.output:
            out_map[o] = node
    return out_map


def fix_gathernd_in_model(model_path):
    print(f"\n[*] Processing: {model_path}")
    if not os.path.exists(model_path):
        print(f"[!] File not found: {model_path}")
        return False

    model = onnx.load(model_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    out_map = build_output_map(graph)

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

    # Find GatherND nodes with indices rank >= 5
    targets = []
    for idx, node in enumerate(graph.node):
        if node.op_type != "GatherND":
            continue
        idx_shape = shapes.get(node.input[1], [])
        if len(idx_shape) >= 5 and idx_shape[-1] == 2:
            targets.append((idx, node))
            print(f"  Found rank-{len(idx_shape)} GatherND: '{node.name}'")

    if not targets:
        print("[+] No high-rank GatherND found.")
        return False

    for node_idx, gnd_node in reversed(targets):
        # The fix_bool_ops patch looks like:
        #   Cast(squeeze_1 -> FLOAT) -> fixed_gathernd_87_in_float   [data wrapper]
        #   <dynamic index helpers>  -> fixed_gathernd_87_indices_dynamic
        #   GatherND(in_float, indices_dynamic) -> fixed_gathernd_87_out_float
        #   Cast(out_float -> BOOL)  -> val_175                      [output wrapper]
        #
        # We want to find the ORIGINAL inputs (squeeze_1, val_174) and the FINAL output (val_175)

        data_patched    = gnd_node.input[0]   # fixed_gathernd_87_in_float
        indices_patched = gnd_node.input[1]   # fixed_gathernd_87_indices_dynamic
        gnd_out         = gnd_node.output[0]  # fixed_gathernd_87_out_float

        # Trace Cast to find original data
        orig_data = data_patched
        if orig_data in out_map and out_map[orig_data].op_type == "Cast":
            orig_data = out_map[orig_data].input[0]  # squeeze_1

        # Trace the Add(val_174, zero) helpers to find original indices
        # The dynamic helper chain: val_174 -> Add(+0) -> fixed_..._indices_dynamic
        orig_indices = indices_patched
        if orig_indices in out_map:
            add_node = out_map[orig_indices]
            if add_node.op_type == "Add":
                orig_indices = add_node.input[0]  # val_174

        # Find the Cast node after GatherND that produces the final BOOL output
        final_out = gnd_out  # fallback
        cast_out_node = None
        for node in graph.node:
            if node.op_type == "Cast" and gnd_out in node.input:
                cast_out_node = node
                final_out = node.output[0]  # val_175
                break

        print(f"  orig_data='{orig_data}', orig_indices='{orig_indices}', final_output='{final_out}'")

        # Collect all nodes to delete
        def collect_ancestors(name, stop_at, visited=None):
            if visited is None:
                visited = set()
            if name in visited or name == stop_at or name not in out_map:
                return visited
            visited.add(name)
            n = out_map[name]
            del_names.add(n.name)
            for inp in n.input:
                collect_ancestors(inp, stop_at, visited)
            return visited

        del_names = set()
        del_names.add(gnd_node.name)                          # GatherND itself
        if cast_out_node is not None:
            del_names.add(cast_out_node.name)                 # Cast after GatherND
        if data_patched != orig_data:
            cast_in = out_map.get(data_patched)
            if cast_in:
                del_names.add(cast_in.name)                   # Cast before GatherND (data)
        if indices_patched != orig_indices:
            collect_ancestors(indices_patched, orig_indices)  # dynamic index helpers

        print(f"  Deleting {len(del_names)} old nodes: {sorted(del_names)}")

        # Build replacement subgraph
        p = f"gnd_fix_{node_idx}"

        c0 = helper.make_node("Constant", [], [f"{p}_c0"],
                              value=helper.make_tensor(f"{p}_c0", TensorProto.INT64, [], [0]),
                              name=f"{p}_const0")
        c1 = helper.make_node("Constant", [], [f"{p}_c1"],
                              value=helper.make_tensor(f"{p}_c1", TensorProto.INT64, [], [1]),
                              name=f"{p}_const1")
        row = helper.make_node("Gather", [orig_indices, f"{p}_c0"], [f"{p}_row"], axis=-1, name=f"{p}_row")
        col = helper.make_node("Gather", [orig_indices, f"{p}_c1"], [f"{p}_col"], axis=-1, name=f"{p}_col")
        dshape = helper.make_node("Shape", [orig_data], [f"{p}_dshape"], name=f"{p}_dshape")
        T = helper.make_node("Gather", [f"{p}_dshape", f"{p}_c1"], [f"{p}_T"], axis=0, name=f"{p}_T")
        mul = helper.make_node("Mul", [f"{p}_row", f"{p}_T"], [f"{p}_mul"], name=f"{p}_mul")
        lin = helper.make_node("Add", [f"{p}_mul", f"{p}_col"], [f"{p}_lin"], name=f"{p}_lin")
        cast_data = helper.make_node("Cast", [orig_data], [f"{p}_dataf"], to=TensorProto.FLOAT, name=f"{p}_cast_data")
        flat_shape = helper.make_node("Constant", [], [f"{p}_fs"],
                                      value=helper.make_tensor(f"{p}_fs", TensorProto.INT64, [1], [-1]),
                                      name=f"{p}_fs")
        flat = helper.make_node("Reshape", [f"{p}_dataf", f"{p}_fs"], [f"{p}_flat"], name=f"{p}_flat")
        gather = helper.make_node("Gather", [f"{p}_flat", f"{p}_lin"], [f"{p}_outf"], axis=0, name=f"{p}_gather")
        cast_back = helper.make_node("Cast", [f"{p}_outf"], [final_out], to=TensorProto.BOOL, name=f"{p}_cast_back")

        new_nodes = [c0, c1, row, col, dshape, T, mul, lin, cast_data, flat_shape, flat, gather, cast_back]

        # Rebuild graph: remove deleted, insert new nodes at GatherND position
        old_nodes = list(graph.node)
        del graph.node[:]
        inserted = False
        for i, n in enumerate(old_nodes):
            if n.name in del_names:
                if n.name == gnd_node.name and not inserted:
                    for nn in new_nodes:
                        graph.node.append(nn)
                    inserted = True
                continue
            graph.node.append(n)

        # If GatherND wasn't found (already removed), append at end
        if not inserted:
            for nn in new_nodes:
                graph.node.append(nn)

        print(f"  Inserted {len(new_nodes)} replacement nodes.")

    # Quick validation: check no undefined inputs
    all_produced = set(i.name for i in graph.initializer) | set(i.name for i in graph.input)
    bad = []
    for n in graph.node:
        for inp in n.input:
            if inp and inp not in all_produced:
                bad.append(f"{n.name} ({n.op_type}): missing '{inp}'")
        all_produced.update(n.output)
    if bad:
        print("[!] Undefined inputs after replacement:")
        for b in bad:
            print(f"    {b}")
        return False

    print("[+] Graph structure valid.")
    return model


def main():
    # Process the standalone onnx file (with inline weights)
    result = fix_gathernd_in_model("onnx/audio_encoder.onnx")
    if not result:
        print("[!] Fix failed or not needed.")
        return

    model = result

    # Verify with OnnxRuntime before saving
    print("\n[*] Verifying with OnnxRuntime...")
    try:
        tmp = "onnx/audio_encoder_fixed_tmp.onnx"
        onnx.save(model, tmp)
        sess = ort.InferenceSession(tmp)
        audio = np.random.randn(1, 2400).astype(np.float32)
        mask  = np.ones((1, 2400), dtype=np.int64)
        out   = sess.run(None, {"audio_features": audio, "audio_attention_mask": mask})
        print(f"[+] OnnxRuntime OK. Output shape: {out[0].shape}")
    except Exception as e:
        print(f"[!] OnnxRuntime verification FAILED: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return

    # Save inline version
    os.rename(tmp, "onnx/audio_encoder.onnx")
    print("[+] Saved: onnx/audio_encoder.onnx")

    # Repackage to pkg dir
    out_dir  = "onnx/audio_encoder_pkg.onnx"
    out_onnx = os.path.join(out_dir, "audio_encoder.onnx")
    out_data = os.path.join(out_dir, "audio_encoder.data")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_data):
        os.remove(out_data)
    onnx.save(model, out_onnx,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location="audio_encoder.data")
    print(f"[+] Saved: {out_onnx}")
    print("[+] Done! Run repackage_models.py is NOT needed — already repackaged.")


if __name__ == "__main__":
    main()
