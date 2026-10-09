#!/usr/bin/env python3
"""
Convert exp/workload/Transformer-1T/transformer1t_layers.txt into 512
per-NPU Chakra .et files, scoping each collective to the correct MP/DP
communicator group via pg_name.

Identical logic to exp/workload/GPT-3/gen_et.py -- see that file's
docstring for the full explanation (why pg_name is needed, the upstream
text_converter.py bug this avoids). Only the input/output paths and the
imported comm_groups module differ.

Must run where `chakra` (and its `protobuf` dependency) is importable --
inside the astra-sim-latest container, after applying the protobuf
gencode/runtime version fix from this session (regenerate et_def_pb2.py
with the container's system protoc and copy it into site-packages -- see
this session's chakra protobuf version-mismatch fix).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comm_groups import mp_group_id, dp_group_id, NUM_NPUS  # noqa: E402

from chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
    ALL_REDUCE, ALL_TO_ALL, ALL_GATHER, REDUCE_SCATTER,
    COMP_NODE, COMM_COLL_NODE, GlobalMetadata, Node as ChakraNode,
    AttributeProto as ChakraAttr,
)
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # noqa: E402

INPUT_FILE = Path(__file__).parent / "transformer1t_layers.txt"
OUTPUT_PREFIX = Path(__file__).parent / "transformer1t_trace"

COMM_TYPE_MAP = {
    "ALLREDUCE": ALL_REDUCE,
    "ALLTOALL": ALL_TO_ALL,
    "ALLGATHER": ALL_GATHER,
    "REDUCESCATTER": REDUCE_SCATTER,
}


class Layer:
    def __init__(self, line: str) -> None:
        col = line.strip().split()
        self.name = col[0]
        self.fwd_comp_time = int(col[2])
        self.fwd_comm_type = col[3]
        self.fwd_comm_size = int(col[4])
        self.bwd_ig_comp_time = int(col[5])
        self.bwd_ig_comm_type = col[6]
        self.bwd_ig_comm_size = int(col[7])
        self.bwd_wg_comp_time = int(col[8])
        self.bwd_wg_comm_type = col[9]
        self.bwd_wg_comm_size = int(col[10])
        self.fwd_comm_node = None
        self.bwd_ig_comm_node = None
        self.bwd_wg_comm_node = None
        self.bwd_wg_comp_node = None


def load_layers(path: Path) -> list:
    with open(path) as f:
        parallelism_type = f.readline().strip()
        assert parallelism_type == "HYBRID_MODEL_DATA", parallelism_type
        num_layers = int(f.readline().strip())
        layers = [Layer(line) for line in f if line.strip()]
    assert len(layers) == num_layers, (len(layers), num_layers)
    return layers


def make_node(next_id: list, name: str, node_type) -> ChakraNode:
    node = ChakraNode()
    node.id = next_id[0]
    next_id[0] += 1
    node.name = name
    node.type = node_type
    return node


def make_comp_node(next_id: list, name: str, phase: str, comp_time: int) -> ChakraNode:
    node = make_node(next_id, f"COMP_{name}_{phase}", COMP_NODE)
    node.duration_micros = comp_time
    return node


def make_comm_node(next_id: list, name: str, phase: str, comm_type: str,
                    comm_size: int, pg_name: int) -> ChakraNode:
    node = make_node(next_id, f"COMM_{name}_{phase}", COMM_COLL_NODE)
    node.attr.append(ChakraAttr(name="comm_type", int64_val=COMM_TYPE_MAP[comm_type]))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
    node.attr.append(ChakraAttr(name="pg_name", string_val=str(pg_name)))
    return node


def add_parent(child: ChakraNode, parent: ChakraNode) -> None:
    child.data_deps.append(parent.id)


def build_npu_trace(npu_id: int, layers: list, out_path: Path) -> None:
    mp_gid = mp_group_id(npu_id)
    dp_gid = dp_group_id(npu_id)
    next_id = [0]
    num_layers = len(layers)

    for layer in layers:
        layer.fwd_comm_node = None
        layer.bwd_ig_comm_node = None
        layer.bwd_wg_comm_node = None
        layer.bwd_wg_comp_node = None

    with open(out_path, "wb") as g:
        encode_message(g, GlobalMetadata(version="0.0.4"))

        fwd_comm_node = None
        for idx, layer in enumerate(layers):
            fwd_comp_node = make_comp_node(next_id, layer.name, "FWD", layer.fwd_comp_time)
            if layer.bwd_wg_comm_node is not None:
                add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
            if idx != 0:
                add_parent(fwd_comp_node, layers[idx - 1].fwd_comm_node)
            encode_message(g, fwd_comp_node)

            fwd_comm_node = make_comm_node(
                next_id, layer.name, "FWD", layer.fwd_comm_type, layer.fwd_comm_size, mp_gid)
            add_parent(fwd_comm_node, fwd_comp_node)
            layer.fwd_comm_node = fwd_comm_node
            encode_message(g, fwd_comm_node)

        for idx, layer in enumerate(reversed(layers)):
            bwd_ig_comp_node = make_comp_node(next_id, layer.name, "BWD_IG", layer.bwd_ig_comp_time)
            if idx == 0:
                add_parent(bwd_ig_comp_node, fwd_comm_node)
            else:
                prev_layer = layers[num_layers - idx]
                add_parent(bwd_ig_comp_node, prev_layer.bwd_wg_comp_node)
                add_parent(bwd_ig_comp_node, prev_layer.bwd_ig_comm_node)
            encode_message(g, bwd_ig_comp_node)

            if idx != (num_layers - 1):
                bwd_ig_comm_node = make_comm_node(
                    next_id, layer.name, "BWD_IG", layer.bwd_ig_comm_type,
                    layer.bwd_ig_comm_size, mp_gid)
                add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                layer.bwd_ig_comm_node = bwd_ig_comm_node
                encode_message(g, bwd_ig_comm_node)

            bwd_wg_comp_node = make_comp_node(next_id, layer.name, "BWD_WG", layer.bwd_wg_comp_time)
            add_parent(bwd_wg_comp_node, bwd_ig_comp_node)
            layer.bwd_wg_comp_node = bwd_wg_comp_node
            encode_message(g, bwd_wg_comp_node)

            bwd_wg_comm_node = make_comm_node(
                next_id, layer.name, "BWD_WG", layer.bwd_wg_comm_type,
                layer.bwd_wg_comm_size, dp_gid)
            add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
            layer.bwd_wg_comm_node = bwd_wg_comm_node
            encode_message(g, bwd_wg_comm_node)


def main() -> None:
    layers = load_layers(INPUT_FILE)
    for npu_id in range(NUM_NPUS):
        out_path = Path(f"{OUTPUT_PREFIX}.{npu_id}.et")
        build_npu_trace(npu_id, layers, out_path)
        if npu_id % 64 == 0:
            print(f"  ...generated npu {npu_id}/{NUM_NPUS}")
    print(f"Wrote {NUM_NPUS} .et files to {OUTPUT_PREFIX}.*.et")


if __name__ == "__main__":
    main()
