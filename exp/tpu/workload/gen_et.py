#!/usr/bin/env python3
"""
Generate per-NPU Chakra .et files for ONE MoE transformer layer (forward pass)
on a 4-chip TPU tray, matching the SCALE-Sim setup used for the per-chip
compute numbers. Each chip has its own workload folder (moe_4npus_tpuv4,
moe_4npus_tpuv5p, ...) holding compute_us.json and the generated moe.*.et.

  - Mixtral-8x7B routing (8 experts, top-2), dims shrunk to Qwen1.5-MoE
    (hidden = 2048, expert FFN = 5632, head_dim = 128).
  - 1024 input tokens, top-2 -> 2048 (token, expert) slots, routed as
    TOKENS_PER_EXPERT = [821, 466, 300, 199, 129, 77, 41, 15].
  - NPU i holds experts 2i and 2i+1 (EXPERTS_PER_NPU = 2).

Per NPU, the 8 GEMMs are the same 8 lines as the SCALE-Sim topology file
("gemm, M, N, K"):
    L1  QKV projection      1024 x 2048 x 2048
    L2  Q x K^T             1024 x 1024 x 128
    L3  score x V           1024 x 128  x 1024
    L4  output projection   1024 x 2048 x 2048
    L5  expert A up         tA   x 5632 x 2048
    L6  expert A down       tA   x 2048 x 5632
    L7  expert B up         tB   x 5632 x 2048
    L8  expert B down       tB   x 2048 x 5632
They run as a dependency chain L1 -> L2 -> ... -> L8.

Communication model (attention replicated: every NPU computes attention on all
1024 tokens, so dispatch needs no traffic -- each NPU picks its own experts'
tokens locally). Only the combine step communicates: after L8, NPU i sends
its experts' outputs (slots_i x HIDDEN x BYTES_PER_ELEM bytes) to each of the
other NPUs, so every NPU ends up with the full 1024 x 2048 layer output.
Sizes differ per NPU, so this is written as point-to-point COMM_SEND_NODE /
COMM_RECV_NODE pairs rather than a built-in collective (which would force
every NPU to send the same amount).

Sends are issued in order dst = i+1, i+2, i+3 (mod N): ASTRA-sim's
HardwareResource allows only one in-flight send per NPU, so node-id order is
send order, and staggering it keeps all NPUs from targeting the same NPU at
the same time.

COMP nodes use duration_micros from <workload>/compute_us.json (written by
scalesim_to_compute_us.py; system config must have
"roofline-enabled": 0, otherwise ASTRA-sim ignores duration_micros). num_ops /
tensor_size are also attached so the same trace can run in roofline mode.

Must run where `chakra` is importable (the astra-sim-latest container):
    docker exec -u $(id -u):$(id -g) astra-sim-latest \
        python3 /app/astra-sim/exp/tpu/workload/gen_et.py moe_4npus_tpuv4
"""
import argparse
import json
from pathlib import Path

from chakra.schema.protobuf.et_def_pb2 import (
    COMP_NODE, COMM_SEND_NODE, COMM_RECV_NODE, GlobalMetadata,
    Node as ChakraNode, AttributeProto as ChakraAttr,
)
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message

SCRIPT_DIR = Path(__file__).parent

TOKENS_PER_EXPERT = [821, 466, 300, 199, 129, 77, 41, 15]
EXPERTS_PER_NPU = 2
ATTN_TOKENS = 1024
HIDDEN = 2048
FFN = 5632
HEAD_DIM = 128
BYTES_PER_ELEM = 2  # bf16
COMBINE_TAG = 1


def gemm_layers(experts: list, tokens: list) -> list:
    """(name, M, N, K) for this NPU, in SCALE-Sim layer order L1..L8."""
    ea, eb = experts
    ta, tb = tokens
    return [
        ("L1_qkv_proj", ATTN_TOKENS, HIDDEN, HIDDEN),
        ("L2_q_kT", ATTN_TOKENS, ATTN_TOKENS, HEAD_DIM),
        ("L3_score_v", ATTN_TOKENS, HEAD_DIM, ATTN_TOKENS),
        ("L4_out_proj", ATTN_TOKENS, HIDDEN, HIDDEN),
        (f"L5_e{ea}_up", ta, FFN, HIDDEN),
        (f"L6_e{ea}_down", ta, HIDDEN, FFN),
        (f"L7_e{eb}_up", tb, FFN, HIDDEN),
        (f"L8_e{eb}_down", tb, HIDDEN, FFN),
    ]


def make_node(next_id: list, name: str, node_type) -> ChakraNode:
    node = ChakraNode()
    node.id = next_id[0]
    next_id[0] += 1
    node.name = name
    node.type = node_type
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    return node


def make_comp_node(next_id: list, name: str, m: int, n: int, k: int,
                   duration_us: int) -> ChakraNode:
    node = make_node(next_id, f"COMP_{name}_{m}x{n}x{k}", COMP_NODE)
    node.duration_micros = duration_us
    node.attr.append(ChakraAttr(name="num_ops", uint64_val=2 * m * n * k))
    node.attr.append(ChakraAttr(
        name="tensor_size", uint64_val=(m * k + k * n + m * n) * BYTES_PER_ELEM))
    return node


def make_p2p_node(next_id: list, node_type, name: str, src: int, dst: int,
                  size: int) -> ChakraNode:
    node = make_node(next_id, name, node_type)
    node.attr.append(ChakraAttr(name="comm_src", int32_val=src))
    node.attr.append(ChakraAttr(name="comm_dst", int32_val=dst))
    node.attr.append(ChakraAttr(name="comm_size", uint64_val=size))
    node.attr.append(ChakraAttr(name="comm_tag", int32_val=COMBINE_TAG))
    return node


def build_npu_trace(npu_id: int, num_npus: int, tokens_per_expert: list,
                    slots_per_npu: list, compute_us: list,
                    out_path: Path) -> None:
    experts = [npu_id * EXPERTS_PER_NPU + e for e in range(EXPERTS_PER_NPU)]
    tokens = [tokens_per_expert[e] for e in experts]
    layers = gemm_layers(experts, tokens)
    assert len(compute_us) == len(layers), (npu_id, len(compute_us))
    next_id = [0]

    with open(out_path, "wb") as g:
        encode_message(g, GlobalMetadata(version="0.0.4"))

        prev = None
        for (name, m, n, k), dur in zip(layers, compute_us):
            comp = make_comp_node(next_id, name, m, n, k, dur)
            if prev is not None:
                comp.data_deps.append(prev.id)
            encode_message(g, comp)
            prev = comp

        # combine: send this NPU's expert outputs to every other NPU
        send_size = slots_per_npu[npu_id] * HIDDEN * BYTES_PER_ELEM
        for offset in range(1, num_npus):
            dst = (npu_id + offset) % num_npus
            send = make_p2p_node(next_id, COMM_SEND_NODE,
                                 f"COMBINE_SEND_{npu_id}to{dst}",
                                 npu_id, dst, send_size)
            send.data_deps.append(prev.id)
            encode_message(g, send)

        # combine: receive every other NPU's expert outputs (pre-posted)
        for offset in range(1, num_npus):
            src = (npu_id - offset) % num_npus
            recv_size = slots_per_npu[src] * HIDDEN * BYTES_PER_ELEM
            recv = make_p2p_node(next_id, COMM_RECV_NODE,
                                 f"COMBINE_RECV_{src}to{npu_id}",
                                 src, npu_id, recv_size)
            encode_message(g, recv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("workload",
                        choices=sorted(p.name for p in SCRIPT_DIR.glob("moe_*") if p.is_dir()),
                        help="workload folder; compute_us.json is read from it")
    parser.add_argument("--out-prefix", type=Path,
                        help="default: <workload>/moe")
    parser.add_argument("--tokens", type=lambda s: [int(x) for x in s.split(",")],
                        default=TOKENS_PER_EXPERT,
                        help="comma-separated tokens per expert, e.g. 256,256,...")
    args = parser.parse_args()

    tokens_per_expert = args.tokens
    assert len(tokens_per_expert) % EXPERTS_PER_NPU == 0
    num_npus = len(tokens_per_expert) // EXPERTS_PER_NPU
    slots_per_npu = [
        sum(tokens_per_expert[i * EXPERTS_PER_NPU:(i + 1) * EXPERTS_PER_NPU])
        for i in range(num_npus)
    ]
    workload_dir = SCRIPT_DIR / args.workload
    out_prefix = args.out_prefix or workload_dir / "moe"
    with open(workload_dir / "compute_us.json") as f:
        compute_us = json.load(f)["npu_compute_us"]

    for npu_id in range(num_npus):
        out_path = Path(f"{out_prefix}.{npu_id}.et")
        build_npu_trace(npu_id, num_npus, tokens_per_expert, slots_per_npu,
                        compute_us[str(npu_id)], out_path)
        send_mb = slots_per_npu[npu_id] * HIDDEN * BYTES_PER_ELEM / 1e6
        print(f"npu {npu_id}: slots={slots_per_npu[npu_id]:5d}  "
              f"combine send {send_mb:.3f} MB x {num_npus - 1}  -> {out_path}")


if __name__ == "__main__":
    main()
