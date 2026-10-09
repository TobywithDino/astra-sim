#!/usr/bin/env python3
"""
Generate 512 per-NPU Chakra .et files for DLRM directly (no intermediate
text-converter file, unlike GPT-3/Transformer-1T) -- DLRM's forward/backward
comm pattern is asymmetric (all-to-all only around the embedding stage,
all-reduce only around the MLP stage) and doesn't fit the generic
"every layer has fwd/bwd_ig/bwd_wg comm" HYBRID_MODEL_DATA shape used for
the other two workloads, so the per-iteration graph is built by hand here.

Modeled structure per NPU, per training iteration (confirmed simplification
-- see this session's "DLRM 真實架構的角色分工問題" discussion):
  1. embed_comp        -- local embedding-table lookup (memory gather, not
                           FLOPs-bound; nominal small comp time)
  2. embed_fwd_alltoall -- redistribute looked-up embeddings so each NPU
                           gets the full vector for its own batch share
                           (MP-scoped: this NPU's 16-member embedding-shard
                           group, see comm_groups.mp_group_id)
  3. mlp_fwd_comp       -- MLP forward (bottom+top MLP, 57M params, Table
                           III's "57M (MLP layers)"; NOT divided by MP,
                           since DLRM's MP only shards embedding tables,
                           not the MLP -- every NPU replicates the full MLP)
  4. mlp_bwd_comp       -- MLP backward (input+weight gradient combined
                           into one node, unlike GPT-3/Transformer-1T's
                           separate ig/wg split, since DLRM here has no
                           multi-layer chain to stagger)
  5. mlp_bwd_allreduce  -- MLP weight-gradient sync (DP-scoped: this NPU's
                           32-member data-parallel group, see
                           comm_groups.dp_group_id)
  6. embed_bwd_alltoall -- route gradients back to the NPUs owning each
                           table (MP-scoped, same group as step 2); runs
                           concurrently with step 5, both depending on
                           step 4
  7. embed_bwd_comp     -- local embedding-table gradient update (no
                           cross-NPU sync needed -- tables are disjoint,
                           not replicated, so there is nothing to
                           all-reduce here)

Free parameters (Table III gives only "#Params: 57M (MLP layers)" -- no
embedding-table count/size, so these are flagged assumptions, not facts):
  BATCH_PER_NPU        = 2048  (DLRM-style per-NPU sample batch, not a
                                seq_len*micro_batch product like the
                                transformer workloads)
  NUM_SPARSE_FEATURES  = 26    (MLPerf DLRM / Criteo Terabyte convention)
  EMBED_DIM            = 128   (MLPerf DLRM convention)
  EMBED_COMP_US        = 20    (nominal -- embedding lookup is a memory
                                gather, not meaningfully FLOPs-bound)
  DTYPE_BYTES          = 2     (fp16/bf16, same as the other two workloads)

Parallelism (this session's confirmed 512-NPU adjustment -- Table III's
MP=1,024 already exceeds 512 NPUs by itself, unlike GPT-3/Transformer-1T
where only MP*DP exceeded 512; both dimensions were scaled down together):
  MP_SIZE = 16, D = 32  (16*32 = 512)

Must run where `chakra`/`protobuf` are importable -- inside the
astra-sim-latest container, with this session's protobuf gencode/runtime
fix already applied (see exp/workload/GPT-3/gen_et.py's docstring).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comm_groups import mp_group_id, dp_group_id, NUM_NPUS  # noqa: E402

from chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
    ALL_REDUCE, ALL_TO_ALL, COMP_NODE, COMM_COLL_NODE,
    GlobalMetadata, Node as ChakraNode, AttributeProto as ChakraAttr,
)
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # noqa: E402

OUTPUT_PREFIX = Path(__file__).parent / "dlrm_trace"

MP_SIZE = 16
MLP_PARAMS = 57_000_000
PEAK_FLOPS = 234e12  # 234 TFLOPS, A100 measured value (see EXP.md)
DTYPE_BYTES = 2

BATCH_PER_NPU = 2048
NUM_SPARSE_FEATURES = 26
EMBED_DIM = 128
EMBED_COMP_US = 20

MLP_FWD_FLOPS = 2 * MLP_PARAMS * BATCH_PER_NPU
MLP_FWD_US = round(MLP_FWD_FLOPS / PEAK_FLOPS * 1e6)
MLP_BWD_US = MLP_FWD_US * 2  # backward ~= 2x forward, same rule of thumb as GPT-3/Transformer-1T
MLP_GRAD_BYTES = round(MLP_PARAMS * DTYPE_BYTES)  # NOT divided by MP -- MLP isn't MP-sharded in DLRM
ALLTOALL_BYTES = round(BATCH_PER_NPU * NUM_SPARSE_FEATURES * EMBED_DIM * DTYPE_BYTES)


def make_node(next_id: list, name: str, node_type) -> ChakraNode:
    node = ChakraNode()
    node.id = next_id[0]
    next_id[0] += 1
    node.name = name
    node.type = node_type
    return node


def make_comp_node(next_id: list, name: str, comp_time: int) -> ChakraNode:
    node = make_node(next_id, f"COMP_{name}", COMP_NODE)
    node.duration_micros = comp_time
    return node


def make_comm_node(next_id: list, name: str, comm_type: int, comm_size: int, pg_name: int) -> ChakraNode:
    node = make_node(next_id, f"COMM_{name}", COMM_COLL_NODE)
    node.attr.append(ChakraAttr(name="comm_type", int64_val=comm_type))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
    node.attr.append(ChakraAttr(name="pg_name", string_val=str(pg_name)))
    return node


def add_parent(child: ChakraNode, parent: ChakraNode) -> None:
    child.data_deps.append(parent.id)


def build_npu_trace(npu_id: int, out_path: Path) -> None:
    mp_gid = mp_group_id(npu_id)
    dp_gid = dp_group_id(npu_id)
    next_id = [0]

    with open(out_path, "wb") as g:
        encode_message(g, GlobalMetadata(version="0.0.4"))

        embed_comp = make_comp_node(next_id, "embed_comp", EMBED_COMP_US)
        encode_message(g, embed_comp)

        embed_fwd_a2a = make_comm_node(next_id, "embed_fwd_a2a", ALL_TO_ALL, ALLTOALL_BYTES, mp_gid)
        add_parent(embed_fwd_a2a, embed_comp)
        encode_message(g, embed_fwd_a2a)

        mlp_fwd_comp = make_comp_node(next_id, "mlp_fwd_comp", MLP_FWD_US)
        add_parent(mlp_fwd_comp, embed_fwd_a2a)
        encode_message(g, mlp_fwd_comp)

        mlp_bwd_comp = make_comp_node(next_id, "mlp_bwd_comp", MLP_BWD_US)
        add_parent(mlp_bwd_comp, mlp_fwd_comp)
        encode_message(g, mlp_bwd_comp)

        mlp_bwd_allreduce = make_comm_node(next_id, "mlp_bwd_allreduce", ALL_REDUCE, MLP_GRAD_BYTES, dp_gid)
        add_parent(mlp_bwd_allreduce, mlp_bwd_comp)
        encode_message(g, mlp_bwd_allreduce)

        embed_bwd_a2a = make_comm_node(next_id, "embed_bwd_a2a", ALL_TO_ALL, ALLTOALL_BYTES, mp_gid)
        add_parent(embed_bwd_a2a, mlp_bwd_comp)
        encode_message(g, embed_bwd_a2a)

        embed_bwd_comp = make_comp_node(next_id, "embed_bwd_comp", EMBED_COMP_US)
        add_parent(embed_bwd_comp, embed_bwd_a2a)
        encode_message(g, embed_bwd_comp)


def main() -> None:
    for npu_id in range(NUM_NPUS):
        out_path = Path(f"{OUTPUT_PREFIX}.{npu_id}.et")
        build_npu_trace(npu_id, out_path)
        if npu_id % 64 == 0:
            print(f"  ...generated npu {npu_id}/{NUM_NPUS}")
    print(f"Wrote {NUM_NPUS} .et files to {OUTPUT_PREFIX}.*.et")
    print(f"  mlp_fwd_us={MLP_FWD_US} mlp_bwd_us={MLP_BWD_US} "
          f"mlp_grad_bytes={MLP_GRAD_BYTES} alltoall_bytes={ALLTOALL_BYTES}")


if __name__ == "__main__":
    main()
