#!/usr/bin/env python3
"""
Generate the Chakra Text-converter layer-description file for GPT-3 175B.

Output format matches extern/graph_frontend/chakra/src/converter/text_converter.py's
Layer class:
    name  _  fwd_comp_time fwd_comm_type fwd_comm_size \
              bwd_ig_comp_time bwd_ig_comm_type bwd_ig_comm_size \
              bwd_wg_comp_time bwd_wg_comm_type bwd_wg_comm_size  bwd_wg_update_time
(col[1] and col[11] are not read by the parser; kept as "0" placeholders so the
column count matches the parser's fixed indexing.)

Architecture (GPT-3 175B, public spec): 96 transformer layers, hidden=12288.
Each layer is split into two rows (attention, mlp), matching Megatron-LM's
tensor-parallel design: each sub-layer does a column-parallel matmul (no
comm) then a row-parallel matmul whose partial sums are combined by an
MP all-reduce -- see the "MP 的實際機制" explanation earlier in this session.

Parallelism (astrasim_v2.md Table III + this session's 512-NPU adjustment):
  MP_SIZE = 16   (unchanged from the paper -- tensor-parallel group size)
  D       = 32   (data-parallel replicas; paper's DP=64 assumes 1024 NPUs,
                  which doesn't fit this 512-NPU topology, so D = 512/MP_SIZE)

Free training-setup parameters (NOT given by the paper -- flagged assumptions):
  SEQ_LEN      = 2048  (GPT-3's context length)
  MICRO_BATCH  = 1     (per data-parallel replica)
  DTYPE_BYTES  = 2     (fp16/bf16 for both activations and gradients)

Per-layer formulas:
  fwd_comp_time (us)   = (2 * layer_params * tokens / MP_SIZE) / PEAK_FLOPS * 1e6
                         (per-NPU FLOPs after MP sharding, divided by peak throughput)
  bwd_ig_comp_time,
  bwd_wg_comp_time      = same as fwd_comp_time (rule of thumb: backward FLOPs
                          ~= 2x forward, split evenly between input-grad and
                          weight-grad passes)
  fwd_comm_size,
  bwd_ig_comm_size (MP all-reduce) = SEQ_LEN * MICRO_BATCH * HIDDEN * DTYPE_BYTES
                                      (this layer's activation tensor; independent
                                      of MP_SIZE -- see this session's MP mechanics
                                      explanation)
  bwd_wg_comm_size (DP all-reduce) = layer_params / MP_SIZE * DTYPE_BYTES
                                      (each NPU's shard of this layer's gradient)

NOTE: this file is the Text-converter INPUT. Running it through the vanilla
`chakra_converter Text` CLI will NOT scope these all-reduces to the MP/DP
groups defined in comm_groups.py -- text_converter.py never sets a node's
pg_name, so every collective defaults to the whole 512-NPU group (see this
session's "Text converter 為什麼無法達到重現平行化" explanation). Turning
this into correctly-grouped .et files needs a follow-up conversion step
that reuses mp_group_id()/dp_group_id() from comm_groups.py.
"""
from pathlib import Path

NUM_LAYERS = 96
HIDDEN = 12288
MP_SIZE = 16
D = 32
NUM_NPUS = MP_SIZE * D

SEQ_LEN = 2048
MICRO_BATCH = 1
DTYPE_BYTES = 2
PEAK_FLOPS = 234e12  # 234 TFLOPS, A100 measured value (see EXP.md)

TOKENS = SEQ_LEN * MICRO_BATCH
ATTN_PARAMS = 4 * HIDDEN**2
MLP_PARAMS = 8 * HIDDEN**2
ACT_BYTES = SEQ_LEN * MICRO_BATCH * HIDDEN * DTYPE_BYTES


def comp_time_us(layer_params: int) -> int:
    fwd_flops_per_npu = 2 * layer_params * TOKENS / MP_SIZE
    return round(fwd_flops_per_npu / PEAK_FLOPS * 1e6)


def grad_bytes(layer_params: int) -> int:
    return round(layer_params / MP_SIZE * DTYPE_BYTES)


def layer_line(name: str, comp_time: int, comm_size: int, grad_size: int) -> str:
    return " ".join(str(x) for x in [
        name, 0,
        comp_time, "ALLREDUCE", comm_size,      # forward
        comp_time, "ALLREDUCE", comm_size,      # backward input-gradient
        comp_time, "ALLREDUCE", grad_size, 0,   # backward weight-gradient
    ])


def main() -> None:
    attn_comp = comp_time_us(ATTN_PARAMS)
    mlp_comp = comp_time_us(MLP_PARAMS)
    attn_grad = grad_bytes(ATTN_PARAMS)
    mlp_grad = grad_bytes(MLP_PARAMS)

    lines = ["HYBRID_MODEL_DATA", str(NUM_LAYERS * 2)]
    for layer_idx in range(NUM_LAYERS):
        lines.append(layer_line(f"L{layer_idx}_attn", attn_comp, ACT_BYTES, attn_grad))
        lines.append(layer_line(f"L{layer_idx}_mlp", mlp_comp, ACT_BYTES, mlp_grad))

    out_path = Path(__file__).parent / "gpt3_layers.txt"
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {NUM_LAYERS * 2} layer rows to {out_path}")
    print(f"  attention: comp={attn_comp}us  mp_comm={ACT_BYTES}B  dp_comm={attn_grad}B")
    print(f"  mlp:       comp={mlp_comp}us  mp_comm={ACT_BYTES}B  dp_comm={mlp_grad}B")
    print(f"  assumes MP={MP_SIZE}, D={D} ({NUM_NPUS} NPUs), "
          f"seq_len={SEQ_LEN}, micro_batch={MICRO_BATCH}, dtype_bytes={DTYPE_BYTES}")


if __name__ == "__main__":
    main()
