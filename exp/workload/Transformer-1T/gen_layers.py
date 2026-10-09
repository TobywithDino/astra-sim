#!/usr/bin/env python3
"""
Generate the Chakra Text-converter layer-description file for Transformer-1T.

Same method as exp/workload/GPT-3/gen_layers.py -- see that file's docstring
for the full formula derivation (MP tensor-parallel mechanics, per-layer
attention/MLP split, comm_size / comp_time formulas). Only the architecture
numbers differ here.

Architecture (ASSUMPTION -- astrasim_v2.md Table III gives only #Params=1T,
MP=128, DP=8, no hidden size / layer count. Using the hidden=25600,
layers=128 configuration commonly cited in Megatron-LM scaling studies for
~1T-parameter models: 12 * 25600^2 * 128 =~ 1.007e12 params, matching
Table III's "1T". This is NOT independently confirmed to be what the
ASTRA-sim authors used -- flagged here as an assumption, not a fact):
  NUM_LAYERS = 128
  HIDDEN     = 25600

Parallelism (astrasim_v2.md Table III + this session's 512-NPU adjustment,
same rule as GPT-3's):
  MP_SIZE = 128  (unchanged from the paper)
  D       = 4    (512 / MP_SIZE; paper's DP=8 assumes 1024 NPUs)

Free training-setup parameters (same defaults used for GPT-3):
  SEQ_LEN      = 2048
  MICRO_BATCH  = 1
  DTYPE_BYTES  = 2  (fp16/bf16)
"""
from pathlib import Path

NUM_LAYERS = 128
HIDDEN = 25600
MP_SIZE = 128
D = 4
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

    total_params = (ATTN_PARAMS + MLP_PARAMS) * NUM_LAYERS

    lines = ["HYBRID_MODEL_DATA", str(NUM_LAYERS * 2)]
    for layer_idx in range(NUM_LAYERS):
        lines.append(layer_line(f"L{layer_idx}_attn", attn_comp, ACT_BYTES, attn_grad))
        lines.append(layer_line(f"L{layer_idx}_mlp", mlp_comp, ACT_BYTES, mlp_grad))

    out_path = Path(__file__).parent / "transformer1t_layers.txt"
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {NUM_LAYERS * 2} layer rows to {out_path}")
    print(f"  total params (sanity check): {total_params:.4e} (~= 1T)")
    print(f"  attention: comp={attn_comp}us  mp_comm={ACT_BYTES}B  dp_comm={attn_grad}B")
    print(f"  mlp:       comp={mlp_comp}us  mp_comm={ACT_BYTES}B  dp_comm={mlp_grad}B")
    print(f"  assumes MP={MP_SIZE}, D={D} ({NUM_NPUS} NPUs), "
          f"seq_len={SEQ_LEN}, micro_batch={MICRO_BATCH}, dtype_bytes={DTYPE_BYTES}")


if __name__ == "__main__":
    main()
