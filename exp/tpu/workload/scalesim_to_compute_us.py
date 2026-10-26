#!/usr/bin/env python3
"""
Convert SCALE-Sim COMPUTE_REPORT.csv files (one per NPU) into compute_us.json
and compute_us_by_expert.json for gen_et.py.

The workload folder argument (moe_4npus_tpuv4, moe_4npus_tpuv5p, ...) picks
both ends: the chip suffix after the last "_" selects the SCALE-Sim runs
scalesim_result/qwen-1.5-dist-<chip>-ep{npu}, and the outputs go to
<workload>/compute_us.json and <workload>/compute_us_by_expert.json.

Uses the "Total Cycles" column: it is the whole layer's runtime and already
includes "Stall Cycles" (cycles the array waits on memory). Check: every row
satisfies Overall Util % = M*N*K / (num_PEs * Total Cycles).

    time_us = Total Cycles / clock_MHz        (TPUv4: 1050 MHz, --clock-mhz)

Chakra's duration_micros is an integer, so each layer is rounded to the
nearest microsecond (minimum 1).

Layer layout per NPU (SCALE-Sim order L1..L8):
    L1..L4  attention (identical on every NPU)
    L5..L6  first local expert  (EXPERT_TOKENS[npu][0] tokens)
    L7..L8  second local expert (EXPERT_TOKENS[npu][1] tokens)

compute_us_by_expert.json keys each expert's [L_a, L_b] times by the number
of tokens routed to it.

Runs on the host (no chakra needed):
    python3 exp/tpu/workload/scalesim_to_compute_us.py moe_4npus_tpuv4
"""
import argparse
import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[2]
NUM_NPUS = 4
NUM_LAYERS = 8
NUM_ATTN_LAYERS = 4
LAYERS_PER_EXPERT = 2

# Tokens routed to each local expert, per NPU (same order as the SCALE-Sim
# topology: first expert = L5..L6, second expert = L7..L8).
EXPERT_TOKENS = {
    0: [821, 466],
    1: [300, 199],
    2: [129, 77],
    3: [41, 15],
}


def read_total_cycles(report: Path) -> list:
    with open(report) as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    header = [h.strip() for h in rows[0]]
    col = header.index("Total Cycles")
    return [int(r[col]) for r in rows[1:] if r and r[0].strip()]


def dump_json(obj, path: Path) -> None:
    """json.dump with indent=4, but keep lists of numbers on one line."""
    text = json.dumps(obj, indent=4)
    text = re.sub(r"\[\s*([-\d.,\s]+?)\s*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
                  text)
    with open(path, "w") as f:
        f.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("workload",
                        choices=sorted(p.name for p in SCRIPT_DIR.glob("moe_*") if p.is_dir()),
                        help="workload folder; the json files are written into it")
    parser.add_argument("--scalesim-dir", type=Path,
                        default=REPO_ROOT / "scalesim_result")
    parser.add_argument("--clock-mhz", type=float, default=1050.0)
    args = parser.parse_args()

    chip = args.workload.rsplit("_", 1)[1]
    run_name = f"qwen-1.5-dist-{chip}-ep{{npu}}"
    out_dir = SCRIPT_DIR / args.workload
    out_path = out_dir / "compute_us.json"
    out_expert_path = out_dir / "compute_us_by_expert.json"
    source = f"SCALE-Sim COMPUTE_REPORT.csv Total Cycles / {args.clock_mhz:g} MHz"

    npu_compute_us = {}
    for npu in range(NUM_NPUS):
        report = args.scalesim_dir / run_name.format(npu=npu) / "COMPUTE_REPORT.csv"
        cycles = read_total_cycles(report)
        assert len(cycles) == NUM_LAYERS, (report, len(cycles))
        npu_compute_us[str(npu)] = [max(1, round(c / args.clock_mhz)) for c in cycles]
        print(f"npu {npu}: {npu_compute_us[str(npu)]}  "
              f"sum={sum(npu_compute_us[str(npu)])} us  <- {report}")

    # ---- compute_us.json (unchanged) ----
    out = {
        "unit": "microseconds (integer), one list per NPU in SCALE-Sim layer order L1..L8",
        "source": source,
        "npu_compute_us": npu_compute_us,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)
        f.write("\n")
    print(f"wrote {out_path}")

    # ---- compute_us_by_expert.json ----
    attention_us = npu_compute_us["0"][:NUM_ATTN_LAYERS]
    for npu in range(1, NUM_NPUS):
        other = npu_compute_us[str(npu)][:NUM_ATTN_LAYERS]
        if other != attention_us:
            print(f"warning: npu {npu} attention {other} != npu 0 {attention_us}; "
                  f"using npu 0")

    expert_us_by_tokens = {}
    for npu in range(NUM_NPUS):
        expert_layers = npu_compute_us[str(npu)][NUM_ATTN_LAYERS:]
        tokens_list = EXPERT_TOKENS[npu]
        assert len(expert_layers) == len(tokens_list) * LAYERS_PER_EXPERT, \
            (npu, expert_layers, tokens_list)
        for i, tokens in enumerate(tokens_list):
            key = str(tokens)
            assert key not in expert_us_by_tokens, f"duplicate token count {tokens}"
            expert_us_by_tokens[key] = \
                expert_layers[i * LAYERS_PER_EXPERT:(i + 1) * LAYERS_PER_EXPERT]

    out_expert = {
        "unit": "microseconds (integer)",
        "source": source,
        "attention_us": attention_us,
        "expert_us_by_tokens": expert_us_by_tokens,
    }
    dump_json(out_expert, out_expert_path)
    print(f"wrote {out_expert_path}")


if __name__ == "__main__":
    main()