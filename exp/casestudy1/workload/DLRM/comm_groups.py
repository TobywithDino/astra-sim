#!/usr/bin/env python3
"""
DLRM communicator-group layout for a 512-NPU ASTRA-sim topology.

Same grid mechanics as exp/workload/GPT-3/comm_groups.py -- see that
file's docstring for the full explanation. The MEANING of MP/DP differs
from GPT-3/Transformer-1T though:

  - GPT-3/Transformer-1T's MP: tensor-parallel -- MP members collaborate
    on the SAME tensor via all-reduce (every layer, both directions).
  - DLRM's MP: embedding-table model-parallel -- MP members hold DISJOINT
    embedding tables and exchange results via all-to-all (not all-reduce).
    DLRM's DP: full data-parallel replicas of the MLP portion (weight
    gradients synced via all-reduce), same concept as the other workloads.

astrasim_v2.md Table III reports DLRM as MP=1,024, DP=1,024 -- both
already exceed this 512-NPU topology on their own (unlike GPT-3/
Transformer-1T where only MP*D exceeded 512 while MP alone fit). Since
DLRM's MP has no hardware-bound justification to hold fixed (unlike
Megatron tensor-parallel groups, which are usually sized to an intra-node
interconnect), both dimensions were scaled down together, keeping them
close to their original 1:1 ratio: MP=16, D=32 (512 = 16*32), a decision
confirmed with the user rather than derived from the paper.

NPU numbering: npu_id = dp_rank * MP_SIZE + mp_rank
  mp_rank = npu_id % MP_SIZE   (0..15, which embedding-table shard)
  dp_rank = npu_id // MP_SIZE  (0..31, which data-parallel replica)

  - MP groups (all-to-all scope): 32 groups, ids 1..32, 16 members each
  - DP groups (all-reduce scope): 16 groups, ids 33..48, 32 members each

Group id 0 is reserved by ASTRA-sim for the default "all NPUs" group
(CommunicatorGroup::set_id asserts id > 0), so real group ids start at 1.
"""
import json
from pathlib import Path

MP_SIZE = 16
D = 32
NUM_NPUS = MP_SIZE * D

MP_GROUP_ID_BASE = 1        # ids 1..D
DP_GROUP_ID_BASE = 1 + D    # ids (D+1)..(D+MP_SIZE)


def mp_rank(npu_id: int) -> int:
    return npu_id % MP_SIZE


def dp_rank(npu_id: int) -> int:
    return npu_id // MP_SIZE


def mp_group_id(npu_id: int) -> int:
    """pg_name for this NPU's embedding all-to-all."""
    return MP_GROUP_ID_BASE + dp_rank(npu_id)


def dp_group_id(npu_id: int) -> int:
    """pg_name for this NPU's MLP gradient all-reduce."""
    return DP_GROUP_ID_BASE + mp_rank(npu_id)


def build_groups() -> dict:
    groups = {}
    for r in range(D):
        gid = MP_GROUP_ID_BASE + r
        groups[str(gid)] = [r * MP_SIZE + i for i in range(MP_SIZE)]
    for r in range(MP_SIZE):
        gid = DP_GROUP_ID_BASE + r
        groups[str(gid)] = [r + i * MP_SIZE for i in range(D)]
    return groups


def main() -> None:
    groups = build_groups()

    assert len(groups) == D + MP_SIZE
    for gid_str, members in groups.items():
        gid = int(gid_str)
        expected_size = MP_SIZE if gid < DP_GROUP_ID_BASE else D
        assert len(members) == expected_size, (gid, len(members), expected_size)
        assert len(set(members)) == len(members), f"duplicate member in group {gid}"
    for npu_id in range(NUM_NPUS):
        assert npu_id in groups[str(mp_group_id(npu_id))]
        assert npu_id in groups[str(dp_group_id(npu_id))]

    out_path = Path(__file__).parent / "comm_group_dlrm.json"
    with open(out_path, "w") as f:
        json.dump(groups, f, indent=2)

    print(f"Wrote {len(groups)} comm groups to {out_path}")
    print(f"  MP groups: ids {MP_GROUP_ID_BASE}..{MP_GROUP_ID_BASE + D - 1} "
          f"({D} groups x {MP_SIZE} members each)")
    print(f"  DP groups: ids {DP_GROUP_ID_BASE}..{DP_GROUP_ID_BASE + MP_SIZE - 1} "
          f"({MP_SIZE} groups x {D} members each)")
    print(f"  Total NPUs covered: {NUM_NPUS}")


if __name__ == "__main__":
    main()
