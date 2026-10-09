#!/usr/bin/env python3
"""
GPT-3 communicator-group layout for a 512-NPU ASTRA-sim topology.

Parallelism strategy (astrasim_v2.md Table III): MP=16 (tensor-parallel;
each transformer layer's matmuls are split 16 ways per astrasim_v2.md's own
definition of MP -- "distributes a model evenly across NPUs and communicates
forward activation and input gradients"). D=32 is the number of data-parallel
replicas we chose so that MP * D == 512 (the paper's own DP=64 assumes a
1024-NPU deployment, which does not fit this 512-NPU topology).

NPU numbering convention:
    npu_id = dp_rank * MP_SIZE + mp_rank
    mp_rank = npu_id % MP_SIZE   (0..15, position within one model replica)
    dp_rank = npu_id // MP_SIZE  (0..31, which model replica)

Two kinds of groups are generated:
  - MP groups: the MP_SIZE=16 NPUs that together hold ONE full model
    replica and run the tensor-parallel all-reduce (forward activation /
    backward input-gradient). One group per dp_rank -> 32 groups.
  - DP groups: the D=32 NPUs that hold the SAME shard position across
    replicas and run the gradient all-reduce (backward weight-gradient).
    One group per mp_rank -> 16 groups.

ASTRA-sim reserves comm-group id 0 for the default "all NPUs" group
(CommunicatorGroup::set_id asserts id > 0, astra-sim/system/CommunicatorGroup.cc:44),
so real group ids here start at 1:
  - MP group ids: 1 .. D          (1..32)
  - DP group ids: D+1 .. D+MP_SIZE (33..48)

Each generated .et node's `pg_name` attribute must be set to the matching
group id returned by mp_group_id()/dp_group_id() below -- see the sibling
et-generation script for GPT-3.
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
    """pg_name for this NPU's tensor-parallel (MP) collectives."""
    return MP_GROUP_ID_BASE + dp_rank(npu_id)


def dp_group_id(npu_id: int) -> int:
    """pg_name for this NPU's data-parallel (DP) collectives."""
    return DP_GROUP_ID_BASE + mp_rank(npu_id)


def build_groups() -> dict:
    groups = {}

    # MP groups: one per data-parallel replica, holding one full model copy.
    for r in range(D):
        gid = MP_GROUP_ID_BASE + r
        groups[str(gid)] = [r * MP_SIZE + i for i in range(MP_SIZE)]

    # DP groups: one per model-shard position, holding all replicas of it.
    for r in range(MP_SIZE):
        gid = DP_GROUP_ID_BASE + r
        groups[str(gid)] = [r + i * MP_SIZE for i in range(D)]

    return groups


def main() -> None:
    groups = build_groups()

    # sanity checks
    assert len(groups) == D + MP_SIZE
    for gid_str, members in groups.items():
        gid = int(gid_str)
        expected_size = MP_SIZE if gid < DP_GROUP_ID_BASE else D
        assert len(members) == expected_size, (gid, len(members), expected_size)
        assert len(set(members)) == len(members), f"duplicate member in group {gid}"
    for npu_id in range(NUM_NPUS):
        assert npu_id in groups[str(mp_group_id(npu_id))]
        assert npu_id in groups[str(dp_group_id(npu_id))]

    out_path = Path(__file__).parent / "comm_group_gpt3.json"
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
