#!/usr/bin/env python3
"""
Transformer-1T communicator-group layout for a 512-NPU ASTRA-sim topology.

Same method as exp/workload/GPT-3/comm_groups.py -- see that file's
docstring for the full MP/DP grid explanation. Only the sizes differ:

  MP_SIZE = 128  (astrasim_v2.md Table III, unchanged -- tensor-parallel
                  group size, independent of total NPU count)
  D       = 4    (data-parallel replicas; paper's DP=8 assumes 128*8=1024
                  NPUs, which doesn't fit this 512-NPU topology, so
                  D = 512 / MP_SIZE, same rule used for GPT-3)

NPU numbering: npu_id = dp_rank * MP_SIZE + mp_rank
  mp_rank = npu_id % MP_SIZE   (0..127, position within one model replica)
  dp_rank = npu_id // MP_SIZE  (0..3, which model replica)

  - MP groups: 4 groups (one per dp_rank), each with the 128 NPUs that
    hold one full model replica -> ids 1..4
  - DP groups: 128 groups (one per mp_rank), each with the 4 NPUs that
    hold the same shard position across replicas -> ids 5..132

Group id 0 is reserved by ASTRA-sim for the default "all NPUs" group
(CommunicatorGroup::set_id asserts id > 0), so real group ids start at 1.
"""
import json
from pathlib import Path

MP_SIZE = 128
D = 4
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

    out_path = Path(__file__).parent / "comm_group_transformer1t.json"
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
