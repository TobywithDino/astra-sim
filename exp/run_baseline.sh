#!/bin/bash
# Case Study 1 (Impact of Scheduling, baseline scheduler only) -> 1GB All-Reduce
# Uses AstraSim_Analytical_Congestion_Unaware
# for all six, matching the paper's own analytical backend (Sec. IV-C):
#   Time = (LinkLatency x Hops) + MessageSize / LinkBandwidth
# (the paper explicitly states congestion modeling was NOT implemented at
# publication time — "Implementing first-order congestion modeling into the
# analytical backend is our future work", astrasim_v2.md:452-454 — so the
# newer AstraSim_Analytical_Congestion_Aware binary is a different, later
# feature and should not be mixed in here.)
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT="${SCRIPT_DIR:?}/.."
BIN="${REPO_ROOT:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"

# Pick which workload to run: ./run_baseline.sh [gpt3|transformer1t|dlrm|all_reduce]
# (defaults to gpt3 if no argument is given).
WORKLOAD_NAME="${1:-gpt3}"
case "$WORKLOAD_NAME" in
    gpt3)
        WORKLOAD="${SCRIPT_DIR:?}/workload/GPT-3/gpt3_trace"
        COMM_GROUP="${SCRIPT_DIR:?}/workload/GPT-3/comm_group_gpt3.json"
        ;;
    transformer1t)
        WORKLOAD="${SCRIPT_DIR:?}/workload/Transformer-1T/transformer1t_trace"
        COMM_GROUP="${SCRIPT_DIR:?}/workload/Transformer-1T/comm_group_transformer1t.json"
        ;;
    dlrm)
        WORKLOAD="${SCRIPT_DIR:?}/workload/DLRM/dlrm_trace"
        COMM_GROUP="${SCRIPT_DIR:?}/workload/DLRM/comm_group_dlrm.json"
        ;;
    all_reduce)
        WORKLOAD="${SCRIPT_DIR:?}/workload/all_reduce/512npus_1024MB/all_reduce"
        COMM_GROUP="empty"  # plain microbenchmark, no MP/DP groups to scope
        ;;
    *)
        echo "[Error] Unknown workload '${WORKLOAD_NAME}'. Use one of: gpt3, transformer1t, dlrm, all_reduce" >&2
        exit 1
        ;;
esac
echo "[workload] ${WORKLOAD_NAME} -> ${WORKLOAD}" >&2
REMOTE_MEMORY="${SCRIPT_DIR:?}/remote_memory/no_memory_expansion.json"

# topology_file:system_config_file pairs (system config's per-dimension
# array length must match the topology's dimension count)
RUNS=(
    "w-1d-350:w1d"
    "w-1d-500:w1d"
    "w-1d-600:w1d"
    "w-2d-500:w2d"
    "conv-3d:conv3d"
    "conv-4d:conv4d"
)

# Normalization reference: W-1D-600 = 1.0 (highest on-wafer BW variant,
# normally the fastest of the six — matches the paper's Fig. 9(a) convention
# of expressing every bar as a ratio against one baseline instead of raw ns,
# so All-Reduce/DLRM/GPT-3/T-1T results with wildly different absolute
# magnitudes can share one axis).
NORM_TOPO="w-1d-600"

topologies=()
wall_times=()
compute_times=()
exposed_comms=()
slowest_npus=()

for pair in "${RUNS[@]}"; do
    topo="${pair%%:*}"
    sys="${pair##*:}"

    cmd=("${BIN}"
        "--workload-configuration=${WORKLOAD}"
        "--comm-group-configuration=${COMM_GROUP}"
        "--system-configuration=${SCRIPT_DIR}/system/${sys}.json"
        "--remote-memory-configuration=${REMOTE_MEMORY}"
        "--network-configuration=${SCRIPT_DIR}/topology/${topo}.yml")
    echo "[cmd] ${cmd[*]}" >&2

    # A known ASTRA-sim exit-time heap-cleanup crash (malloc_consolidate)
    # can follow a fully-correct run -- every per-NPU Wall time line is
    # already printed by then, so "|| true" stops set -e from killing the
    # script on that nonzero exit; the check below still catches genuinely
    # missing output.
    out=$("${cmd[@]}" 2>&1) || true

    # Take the max Wall time across all NPUs (sys[N]), not just whichever
    # line happened to print last — the collective isn't "done" until its
    # slowest (straggler) NPU finishes. sys_id is kept for transparency.
    max_line=$(echo "$out" | grep "Wall time" \
        | sed -E 's/.*sys\[([0-9]+)\], Wall time: ([0-9]+).*/\1 \2/' \
        | sort -k2 -n | tail -1)

    if [ -z "$max_line" ]; then
        echo "[Error] $topo did not produce a Wall time line. Raw output:" >&2
        echo "$out" >&2
        exit 1
    fi

    sid=$(echo "$max_line" | awk '{print $1}')
    wall_time=$(echo "$max_line" | awk '{print $2}')

    # Pull the SAME NPU's compute (GPU time) and exposed-communication
    # cycles, so wall_time == compute_time + exposed_comm exactly for this
    # row (verified against real output: GPU time + exposed communication
    # == Wall time). "GPU time" is ASTRA-sim's own log label for per-NPU
    # compute time (not a claim about real GPU hardware).
    compute_time=$(echo "$out" | grep "sys\[${sid}\], GPU time:" \
        | sed -E 's/.*GPU time: ([0-9]+).*/\1/')
    exposed_comm=$(echo "$out" | grep "sys\[${sid}\] finished,.*exposed communication" \
        | sed -E 's/.*exposed communication ([0-9]+) cycles.*/\1/')

    # Communication-only microbenchmarks (e.g. all_reduce) have no compute
    # nodes, so ASTRA-sim never logs a "GPU time" line for any NPU -- that's
    # not an error, compute_time is genuinely 0 for these workloads.
    if [ -z "$compute_time" ]; then
        compute_time=0
    fi

    if [ -z "$exposed_comm" ]; then
        echo "[Error] $topo: could not find exposed communication for sys[${sid}]. Raw output:" >&2
        echo "$out" >&2
        exit 1
    fi

    topologies+=("$topo")
    slowest_npus+=("sys[${sid}]")
    wall_times+=("$wall_time")
    compute_times+=("$compute_time")
    exposed_comms+=("$exposed_comm")
done

# Find the normalization baseline's wall time among what we just ran.
baseline_wall_time=""
for i in "${!topologies[@]}"; do
    if [ "${topologies[$i]}" == "$NORM_TOPO" ]; then
        baseline_wall_time="${wall_times[$i]}"
        break
    fi
done
if [ -z "$baseline_wall_time" ]; then
    echo "[Error] Normalization baseline topology '${NORM_TOPO}' was not run." >&2
    exit 1
fi

# compute_ratio and exposed_ratio are both divided by the SAME baseline
# (not by each row's own wall_time), so compute_ratio + exposed_ratio ==
# normalized exactly -- this is what lets them stack into one bar matching
# the paper's Fig. 9(a) convention.
printf "%-10s %-16s %-10s %-14s %-14s %s\n" \
    "topology" "max_wall_time_ns" "normalized" "compute_ratio" "exposed_ratio" "slowest_npu"
for i in "${!topologies[@]}"; do
    normalized=$(awk -v w="${wall_times[$i]}" -v b="$baseline_wall_time" 'BEGIN { printf "%.3f", w / b }')
    compute_ratio=$(awk -v c="${compute_times[$i]}" -v b="$baseline_wall_time" 'BEGIN { printf "%.3f", c / b }')
    exposed_ratio=$(awk -v e="${exposed_comms[$i]}" -v b="$baseline_wall_time" 'BEGIN { printf "%.3f", e / b }')
    printf "%-10s %-16s %-10s %-14s %-14s %s\n" \
        "${topologies[$i]}" "${wall_times[$i]}" "$normalized" "$compute_ratio" "$exposed_ratio" "${slowest_npus[$i]}"
done
