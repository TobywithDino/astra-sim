#!/bin/bash
# ./run_analytical.sh TOPOLOGY ROOFLINE BINARY SWAP_EXPERT
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT="${SCRIPT_DIR:?}/../.."
BIN_NAME="${3:-aware}"
case "$BIN_NAME" in
    (unaware)
        BIN="${REPO_ROOT:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
        ;;
    (aware)
        BIN="${REPO_ROOT:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
        ;;
    (*)
        echo "[Error] Unknown network backend '${BIN_NAME}'." >&2
        exit 1
        ;;
esac

TOPOLOGY="${1:-tpuv4}"
SYSTEM="$TOPOLOGY"
case "${2:-}" in
    ("")
        ;;
    (roofline)
        SYSTEM="${SYSTEM}_roofline"
        ;;
    (*)
        echo "[Error] Unknown '${2}' (expected 'roofline' or '(empty)')." >&2
        exit 1
        ;;
esac
if [ ! -f "${SCRIPT_DIR}/system/${SYSTEM}.json" ]; then
    echo "[Error] System config '${SCRIPT_DIR}/system/${SYSTEM}.json' not found." >&2
    exit 1
fi

case "$TOPOLOGY" in
    (tpuv4)
        WORKLOAD="${SCRIPT_DIR:?}/workload/moe_4npus_tpuv4/moe"
        COMM_GROUP="empty"
        ;;
    (tpuv5p)
        WORKLOAD="${SCRIPT_DIR:?}/workload/moe_4npus_tpuv5p/moe"
        COMM_GROUP="empty"
        ;;
    (*)
        echo "[Error] No workload corresponds to this topology'${TOPOLOGY}'." >&2
        exit 1
        ;;
esac


mkdir -p "${SCRIPT_DIR}/output"
OUTPUT="${SCRIPT_DIR}/output/${BIN_NAME}_${SYSTEM}"

case "${4:-}" in
    ("")
        OUTPUT="${OUTPUT}.log"
        ;;
    (swap)
        WORKLOAD="${WORKLOAD:?}_swap"
        OUTPUT="${OUTPUT}_swap.log"
        ;;
    (*)
        echo "[Error] Unknown '${4}' (expected 'swap' or '(empty)')." >&2
        exit 1
        ;;
esac

cmd=("${BIN}"
    "--workload-configuration=${WORKLOAD}"
    "--comm-group-configuration=${COMM_GROUP}"
    "--system-configuration=${SCRIPT_DIR}/system/${SYSTEM}.json"
    "--remote-memory-configuration=${SCRIPT_DIR}/remote_memory/no_memory_expansion.json"
    "--network-configuration=${SCRIPT_DIR}/topology/${TOPOLOGY}.yml")
echo "[cmd] ${cmd[*]}" >&2
echo "[output] ${OUTPUT}" >&2
# ASTRA-sim writes log/log.log relative to the cwd
cd "${SCRIPT_DIR}"
"${cmd[@]}" > "$OUTPUT" 2>&1
