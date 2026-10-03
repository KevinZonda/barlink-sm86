#!/usr/bin/env bash
# Qwen3.8-27B INT8 W8A16 TP=2 decode benchmark: barlink PG vs NCCL.
#
#   bash demos/qwen_tp/qwen38_int8/run.sh barlink
#   bash demos/qwen_tp/qwen38_int8/run.sh nccl
#   bash demos/qwen_tp/qwen38_int8/run.sh both     # barlink -> nccl -> compare
#   bash demos/qwen_tp/qwen38_int8/run.sh compare
#
# Two-process launch templates:
#   barlink: tools/blrun (LOCAL_RANK card selection, BL_SOCK_PATH)
#   nccl:    CUDA_VISIBLE_DEVICES per process (stock torch, no caps)
#
# GPU 0 may be shared with other local experiments; each phase waits for a
# free card and retries on failure.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
HERE=demos/qwen_tp/qwen38_int8
OUT="$HERE/results"
MODE="${1:-both}"
mkdir -p "$OUT"

export BL_POOL_MB=192

gpu0_free() {
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 500 ]
}

wait_gpu() {
    until gpu0_free; do sleep 5; done
}

run_one() {  # run_one <barlink|nccl> <tag> [extra worker args...]
    local backend=$1 tag=$2; shift 2
    local port=$(( (RANDOM % 20000) + 20000 ))
    local sock="/tmp/bl_qwen38_$$_$RANDOM.sock"
    wait_gpu
    # NB: no `| tee` on the workers -- $! would be tee's pid and the real
    # exit code would be lost (this once marked a crashed phase as OK).
    # Logs stream to file; the tail is echoed afterwards.
    if [ "$backend" = "barlink" ]; then
        BL_SOCK_PATH="$sock" LOCAL_RANK=0 BL_SKIP_INIT=1 \
            tools/blrun "$HERE/worker.py" --backend barlink --rank 0 \
            --port "$port" --tag "$tag" --out "$OUT" "$@" \
            > "$OUT/${tag}_rank0.log" 2>&1 &
        local p0=$!
        BL_SOCK_PATH="$sock" LOCAL_RANK=1 BL_SKIP_INIT=1 \
            tools/blrun "$HERE/worker.py" --backend barlink --rank 1 \
            --port "$port" --tag "$tag" --out "$OUT" "$@" \
            > "$OUT/${tag}_rank1.log" 2>&1 &
        local p1=$!
    else
        NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
            CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
            .venv/bin/python "$HERE/worker.py" --backend nccl --rank 0 \
            --port "$port" --tag "$tag" --out "$OUT" "$@" \
            > "$OUT/${tag}_rank0.log" 2>&1 &
        local p0=$!
        NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
            CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
            .venv/bin/python "$HERE/worker.py" --backend nccl --rank 1 \
            --port "$port" --tag "$tag" --out "$OUT" "$@" \
            > "$OUT/${tag}_rank1.log" 2>&1 &
        local p1=$!
    fi
    wait "$p0"; local r0=$?
    wait "$p1"; local r1=$?
    rm -f "$sock"
    tail -n 4 "$OUT/${tag}_rank0.log"
    tail -n 4 "$OUT/${tag}_rank1.log"
    [ "$r0" -eq 0 ] && [ "$r1" -eq 0 ]
}

do_compare() {
    .venv/bin/python "$HERE/compare.py" --dir "$OUT" \
        --tag-a barlink --tag-b nccl
}

retry() {  # retry <phase-name> <fn...>
    local name=$1; shift
    for attempt in $(seq 1 10); do
        echo "=== $name (attempt $attempt) $(date +%T) ==="
        if "$@"; then
            echo "=== $name OK ==="
            return 0
        fi
        echo "=== $name failed, retrying ==="
        sleep 5
    done
    echo "=== $name FAILED after 10 attempts ==="
    return 1
}

case "$MODE" in
    barlink)  retry TP2-barlink run_one barlink barlink || exit 1 ;;
    nccl)     retry TP2-nccl run_one nccl nccl || exit 1 ;;
    nocomm-barlink) retry TP2-barlink-nocomm \
                        run_one barlink barlink_nc --no-comm || exit 1 ;;
    nocomm-nccl)    retry TP2-nccl-nocomm \
                        run_one nccl nccl_nc --no-comm || exit 1 ;;
    all)
        retry TP2-barlink run_one barlink barlink || exit 1
        retry TP2-nccl run_one nccl nccl || exit 1
        retry TP2-barlink-nocomm run_one barlink barlink_nc --no-comm || exit 1
        retry TP2-nccl-nocomm run_one nccl nccl_nc --no-comm || exit 1
        do_compare || exit 1
        ;;
    both)
        retry TP2-barlink run_one barlink barlink || exit 1
        retry TP2-nccl run_one nccl nccl || exit 1
        do_compare || exit 1
        ;;
    compare) do_compare || exit 1 ;;
    *) echo "usage: $0 [barlink|nccl|nocomm-barlink|nocomm-nccl|both|all|compare]";
       exit 2 ;;
esac
