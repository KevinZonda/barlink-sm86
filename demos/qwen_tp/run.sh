#!/usr/bin/env bash
# Qwen3.5-27B-GPTQ-Int4 TP decode benchmark.
#
#   bash demos/qwen_tp/run.sh tp1          # single card
#   bash demos/qwen_tp/run.sh barlink      # TP=2 over the barlink PG
#   bash demos/qwen_tp/run.sh nccl         # TP=2 over NCCL (no P2P -> SHM)
#
# Two-process launch templates:
#   barlink: torch_ext/tests/run_pg.sh (blrun, LOCAL_RANK card selection)
#   nccl:    CUDA_VISIBLE_DEVICES per process (stock torch, no caps)
#
# GPU 0 may be shared with other local experiments; each phase waits for a
# free card and retries on failure.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT=demos/qwen_tp/results
MODE="${1:-barlink}"
mkdir -p "$OUT"

export BL_POOL_MB=192
export QWEN_SAVE_ALL_RANKS=1

gpu0_free() {
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 500 ]
}

wait_gpu() {
    until gpu0_free; do sleep 1; done
}

run_tp1() {
    wait_gpu
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python demos/qwen_tp/worker.py --tp 1 \
        --out "$OUT" 2>&1 | tee "$OUT/tp1.log"
    return ${PIPESTATUS[0]}
}

run_tp2() {  # run_tp2 <barlink|nccl> <tag>
    local backend=$1 tag=$2
    local port=$(( (RANDOM % 20000) + 20000 ))
    local sock="/tmp/bl_qwen_$$_$RANDOM.sock"
    wait_gpu
    if [ "$backend" = "barlink" ]; then
        BL_SOCK_PATH="$sock" LOCAL_RANK=0 BL_SKIP_INIT=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True tools/blrun \
            demos/qwen_tp/worker.py --tp 2 --backend barlink --rank 0 \
            --port "$port" --tag "$tag" --out "$OUT" \
            > "$OUT/${tag}_rank0.log" 2>&1 &
        local p0=$!
        BL_SOCK_PATH="$sock" LOCAL_RANK=1 BL_SKIP_INIT=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True tools/blrun \
            demos/qwen_tp/worker.py --tp 2 --backend barlink --rank 1 \
            --port "$port" --tag "$tag" --out "$OUT" \
            > "$OUT/${tag}_rank1.log" 2>&1 &
        local p1=$!
    else
        NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
            CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            .venv/bin/python demos/qwen_tp/worker.py --tp 2 --backend nccl \
            --rank 0 --port "$port" --tag "$tag" --out "$OUT" \
            > "$OUT/${tag}_rank0.log" 2>&1 &
        local p0=$!
        NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
            CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            .venv/bin/python demos/qwen_tp/worker.py --tp 2 --backend nccl \
            --rank 1 --port "$port" --tag "$tag" --out "$OUT" \
            > "$OUT/${tag}_rank1.log" 2>&1 &
        local p1=$!
    fi
    wait "$p0"; local r0=$?
    wait "$p1"; local r1=$?
    rm -f "$sock"
    cat "$OUT/${tag}_rank0.log"; cat "$OUT/${tag}_rank1.log"
    [ "$r0" -eq 0 ] && [ "$r1" -eq 0 ]
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
        sleep 2
    done
    echo "=== $name FAILED after 10 attempts ==="
    return 1
}

case "$MODE" in
    tp1)     retry TP1 run_tp1 || exit 1 ;;
    barlink) retry TP2-barlink run_tp2 barlink tp2 || exit 1 ;;
    nccl)    retry TP2-nccl run_tp2 nccl tp2_nccl || exit 1 ;;
    *) echo "usage: $0 [tp1|barlink|nccl]"; exit 2 ;;
esac
