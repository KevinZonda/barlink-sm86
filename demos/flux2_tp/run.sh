#!/usr/bin/env bash
# FLUX.2 Klein 9B DiT TP benchmark: TP=1 baseline + TP=2 over the barlink PG.
# Two-process launch template: torch_ext/tests/run_pg.sh (blrun, per-rank
# LOCAL_RANK card selection, shared BL_SOCK_PATH).
#
# GPU 0 may be shared with other local experiments; each phase waits for a
# free card and retries on failure (e.g. a collision mid-run).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT=demos/flux2_tp/results
mkdir -p "$OUT"

export BL_POOL_MB=192

gpu0_free() {
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 500 ]
}

wait_gpu() {
    until gpu0_free; do sleep 1; done
}

run_tp1() {
    # TP=1 uses no bl at all: plain python, no pool memory on the card.
    wait_gpu
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        .venv/bin/python demos/flux2_tp/worker.py --tp 1 \
        --out "$OUT" 2>&1 | tee "$OUT/tp1.log"
    return ${PIPESTATUS[0]}
}

run_tp2() {
    local port=$(( (RANDOM % 20000) + 20000 ))
    local sock="/tmp/bl_flux2_$$_$RANDOM.sock"
    wait_gpu
    BL_SOCK_PATH="$sock" LOCAL_RANK=0 BL_SKIP_INIT=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True tools/blrun \
        demos/flux2_tp/worker.py --tp 2 --rank 0 --port "$port" \
        --out "$OUT" 2>&1 | tee "$OUT/tp2_rank0.log" &
    local p0=$!
    BL_SOCK_PATH="$sock" LOCAL_RANK=1 BL_SKIP_INIT=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True tools/blrun \
        demos/flux2_tp/worker.py --tp 2 --rank 1 --port "$port" \
        --out "$OUT" 2>&1 | tee "$OUT/tp2_rank1.log" &
    local p1=$!
    wait "$p0"; local r0=$?
    wait "$p1"; local r1=$?
    rm -f "$sock"
    [ "$r0" -eq 0 ] && [ "$r1" -eq 0 ]
}

retry() {  # retry <phase-name> <fn>
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

retry TP1 run_tp1 || exit 1
retry TP2 run_tp2 || exit 1

echo "=== correctness ==="
.venv/bin/python demos/flux2_tp/compare.py --dir "$OUT" | tee "$OUT/correctness.log"
