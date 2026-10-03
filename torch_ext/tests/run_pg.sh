#!/usr/bin/env bash
# torch.distributed ProcessGroup backend test: two processes, one per GPU.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SOCK="/tmp/bl_pg_$$.sock"
PORT=$(( (RANDOM % 20000) + 20000 ))

export BL_SOCK_PATH="$SOCK"
export BL_POOL_MB=192

LOCAL_RANK=0 BL_SKIP_INIT=1 tools/blrun torch_ext/tests/test_process_group.py \
    --rank 0 --port "$PORT" &
P0=$!
LOCAL_RANK=1 BL_SKIP_INIT=1 tools/blrun torch_ext/tests/test_process_group.py \
    --rank 1 --port "$PORT" &
P1=$!

wait "$P0"; R0=$?
wait "$P1"; R1=$?
rm -f "$SOCK"

if [ "$R0" -eq 0 ] && [ "$R1" -eq 0 ]; then
    echo "PG TESTS PASSED"
else
    echo "PG TESTS FAILED (rank0=$R0 rank1=$R1)"
    exit 1
fi
