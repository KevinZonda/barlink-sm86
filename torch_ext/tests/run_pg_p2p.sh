#!/usr/bin/env bash
# torch.distributed ProcessGroup point-to-point test: two processes, one per
# GPU. Template = tests/run_pg.sh.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SOCK="/tmp/bl_p2p_$$.sock"
PORT=$(( (RANDOM % 20000) + 20000 ))

export BL_SOCK_PATH="$SOCK"
export BL_POOL_MB=192

LOCAL_RANK=0 BL_SKIP_INIT=1 tools/blrun torch_ext/tests/test_pg_p2p.py \
    --rank 0 --port "$PORT" &
P0=$!
LOCAL_RANK=1 BL_SKIP_INIT=1 tools/blrun torch_ext/tests/test_pg_p2p.py \
    --rank 1 --port "$PORT" &
P1=$!

wait "$P0"; R0=$?
wait "$P1"; R1=$?
rm -f "$SOCK"

if [ "$R0" -eq 0 ] && [ "$R1" -eq 0 ]; then
    echo "P2P TESTS PASSED"
else
    echo "P2P TESTS FAILED (rank0=$R0 rank1=$R1)"
    exit 1
fi
