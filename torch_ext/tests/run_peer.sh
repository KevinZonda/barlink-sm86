#!/usr/bin/env bash
# Cross-process peer test: one process per GPU, orchestrated symmetrically.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SOCK="/tmp/bl_peer_$$.sock"

BL_SKIP_INIT=1 tools/blrun torch_ext/tests/test_peer.py --rank 0 --sock "$SOCK" &
P0=$!
BL_SKIP_INIT=1 tools/blrun torch_ext/tests/test_peer.py --rank 1 --sock "$SOCK" &
P1=$!

wait "$P0"; R0=$?
wait "$P1"; R1=$?
rm -f "$SOCK"

if [ "$R0" -eq 0 ] && [ "$R1" -eq 0 ]; then
    echo "PEER TESTS PASSED"
else
    echo "PEER TESTS FAILED (rank0=$R0 rank1=$R1)"
    exit 1
fi
