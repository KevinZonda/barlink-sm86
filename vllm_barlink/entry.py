# SPDX-License-Identifier: MIT
#
# blrun entry trampoline: blrun hardcodes <repo>/.venv/bin/python, but the
# ambient CAP_SYS_ADMIN it raises survives execve. Re-exec the target under
# the vllm venv interpreter, keeping caps ambient for every vllm-spawned
# worker (each rank's init_peer drops its own caps afterwards).
#
#   BL_SKIP_INIT=1 BL_POOL_MB=192 tools/blrun \
#       vllm_barlink/entry.py <script.py> [args...]
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
VENV = os.environ.get("BL_VLLM_VENV",
                      os.path.join(_REPO, ".venv-vllm"))
PY = os.path.join(VENV, "bin", "python")

if len(sys.argv) < 2:
    sys.stderr.write("usage: entry.py <script.py> [args...]\n")
    sys.exit(2)
os.execv(PY, [PY] + sys.argv[1:])
