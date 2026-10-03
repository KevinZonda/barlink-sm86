#!/usr/bin/env bash
# Install a .pth into the vllm venv so barlink-vllm integration loads at
# every python startup in that venv (including vllm-spawned workers).
set -eu
VENV="${1:?usage: install_pth.sh <venv-dir>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SP="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
"$VENV/bin/python" - "$ROOT" "$SP" <<'PYEOF'
import sys
root, sp = sys.argv[1], sys.argv[2]
with open(sp + "/vllm_barlink.pth", "w") as f:
    f.write("import sys; sys.path.insert(0, %r); import vllm_barlink\n" % root)
PYEOF
echo "installed $SP/vllm_barlink.pth"
