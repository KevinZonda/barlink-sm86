# SPDX-License-Identifier: MIT
#
# sitecustomize shim: with the REPO ROOT on PYTHONPATH, every
# python startup (including vllm's spawned engine-core processes, which
# inherit PYTHONPATH) auto-installs the barlink vllm integration.
# See vllm_barlink/__init__.py for what install() does and the launch
# requirements (tools/blrun + BL_SKIP_INIT=1).

try:
    import vllm_barlink  # noqa: F401  (install() runs at import)
except Exception as e:    # never break unrelated python startups
    import sys
    sys.stderr.write("sitecustomize: vllm_barlink install skipped: %s\n" % e)
