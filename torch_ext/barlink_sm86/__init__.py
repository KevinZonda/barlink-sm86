# SPDX-License-Identifier: MIT
#
# barlink_sm86 -- lazy-loading front for the compiled extension.
#
# The mechanism module is barlink_sm86._C (built by ../setup.py). It is
# imported lazily so that `import barlink_sm86` gives an actionable error
# instead of a raw ImportError when the extension was not built yet or
# torch is missing/misconfigured.

import importlib
import os
import sys


def _load():
    # torch sanity first: actionable message instead of a downstream crash
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "barlink_sm86 requires PyTorch. Install it in this environment "
            "(see barlink-torch/BUILD_AND_TEST.md). Original error: %s" % e
        ) from e

    import torch
    if not torch.cuda.is_available():
        sys.stderr.write(
            "barlink_sm86: warning: torch.cuda.is_available() is False; "
            "bl.init() will fail unless CUDA becomes available.\n"
        )

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        mod = importlib.import_module("barlink_sm86._C")
    except ImportError as e:
        raise ImportError(
            "barlink_sm86: the compiled extension barlink_sm86._C was not "
            "found. Build it with:\n"
            "    cd %s/.. && python setup.py build_ext --inplace\n"
            "(needs nvcc, the CUDA toolkit, and torch with CUDA support). "
            "Original error: %s" % (os.path.dirname(here), e)
        ) from e
    return mod


_C = _load()

init = _C.init
shutdown = _C.shutdown
empty = _C.empty
copy_ = _C.copy_
allreduce_ = _C.allreduce_
verify = _C.verify
readback = _C.readback

__all__ = ["init", "shutdown", "empty", "copy_", "allreduce_", "verify",
           "readback"]
