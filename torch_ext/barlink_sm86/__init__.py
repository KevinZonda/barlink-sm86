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
    # torch-major-version-specific prebuilt extension (e.g. _C.torch213.so,
    # built for the vllm venv's torch): the pybind ABI is NOT stable across
    # torch versions, so multiple builds can coexist next to the default
    # _C*.so (always built against the repo's primary venv).
    tv = torch.__version__.split("+")[0].split(".")
    tagged = os.path.join(here, "_C.torch%s%s.so" % (tv[0], tv[1]))
    if os.path.exists(tagged):
        import importlib.util as ilutil
        spec = ilutil.spec_from_file_location("barlink_sm86._C", tagged)
        mod = ilutil.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
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
init_peer = _C.init_peer
shutdown = _C.shutdown
empty = _C.empty
copy_ = _C.copy_
allreduce_ = _C.allreduce_
allreduce_into = _C.allreduce_into
send_into = _C.send_into
recv_into = _C.recv_into
verify = _C.verify
readback = _C.readback
pool_move = _C.pool_move


def _chunk_elems(itemsize, chunk_bytes):
    # per-slice element count whose byte size is a multiple of 16 (the copy_
    # granularity); itemsize always divides 16 for the supported dtypes
    per = 16 // itemsize
    return max(per, (max(16, chunk_bytes) // 16) * per)


def copy_large(dst, src, chunk_bytes=16 << 20):
    """Chunked bl.copy_ for tensors bigger than the pool slice that fits a
    single call. Slices on the 16-byte copy granularity; identical
    semantics to copy_ (SPMD symmetric discipline applies in peer mode)."""
    total = src.numel()
    if total == 0:
        return
    if src.element_size() != dst.element_size():
        raise TypeError("barlink_sm86 copy_large: dtype mismatch")
    step = _chunk_elems(src.element_size(), chunk_bytes)
    if total * src.element_size() <= step * src.element_size():
        copy_(dst, src)
        return
    for i in range(0, total, step):
        j = min(i + step, total)
        copy_(dst[i:j], src[i:j])


def allreduce_large(a, b, chunk_bytes=16 << 20):
    """Chunked bl.allreduce_; same SPMD discipline as allreduce_."""
    total = a.numel()
    if total == 0:
        return
    if a.element_size() != b.element_size():
        raise TypeError("barlink_sm86 allreduce_large: dtype mismatch")
    step = _chunk_elems(a.element_size(), chunk_bytes)
    if total * a.element_size() <= step * a.element_size():
        allreduce_(a, b)
        return
    for i in range(0, total, step):
        j = min(i + step, total)
        allreduce_(a[i:j], b[i:j])


__all__ = ["init", "init_peer", "shutdown", "empty", "copy_", "allreduce_",
           "allreduce_into", "send_into", "recv_into", "copy_large",
           "allreduce_large", "verify", "readback", "pool_move"]
