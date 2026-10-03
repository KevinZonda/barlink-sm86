# SPDX-License-Identifier: MIT
#
# vLLM <-> barlink PG integration shim. See the repo docs (§10 of
# BUILD_AND_TEST.md) for the launch recipe. BL_SHIM_STEPS="1,2,3,4" (env)
# selects which install steps run -- used to bisect integration issues.
#
#   1. register the "barlink" torch PG backend (extended API; world-2 link
#      + size-1 dummy subgroups)
#   2. wrap torch.distributed.init_process_group: any backend string naming
#      nccl for CUDA is rewritten to barlink (vllm 0.30 hardcodes
#      "cpu:gloo,cuda:nccl" and looks the backend up as an attribute of
#      torch.distributed at call time, so one wrap covers every path)
#   3. default VLLM_DISABLE_PYNCCL=1 + set the platform dist_backend attr,
#      so CudaCommunicator's all_reduce/all_gather dispatch falls through
#      to the torch-PG fallbacks (== barlink)
#   4. defensively neutralize CudaCommunicator.maybe_init_pynccl_communicator

import os

_DONE = False


def _step(n):
    steps = os.environ.get("BL_SHIM_STEPS", "1,2,3,4")
    return str(n) in steps.split(",")


def install():
    global _DONE
    if _DONE or os.environ.get("BL_SHIM_OFF") == "1":
        return
    _DONE = True

    if _step(1) or _step(2):
        # registration is LAZY (done inside the init_process_group wrapper):
        # importing barlink_sm86 at interpreter startup made vllm's
        # engine-core process initialize CUDA before forking its TP workers
        # (bisected: only the site-time import triggers it; the wrapper
        # alone is clean). Registering on first use avoids the poisoning
        # entirely -- every process that inits a PG registers its own copy.
        _registered = {"done": False}

        def _ensure_registered():
            if _registered["done"]:
                return
            _registered["done"] = True
            here = os.path.dirname(os.path.abspath(__file__))
            for p in (os.path.join(here, "..", "torch_ext"),
                      os.path.join(here, "..", "..", "torch_ext")):
                p = os.path.abspath(p)
                if os.path.isdir(p):
                    import sys
                    if p not in sys.path:
                        sys.path.insert(0, p)
            import barlink_sm86.process_group as _blpg
            _blpg.register()

        import torch.distributed as _dist
        _orig_ipg = _dist.init_process_group

        def init_process_group(*args, **kwargs):
            backend = kwargs.get("backend", args[0] if args else None)
            if isinstance(backend, str) and "nccl" in backend:
                if _step(1):
                    _ensure_registered()
                if _step(2):
                    rewritten = backend.replace("nccl", "barlink")
                    if "backend" in kwargs:
                        kwargs["backend"] = rewritten
                    else:
                        args = (rewritten,) + args[1:]
            return _orig_ipg(*args, **kwargs)

        _dist.init_process_group = init_process_group

        # vllm checks torch.distributed.is_backend_available("barlink")
        # BEFORE init_process_group -- hook it so the lazy registration
        # fires there too (still not at interpreter startup, which would
        # make vllm's engine-core initialize CUDA before forking workers).
        _orig_iba = _dist.is_backend_available

        def is_backend_available(backend):
            if backend == "barlink" and _step(1):
                _ensure_registered()
            return _orig_iba(backend)

        _dist.is_backend_available = is_backend_available

    if _step(3):
        os.environ.setdefault("VLLM_DISABLE_PYNCCL", "1")
        try:
            from vllm.platforms.cuda import CudaPlatformBase
            CudaPlatformBase.dist_backend = "barlink"
        except Exception as e:
            import sys as _s
            _s.stderr.write(
                "vllm_barlink: CudaPlatformBase patch skipped: %s\n" % e)

    if _step(4):
        try:
            from vllm.distributed.device_communicators import cuda_communicator
            cc = cuda_communicator.CudaCommunicator
            if hasattr(cc, "maybe_init_pynccl_communicator"):
                cc.maybe_init_pynccl_communicator = lambda self: None
        except Exception as e:
            import sys as _s
            _s.stderr.write(
                "vllm_barlink: CudaCommunicator patch skipped: %s\n" % e)


install()
