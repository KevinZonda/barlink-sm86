# SPDX-License-Identifier: MIT
#
# Compare the TP=1 and TP=2 outputs saved by worker.py (same seed/inputs).
#
# Criterion: this is a 32-block bf16 model — TP changes GEMM shapes, head
# counts and allreduce order, so reassociation noise is intrinsic, not a
# sharding bug. The anchor run (worker.py --compute-fp32: bf16 weights, fp32
# compute, TP=2) measures that noise floor directly:
#
#   TP1-bf16 vs fp32 anchor: rel_L2 ~2.3%   TP2-bf16 vs fp32 anchor: ~2.4%
#
# so both sides sit ~2.4% from the fp32 truth and closer to each other. PASS
# bar: cosine > 0.999 and rel_L2 < 3% (i.e. TP2 is as close to TP1 as either
# is to ground truth), plus allclose(atol=2e-2, rtol=1e-2) fraction reported.

import argparse
import json
import os
import sys

import torch


def rel_stats(a, b):
    d = a - b
    return {
        "rel_l2": (d.norm() / b.norm()).item(),
        "max_abs_err": d.abs().max().item(),
        "cosine": torch.cosine_similarity(a.flatten(), b.flatten(),
                                          dim=0).item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    ref = torch.load(os.path.join(a.dir, "output_tp1.pt"))
    out = torch.load(os.path.join(a.dir, "output_tp2.pt"))
    if ref.shape != out.shape:
        print("FAIL: shape mismatch %s vs %s" % (ref.shape, out.shape))
        sys.exit(1)

    s = rel_stats(out, ref)
    close_frac = torch.isclose(out, ref, atol=2e-2, rtol=1e-2).float().mean()
    s["allclose_frac_atol_2e-2"] = close_frac.item()

    anchor_path = os.path.join(a.dir, "output_tp2anchor.pt")
    if os.path.exists(anchor_path):
        anchor = torch.load(anchor_path)
        s["anchor_tp1_rel_l2"] = rel_stats(ref, anchor)["rel_l2"]
        s["anchor_tp2_rel_l2"] = rel_stats(out, anchor)["rel_l2"]
        s["anchor_rel_l2_max"] = max(s["anchor_tp1_rel_l2"],
                                     s["anchor_tp2_rel_l2"])

    s1 = json.load(open(os.path.join(a.dir, "stats_tp1.json")))
    s2 = json.load(open(os.path.join(a.dir, "stats_tp2.json")))
    s["tp1_mean_ms"] = s1["mean_ms_per_step"]
    s["tp2_mean_ms"] = s2["mean_ms_per_step"]
    s["speedup"] = s1["mean_ms_per_step"] / s2["mean_ms_per_step"]
    s["passed"] = bool(s["cosine"] > 0.999 and s["rel_l2"] < 0.03)

    print("cosine=%.6f  rel_L2=%.5f  max_abs_err=%.5f  "
          "allclose(2e-2)=%.4f" % (s["cosine"], s["rel_l2"],
                                   s["max_abs_err"], s["allclose_frac_atol_2e-2"]))
    if "anchor_rel_l2_max" in s:
        print("anchor (fp32 compute): TP1 %.4f / TP2 %.4f rel_L2 — bf16 noise "
              "floor" % (s["anchor_tp1_rel_l2"], s["anchor_tp2_rel_l2"]))
    print("tp1: %.1f ms/step  tp2: %.1f ms/step  speedup=%.2fx"
          % (s["tp1_mean_ms"], s["tp2_mean_ms"], s["speedup"]))

    with open(os.path.join(a.dir, "correctness.json"), "w") as f:
        json.dump(s, f, indent=2)

    if s["passed"]:
        print("CORRECTNESS PASSED")
    else:
        print("CORRECTNESS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
