# SPDX-License-Identifier: MIT
#
# Correctness check for the Qwen3.8-27B INT8 TP decode benchmark.
#
# What must be EXACT: rank0 vs rank1 within each backend (same backend,
# same math -> bitwise identical greedy trajectory).
#
# What may legitimately differ across backends: the reduction rounding of
# the rowwise all_reduce (barlink PG adds in fp32 and rounds once to
# bf16; NCCL's bf16 SUM path may round differently), which can flip a
# near-tie argmax and send the greedy trajectories apart. So:
#   - report token match rate + first divergence index (no hard failure)
#   - the step-0 dump (first generated token, both backends see the
#     identical prompt) must agree tightly -> the numeric delta
#     attributable to the backend alone.

import argparse
import json
import os
import sys

import torch


def load_ids(d, tag, rank):
    return torch.load(os.path.join(d, "genids_%s_rank%d.pt" % (tag, rank)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag-a", default="barlink")
    ap.add_argument("--tag-b", default="nccl")
    a = ap.parse_args()

    res = {}
    ok = True

    ids_a0 = load_ids(a.dir, a.tag_a, 0)
    ids_a1 = load_ids(a.dir, a.tag_a, 1)
    ids_b0 = load_ids(a.dir, a.tag_b, 0)
    ids_b1 = load_ids(a.dir, a.tag_b, 1)

    ra = bool(torch.equal(ids_a0, ids_a1))
    rb = bool(torch.equal(ids_b0, ids_b1))
    res["rank_consistent"] = {a.tag_a: ra, a.tag_b: rb}
    ok = ok and ra and rb

    n = min(len(ids_a0), len(ids_b0))
    match = (ids_a0[:n] == ids_b0[:n])
    rate = match.float().mean().item()
    first_bad = int((~match).nonzero()[0]) if not bool(match.all()) else -1
    res["token_match_rate"] = rate
    res["first_divergence"] = first_bad
    # trajectories may legitimately part after a near-tie; log, not fail

    for kind in ("logits", "hidden"):
        pa = os.path.join(a.dir, "dump_%s_rank0.pt" % a.tag_a)
        pb = os.path.join(a.dir, "dump_%s_rank0.pt" % a.tag_b)
        if os.path.exists(pa) and os.path.exists(pb):
            da = torch.load(pa)[kind].float()
            db = torch.load(pb)[kind].float()
            d = (da - db).abs()
            rel = (d.norm() / db.norm().clamp(min=1e-12)).item()
            res["%s_step0_max_abs_diff" % kind] = d.max().item()
            res["%s_step0_rel_l2" % kind] = rel
            if kind == "logits":
                # bf16 rounding of two equivalent reductions: ~1e-3 band
                ok = ok and rel < 1e-2
                top_a = da.argmax(-1)
                top_b = db.argmax(-1)
                res["step0_argmax_equal"] = bool((top_a == top_b).all().item())

    res["passed"] = ok
    with open(os.path.join(a.dir, "correctness_%s_vs_%s.json"
                           % (a.tag_a, a.tag_b)), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("CORRECTNESS %s" % ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
