# SPDX-License-Identifier: MIT
#
# Correctness check for the Qwen TP decode benchmark:
#   - TP=2 rank0 vs rank1 generated ids identical (same greedy trajectory)
#   - TP=2 vs TP=1 generated ids: token match rate + (loosely) divergence point
# Greedy decode may legitimately diverge at some token once in a while when
# two candidates are within int8 reassociation noise; the match rate and the
# first-divergence index tell the story.

import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tp2-tag", default="tp2")
    a = ap.parse_args()

    ids1 = torch.load(os.path.join(a.dir, "genids_tp1.pt"))
    ids2 = torch.load(os.path.join(a.dir, "genids_%s.pt" % a.tp2_tag))
    n = min(len(ids1), len(ids2))
    match = (ids1[:n] == ids2[:n])
    rate = match.float().mean().item()
    first_bad = int((~match).nonzero()[0]) if not bool(match.all()) else -1

    rank_consistent = None
    p = os.path.join(a.dir, "genids_%s_rank1.pt" % a.tp2_tag)
    if os.path.exists(p):
        ids2r = torch.load(p)
        rank_consistent = bool(torch.equal(ids2, ids2r))

    # greedy decode may legitimately flip at a near-tie token once int8
    # reassociation noise tips the balance; the meaningful bars are: same
    # trajectory for the vast majority of tokens + both TP ranks identical
    passed = rate >= 0.90 and (rank_consistent in (None, True))
    print("tp1 vs %s: token match %.4f (%d/%d), first divergence at %s"
          % (a.tp2_tag, rate, int(match.sum()), n,
             first_bad if first_bad >= 0 else "-"))
    if rank_consistent is not None:
        print("tp2 rank0 vs rank1 identical: %s" % rank_consistent)

    with open(os.path.join(a.dir, "correctness_%s.json" % a.tp2_tag),
              "w") as f:
        json.dump({
            "tp2_tag": a.tp2_tag,
            "token_match_rate": rate,
            "n_compared": n,
            "first_divergence": first_bad,
            "rank_consistent": rank_consistent,
            "passed": passed,
        }, f, indent=2)
    print("CORRECTNESS %s" % ("PASSED" if passed else "FAILED"))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
