#!/usr/bin/env python3
"""Compare two GSM8K runs of bench/rexl3/gsm8k.py (--out files) on the same problems, by discordant pairs (stdlib only).

  gsm8k_compare.py BASELINE.json CANDIDATE.json

Only the problems the two runs disagree on carry information: b = right in the baseline only, c = right in the
candidate only. Under "no difference" each discordant problem is a fair coin, so the exact two-sided binomial (McNemar)
p-value of min(b, c) out of b + c says whether the candidate is worse or better beyond noise. Uncommitted answers
count as wrong. Prints the counts, p, a verdict line, and METRIC lines; exit 1 when the candidate is worse at p < 0.05.
"""
import json
import math
import sys


def exact_two_sided(k, n):
    """P(X <= k or X >= n - k) for X ~ Binomial(n, 1/2), k <= n / 2."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    base, cand = (json.load(open(p)) for p in sys.argv[1:3])
    right = lambda rows: {r["i"]: bool(r["ok"] and r["committed"]) for r in rows}  # noqa: E731
    rb, rc = right(base), right(cand)
    common = sorted(set(rb) & set(rc))
    if not common:
        sys.exit("no common problems")
    b = [i for i in common if rb[i] and not rc[i]]
    c = [i for i in common if rc[i] and not rb[i]]
    p = exact_two_sided(min(len(b), len(c)), len(b) + len(c))
    nb, nc = sum(rb[i] for i in common), sum(rc[i] for i in common)
    print(f"{len(common)} problems: baseline {nb} right, candidate {nc} right")
    print(f"discordant: baseline only {len(b)} {b}; candidate only {len(c)} {c}; exact two-sided p = {p:.3f}")
    worse = len(b) > len(c) and p < 0.05
    print("verdict: candidate WORSE beyond noise" if worse else
          "verdict: candidate better beyond noise" if len(c) > len(b) and p < 0.05 else "verdict: no difference beyond noise")
    print(f"METRIC gsm8k_discordant_base_only={len(b)}")
    print(f"METRIC gsm8k_discordant_cand_only={len(c)}")
    print(f"METRIC gsm8k_mcnemar_p={p:.4f}")
    sys.exit(1 if worse else 0)


if __name__ == "__main__":
    main()
