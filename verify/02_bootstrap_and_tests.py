#!/usr/bin/env python3
"""Bootstrap CIs and paired tests for the UnsafeBench eval.

- Percentile bootstrap (B=5000, seed=42, resample images with replacement)
  for F1 of: overall per mode, core per mode, Sexual per source (strict &
  super_strict).
- Bootstrap CI on the real-minus-AI F1 DIFFERENCE per mode (all & core, and
  Sexual strict) — the statistic behind the "no gap" claim.
- Exact McNemar tests between modes (paired on the same 2,037 images),
  overall and core.

Run: python3 02_bootstrap_and_tests.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results.jsonl")

CORE = {"Sexual", "Violence", "Shocking", "Self-harm", "Hate"}
MODES = ["strict", "moderate", "super_strict"]
B = 5000
SEED = 42

rows = [json.loads(l) for l in open(RESULTS)]


def to_arrays(sub):
    y = np.array([1 if r["label"] == "Unsafe" else 0 for r in sub], dtype=np.int8)
    p = np.array([r["pred"] for r in sub], dtype=np.int8)
    return y, p


def f1(y, p):
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def boot_ci(y, p, rng, b=B):
    n = len(y)
    stats = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        stats[i] = f1(y[idx], p[idx])
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def boot_diff_ci(y1, p1, y2, p2, rng, b=B):
    """CI on F1(sub1) - F1(sub2), resampling each stratum independently."""
    n1, n2 = len(y1), len(y2)
    stats = np.empty(b)
    for i in range(b):
        i1 = rng.integers(0, n1, n1)
        i2 = rng.integers(0, n2, n2)
        stats[i] = f1(y1[i1], p1[i1]) - f1(y2[i2], p2[i2])
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


rng = np.random.default_rng(SEED)

print("=" * 76)
print(f"BOOTSTRAP 95% CIs — percentile, B={B}, seed={SEED}, resample images")
print("=" * 76)

CLAIMED_CI = {
    ("overall", "strict"): (0.542, 0.605),
    ("overall", "moderate"): (0.401, 0.476),
    ("overall", "super_strict"): (0.668, 0.716),
    ("core", "strict"): (0.694, 0.766),
    ("core", "moderate"): (0.499, 0.597),
    ("core", "super_strict"): (0.704, 0.765),
}

for scope in ("overall", "core"):
    for m in MODES:
        sub = [r for r in rows if r["mode"] == m
               and (scope == "overall" or r["category"] in CORE)]
        y, p = to_arrays(sub)
        lo, hi = boot_ci(y, p, rng)
        cl, ch = CLAIMED_CI[(scope, m)]
        d = max(abs(lo - cl), abs(hi - ch))
        tag = "OK(~)" if d <= 0.01 else "CHECK"
        print(f"{tag} {scope:<8} {m:<13} F1={f1(y,p):.3f}  "
              f"claimed [{cl:.3f},{ch:.3f}]  recomputed [{lo:.3f},{hi:.3f}]  maxdev={d:.3f}")

print()
print("Sexual-category F1 CIs (not reported in REPORT.md — computed here):")
for m in ("strict", "super_strict"):
    for src in ("Laion5B", "Lexica"):
        sub = [r for r in rows if r["mode"] == m and r["category"] == "Sexual"
               and r["source"] == src]
        y, p = to_arrays(sub)
        lo, hi = boot_ci(y, p, rng)
        print(f"  Sexual {m:<13} {src:<9} n={len(y):<4} F1={f1(y,p):.3f}  95% CI [{lo:.3f},{hi:.3f}]")
    sub = [r for r in rows if r["mode"] == m and r["category"] == "Sexual"]
    y, p = to_arrays(sub)
    lo, hi = boot_ci(y, p, rng)
    print(f"  Sexual {m:<13} {'both':<9} n={len(y):<4} F1={f1(y,p):.3f}  95% CI [{lo:.3f},{hi:.3f}]")

print()
print("=" * 76)
print("REAL-minus-AI F1 DIFFERENCE — bootstrap CI (independent strata)")
print("  (the 'no real-vs-AI gap' claim; CI containing 0 = gap not detected,")
print("   NOT evidence of no gap — note the CI width)")
print("=" * 76)
for scope in ("all", "core"):
    for m in MODES:
        s_real = [r for r in rows if r["mode"] == m and r["source"] == "Laion5B"
                  and (scope == "all" or r["category"] in CORE)]
        s_ai = [r for r in rows if r["mode"] == m and r["source"] == "Lexica"
                and (scope == "all" or r["category"] in CORE)]
        y1, p1 = to_arrays(s_real)
        y2, p2 = to_arrays(s_ai)
        d = f1(y1, p1) - f1(y2, p2)
        lo, hi = boot_diff_ci(y1, p1, y2, p2, rng)
        print(f"  {scope:<5} {m:<13} diff(real-AI)={d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  width={hi-lo:.3f}")
# Sexual strict diff
s_real = [r for r in rows if r["mode"] == "strict" and r["category"] == "Sexual" and r["source"] == "Laion5B"]
s_ai = [r for r in rows if r["mode"] == "strict" and r["category"] == "Sexual" and r["source"] == "Lexica"]
y1, p1 = to_arrays(s_real)
y2, p2 = to_arrays(s_ai)
d = f1(y1, p1) - f1(y2, p2)
lo, hi = boot_diff_ci(y1, p1, y2, p2, rng)
print(f"  Sexual strict diff(real-AI)={d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  width={hi-lo:.3f}")

print()
print("Reference: SD Safety Filter Sexual real-AI gap cited in REPORT = 0.833-0.727 = +0.106")

print()
print("=" * 76)
print("McNEMAR (exact, two-sided) BETWEEN MODES — paired on identical images")
print("=" * 76)
by_mode = defaultdict(dict)
for r in rows:
    by_mode[r["mode"]][r["row_idx"]] = r
idxs = sorted(by_mode["strict"])
for scope in ("overall", "core"):
    ids = [i for i in idxs if scope == "overall" or by_mode["strict"][i]["category"] in CORE]
    for a, bm in (("strict", "moderate"), ("strict", "super_strict"), ("moderate", "super_strict")):
        b_cnt = c_cnt = 0
        for i in ids:
            ra, rb = by_mode[a][i], by_mode[bm][i]
            ca = (ra["pred"] == 1) == (ra["label"] == "Unsafe")
            cb = (rb["pred"] == 1) == (rb["label"] == "Unsafe")
            if ca and not cb:
                b_cnt += 1
            elif cb and not ca:
                c_cnt += 1
        pv = mcnemar_exact(b_cnt, c_cnt)
        print(f"  {scope:<8} {a} vs {bm}: discordant {b_cnt}/{c_cnt}  p={pv:.2e}")

print()
print("=" * 76)
print("super_strict FPR ON SAFE IMAGES BY CATEGORY — tests the 'by design' story")
print("=" * 76)
for cat in sorted({r["category"] for r in rows}):
    sub = [r for r in rows if r["mode"] == "super_strict" and r["category"] == cat
           and r["label"] == "Safe"]
    fpn = sum(1 for r in sub if r["pred"] == 1)
    print(f"  {cat:<30} nSafe={len(sub):<4} FP={fpn:<4} FPR={fpn/len(sub):.3f}")
