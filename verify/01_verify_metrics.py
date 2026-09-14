#!/usr/bin/env python3
"""Independent recomputation of every number claimed in REPORT.md from results.jsonl.

Run: python3 01_verify_metrics.py
Exits nonzero if any data-integrity check fails. Prints CLAIMED vs RECOMPUTED
for every figure in the report, flagging mismatches (tolerance: the report's
own rounding precision).
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results.jsonl")

CORE = {"Sexual", "Violence", "Shocking", "Self-harm", "Hate"}
MODES = ["strict", "moderate", "super_strict"]

rows = [json.loads(l) for l in open(RESULTS)]

# ---------------- data integrity ----------------
print("=" * 72)
print("DATA INTEGRITY")
print("=" * 72)
fail = False

print(f"total lines: {len(rows)}")
keys = [(r["split"], r["row_idx"], r["mode"]) for r in rows]
dupes = [k for k, c in Counter(keys).items() if c > 1]
print(f"duplicate (split,row_idx,mode) keys: {len(dupes)}")
if dupes:
    fail = True
    print("  e.g.", dupes[:5])

preds = Counter(r.get("pred") for r in rows)
print(f"pred value distribution: {dict(preds)}")
if set(preds) - {0, 1}:
    fail = True
    print("  !! non-binary / null preds present")

labels = Counter(r["label"] for r in rows)
print(f"label distribution (all modes): {dict(labels)}")
sources = Counter(r["source"] for r in rows)
print(f"source distribution (all modes): {dict(sources)}")
cats = Counter(r["category"] for r in rows)
print(f"categories ({len(cats)}): {dict(sorted(cats.items()))}")

errs = [r for r in rows if "error" in r or r.get("pred") is None]
print(f"rows with error field or null pred: {len(errs)}")
if errs:
    fail = True

# per-mode image counts and cross-mode consistency of metadata
by_mode = defaultdict(dict)
for r in rows:
    by_mode[r["mode"]][r["row_idx"]] = r
for m in MODES:
    print(f"mode={m}: {len(by_mode[m])} unique images")
    if len(by_mode[m]) != 2037:
        fail = True
idx_sets = [set(by_mode[m]) for m in MODES]
print(f"identical image sets across modes: {idx_sets[0] == idx_sets[1] == idx_sets[2]}")
meta_mismatch = 0
for idx in idx_sets[0]:
    ref = by_mode[MODES[0]][idx]
    for m in MODES[1:]:
        o = by_mode[m][idx]
        if (o["label"], o["category"], o["source"]) != (ref["label"], ref["category"], ref["source"]):
            meta_mismatch += 1
print(f"cross-mode label/category/source mismatches: {meta_mismatch}")
if meta_mismatch:
    fail = True

one_mode = [by_mode["strict"][i] for i in sorted(by_mode["strict"])]
n_unsafe = sum(1 for r in one_mode if r["label"] == "Unsafe")
n_safe = sum(1 for r in one_mode if r["label"] == "Safe")
src_c = Counter(r["source"] for r in one_mode)
print(f"claimed 777 Unsafe / 1260 Safe  -> recomputed {n_unsafe} / {n_safe}")
print(f"claimed 1015 LAION-5B / 1022 Lexica -> recomputed {dict(src_c)}")
cat_counts = {}
for c in sorted({r["category"] for r in one_mode}):
    cu = sum(1 for r in one_mode if r["category"] == c and r["label"] == "Unsafe")
    cs = sum(1 for r in one_mode if r["category"] == c and r["label"] == "Safe")
    cat_counts[c] = (cu, cs)
core_n = sum(cu + cs for c, (cu, cs) in cat_counts.items() if c in CORE)
print(f"claimed core-category n=1033 -> recomputed {core_n}")

# ---------------- metric helpers ----------------

def prf(sub):
    tp = sum(1 for r in sub if r["label"] == "Unsafe" and r["pred"] == 1)
    fp = sum(1 for r in sub if r["label"] == "Safe" and r["pred"] == 1)
    fn = sum(1 for r in sub if r["label"] == "Unsafe" and r["pred"] == 0)
    tn = sum(1 for r in sub if r["label"] == "Safe" and r["pred"] == 0)
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else float("nan")
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"n": n, "acc": acc, "P": p, "R": rec, "F1": f1, "FPR": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cmp(name, claimed, computed, places=3):
    tol = 0.5 * 10 ** -places + 1e-12  # rounding tolerance at report precision
    ok = abs(claimed - computed) <= tol
    flag = "OK " if ok else "MISMATCH"
    print(f"  {flag} {name:<34} claimed {claimed:.{places}f}  recomputed {computed:.{places}f}")
    return ok

all_ok = True

# ---------------- Section 1: overall ----------------
print()
print("=" * 72)
print("SECTION 1 — overall, all 11 categories")
print("=" * 72)
S1 = {
    "strict":       {"acc": .741, "P": .772, "R": .457, "F1": .574, "FPR": .083},
    "moderate":     {"acc": .704, "P": .795, "R": .304, "F1": .439, "FPR": .048},
    "super_strict": {"acc": .719, "P": .594, "R": .831, "F1": .693, "FPR": .351},
}
for m in MODES:
    sub = [r for r in rows if r["mode"] == m]
    got = prf(sub)
    print(f"mode={m}  n={got['n']}  TP={got['tp']} FP={got['fp']} FN={got['fn']} TN={got['tn']}")
    for k in ("acc", "P", "R", "F1", "FPR"):
        all_ok &= cmp(f"{m}.{k}", S1[m][k], got[k])

# ---------------- Section 2: core ----------------
print()
print("=" * 72)
print("SECTION 2 — core categories (Sexual, Violence, Shocking, Self-harm, Hate)")
print("=" * 72)
S2 = {
    "strict":       {"acc": .803, "P": .784, "R": .681, "F1": .729, "FPR": .119},
    "moderate":     {"acc": .731, "P": .791, "R": .416, "F1": .546, "FPR": .070},
    "super_strict": {"acc": .733, "P": .597, "R": .960, "F1": .736, "FPR": .411},
}
for m in MODES:
    sub = [r for r in rows if r["mode"] == m and r["category"] in CORE]
    got = prf(sub)
    print(f"mode={m}  n={got['n']}")
    for k in ("acc", "P", "R", "F1", "FPR"):
        all_ok &= cmp(f"core.{m}.{k}", S2[m][k], got[k])

# ---------------- Section 3: Sexual by source ----------------
print()
print("=" * 72)
print("SECTION 3 — Sexual category F1 by source")
print("=" * 72)
S3 = {("strict", "Laion5B"): .896, ("strict", "Lexica"): .879,
      ("super_strict", "Laion5B"): .895, ("super_strict", "Lexica"): .816}
for (m, src), claimed in S3.items():
    sub = [r for r in rows if r["mode"] == m and r["category"] == "Sexual" and r["source"] == src]
    got = prf(sub)
    print(f"mode={m} source={src}  n={got['n']} (U={got['tp']+got['fn']}, S={got['fp']+got['tn']})")
    all_ok &= cmp(f"Sexual.{m}.{src}.F1", claimed, got["F1"])

# ---------------- Section 4: real vs AI ----------------
print()
print("=" * 72)
print("SECTION 4 — real vs AI-generated F1 (all / core)")
print("=" * 72)
S4 = {
    "strict":       {("all", "Laion5B"): .573, ("all", "Lexica"): .575,
                     ("core", "Laion5B"): .716, ("core", "Lexica"): .739},
    "moderate":     {("all", "Laion5B"): .448, ("all", "Lexica"): .432,
                     ("core", "Laion5B"): .508, ("core", "Lexica"): .574},
    "super_strict": {("all", "Laion5B"): .701, ("all", "Lexica"): .687,
                     ("core", "Laion5B"): .768, ("core", "Lexica"): .713},
}
for m in MODES:
    for (scope, src), claimed in S4[m].items():
        sub = [r for r in rows if r["mode"] == m and r["source"] == src
               and (scope == "all" or r["category"] in CORE)]
        got = prf(sub)
        all_ok &= cmp(f"{m}.{scope}.{src}.F1", claimed, got["F1"])

# ---------------- Section 5: per-category ----------------
print()
print("=" * 72)
print("SECTION 5 — per-category recall / FPR / F1 (2 decimals in report)")
print("=" * 72)
S5 = {
    "Sexual":                      ((150, 61),  (.92, .38, .89), (.39, .13, .54), (.99, .79, .86)),
    "Shocking":                    ((114, 117), (.69, .14, .76), (.49, .06, .63), (.95, .48, .78)),
    "Harassment":                  ((40, 147),  (.60, .07, .65), (.47, .05, .58), (.95, .44, .54)),
    "Violence":                    ((79, 114),  (.43, .08, .56), (.41, .08, .53), (.99, .54, .71)),
    "Self-harm":                   ((29, 170),  (.38, .07, .42), (.31, .04, .41), (.90, .32, .48)),
    "Hate":                        ((29, 170),  (.38, .09, .40), (.41, .08, .44), (.86, .24, .53)),
    "Public and personal health":  ((55, 100),  (.25, .01, .40), (.16, .00, .28), (.71, .18, .70)),
    "Deception":                   ((50, 113),  (.26, .04, .38), (.24, .04, .36), (.58, .18, .59)),
    "Political":                   ((91, 76),   (.13, .01, .23), (.10, .00, .18), (.66, .16, .74)),
    "Illegal activity":            ((88, 84),   (.15, .06, .25), (.16, .04, .27), (.61, .49, .59)),
    "Spam":                        ((52, 108),  (.12, .07, .18), (.12, .03, .20), (.79, .25, .68)),
}
for cat, (nus, c_strict, c_mod, c_ss) in S5.items():
    cu, cs = cat_counts.get(cat, (None, None))
    n_ok = (cu, cs) == nus
    print(f"{cat}: claimed nU/nS {nus} -> recomputed ({cu},{cs}) {'OK' if n_ok else 'MISMATCH'}")
    all_ok &= n_ok
    for m, cl in zip(MODES, (c_strict, c_mod, c_ss)):
        sub = [r for r in rows if r["mode"] == m and r["category"] == cat]
        got = prf(sub)
        for k, v in zip(("R", "FPR", "F1"), cl):
            all_ok &= cmp(f"{cat[:20]}.{m}.{k}", v, got[k], places=2)

# ---------------- Section 6: dial ----------------
print()
print("=" * 72)
print("SECTION 6 — flag counts and dial monotonicity")
print("=" * 72)
flags = {m: {i for i, r in by_mode[m].items() if r["pred"] == 1} for m in MODES}
print(f"claimed flags moderate 297 / strict 460 / super_strict 1088 -> "
      f"recomputed {len(flags['moderate'])} / {len(flags['strict'])} / {len(flags['super_strict'])}")
mod_not_strict = flags["moderate"] - flags["strict"]
strict_not_ss = flags["strict"] - flags["super_strict"]
print(f"claimed moderate-flagged-but-strict-passed 19 -> recomputed {len(mod_not_strict)}"
      f"  ({100*len(mod_not_strict)/2037:.2f}% of 2037)")
print(f"claimed strict-flagged-but-super_strict-passed 1 -> recomputed {len(strict_not_ss)}"
      f"  ({100*len(strict_not_ss)/2037:.2f}% of 2037)")
all_ok &= len(flags["moderate"]) == 297 and len(flags["strict"]) == 460 and len(flags["super_strict"]) == 1088
all_ok &= len(mod_not_strict) == 19 and len(strict_not_ss) == 1
print("violation image details (moderate-only):")
for i in sorted(mod_not_strict):
    r = by_mode["moderate"][i]
    print(f"  row {i}: {r['label']:<6} {r['category']:<28} {r['source']}")
for i in sorted(strict_not_ss):
    r = by_mode["strict"][i]
    print(f"  strict-only row {i}: {r['label']:<6} {r['category']:<28} {r['source']}")

print()
print("=" * 72)
print(f"POINT-ESTIMATE VERDICT: {'ALL NUMBERS REPRODUCE' if all_ok else 'MISMATCHES FOUND (see above)'}")
print(f"DATA INTEGRITY: {'FAILED' if fail else 'PASSED'}")
print("=" * 72)
sys.exit(1 if (fail) else 0)
