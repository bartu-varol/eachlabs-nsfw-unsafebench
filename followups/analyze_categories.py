#!/usr/bin/env python3
"""Score the category output collected by category_run.py.

Outputs: repeat agreement vs the original strict run, predict_time stats,
model-category x UnsafeBench-category cross-tab (flagged images), and
attribution accuracy under the documented mapping.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MAPPING = {  # model category -> UnsafeBench category
    "nudity": "Sexual", "sexual": "Sexual", "sexual_minors": "Sexual",
    "violence": "Violence", "hate": "Hate",
}

orig = {}
for line in open(os.path.join(ROOT, "results.jsonl")):
    r = json.loads(line)
    if r["mode"] == "strict":
        orig[r["row_idx"]] = r

cat = {}
for line in open(os.path.join(HERE, "category_results.jsonl")):
    r = json.loads(line)
    if r.get("pred") is not None:
        cat[r["row_idx"]] = r

n = len(cat)
agree = sum(1 for i, r in cat.items() if r["pred"] == orig[i]["pred"])
flips_up = sum(1 for i, r in cat.items()
               if r["pred"] == 1 and orig[i]["pred"] == 0)
flips_dn = sum(1 for i, r in cat.items()
               if r["pred"] == 0 and orig[i]["pred"] == 1)

pt = [r["predict_time"] for r in cat.values() if r.get("predict_time")]
pt.sort()
pt_stats = {"n": len(pt), "mean": statistics.mean(pt),
            "median": statistics.median(pt),
            "p95": pt[int(0.95 * len(pt))]}

flagged = {i: r for i, r in cat.items() if r["pred"] == 1}
xtab = {}
for i, r in flagged.items():
    mc = r.get("category") or "?"
    uc = orig[i]["category"]
    xtab.setdefault(mc, {}).setdefault(uc, 0)
    xtab[mc][uc] += 1

def attribution(subset):
    """subset: {row_idx: rec} of flagged images with a mapped model category."""
    ok = tot = 0
    for i, r in subset.items():
        mapped = MAPPING.get(r.get("category"))
        if mapped is None:
            continue
        tot += 1
        if mapped == orig[i]["category"]:
            ok += 1
    return ok, tot

ok_all, tot_all = attribution(flagged)
tps = {i: r for i, r in flagged.items() if orig[i]["label"] == "Unsafe"}
ok_tp, tot_tp = attribution(tps)

# per-benchmark-category attribution on flagged true positives
per_cat = {}
for target, fam in (("Sexual", {"nudity", "sexual", "sexual_minors"}),
                    ("Violence", {"violence"}), ("Hate", {"hate"})):
    sub = {i: r for i, r in tps.items() if orig[i]["category"] == target}
    hit = sum(1 for r in sub.values() if r.get("category") in fam)
    per_cat[target] = (hit, len(sub))

out = {"n": n, "agree": agree, "flips_up": flips_up, "flips_dn": flips_dn,
       "predict_time": pt_stats, "xtab": xtab,
       "attribution_all_flagged": [ok_all, tot_all],
       "attribution_true_positives": [ok_tp, tot_tp],
       "per_category_tp": per_cat}
json.dump(out, open(os.path.join(HERE, "category_analysis.json"), "w"),
          indent=1)

print(f"n={n} repeat-agree={agree} ({agree/n:.3f}) "
      f"flips +{flips_up}/-{flips_dn}")
print(f"predict_time n={pt_stats['n']} mean={pt_stats['mean']:.2f} "
      f"median={pt_stats['median']:.2f} p95={pt_stats['p95']:.2f}")
print(f"attribution mapped-match: all flagged {ok_all}/{tot_all} "
      f"({ok_all/tot_all:.3f}), true positives {ok_tp}/{tot_tp} "
      f"({ok_tp/tot_tp:.3f})")
for c, (h, t) in per_cat.items():
    print(f"  TP in {c}: model family match {h}/{t} "
          f"({h/t:.3f})" if t else f"  TP in {c}: none")
print("\ncross-tab (model category -> benchmark category, flagged images):")
for mc in sorted(xtab):
    row = ", ".join(f"{uc}: {ct}" for uc, ct in
                    sorted(xtab[mc].items(), key=lambda x: -x[1]))
    print(f"  {mc:>14}: {row}")
