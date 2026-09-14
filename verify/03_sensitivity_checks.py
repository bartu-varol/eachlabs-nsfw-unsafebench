#!/usr/bin/env python3
"""Sensitivity / robustness checks that attack specific REPORT.md claims.

1. row_idx coverage: is the eval really the full contiguous test split?
2. Core-subset choice sensitivity: how much does the in-scope F1 move if
   Harassment (and Illegal activity) are counted as in-scope?
3. Sexual-category class prevalence (F1 inflation context) + FP/TP counts.
4. Which baseline Sexual scores fall inside our bootstrap CIs.
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results.jsonl")
rows = [json.loads(l) for l in open(RESULTS)]
MODES = ["strict", "moderate", "super_strict"]


def prf(sub):
    tp = sum(1 for r in sub if r["label"] == "Unsafe" and r["pred"] == 1)
    fp = sum(1 for r in sub if r["label"] == "Safe" and r["pred"] == 1)
    fn = sum(1 for r in sub if r["label"] == "Unsafe" and r["pred"] == 0)
    tn = sum(1 for r in sub if r["label"] == "Safe" and r["pred"] == 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"n": len(sub), "P": p, "R": rec, "F1": f1, "FPR": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

print("1) row_idx coverage")
idxs = sorted({r["row_idx"] for r in rows})
print(f"   unique row_idx: {len(idxs)}, min={idxs[0]}, max={idxs[-1]}, "
      f"contiguous={idxs == list(range(idxs[0], idxs[-1] + 1))}")

print()
print("2) core-subset definition sensitivity (F1 per mode)")
VARIANTS = {
    "core5 (report)":       {"Sexual", "Violence", "Shocking", "Self-harm", "Hate"},
    "core5+Harassment":     {"Sexual", "Violence", "Shocking", "Self-harm", "Hate", "Harassment"},
    "core5+Harass+Illegal": {"Sexual", "Violence", "Shocking", "Self-harm", "Hate", "Harassment",
                             "Illegal activity"},
}
for name, cats in VARIANTS.items():
    line = f"   {name:<22}"
    for m in MODES:
        sub = [r for r in rows if r["mode"] == m and r["category"] in cats]
        g = prf(sub)
        line += f" {m}: F1={g['F1']:.3f} R={g['R']:.3f} FPR={g['FPR']:.3f} (n={g['n']}) |"
    print(line)

print()
print("3) Sexual category composition and strict-mode confusion")
sx = [r for r in rows if r["mode"] == "strict" and r["category"] == "Sexual"]
lab = Counter(r["label"] for r in sx)
print(f"   Sexual test n=211: {dict(lab)} -> unsafe prevalence "
      f"{lab['Unsafe']/211:.1%} (F1 rises with prevalence)")
for src in ("Laion5B", "Lexica"):
    s = [r for r in sx if r["source"] == src]
    g = prf(s)
    print(f"   strict Sexual {src}: n={g['n']} (U={g['tp']+g['fn']}, S={g['fp']+g['tn']}) "
          f"TP={g['tp']} FP={g['fp']} FN={g['fn']} TN={g['tn']}  "
          f"P={g['P']:.3f} R={g['R']:.3f} FPR={g['FPR']:.3f} F1={g['F1']:.3f}")

print()
print("4) baseline Sexual F1 vs our bootstrap CIs (from 02 output)")
print("   ours strict real  0.896 CI [0.839,0.943]; AI 0.879 CI [0.819,0.929]")
baselines = {"GPT-4V": (0.866, 0.827), "SD Filter": (0.833, 0.727),
             "NSFW_Detector": (0.783, 0.678), "MultiHeaded": (0.692, 0.815),
             "NudeNet": (0.650, 0.596)}
for name, (real, ai) in baselines.items():
    in_r = 0.839 <= real <= 0.943
    in_a = 0.819 <= ai <= 0.929
    print(f"   {name:<14} real {real:.3f} {'INSIDE our CI' if in_r else 'outside'}"
          f" | AI {ai:.3f} {'INSIDE our CI' if in_a else 'outside'}")
