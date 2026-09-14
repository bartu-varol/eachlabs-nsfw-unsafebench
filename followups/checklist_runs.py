#!/usr/bin/env python3
"""A.7 pre-publication checklist runs.

1. Determinism re-run: 200 seeded-random images, strict mode, batch 10
   (same conditions as the original run); compare against results.jsonl.
2. Alignment spot-check: 50 different seeded-random images, strict mode,
   batch size 1 (removes any ordering ambiguity); compare per image.
3. FP-audit fetch: download super_strict false positives on Safe images in
   out-of-scope categories (up to 5 per category) to fp_audit/ for review.
4. Raw-response capture: full API JSON of one batch, to establish whether
   the response exposes a category output at all.

Cost: ~250 classifications = ~$0.25. Everything is logged to
checklist_results.json; the script is idempotent per section via flags.
"""
import json
import os
import random
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from unsafebench_eval import fetch_page, hf_token, each_key, EACH_API  # noqa: E402

RESULTS = os.path.join(ROOT, "results.jsonl")
OUT = os.path.join(HERE, "checklist_results.json")
RAW = os.path.join(HERE, "checklist_raw_response.json")
FPDIR = os.path.join(HERE, "fp_audit")

OUT_OF_SCOPE = {"Harassment", "Illegal activity", "Deception", "Political",
                "Public and personal health", "Spam"}


def load_orig():
    orig = {}
    for line in open(RESULTS):
        r = json.loads(line)
        orig[(r["mode"], r["row_idx"])] = r
    return orig


def pages_for(row_idxs, tok, cache):
    """Fetch HF pages covering the given row_idxs; return row_idx -> row."""
    need_offsets = sorted({(i // 100) * 100 for i in row_idxs})
    rows = {}
    for off in need_offsets:
        if off not in cache:
            cache[off] = fetch_page("test", off, tok)
            time.sleep(0.3)
        for b in cache[off]:
            rows[b["row_idx"]] = b
    return rows


def predict_raw(key, urls, mode):
    body = {"model": "eachlabs-nsfw-checker",
            "input": {"images": urls, "mode": mode}, "webhook_url": ""}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.post(EACH_API, json=body, headers=hdr, timeout=60)
    r.raise_for_status()
    pid = r.json()["predictionID"]
    t0 = time.time()
    while time.time() - t0 < 120:
        time.sleep(2.5)
        g = requests.get(EACH_API + pid, headers=hdr, timeout=60)
        if g.status_code >= 500:
            continue
        j = g.json()
        st = j.get("status")
        if st == "success":
            return j
        if st == "error":
            raise RuntimeError(j.get("message") or "prediction error")
    raise RuntimeError("poll timeout")


def main():
    tok, key = hf_token(), each_key()
    orig = load_orig()
    all_idx = sorted({i for (_, i) in orig})
    state = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cache = {}

    # ---- 1. determinism: 200 images, strict, batch 10 ----------------
    if "determinism" not in state:
        rng = random.Random(42)
        sample = sorted(rng.sample(all_idx, 200))
        rows = pages_for(sample, tok, cache)
        recs, raw_saved = [], False
        for i in range(0, len(sample), 10):
            chunk = [rows[j] for j in sample[i:i + 10]]
            j = predict_raw(key, [c["url"] for c in chunk], "strict")
            if not raw_saved:
                json.dump(j, open(RAW, "w"), indent=1)
                raw_saved = True
            preds = j["output"].get("images")
            for k, c in enumerate(chunk):
                new = preds[k] if preds and k < len(preds) else None
                old = orig[("strict", c["row_idx"])]["pred"]
                recs.append({"row_idx": c["row_idx"], "old": old, "new": new})
            done = sum(1 for r in recs)
            print(f"determinism {done}/200", flush=True)
        agree = sum(1 for r in recs if r["old"] == r["new"])
        state["determinism"] = {"n": len(recs), "agree": agree,
                                "mismatches": [r for r in recs
                                               if r["old"] != r["new"]]}
        json.dump(state, open(OUT, "w"), indent=1)
        print(f"DETERMINISM: {agree}/{len(recs)} agree")

    # ---- 2. alignment: 50 images, strict, batch size 1 ---------------
    if "alignment" not in state:
        rng = random.Random(43)
        det_idx = {r["row_idx"] for r in state["determinism"]["mismatches"]}
        pool = [i for i in all_idx if i not in det_idx]
        sample = sorted(rng.sample(pool, 50))
        rows = pages_for(sample, tok, cache)
        recs = []
        for n, j_idx in enumerate(sample, 1):
            c = rows[j_idx]
            j = predict_raw(key, [c["url"]], "strict")
            preds = j["output"].get("images")
            new = preds[0] if preds else None
            old = orig[("strict", c["row_idx"])]["pred"]
            recs.append({"row_idx": c["row_idx"], "old": old, "new": new})
            if n % 10 == 0:
                print(f"alignment {n}/50", flush=True)
        agree = sum(1 for r in recs if r["old"] == r["new"])
        state["alignment"] = {"n": len(recs), "agree": agree,
                              "mismatches": [r for r in recs
                                             if r["old"] != r["new"]]}
        json.dump(state, open(OUT, "w"), indent=1)
        print(f"ALIGNMENT: {agree}/{len(recs)} agree")

    # ---- 3. FP audit image fetch (no inference) -----------------------
    if "fp_audit_fetched" not in state:
        os.makedirs(FPDIR, exist_ok=True)
        fps = [r for (m, i), r in orig.items()
               if m == "super_strict" and r["label"] == "Safe"
               and r["pred"] == 1 and r["category"] in OUT_OF_SCOPE]
        rng = random.Random(44)
        picked = []
        by_cat = {}
        for r in fps:
            by_cat.setdefault(r["category"], []).append(r)
        for cat, lst in sorted(by_cat.items()):
            picked += rng.sample(lst, min(5, len(lst)))
        rows = pages_for([p["row_idx"] for p in picked], tok, cache)
        saved = []
        for p in picked:
            c = rows.get(p["row_idx"])
            if not c:
                continue
            cat = p["category"].replace(" ", "_")
            fn = os.path.join(FPDIR, f"{p['row_idx']}_{cat}.jpg")
            img = requests.get(c["url"], timeout=60)
            open(fn, "wb").write(img.content)
            saved.append({"row_idx": p["row_idx"], "category": p["category"],
                          "source": p["source"], "file": os.path.basename(fn)})
        state["fp_audit_fetched"] = saved
        json.dump(state, open(OUT, "w"), indent=1)
        print(f"FP AUDIT: saved {len(saved)} images to fp_audit/")

    print("ALL DONE")


if __name__ == "__main__":
    main()
