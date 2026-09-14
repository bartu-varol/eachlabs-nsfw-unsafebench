#!/usr/bin/env python3
"""Benchmark eachlabs-nsfw-checker on UnsafeBench (HF: yiting/UnsafeBench).

Pulls rows (with signed image URLs) from the HF datasets-server rows API,
feeds them to the each::labs NSFW Checker in batches of 10, and writes one
JSONL record per (image, mode). Re-runs skip already-done work, so the
script is safe to interrupt and resume.

Usage:
  python3 unsafebench_eval.py --modes strict,moderate,super_strict --limit 0
  python3 unsafebench_eval.py --report            # metrics from results.jsonl

Auth files next to this script:
  .hf_token   HF read token of an account that accepted the UnsafeBench gate
  .each_env   line: EACH_API_KEY=...
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DS = "yiting/UnsafeBench"
ROWS_API = "https://datasets-server.huggingface.co/rows"
EACH_API = "https://api.eachlabs.ai/v1/prediction/"
RESULTS = os.path.join(HERE, "results.jsonl")
PAGE = 100
BATCH = 10
POLL_S = 2.5
POLL_MAX = 120


def hf_token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        p = os.path.join(HERE, ".hf_token")
        if os.path.exists(p):
            t = open(p).read().strip()
    if not t:
        sys.exit("No HF token: put it in .hf_token or export HF_TOKEN")
    return t


def each_key():
    p = os.path.join(HERE, ".each_env")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("EACH_API_KEY="):
                return line.split("=", 1)[1].strip()
    k = os.environ.get("EACH_API_KEY")
    if not k:
        sys.exit("No EACH_API_KEY: put .each_env next to script")
    return k


def fetch_page(split, offset, tok):
    r = requests.get(
        ROWS_API,
        params={"dataset": DS, "config": "default", "split": split,
                "offset": offset, "length": PAGE},
        headers={"Authorization": f"Bearer {tok}"},
        timeout=60,
    )
    r.raise_for_status()
    out = []
    for row in r.json()["rows"]:
        d = row["row"]
        img = d.get("image") or {}
        src = img.get("src")
        if not src:
            continue
        out.append({
            "row_idx": row["row_idx"],
            "url": src,
            "label": d.get("safety_label"),
            "category": d.get("category"),
            "source": d.get("source"),
        })
    return out


def predict(key, urls, mode, retries=2):
    body = {"model": "eachlabs-nsfw-checker",
            "input": {"images": urls, "mode": mode}, "webhook_url": ""}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            r = requests.post(EACH_API, json=body, headers=hdr, timeout=60)
            if r.status_code == 429:
                time.sleep(10)
                continue
            r.raise_for_status()
            pid = r.json()["predictionID"]
            t0 = time.time()
            while time.time() - t0 < POLL_MAX:
                time.sleep(POLL_S)
                g = requests.get(EACH_API + pid, headers=hdr, timeout=60)
                if g.status_code >= 500:
                    continue
                j = g.json()
                st = j.get("status")
                if st == "success":
                    return j["output"].get("images"), j.get("metrics", {})
                if st == "error":
                    raise RuntimeError(j.get("message") or "prediction error")
            raise RuntimeError("poll timeout")
        except Exception as e:
            if attempt == retries:
                return None, {"error": str(e)}
            time.sleep(3 * (attempt + 1))
    return None, {"error": "unreachable"}


def load_done():
    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                if r.get("pred") is not None:
                    done.add((r["split"], r["row_idx"], r["mode"]))
            except Exception:
                pass
    return done


def run(args):
    tok, key = hf_token(), each_key()
    modes = args.modes.split(",")
    done = load_done()
    print(f"{len(done)} results already on disk")
    fout = open(RESULTS, "a")
    lock = threading.Lock()
    n_new = 0
    spent = 0.0

    def handle_batch(batch, mode, split):
        nonlocal n_new, spent
        preds, metrics = predict(key, [b["url"] for b in batch], mode)
        with lock:
            for i, b in enumerate(batch):
                rec = {"split": split, "row_idx": b["row_idx"], "mode": mode,
                       "label": b["label"], "category": b["category"],
                       "source": b["source"]}
                if preds is not None and i < len(preds):
                    rec["pred"] = preds[i]
                else:
                    rec["pred"] = None
                    rec["error"] = metrics.get("error", "no output")
                fout.write(json.dumps(rec) + "\n")
                n_new += 1
            fout.flush()
            if preds is not None:
                spent += metrics.get("cost", 0) or 0

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        offset, seen = 0, 0
        while True:
            page = fetch_page(args.split, offset, tok)
            if not page:
                break
            if args.limit:
                page = page[:max(0, args.limit - seen)]
            futures = []
            for mode in modes:
                todo = [b for b in page
                        if (args.split, b["row_idx"], mode) not in done]
                for i in range(0, len(todo), BATCH):
                    futures.append(ex.submit(handle_batch,
                                             todo[i:i + BATCH], mode,
                                             args.split))
            for f in cf.as_completed(futures):
                f.result()
            seen += len(page)
            print(f"offset {offset}: done, {n_new} new results, "
                  f"~${spent:.2f} this run", flush=True)
            offset += PAGE
            if args.limit and seen >= args.limit:
                break
    fout.close()
    print(f"DONE: {n_new} new results, ~${spent:.2f} spent this run")


def report():
    if not os.path.exists(RESULTS):
        sys.exit("no results.jsonl yet")
    rows = [json.loads(l) for l in open(RESULTS)]
    uniq = {}
    for r in rows:
        k = (r["split"], r["row_idx"], r["mode"])
        if r.get("pred") is not None or k not in uniq:
            uniq[k] = r
    rows = list(uniq.values())
    errs = [r for r in rows if r["pred"] is None]
    rows = [r for r in rows if r["pred"] is not None]
    print(f"{len(rows)} scored, {len(errs)} errored\n")

    def prf(sub):
        tp = sum(1 for r in sub if r["label"] == "Unsafe" and r["pred"] == 1)
        fp = sum(1 for r in sub if r["label"] == "Safe" and r["pred"] == 1)
        fn = sum(1 for r in sub if r["label"] == "Unsafe" and r["pred"] == 0)
        tn = sum(1 for r in sub if r["label"] == "Safe" and r["pred"] == 0)
        n = tp + fp + fn + tn
        acc = (tp + tn) / n if n else 0
        p = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0
        return n, acc, p, rec, f1

    for mode in sorted({r["mode"] for r in rows}):
        sub = [r for r in rows if r["mode"] == mode]
        n, acc, p, rec, f1 = prf(sub)
        print(f"== mode={mode}  n={n}  acc={acc:.3f}  P={p:.3f}  "
              f"R={rec:.3f}  F1={f1:.3f}")
        for key, name in (("source", "source"), ("category", "category")):
            groups = collections.defaultdict(list)
            for r in sub:
                groups[r[key]].append(r)
            for g in sorted(groups):
                n, acc, p, rec, f1 = prf(groups[g])
                print(f"   {name}={g:<22} n={n:<5} acc={acc:.3f} "
                      f"P={p:.3f} R={rec:.3f} F1={f1:.3f}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", default="strict,moderate,super_strict")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run(a)
