#!/usr/bin/env python3
"""Per-image category collection: strict mode, batch size 1, full test split.

The API returns one category per request, so per-image category attribution
requires batch-size-1 calls. Also records per-call predict_time and doubles
as a full-split alignment check (batch-1 removes ordering ambiguity).

Resumable: skips row_idxs already present in category_results.jsonl.
Cost: ~$0.001/image, ~$2.04 for the full 2,037-image split.
"""
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from unsafebench_eval import fetch_page, hf_token, each_key, EACH_API  # noqa: E402

OUTF = os.path.join(HERE, "category_results.jsonl")
MODE = "strict"


def predict_one(key, url):
    body = {"model": "eachlabs-nsfw-checker",
            "input": {"images": [url], "mode": MODE}, "webhook_url": ""}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(EACH_API, json=body, headers=hdr, timeout=60)
            if r.status_code == 429:
                time.sleep(10)
                continue
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
                    out = j.get("output") or {}
                    preds = out.get("images") or []
                    return {"pred": preds[0] if preds else None,
                            "category": out.get("category"),
                            "flagged": out.get("flagged"),
                            "predict_time": (j.get("metrics") or {}).get("predict_time")}
                if st == "error":
                    raise RuntimeError(j.get("message") or "prediction error")
            raise RuntimeError("poll timeout")
        except Exception as e:
            if attempt == 2:
                return {"pred": None, "error": str(e)}
            time.sleep(3 * (attempt + 1))
    return {"pred": None, "error": "unreachable"}


def main():
    tok, key = hf_token(), each_key()
    done = set()
    if os.path.exists(OUTF):
        for line in open(OUTF):
            try:
                r = json.loads(line)
                if r.get("pred") is not None:
                    done.add(r["row_idx"])
            except Exception:
                pass
    print(f"{len(done)} already collected")
    fout = open(OUTF, "a")
    lock = threading.Lock()
    n = 0

    def work(b):
        nonlocal n
        res = predict_one(key, b["url"])
        rec = {"row_idx": b["row_idx"], "mode": MODE, **res}
        with lock:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n} done", flush=True)

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        offset = 0
        futures = []
        while True:
            page = fetch_page("test", offset, tok)
            if not page:
                break
            for b in page:
                if b["row_idx"] not in done:
                    futures.append(ex.submit(work, b))
            offset += 100
        for f in cf.as_completed(futures):
            f.result()
    fout.close()
    print(f"DONE: {n} new records in {OUTF}")


if __name__ == "__main__":
    main()
