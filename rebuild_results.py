#!/usr/bin/env python3
"""Rebuild the full results.jsonl by joining predictions.jsonl with
UnsafeBench labels fetched from the Hugging Face datasets-server.

UnsafeBench is a gated dataset; its labels are not redistributed here.
You need an HF account that has accepted the dataset terms and a read
token in the HF_TOKEN environment variable (or a .hf_token file).

Usage: python3 rebuild_results.py   ->  writes results.jsonl
"""
import json
import os
import sys
import time

import requests

DS = "yiting/UnsafeBench"
ROWS_API = "https://datasets-server.huggingface.co/rows"


def hf_token():
    t = os.environ.get("HF_TOKEN")
    if not t and os.path.exists(".hf_token"):
        t = open(".hf_token").read().strip()
    if not t:
        sys.exit("Set HF_TOKEN (account must have accepted the UnsafeBench gate)")
    return t


def fetch_meta(tok):
    meta, offset = {}, 0
    while True:
        r = requests.get(ROWS_API,
                         params={"dataset": DS, "config": "default",
                                 "split": "test", "offset": offset,
                                 "length": 100},
                         headers={"Authorization": f"Bearer {tok}"},
                         timeout=60)
        r.raise_for_status()
        rows = r.json()["rows"]
        if not rows:
            break
        for row in rows:
            d = row["row"]
            meta[row["row_idx"]] = {"label": d.get("safety_label"),
                                    "category": d.get("category"),
                                    "source": d.get("source")}
        offset += 100
        time.sleep(0.3)
    return meta


def main():
    meta = fetch_meta(hf_token())
    n = 0
    with open("results.jsonl", "w") as out:
        for line in open("predictions.jsonl"):
            p = json.loads(line)
            m = meta.get(p["row_idx"])
            if not m:
                continue
            out.write(json.dumps({**p, **m}) + "\n")
            n += 1
    print(f"results.jsonl rebuilt: {n} rows")


if __name__ == "__main__":
    main()
