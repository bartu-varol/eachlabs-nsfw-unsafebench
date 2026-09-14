# each::labs NSFW Checker on UnsafeBench: evaluation artifacts

Raw predictions and analysis code for the technical report
"each::labs NSFW Checker on UnsafeBench: A Transparent Evaluation of a
Production Image-Safety API at Three Policy Operating Points"
(each::labs, September 2026).

## Contents

- `predictions.jsonl`: 6,111 rows, one per (image, mode). Fields:
  `split`, `row_idx` (index into the official UnsafeBench test split),
  `mode` (moderate / strict / super_strict), `pred` (1 = flagged Unsafe).
- `unsafebench_eval.py`: the evaluation client used for the run
  (2026-09-10, batches of 10, images-only input).
- `rebuild_results.py`: joins `predictions.jsonl` with UnsafeBench labels
  to produce `results.jsonl`, the input for the verification scripts.
- `verify/01_verify_metrics.py`: data integrity and every point estimate
  in the report.
- `verify/02_bootstrap_and_tests.py`: bootstrap CIs (percentile, B = 5,000,
  seed 42), real-minus-AI difference CIs, exact McNemar tests between modes,
  per-category FPR.
- `verify/03_sensitivity_checks.py`: coverage checks, core-subset
  sensitivity, Sexual-category confusion detail, CI-versus-baseline checks.

## Reproduce

UnsafeBench is a gated dataset. Labels are not redistributed in this
repository; you need a Hugging Face account that has accepted the dataset
terms at https://huggingface.co/datasets/yiting/UnsafeBench.

```
export HF_TOKEN=hf_...        # account with UnsafeBench access
python3 rebuild_results.py    # writes results.jsonl (predictions + labels)
python3 verify/01_verify_metrics.py
python3 verify/02_bootstrap_and_tests.py
python3 verify/03_sensitivity_checks.py
```

Re-running the evaluation itself requires an each::labs API key
(`EACH_API_KEY`) and costs about $6 for the full 3-mode run:

```
python3 unsafebench_eval.py --split test
```

## Citation

Benchmark: Qu et al., "UnsafeBench: Benchmarking Image Safety Classifiers
on Real-World and AI-Generated Images", ACM CCS 2025, arXiv:2405.03486.
