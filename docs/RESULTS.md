# Benchmark results (inventory snapshot)

**Not a SOTA claim.** These numbers are internal hard-budget Acc/F1 curves from
`jevtree eval-afa`, compared only to in-repo baselines (`random`, `sequential`)
and IG policy variants — **not** vs GDFS / DIME / official AFABench leaderboard
entries. Demo CSVs under `examples/data/` (loan, iris, tennis, …) are **not**
benchmarks; see [`examples/data/SOURCES.md`](../examples/data/SOURCES.md).

Frozen JSON cited below lives in [`docs/snapshots/`](snapshots/) (copied from
local gitignored `results/` dumps). Every cell is attributable to a path in that
folder.

## Protocol (shared)

| Item | Value |
|------|--------|
| Scoring entry | `jevtree eval-afa --config configs/…` |
| Metrics | Acc / macro-F1 vs hard feature budget (`src/jevtree/eval/metrics.py`) |
| MiniBooNE | UCI MiniBooNE PID; `n_train=2000`, `n_test=500`, `split_seed=0`, 50 features |
| Predictor (fair table) | Shared `logistic_impute` (`MaskedLogisticPredictor`) |
| Cube smoke | Synthetic `cube_without_noise`, `n_samples=256`, budgets `{1,2,3}`, `split_seed=0` |
| Run dates (local) | 2026-09-22 (UTC timestamps in filenames) |
| Aggregation | `src/jevtree/tools/compare_results.py` (reads existing `summary.json` only; no rescoring) |

## MiniBooNE — Acc @ budget (shared logistic predictor)

Source table:
[`docs/snapshots/ablation_miniboone_disc_logistic_20260922T181304__comparison.json`](snapshots/ablation_miniboone_disc_logistic_20260922T181304__comparison.json)
(created `2026-09-22T18:13:04Z`; note: *Aggregated from existing summary.json only*).

Underlying runs (same predictor `logistic_impute`):

| Label | Policy | Snapshot |
|-------|--------|----------|
| Disc | `ig_discriminative` | [`…181256Z__summary.json`](snapshots/miniboone_ig_discriminative_20260922T181256Z__summary.json) |
| IG_static | `ig_static` | [`…180735Z__summary.json`](snapshots/miniboone_ig_static_20260922T180735Z__summary.json) |
| Random | `random` | [`…180736Z__summary.json`](snapshots/miniboone_random_20260922T180736Z__summary.json) |
| Sequential | `sequential` | [`…180737Z__summary.json`](snapshots/miniboone_sequential_20260922T180737Z__summary.json) |

| Budget | Disc Acc | IG_static Acc | Random Acc | Sequential Acc |
|-------:|---------:|--------------:|-----------:|---------------:|
| 5 | 0.824 | 0.814 | 0.746 | 0.724 |
| 10 | 0.820 | 0.822 | 0.766 | 0.728 |
| 20 | 0.814 | 0.822 | 0.760 | 0.792 |
| 40 | 0.828 | 0.814 | 0.800 | 0.822 |

Corresponding F1 (same file):

| Budget | Disc F1 | IG_static F1 | Random F1 | Sequential F1 |
|-------:|--------:|-------------:|----------:|--------------:|
| 5 | 0.738 | 0.697 | 0.499 | 0.420 |
| 10 | 0.734 | 0.733 | 0.567 | 0.421 |
| 20 | 0.730 | 0.733 | 0.571 | 0.659 |
| 40 | 0.753 | 0.735 | 0.694 | 0.738 |

`ig_conditional` on the same predictor (separate aggregation
[`…logistic_20260922T180805__comparison.json`](snapshots/ablation_miniboone_logistic_20260922T180805__comparison.json);
shares the same Random / Sequential / IG_static rows):

| Budget | IG_cond Acc | IG_cond F1 |
|-------:|------------:|-----------:|
| 5 | 0.798 | 0.647 |
| 10 | 0.796 | 0.657 |
| 20 | 0.796 | 0.671 |
| 40 | 0.832 | 0.752 |

## Cube without noise — Acc @ budget (smoke)

Same-wave summaries (`policy.predict` / match-majority style builtin; `n_test=77`):

| Policy | Acc@1 | Acc@2 | Acc@3 | Snapshot |
|--------|------:|------:|------:|----------|
| `ig_static` | 0.688 | 0.909 | **1.000** | [`…ig_static_…175144Z__summary.json`](snapshots/cube_without_noise_ig_static_20260922T175144Z__summary.json) |
| `sequential` | 0.688 | 0.909 | **1.000** | [`…sequential_…175144Z__summary.json`](snapshots/cube_without_noise_sequential_20260922T175144Z__summary.json) |
| `random` | 0.701 | 0.740 | 0.766 | [`…random_…175144Z__summary.json`](snapshots/cube_without_noise_random_20260922T175144Z__summary.json) |

On this synthetic cube, static IG order and sequential feature order both reach
perfect Acc@3; random does not. Treat as a protocol smoke, not a public SOTA curve.

## What was **not** evaluated (gaps)

| Gap | Status |
|-----|--------|
| Official AFABench snakemake / GDFS / DIME numbers | **Missing** — adapter is protocol-shaped only; no imported AFABench scores |
| Diabetes / BankMarketing / CUBE-NM public curves | Config exists for diabetes (`configs/eval_diabetes_hard.json`); **no** `results/` dump found |
| Loan / iris / tennis / SMS demos | Teaching data only — **no** Acc@budget tables |
| Clean multi-seed / full-dataset MiniBooNE | Single `split_seed=0` subsample (2k/500); later dirty-tree re-runs exist under gitignored `results/` but were **not** used for the README table |
| Claimed SOTA wording in repo/docs | **None** (examples explicitly forbid AFABench SOTA claims) |

## How to reproduce / refresh

```bash
# Cube smoke
jevtree eval-afa --config configs/eval_cube_hard.json
jevtree eval-afa --config configs/eval_cube_random.json
jevtree eval-afa --config configs/eval_cube_sequential.json

# MiniBooNE shared-predictor set (needs cached UCI file under data/cache/)
jevtree eval-afa --config configs/eval_miniboone_disc_logistic.json
jevtree eval-afa --config configs/eval_miniboone_ig_static_logistic.json
jevtree eval-afa --config configs/eval_miniboone_random_logistic.json
jevtree eval-afa --config configs/eval_miniboone_sequential_logistic.json
jevtree eval-afa --config configs/eval_miniboone_ig_conditional_logistic.json

# Aggregate without rescoring
PYTHONPATH=src python3 -m jevtree.tools.compare_results \
  --label Disc=results/<disc_run>/summary.json \
  --label IG_static=results/<static_run>/summary.json \
  …
```

To turn a fresh `results/*/summary.json` wave into a README table: run
`compare_results`, copy the new `comparison.json` + cited `summary.json` into
`docs/snapshots/`, update this file and the short README tables, and keep the
footnote paths in sync. Do **not** invent cells.
