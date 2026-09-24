# Benchmark results

Hard-budget accuracy / macro-F1 from `jevtree eval-afa`. Raw summaries are in [`snapshots/`](snapshots/).

## Protocol

| Item | Value |
|------|--------|
| Scoring entry | `jevtree eval-afa --config configs/…` |
| Metrics | Acc / macro-F1 vs hard feature budget (`src/jevtree/eval/metrics.py`) |
| MiniBooNE | UCI MiniBooNE PID; `n_train=2000`, `n_test=500`, `split_seed=0`, 50 features |
| Predictor (fair table) | Shared `logistic_impute` (`MaskedLogisticPredictor`) |
| Cube | Synthetic `cube_without_noise`, `n_samples=256`, budgets `{1,2,3}`, `split_seed=0` |

## MiniBooNE — Acc @ budget (shared logistic predictor)

Source table:
[`docs/snapshots/ablation_miniboone_disc_logistic_20260922T181304__comparison.json`](snapshots/ablation_miniboone_disc_logistic_20260922T181304__comparison.json)

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

## Cube without noise — Acc @ budget

Same-wave summaries (`policy.predict` / match-majority style builtin; `n_test=77`):

| Policy | Acc@1 | Acc@2 | Acc@3 | Snapshot |
|--------|------:|------:|------:|----------|
| `ig_static` | 0.688 | 0.909 | **1.000** | [`…ig_static_…175144Z__summary.json`](snapshots/cube_without_noise_ig_static_20260922T175144Z__summary.json) |
| `sequential` | 0.688 | 0.909 | **1.000** | [`…sequential_…175144Z__summary.json`](snapshots/cube_without_noise_sequential_20260922T175144Z__summary.json) |
| `random` | 0.701 | 0.740 | 0.766 | [`…random_…175144Z__summary.json`](snapshots/cube_without_noise_random_20260922T175144Z__summary.json) |

On this synthetic cube, static IG order and sequential feature order both reach
perfect Acc@3; random does not.

## Reproduce

```bash
# Cube
jevtree eval-afa --config configs/eval_cube_hard.json
jevtree eval-afa --config configs/eval_cube_random.json
jevtree eval-afa --config configs/eval_cube_sequential.json

# MiniBooNE shared-predictor set (needs cached UCI file under data/cache/)
jevtree eval-afa --config configs/eval_miniboone_disc_logistic.json
jevtree eval-afa --config configs/eval_miniboone_ig_static_logistic.json
jevtree eval-afa --config configs/eval_miniboone_random_logistic.json
jevtree eval-afa --config configs/eval_miniboone_sequential_logistic.json
jevtree eval-afa --config configs/eval_miniboone_ig_conditional_logistic.json

# Aggregate
PYTHONPATH=src python3 -m jevtree.tools.compare_results \
  --label Disc=results/<disc_run>/summary.json \
  --label IG_static=results/<static_run>/summary.json \
  …
```
