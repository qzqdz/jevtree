#!/usr/bin/env bash
# Verified 5-minute smoke for jevtree decide (loan CSV demo).
# Requires: Python >= 3.11, editable install, and .env with JEVTREE_LLM_*
# (legacy META_JEV_LLM_* also accepted). Writes under results/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="examples/data/bring_your_csv/loan_approve.csv"
OUT="results/loan_demo"
GOAL="按是否批准贷款做决策树 / Grow a loan-approval decision tree"
SAMPLE_OBS="examples/artifacts/bring_your_csv/sample_obs.json"

if [[ ! -f "$DATA" ]]; then
  echo "quickstart: missing demo data: $DATA" >&2
  exit 2
fi

if [[ ! -f .env ]]; then
  echo "quickstart: missing .env — copy .env.example and set JEVTREE_LLM_API_KEY" >&2
  exit 2
fi

if [[ -x .venv/bin/jevtree ]]; then
  JEV=".venv/bin/jevtree"
elif command -v jevtree >/dev/null 2>&1; then
  JEV="$(command -v jevtree)"
else
  echo "quickstart: jevtree not found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
  exit 2
fi

rm -rf "$OUT"
echo "quickstart: $JEV decide --data $DATA --out $OUT"
"$JEV" decide --data "$DATA" --goal "$GOAL" --out "$OUT"

for f in feature_table.json tree.json sop.json story.md; do
  if [[ ! -s "$OUT/$f" ]]; then
    echo "quickstart: missing or empty artifact: $OUT/$f" >&2
    exit 1
  fi
done

if [[ -f "$SAMPLE_OBS" ]]; then
  echo "quickstart: jevtree run --sop $OUT/sop.json --trace"
  "$JEV" run --sop "$OUT/sop.json" --input "$SAMPLE_OBS" --trace >/dev/null
fi

echo "quickstart: OK → $OUT"
exit 0
