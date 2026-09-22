# Meta-Jev

**Bring your data or materials + a goal → grow an auditable decision tree / DecisionSOP.**

Meta-Jev is a universal decision product. Vertical stories (loan approve, support routing, homework scoring, tickets) are **presets/examples** on one spine — not separate apps.

```
Input (any of):  table | labeled text batch | scored answers | messy corpus + goal
        ↓ normalize
   FeatureTable  (rows, feature_keys, label_key, label_kind)
        ↓ grow (IG)
   Decision tree / SOP  (ask fewer cues → decide)
        ↓ run
   Budgeted acquisition + --trace + exportable SOP artifact
```

> Research / AFABench iteration continues under `meta-jev eval-afa`. Demos must **not** claim Acc@budget / SOTA.

## Project docs (Feishu Wiki)

- **Wiki**: [Meta-Jev](https://hcnzpmamw7fl.feishu.cn/wiki/PeOfwAaTOik9cDkCOlWcschanEf)
- **Index**: [项目文档索引](https://hcnzpmamw7fl.feishu.cn/wiki/EmQ2wJRoqiQIcRk7nZsc5Eb7ntf)

## Five-minute product path

```bash
pip install -e .
# or: PYTHONPATH=src python3 -m meta_jev --help

# A) Clean feature CSV (offline, no LLM)
meta-jev grow --csv examples/data/bring_your_csv/loan_approve.csv \
  --label approve --out /tmp/tree.json --sop-out /tmp/tree.sop.json --seed 42
# prints: 从你的表长出的决策树：先问 X，再问 Y，决定 approve
printf '{"income_band":"high","credit_history":"good","employment_years":"5plus","debt_ratio":"low","collateral":"yes"}' \
  > /tmp/obs.json
meta-jev run --sop /tmp/tree.sop.json --input /tmp/obs.json --trace

# one-command demo
.venv/bin/python examples/bring_your_csv.py

# B) Batch texts + labels (classification) — offline keyword cues
meta-jev ingest-batch --csv examples/data/batch_texts/support_emails.csv \
  --label label --out-csv /tmp/email_feats.csv \
  --out /tmp/email_tree.json --sop-out /tmp/email.sop.json
# or: .venv/bin/python examples/batch_texts_to_tree.py

# C) Grading / scoring batch (homework, exams) — same door, ordinal scores
meta-jev ingest-batch --csv examples/data/homework_scoring/short_answers.csv \
  --label score --label-kind ordinal \
  --out-csv /tmp/hw_feats.csv --out /tmp/hw_tree.json --sop-out /tmp/hw.sop.json
# Framing: interpretable scoring SOP over cues — not an automatic grader SOTA.

# D) Messy paste + one-line goal (LLM optional; needs .env key)
# meta-jev ingest-text --goal "分流到哪个团队" --input examples/data/messy_notes_sample.txt \
#   --out-csv /tmp/messy.csv --out /tmp/messy_tree.json --sop-out /tmp/messy.sop.json
# or: .venv/bin/python examples/messy_to_tree.py
```

If the LLM key is missing, ingest-text fails clearly and points you to CSV / text-batch — it never invents silent fake rows.

## How scenarios map to the spine

| Scenario | Ingest door | `label_kind` | Example data |
|----------|-------------|--------------|--------------|
| Bring clean CSV | `grow --csv` | categorical / ordinal | `bring_your_csv/loan_approve.csv` |
| Text classification batch | `ingest-batch` | categorical | `batch_texts/support_emails.csv` |
| Homework / exam scoring | `ingest-batch` | ordinal (or `numeric_binned`) | `homework_scoring/short_answers.csv` |
| Messy notes + goal | `ingest-text` | from LLM JSON | `messy_notes_sample.txt` |
| Ticket routing (optional) | `--dataset ticket_routing` | categorical | `ticket_routing.csv` |

## Official scoring (research only)

```bash
meta-jev eval-afa --config configs/eval_miniboone_hard.yaml
```

`eval-afa` is the **only** path to Acc/F1-vs-budget. Examples must not claim leaderboard numbers.

## CLI surface

```bash
meta-jev grow --help          # CSV / cube / ticket_routing → tree+SOP+story
meta-jev ingest-batch --help  # texts+labels/scores → keyword FeatureTable → grow
meta-jev ingest-text --help   # messy + goal → LLM table → grow
meta-jev run --sop ... --trace
meta-jev validate --help
meta-jev eval-afa --help
meta-jev version
```

## Environment

Python ≥3.11.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.lock
# optional LLM for messy path — never commit secrets
cp .env.example .env   # fill META_JEV_LLM_API_KEY
```

## Honest limits

- Text-batch v1 uses **keyword presence** features (offline) — interpretable cues, not embeddings / SOTA NLP.
- Messy ingest needs a working mimo-compatible key; otherwise use CSV or text-batch.
- Grown trees are small IG demos for auditable SOPs under a question/cue budget metaphor — not production classifiers or auto-graders.
- Do not claim Acc@budget outside `eval-afa`.

## Optional older demo

```bash
.venv/bin/python examples/ticket_routing.py   # ticket preset on the same spine
```
