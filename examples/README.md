# Examples (demos only — not Acc@budget)

Meta-Jev product spine: **any materials + goal → FeatureTable → IG tree/SOP → trace**.

Do **not** claim leaderboard metrics here. Scoring stays in `meta-jev eval-afa`.

## Generic (start here)

```bash
# 1) Clean feature CSV
python examples/bring_your_csv.py

# 2) Batch labeled texts (classification)
python examples/batch_texts_to_tree.py \
  --csv examples/data/batch_texts/support_emails.csv --label label

# 3) Homework / exam scoring batch (ordinal scores)
python examples/batch_texts_to_tree.py \
  --csv examples/data/homework_scoring/short_answers.csv \
  --label score --label-kind ordinal \
  --out-dir examples/artifacts/homework_scoring

# 4) Messy paste + goal (needs .env LLM key)
python examples/messy_to_tree.py --goal "分流到哪个团队"
```

## Optional vertical example

- `ticket_routing.py` — older five-minute ticket story (preset on the same spine)

## Data

| Path | Role |
|------|------|
| `data/bring_your_csv/loan_approve.csv` | structured tabular |
| `data/bring_your_csv/course_pass.csv` | structured tabular |
| `data/batch_texts/support_emails.csv` | text + class label |
| `data/homework_scoring/short_answers.csv` | text + score 1–5 |
| `data/messy_notes_sample.txt` | free-form paste |
| `data/ticket_routing.csv` | optional ticket preset |
