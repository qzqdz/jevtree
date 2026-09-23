# Examples

**唯一产品入口 / only product entry:**

```bash
meta-jev decide --data <path> --goal "自然语言需求" --out results/demo
```

Aliases: `meta-jev run-job` / `meta-jev from-data`.

Offline mocked demo (no API key):

```bash
PYTHONPATH=src python3 examples/decide_demo.py
```

Core entropy textbook check (not a product door):

```bash
PYTHONPATH=src python3 examples/entropy_split_demo.py
```

## Sample data

| Path | Role |
|------|------|
| `data/bring_your_csv/loan_approve.csv` | Structured table (synthetic) |
| `data/bring_your_csv/course_pass.csv` | Structured table (synthetic) |
| `data/bring_your_csv/play_tennis.csv` | Classic Play Tennis |
| `data/bring_your_csv/iris_banded.csv` | Binned numeric features |
| `data/batch_texts/support_emails.csv` | Support email labels |
| `data/batch_texts/sms_spam_excerpt.csv` | spam/ham excerpt |
| `data/homework_scoring/short_answers.csv` | Short-answer scores |
| `data/messy_notes_sample.txt` | Messy notes paste |
| `data/ticket_routing.csv` | Ticket routing sample (use via `decide`, not a separate CLI) |

Sources / licenses: [`data/SOURCES.md`](data/SOURCES.md).
