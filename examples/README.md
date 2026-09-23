# Examples

产品入口优先走 CLI：

```bash
meta-jev decide --data <path> --goal "自然语言需求" --out results/demo
```

本目录脚本是同一脊柱上的快捷写法（内部仍可调用 grow / ingest）。

```bash
# 干净特征 CSV → tree（底层 grow，无需 LLM）
python examples/bring_your_csv.py

# 批量标注文本
python examples/batch_texts_to_tree.py \
  --csv examples/data/batch_texts/support_emails.csv --label label

# 作业短答打分
python examples/batch_texts_to_tree.py \
  --csv examples/data/homework_scoring/short_answers.csv \
  --label score --label-kind ordinal \
  --out-dir examples/artifacts/homework_scoring

# 杂乱粘贴 + 目标（LLM）
python examples/messy_to_tree.py --goal "分流到哪个团队"
```

## 数据一览

| Path | Role |
|------|------|
| `data/bring_your_csv/loan_approve.csv` | 结构化表格（合成教学） |
| `data/bring_your_csv/course_pass.csv` | 结构化表格（合成教学） |
| `data/bring_your_csv/play_tennis.csv` | 经典教材 Play Tennis |
| `data/bring_your_csv/iris_banded.csv` | 分箱数值特征 |
| `data/batch_texts/support_emails.csv` | 支持邮件分类 |
| `data/batch_texts/sms_spam_excerpt.csv` | spam/ham 摘录 |
| `data/homework_scoring/short_answers.csv` | 短答打分 |
| `data/messy_notes_sample.txt` | 杂乱笔记 |
| `data/ticket_routing.csv` | 工单预设数据 |

出处与许可见 [`data/SOURCES.md`](data/SOURCES.md)。
