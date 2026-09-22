# Examples（演示 only — 不是 Acc@budget）

Meta-Jev 产品脊柱：**任意材料 + 目标 → FeatureTable → IG tree/SOP → trace**。

禁止在这里宣称榜单指标。正式评分只走 `meta-jev eval-afa`。

## 通用路径（从这里开始）

```bash
# 1) 干净特征 CSV
python examples/bring_your_csv.py
# 可选: --csv examples/data/bring_your_csv/play_tennis.csv --label play
# 可选: --csv examples/data/bring_your_csv/iris_banded.csv --label variety

# 2) 批量标注文本（分类）
python examples/batch_texts_to_tree.py \
  --csv examples/data/batch_texts/support_emails.csv --label label
# 可选: --csv examples/data/batch_texts/sms_spam_excerpt.csv --label label

# 3) 作业 / 考试打分批次（ordinal）
python examples/batch_texts_to_tree.py \
  --csv examples/data/homework_scoring/short_answers.csv \
  --label score --label-kind ordinal \
  --out-dir examples/artifacts/homework_scoring

# 4) 杂乱粘贴 + 目标（需要 .env LLM key）
python examples/messy_to_tree.py --goal "分流到哪个团队"
```

## 可选垂直示例

- `ticket_routing.py` — 较早的五分钟工单故事（同一脊柱上的 preset）

## 数据一览

| Path | Role |
|------|------|
| `data/bring_your_csv/loan_approve.csv` | 结构化表格（合成教学） |
| `data/bring_your_csv/course_pass.csv` | 结构化表格（合成教学） |
| `data/bring_your_csv/play_tennis.csv` | 经典教材 Play Tennis |
| `data/bring_your_csv/iris_banded.csv` | UCI Iris 分箱版（CC BY 4.0） |
| `data/batch_texts/support_emails.csv` | 文本 + 类别 |
| `data/batch_texts/sms_spam_excerpt.csv` | UCI SMS Spam 摘录（CC BY 4.0） |
| `data/homework_scoring/short_answers.csv` | 文本 + 分数 1–5（多题） |
| `data/homework_scoring/photosynthesis_scores.csv` | 单题子集 |
| `data/messy_notes_sample.txt` | 自由粘贴 |
| `data/ticket_routing.csv` | 可选工单 preset |

出处与许可：[`data/SOURCES.md`](data/SOURCES.md)。
