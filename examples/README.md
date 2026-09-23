# Examples

语言: 中文 | [English](README_EN.md)

**唯一产品入口：**

```bash
jevtree decide --data <path> --goal "自然语言需求" --out results/demo
```

别名：`jevtree run-job` / `jevtree from-data`。

离线 mock 演示（无需 API key）：

```bash
PYTHONPATH=src python3 examples/decide_demo.py
```

核心信息熵教材校验（不是产品入口）：

```bash
PYTHONPATH=src python3 examples/entropy_split_demo.py
```

## 样例数据

| Path | Role |
|------|------|
| `data/bring_your_csv/loan_approve.csv` | 结构化表（合成教学） |
| `data/bring_your_csv/course_pass.csv` | 结构化表（合成教学） |
| `data/bring_your_csv/play_tennis.csv` | 经典 Play Tennis |
| `data/bring_your_csv/iris_banded.csv` | 分箱数值特征 |
| `data/batch_texts/support_emails.csv` | 支持邮件分类标签 |
| `data/batch_texts/sms_spam_excerpt.csv` | spam/ham 摘录 |
| `data/homework_scoring/short_answers.csv` | 短答打分 |
| `data/messy_notes_sample.txt` | 杂乱笔记粘贴 |
| `data/ticket_routing.csv` | 工单分流样例（经 `decide` 使用，无独立 CLI） |

出处与许可见 [`data/SOURCES.md`](data/SOURCES.md)。
