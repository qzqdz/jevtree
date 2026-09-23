# Example data sources

Small demo corpora under `examples/data/`. Prefer dozens–low hundreds of rows so `grow` stays instant. **Do not** treat these as benchmarks or Acc@budget evidence.

## Bring-your-CSV (`bring_your_csv/`)

| File | Rows | Origin / license | Notes |
|------|------|------------------|-------|
| `loan_approve.csv` | ~88 | **Original** jevtree teaching sample | Synthetic categorical credit-style features for the product spine demo. Not a real lender dataset. |
| `course_pass.csv` | ~70 | **Original** jevtree teaching sample | Synthetic course-outcome table. |
| `play_tennis.csv` | 14 | Classic Quinlan / ID3 “Play Tennis” textbook table | Widely reproduced public teaching example (no proprietary claim). Cite: Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*. |
| `iris_banded.csv` | 150 | [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) (Fisher 1936; Anderson) | Continuous features **discretized into low/mid/high tertile bands** for jevtree’s discrete IG grower. UCI distribution: **CC BY 4.0**. Cite: Fisher, R. A. (1936); UCI DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76). Full continuous original: UCI / many mirrors (e.g. [gist](https://gist.github.com/netj/8836201)). |

## Batch text classification (`batch_texts/`)

| File | Rows | Origin / license | Notes |
|------|------|------------------|-------|
| `support_emails.csv` | 60 | **Original** jevtree teaching sample | Synthetic support tickets → `{billing, engineering, trust_safety, general}`. |
| `sms_spam_excerpt.csv` | 80 | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | Curated **balanced excerpt** (40 ham / 40 spam). Original corpus free for research use; UCI lists **CC BY 4.0**. Cite: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84). Full corpus: UCI (do not ship the full ~5.5k file in-repo). |

## Homework / exam scoring (`homework_scoring/`)

| File | Rows | Origin / license | Notes |
|------|------|------------------|-------|
| `short_answers.csv` | 52 | **Original** jevtree teaching sample | Multi-prompt short answers with rubric scores 1–5 (`prompt_id` column). Framing inspired by public ASAG/AES practice (e.g. [ASAP-AES](https://www.kaggle.com/c/asap-aes), Mohler ASAG) — **no student essays copied**. |
| `photosynthesis_scores.csv` | 20 | Same original sample | Single-prompt subset for a minimal CLI demo. |

## Other

| File | Origin |
|------|--------|
| `messy_notes_sample.txt` | Original jevtree messy-paste fixture |
| `ticket_routing.csv` | Sample table for `jevtree decide` (not a separate CLI) |

## Research pointers (citations only — not product claims)

Interpretable / SOP-style decision processes:

- Molnar, C. *Interpretable Machine Learning* — [Decision Rules](https://christophm.github.io/interpretable-ml-book/rules.html)
- CORELS / rule lists overview: [UBC Systopia](https://systopia.cs.ubc.ca/rule_lists)
- Yu et al., Learning Optimal Decision Sets and Lists with SAT, *JAIR* (2021): https://www.jair.org/index.php/jair/article/view/12719

Active feature acquisition / budgeted questioning (high-level):

- *A Survey on Active Feature Acquisition Strategies* (2025): https://arxiv.org/abs/2502.11067
- Lomax & Vadera, A survey of cost-sensitive decision tree induction algorithms, *ACM Computing Surveys* (2011)
- Online DT + AFA (IJCAI 2023): https://www.ijcai.org/proceedings/2023/0463.pdf

jevtree’s product story is: **materials + goal → auditable tree/SOP under a question/cue budget metaphor**. Official Acc/F1-vs-budget lives only in `jevtree eval-afa` — demos must not claim AFABench SOTA.
