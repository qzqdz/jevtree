"""
示例：客服工单路由 SOP

输入：{"ticket": "我的发票开错了，需要重新开一张"}
输出：{"choice": "billing", "probabilities": {...}, "confidence": 0.87}
"""
from meta_jev.core.sop import (
    DecisionSOP,
    FeatureNode,
    JevNode,
    JevQuestion,
    bool_sop,
    classify_sop,
    score_sop,
)


# ── 示例 1：最简分类 SOP ──────────────────────────────────────────
simple_routing = classify_sop(
    name="ticket-routing-simple",
    description="客服工单三分类：billing / tech / other",
    categories={
        "billing": "计费、退款、发票、账单相关问题",
        "tech": "技术故障、bug、无法使用、报错相关问题",
        "other": "其他咨询、建议、投诉、非以上两类",
    },
    question_instructions=(
        "Read the customer ticket below and choose the single best team to handle it. "
        "Pick billing if it involves money, invoices, refunds, or charges. "
        "Pick tech if it describes a product malfunction, error, or broken feature. "
        "Pick other if it's unclear or doesn't fit either."
    ),
)


# ── 示例 2：带特征工程的分类 SOP ──────────────────────────────────
featured_routing = DecisionSOP(
    name="ticket-routing-v2",
    description="带特征提取的工单路由",
    nodes=[
        # 特征节点：从工单文本提取信号
        FeatureNode(
            id="features",
            expression=(
                "{'ticket_len': len(obs['ticket']), "
                "'has_refund': '退款' in obs['ticket'], "
                "'has_invoice': '发票' in obs['ticket'], "
                "'has_error': '错误' in obs['ticket'] or 'bug' in obs['ticket'].lower(), "
                "'has_money': any(w in obs['ticket'] for w in ['钱','费','付','扣'])}"
            ),
        ),
        # 主分类节点
        JevNode(
            id="router",
            state="{'ticket': obs['ticket'], 'features': nodes['features']}",
            questions={
                "team": JevQuestion(
                    qid="team",
                    type="choice",
                    instructions=(
                        "Based on the ticket text and extracted features, "
                        "choose the best team. Use the feature flags as hints, "
                        "but rely primarily on the ticket content itself."
                    ),
                    criteria={
                        "billing": "计费/退款/发票/账单",
                        "tech": "技术故障/bug/报错/无法使用",
                        "other": "其他咨询/投诉/建议",
                    },
                ),
            },
        ),
    ],
    output="nodes['router']['team']['choice']",
)


# ── 示例 3：紧急程度打分 SOP ──────────────────────────────────────
urgency = score_sop(
    name="ticket-urgency",
    description="工单紧急程度 1-5 分",
    levels=[
        "1: 不紧急，排队处理即可",
        "2: 一般，今天内处理",
        "3: 较紧急，2小时内回复",
        "4: 紧急，30分钟内响应",
        "5: 非常紧急，影响生产/大量用户，立即处理",
    ],
    question_instructions=(
        "Read the ticket and rate its urgency level. "
        "Consider: how many users are affected, is it blocking work, "
        "is it revenue-impacting, and what does the customer's tone suggest."
    ),
)


# ── 示例 4：是否死循环判断（Agent 防卡死）─────────────────────────
loop_detector = bool_sop(
    name="loop-detector",
    description="检测 Agent 是否陷入重复循环",
    question=(
        "Given the recent actions and their results, is the agent stuck in a loop? "
        "A loop means repeating the same action with the same failing result 3+ times "
        "without making progress."
    ),
    state_expr="{'recent_actions': obs['recent_actions'], 'last_error': obs['last_error']}",
)


if __name__ == "__main__":
    for sop in [simple_routing, featured_routing, urgency, loop_detector]:
        print(f"=== {sop.name} ===")
        print(f"  {sop.description}")
        errors = sop.validate()
        print(f"  validation: {'PASS' if not errors else errors}")
        print(f"  hash: {sop.hash}")
        print(f"  nodes: {len(sop.nodes)}")
        print()
