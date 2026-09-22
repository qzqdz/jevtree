"""
Meta-Jev 决定 SOP 数据结构与校验。

一个 SOP 就是一套"看什么数据 → 问什么问题 → 怎么决策"的标准流程。
底层格式兼容 JevHarness 的 PipelineSpec，但对外提供更简洁的分类/决策抽象。
"""
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal


# ── 三种 Jev 问题类型 ──────────────────────────────────────────────
QuestionType = Literal["choice", "score", "noul"]


@dataclass
class JevQuestion:
    """一个 Jev 判断点。"""
    qid: str                          # 问题 ID，如 "is_billing"
    type: QuestionType               # choice / score / noul
    instructions: str                 # 给 Jev 的判断指令
    criteria: dict | list | None = None
        # choice: {选项ID: 选项描述}
        # score:  [等级1描述, 等级2描述, ...]
        # noul:   None


@dataclass
class FeatureNode:
    """特征计算节点：从原始 obs 提取信号。"""
    id: str
    expression: str                  # 表达式，如 "len(obs['ticket'].split())"


@dataclass
class JevNode:
    """Jev 决策节点：问 Jev 一个或多个问题。"""
    id: str
    state: str                        # 表达式，决定传给 Jev 的 state
    questions: dict[str, JevQuestion]  # 这个节点里并行问的所有问题


@dataclass
class DecisionSOP:
    """一套完整的决定 SOP。"""
    name: str                         # SOP 名称，如 "ticket-routing"
    description: str                  # 一句话描述
    nodes: list[FeatureNode | JevNode] # 决策流程图（DAG）
    output: str                       # 最终决策表达式，如 "nodes['router']['pick']['choice']"
    jev_model: str = "typesafe-ai/jev"
    # Optional deterministic tree payload produced by IGDecisionTreeGrower.
    # This preserves normal Jev SOP JSON while enabling local tree execution.
    tree: dict[str, Any] | None = None

    @property
    def hash(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """转成 JevHarness 兼容的 PipelineSpec JSON。"""
        nodes = []
        for n in self.nodes:
            if isinstance(n, FeatureNode):
                nodes.append({
                    "id": n.id,
                    "kind": "expression",
                    "expression": n.expression,
                })
            elif isinstance(n, JevNode):
                qs = {}
                for qid, q in n.questions.items():
                    qd: dict = {"type": q.type, "instructions": q.instructions}
                    if q.criteria is not None:
                        qd["criteria"] = q.criteria
                    qs[qid] = qd
                nodes.append({
                    "id": n.id,
                    "kind": "jev",
                    "state": n.state,
                    "questions": qs,
                })
        payload = {
            "version": 2,
            "name": self.name,
            "jev_model": self.jev_model,
            "nodes": nodes,
            "output": self.output,
        }
        if self.tree is not None:
            payload["kind"] = "tree"
            payload["tree"] = self.tree
        return payload

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionSOP":
        """从 PipelineSpec dict 反序列化。"""
        nodes = []
        for n in d.get("nodes", []):
            if n["kind"] == "expression":
                nodes.append(FeatureNode(id=n["id"], expression=n["expression"]))
            elif n["kind"] == "jev":
                qs = {}
                for qid, qd in n["questions"].items():
                    qs[qid] = JevQuestion(
                        qid=qid,
                        type=qd["type"],
                        instructions=qd["instructions"],
                        criteria=qd.get("criteria"),
                    )
                nodes.append(JevNode(id=n["id"], state=n["state"], questions=qs))
        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            nodes=nodes,
            output=d["output"],
            jev_model=d.get("jev_model", "typesafe-ai/jev"),
            tree=d.get("tree"),
        )

    def validate(self) -> list[str]:
        """校验 SOP 合法性，返回错误列表（空 = 通过）。"""
        errors: list[str] = []
        node_ids = set()

        if not self.name.strip():
            errors.append("SOP name is empty")

        if not self.nodes and self.tree is None:
            errors.append("SOP has no nodes")

        if self.tree is not None:
            if not isinstance(self.tree, dict) or self.tree.get("kind") != "IGDecisionTree":
                errors.append("tree payload must be an IGDecisionTree")

        for n in self.nodes:
            if n.id in node_ids:
                errors.append(f"duplicate node id: {n.id}")
            node_ids.add(n.id)

            if isinstance(n, JevNode):
                for qid, q in n.questions.items():
                    if q.type == "choice":
                        if not isinstance(q.criteria, dict) or len(q.criteria) < 2:
                            errors.append(f"{n.id}.{qid}: choice needs >=2 options")
                    elif q.type == "score":
                        if not isinstance(q.criteria, list) or not (2 <= len(q.criteria) <= 10):
                            errors.append(f"{n.id}.{qid}: score needs 2-10 levels")
                    elif q.type == "noul":
                        if q.criteria is not None:
                            errors.append(f"{n.id}.{qid}: noul should have no criteria")

        if not self.output.strip():
            errors.append("output expression is empty")

        return errors


# ── 便捷构造函数 ────────────────────────────────────────────────────
def classify_sop(
    name: str,
    description: str,
    categories: dict[str, str],   # {类别ID: 类别描述}
    feature_exprs: dict[str, str] | None = None,  # {特征名: 表达式}
    state_expr: str = "obs",      # 传给 Jev 的 state 表达式
    question_instructions: str = "",
) -> DecisionSOP:
    """快速构造一个单节点分类 SOP。

    例子：
        sop = classify_sop(
            name="ticket-routing",
            description="客服工单三分类",
            categories={"billing": "计费/退款/发票问题", "tech": "技术故障/bug", "other": "其他"},
            feature_exprs={"text_len": "len(obs['ticket'])", "has_refund": "'退款' in obs['ticket']"},
            state_expr="{'ticket': obs['ticket'], 'features': nodes['features']}",
            question_instructions="根据工单内容，选择最匹配的团队。",
        )
    """
    nodes: list[FeatureNode | JevNode] = []

    # 特征节点
    if feature_exprs:
        merged = "{" + ", ".join(f"'{k}': {v}" for k, v in feature_exprs.items()) + "}"
        nodes.append(FeatureNode(id="features", expression=merged))

    # 主分类节点
    nodes.append(JevNode(
        id="classify",
        state=state_expr,
        questions={
            "category": JevQuestion(
                qid="category",
                type="choice",
                instructions=question_instructions,
                criteria=categories,
            )
        },
    ))

    return DecisionSOP(
        name=name,
        description=description,
        nodes=nodes,
        output="nodes['classify']['category']['choice']",
    )


def score_sop(
    name: str,
    description: str,
    levels: list[str],             # 等级描述列表，2-10 个
    state_expr: str = "obs",
    question_instructions: str = "",
) -> DecisionSOP:
    """快速构造一个打分 SOP。

    例子：
        sop = score_sop(
            name="urgency-scoring",
            description="工单紧急程度 1-5 分",
            levels=["1: 不紧急，可排队", "2: 一般", "3: 较紧急", "4: 紧急", "5: 非常紧急，影响生产"],
        )
    """
    return DecisionSOP(
        name=name,
        description=description,
        nodes=[
            JevNode(
                id="urgency",
                state=state_expr,
                questions={
                    "level": JevQuestion(
                        qid="level",
                        type="score",
                        instructions=question_instructions,
                        criteria=levels,
                    )
                },
            )
        ],
        output="nodes['urgency']['level']['score']",
    )


def bool_sop(
    name: str,
    description: str,
    question: str,                 # 是非题
    state_expr: str = "obs",
) -> DecisionSOP:
    """快速构造一个是非判断 SOP。"""
    return DecisionSOP(
        name=name,
        description=description,
        nodes=[
            JevNode(
                id="judge",
                state=state_expr,
                questions={
                    "verdict": JevQuestion(
                        qid="verdict",
                        type="noul",
                        instructions=question,
                    )
                },
            )
        ],
        output="nodes['judge']['verdict']['noul']",
    )
