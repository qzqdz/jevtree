"""Human-readable story from a grown IG tree (product UX)."""

from __future__ import annotations

from typing import Any

from meta_jev.core.tree import TreeLeaf, TreeNode


def _walk_priority(node: TreeNode | TreeLeaf, acc: list[str], depth: int = 0) -> None:
    if isinstance(node, TreeLeaf):
        return
    if isinstance(node, TreeNode):
        if node.feature not in acc:
            acc.append(node.feature)
        # prefer larger child first for "story order"
        kids = sorted(
            node.children.items(),
            key=lambda kv: (-getattr(kv[1], "n_samples", 0), str(kv[0])),
        )
        for _, child in kids:
            _walk_priority(child, acc, depth + 1)


def tree_question_order(tree: TreeNode | TreeLeaf) -> list[str]:
    order: list[str] = []
    _walk_priority(tree, order)
    return order


def format_tree_story(
    tree: TreeNode | TreeLeaf,
    *,
    label_key: str = "label",
    max_ask: int = 4,
    lang: str = "zh",
) -> str:
    """Return a short bilingual-friendly story line for classmates."""
    order = tree_question_order(tree)[:max_ask]
    if not order:
        if isinstance(tree, TreeLeaf):
            pred = tree.prediction
            if lang == "en":
                return f"From your data: always decide {pred!r} (no questions needed)."
            return f"从你的表长出的决策树：无需再问，直接决定 {pred!r}。"
        return "empty tree"

    if lang == "en":
        if len(order) == 1:
            body = f"first ask {order[0]}"
        elif len(order) == 2:
            body = f"first ask {order[0]}, then {order[1]}"
        else:
            mid = ", then ".join(order[:-1])
            body = f"first ask {mid}, then {order[-1]}"
        return f"Decision tree grown from your table: {body}, then decide {label_key}."

    if len(order) == 1:
        body = f"先问 {order[0]}"
    elif len(order) == 2:
        body = f"先问 {order[0]}，再问 {order[1]}"
    else:
        body = "先问 " + "，再问 ".join(order)
    return f"从你的表长出的决策树：{body}，决定 {label_key}。"


def format_trace_story(
    path: list[dict[str, Any]],
    decision: Any,
    *,
    lang: str = "zh",
) -> str:
    if lang == "en":
        steps = " → ".join(f"{s.get('feature')}={s.get('value')}" for s in path) or "(none)"
        return f"Trace: {steps} ⇒ {decision}"
    steps = " → ".join(f"{s.get('feature')}={s.get('value')}" for s in path) or "（无）"
    return f"路径：{steps} ⇒ 决定 {decision}"


__all__ = ["format_trace_story", "format_tree_story", "tree_question_order"]
