"""Local runtime for deterministic SOP / tree execution."""

from __future__ import annotations

from typing import Any

from jevtree.core.sop import DecisionSOP
from jevtree.core.tree import IGDecisionTreeGrower, TreeLeaf, TreeNode


class RuntimeEngine:
    """Run a deterministic tree or exported tree SOP on one observation.

    Jev-backed nodes remain declarative and are rejected with a clear error;
    this runtime deliberately never calls a remote model or evaluates arbitrary
    Python expressions.
    """

    def run_sop(self, sop: DecisionSOP | dict[str, Any], obs: dict[str, Any]) -> Any:
        if isinstance(sop, dict):
            sop = DecisionSOP.from_dict(sop)
        errors = sop.validate()
        if errors:
            raise ValueError("invalid SOP: " + "; ".join(errors))
        if sop.tree is not None:
            return self.run_tree_payload(sop.tree, obs)
        raise NotImplementedError(
            "JevNode execution requires a Jev provider; use an exported IG tree SOP"
        )

    def run_tree(self, tree: Any, obs: dict[str, Any]) -> Any:
        if isinstance(tree, IGDecisionTreeGrower):
            return tree.predict(obs)
        if isinstance(tree, (TreeNode, TreeLeaf)):
            return tree.predict(obs)
        if isinstance(tree, dict):
            if tree.get("type") == "IGDecisionTree":
                return self.run_tree_payload(tree, obs)
            if tree.get("kind") in {"node", "leaf"}:
                node = TreeLeaf.from_dict(tree) if tree.get("kind") == "leaf" else TreeNode.from_dict(tree)
                return node.predict(obs)
        raise TypeError(f"unsupported tree type: {type(tree).__name__}")

    def run_tree_payload(self, payload: dict[str, Any], obs: dict[str, Any]) -> Any:
        """Run a serialized IG tree, applying its train-time binning policy."""
        if payload.get("kind") == "IGDecisionTree":
            payload = {"type": "IGDecisionTree", **payload}
        if payload.get("type") != "IGDecisionTree":
            raise ValueError("expected an IGDecisionTree payload")
        return IGDecisionTreeGrower.from_json(payload).predict(obs)


    def run_tree_payload_traced(
        self, payload: dict[str, Any], obs: dict[str, Any]
    ) -> dict[str, Any]:
        """Like run_tree_payload but also returns the feature/question path."""
        if payload.get("kind") == "IGDecisionTree":
            payload = {"type": "IGDecisionTree", **payload}
        if payload.get("type") != "IGDecisionTree":
            raise ValueError("expected an IGDecisionTree payload")
        grower = IGDecisionTreeGrower.from_json(payload)
        decision, path = grower.predict_path(obs)
        return {
            "decision": decision,
            "path": path,
            "questions_used": len(path),
        }

    def run_sop_traced(
        self, sop: DecisionSOP | dict[str, Any], obs: dict[str, Any]
    ) -> dict[str, Any]:
        """Run SOP and return decision + auditable question path.

        Only IG tree SOPs are supported locally (same constraint as run_sop).
        """
        if isinstance(sop, dict):
            sop = DecisionSOP.from_dict(sop)
        errors = sop.validate()
        if errors:
            raise ValueError("invalid SOP: " + "; ".join(errors))
        if sop.tree is not None:
            return self.run_tree_payload_traced(sop.tree, obs)
        raise NotImplementedError(
            "JevNode execution requires a Jev provider; use an exported IG tree SOP"
        )
