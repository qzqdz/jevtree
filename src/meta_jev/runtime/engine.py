"""Runtime: execute SOP / tree on observations (stub — P3)."""

from __future__ import annotations

from typing import Any


class RuntimeEngine:
    """Run a DecisionSOP or grown tree against an observation dict.

    Does not score Acc/F1 — callers that need hard-budget metrics must use
    `meta-jev eval-afa` / meta_jev.eval.protocol.
    """

    def run_sop(self, sop: Any, obs: dict[str, Any]) -> Any:
        raise NotImplementedError("RuntimeEngine.run_sop — P3")

    def run_tree(self, tree: Any, obs: dict[str, Any]) -> Any:
        raise NotImplementedError("RuntimeEngine.run_tree — P3")
