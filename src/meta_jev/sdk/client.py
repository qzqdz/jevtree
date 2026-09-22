"""Small local SDK for deterministic Meta-Jev tree decisions."""

from __future__ import annotations

from typing import Any

from meta_jev.core.sop import DecisionSOP
from meta_jev.runtime.engine import RuntimeEngine


def decide(obs: dict[str, Any], sop: Any | None = None) -> Any:
    """Run a deterministic tree/SOP on one observation.

    ``sop`` may be a ``DecisionSOP``, serialized SOP dict, serialized tree
    payload, or a fitted ``IGDecisionTreeGrower``.
    """
    if sop is None:
        raise ValueError("sop is required")
    engine = RuntimeEngine()
    if isinstance(sop, DecisionSOP) or (
        isinstance(sop, dict) and ("nodes" in sop or sop.get("kind") == "tree")
    ):
        return engine.run_sop(sop, obs)
    return engine.run_tree(sop, obs)
