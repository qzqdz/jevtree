"""Product SDK stub — decide() surface deferred.

Current tree is a research / AFABench iteration skeleton.
Final product layout: thin sdk + grow/run CLI; eval becomes optional extra.
"""

from __future__ import annotations

from typing import Any


def decide(obs: dict[str, Any], sop: Any | None = None) -> Any:
    """Run a decision SOP / tree on one observation (product API — stub)."""
    raise NotImplementedError(
        "sdk.client.decide — deferred until product layout; "
        "use meta-jev run / runtime once P3 lands"
    )
