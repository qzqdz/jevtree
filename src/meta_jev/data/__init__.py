"""Synthetic / tabular datasets for Meta-Jev smoke evals."""

from meta_jev.data.cube import generate_cube, load_cube_split
from meta_jev.data.tabular import load_tabular_split
from meta_jev.data.ticket_routing import (
    generate_ticket_routing,
    load_ticket_routing_split,
)

__all__ = [
    "generate_cube",
    "load_cube_split",
    "load_tabular_split",
    "generate_ticket_routing",
    "load_ticket_routing_split",
]
