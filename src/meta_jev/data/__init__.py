"""Datasets + universal ingest → FeatureTable for Meta-Jev."""

from meta_jev.data.cube import generate_cube, load_cube_split
from meta_jev.data.csv_ingest import ingest_csv
from meta_jev.data.feature_table import FeatureTable, IngestResult
from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.data.messy_ingest import ingest_messy_text
from meta_jev.data.tabular import load_tabular_split
from meta_jev.data.text_batch import ingest_text_batch
from meta_jev.data.ticket_routing import (
    generate_ticket_routing,
    load_ticket_routing_split,
)

__all__ = [
    "FeatureTable",
    "IngestResult",
    "generate_cube",
    "generate_ticket_routing",
    "grow_from_table",
    "ingest_csv",
    "ingest_messy_text",
    "ingest_text_batch",
    "load_cube_split",
    "load_tabular_split",
    "load_ticket_routing_split",
]
