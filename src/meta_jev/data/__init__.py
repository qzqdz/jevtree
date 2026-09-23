"""Datasets + universal ingest → FeatureTable for Meta-Jev."""

from meta_jev.data.cube import generate_cube, load_cube_split
from meta_jev.data.feature_table import FeatureTable, IngestResult
from meta_jev.data.grow_pipeline import grow_from_table
from meta_jev.data.messy_ingest import ingest_messy_text, validate_extracted_table
from meta_jev.data.universal_ingest import ingest_data_with_goal
from meta_jev.data.decide_pipeline import run_decide
from meta_jev.data.tabular import load_tabular_split

__all__ = [
    "FeatureTable",
    "IngestResult",
    "generate_cube",
    "grow_from_table",
    "ingest_messy_text",
    "validate_extracted_table",
    "ingest_data_with_goal",
    "run_decide",
    "load_cube_split",
    "load_tabular_split",
]
