"""Datasets + universal ingest → FeatureTable for jevtree."""

from jevtree.data.cube import generate_cube, load_cube_split
from jevtree.data.feature_table import FeatureTable, IngestResult
from jevtree.data.grow_pipeline import grow_from_table
from jevtree.data.messy_ingest import ingest_messy_text, validate_extracted_table
from jevtree.data.universal_ingest import ingest_data_with_goal
from jevtree.data.decide_pipeline import run_decide
from jevtree.data.tabular import load_tabular_split

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
