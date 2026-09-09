"""
EvoScan: Zero-Shot DNA Mutation Map Visualizer & Deep Mutational Scanning Engine.
"""

__version__ = "1.0.0"
__author__ = "EvoScan Contributors"

from evoscan.utils import (
    clean_and_validate_dna,
    parse_fasta,
    compute_sequence_stats,
    format_results_to_dataframe,
)
from evoscan.model import EvoScanEngine
from evoscan.alphagenome_engine import (
    AlphaGenomeEngine,
    MODALITY_OPTIONS,
    GENOMIC_PRESETS,
    parse_genomic_region,
    format_atlas_url,
)
from evoscan.viz import (
    create_saturation_heatmap,
    create_position_sensitivity_plot,
    create_score_distribution_plot,
)

__all__ = [
    "clean_and_validate_dna",
    "parse_fasta",
    "compute_sequence_stats",
    "format_results_to_dataframe",
    "EvoScanEngine",
    "AlphaGenomeEngine",
    "MODALITY_OPTIONS",
    "GENOMIC_PRESETS",
    "parse_genomic_region",
    "format_atlas_url",
    "create_saturation_heatmap",
    "create_position_sensitivity_plot",
    "create_score_distribution_plot",
]
