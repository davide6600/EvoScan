"""
Unit and Integration Tests for EvoScan AlphaGenome Atlas Engine & Visualization.
Validates coordinate parsing, deep-linking, preset configurations,
and live gRPC connection using the designated test credentials.
"""

import os
import numpy as np
import pandas as pd
import pytest

from evoscan.alphagenome_engine import (
    AlphaGenomeEngine,
    MODALITY_OPTIONS,
    GENOMIC_PRESETS,
    parse_genomic_region,
    format_atlas_url,
)
from evoscan.utils import format_results_to_dataframe, classify_phred_effect
from evoscan.viz import (
    create_saturation_heatmap,
    create_position_sensitivity_plot,
    create_score_distribution_plot,
    create_substitution_matrix_plot,
)

# Optional test API key read strictly from environment variable for live verification
TEST_API_KEY = os.environ.get("ALPHAGENOME_API_KEY", os.environ.get("ALPHAGENOME_TEST_API_KEY", ""))


def test_parse_genomic_region_valid():
    chrom, start, end = parse_genomic_region("chr11:5225720-5225780")
    assert chrom == "chr11"
    assert start == 5225720
    assert end == 5225780

    # Comma-formatted and prefix-less
    chrom2, start2, end2 = parse_genomic_region("11:5,225,720-5,225,780")
    assert chrom2 == "chr11"
    assert start2 == 5225720
    assert end2 == 5225780

    # Sex chromosome
    chrom3, start3, end3 = parse_genomic_region("chrX:101386210-101386270")
    assert chrom3 == "chrX"
    assert start3 == 101386210
    assert end3 == 101386270


def test_parse_genomic_region_invalid():
    with pytest.raises(ValueError, match="Formato regione non valido"):
        parse_genomic_region("invalid_region_format")

    with pytest.raises(ValueError, match="non può essere minore"):
        parse_genomic_region("chr11:500-400")

    with pytest.raises(ValueError, match="supera il limite"):
        parse_genomic_region("chr11:1000-5000", max_window=1000)


def test_format_atlas_url():
    url = format_atlas_url("chr11", 5225725, "A", "T")
    assert "deepmind.google.com/science/alphagenome/atlas" in url
    assert "chr11%3A5225725%3AA%3ET" in url
    assert "m=variant" in url


def test_genomic_presets_integrity():
    assert len(GENOMIC_PRESETS) >= 6
    for name, data in GENOMIC_PRESETS.items():
        assert "region" in data
        assert "gene" in data
        assert "description" in data
        chrom, start, end = parse_genomic_region(data["region"])
        assert chrom == data["chrom"]
        assert start == data["start"]
        assert end == data["end"]


def test_classify_phred_effect():
    assert classify_phred_effect(0.0, "A", "A") == "Wild-Type (Neutral)"
    assert "Extremely Disruptive" in classify_phred_effect(32.5, "A", "G")
    assert "High Impact" in classify_phred_effect(22.1, "A", "G")
    assert "Moderate Impact" in classify_phred_effect(12.4, "A", "G")
    assert "Mild Impact" in classify_phred_effect(6.0, "A", "G")
    assert "Tolerated" in classify_phred_effect(2.0, "A", "G")


def test_alphagenome_engine_unconfigured():
    engine = AlphaGenomeEngine(api_key="")
    assert not engine.configured
    valid, msg = engine.validate_connection()
    assert not valid
    assert "non configurata" in msg

    with pytest.raises(ValueError, match="mancante"):
        engine.score_genomic_interval("chr11", 5225720, 5225730)


def test_alphagenome_visualization_with_phred_data():
    """Validates that Plotly visualizers handle Phred formatted matrices without errors."""
    seq = "ATGCATGCATGC"
    length = len(seq)
    phred_scores = np.zeros((4, length), dtype=np.float32)
    # Put sample Phred scores
    phred_scores[1, 0] = 25.4  # High impact
    phred_scores[2, 3] = 34.1  # Extreme impact
    phred_scores[3, 5] = 12.0  # Moderate

    wide_df, tidy_df = format_results_to_dataframe(
        phred_scores, seq, is_phred=True, metric_name="AVI_Score"
    )

    fig_heat = create_saturation_heatmap(
        wide_df=wide_df,
        sequence=seq,
        is_phred=True,
        metric_label="AVI Score (Phred)",
        tidy_df=tidy_df,
    )
    assert fig_heat is not None
    assert len(fig_heat.data) >= 2

    fig_prof = create_position_sensitivity_plot(
        tidy_df=tidy_df, sequence=seq, is_phred=True
    )
    assert fig_prof is not None

    fig_dist = create_score_distribution_plot(tidy_df, is_phred=True)
    assert fig_dist is not None

    fig_sub = create_substitution_matrix_plot(tidy_df, is_phred=True)
    assert fig_sub is not None


@pytest.mark.skipif(not TEST_API_KEY, reason="AlphaGenome test API key not available")
def test_alphagenome_live_grpc_integration():
    """
    Live test against Google DeepMind AlphaGenome Atlas gRPC cluster.
    Uses the provided test credentials on a micro-locus (15 bp) of HBB promoter.
    """
    engine = AlphaGenomeEngine(api_key=TEST_API_KEY)
    assert engine.configured

    # Verify lightweight ping / connectivity
    connected, conn_msg = engine.validate_connection()
    assert connected, f"AlphaGenome Atlas gRPC ping failed: {conn_msg}"

    # Micro-interval: HBB promoter core (15 bp)
    chrom = "chr11"
    start = 5225720
    end = 5225734  # 15 bp
    score_matrix, ref_seq, tidy_df, meta = engine.score_genomic_interval(
        chrom=chrom,
        start_1_based=start,
        end_1_based=end,
        modality="AVI_SCORE",
    )

    assert score_matrix.shape == (4, 15)
    assert len(ref_seq) == 15
    assert not tidy_df.empty
    assert meta["variants_count"] > 0
    assert meta["max_phred"] >= 0.0

    # Ensure all variants have a valid deep-link
    for url in tidy_df["atlas_url"]:
        assert "deepmind.google.com/science/alphagenome/atlas" in url
