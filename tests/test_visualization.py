"""
Unit tests for EvoScan visualization components and Plotly figures.
"""

import numpy as np
import pandas as pd
import pytest
from evoscan.utils import format_results_to_dataframe
from evoscan.viz import (
    create_saturation_heatmap,
    create_position_sensitivity_plot,
    create_score_distribution_plot,
    create_substitution_matrix_plot,
)


@pytest.fixture
def sample_data():
    seq = "CACGTGTCACGTGTGACTCAGTGACTCA"
    length = len(seq)
    # Generate realistic dummy matrix
    np.random.seed(42)
    scores = np.random.uniform(-3.0, 0.5, size=(4, length)).astype(np.float32)
    # Zero out wild-type
    alphabet = ["A", "C", "G", "T"]
    for i, base in enumerate(seq):
        scores[alphabet.index(base), i] = 0.0

    wide_df, tidy_df = format_results_to_dataframe(scores, seq)
    return seq, wide_df, tidy_df


def test_create_saturation_heatmap(sample_data):
    seq, wide_df, _ = sample_data
    fig = create_saturation_heatmap(wide_df=wide_df, sequence=seq, show_values=True)
    assert fig is not None
    assert len(fig.data) >= 2  # Heatmap trace + Scatter trace for WT points
    assert fig.data[0].type == "heatmap"
    assert fig.data[1].type == "scatter"


def test_create_position_sensitivity_plot(sample_data):
    seq, _, tidy_df = sample_data
    fig = create_position_sensitivity_plot(tidy_df=tidy_df, sequence=seq)
    assert fig is not None
    assert len(fig.data) >= 2  # Bar trace + Line trace


def test_create_score_distribution_plot(sample_data):
    _, _, tidy_df = sample_data
    fig = create_score_distribution_plot(tidy_df=tidy_df)
    assert fig is not None
    assert len(fig.data) >= 1


def test_create_substitution_matrix_plot(sample_data):
    _, _, tidy_df = sample_data
    fig = create_substitution_matrix_plot(tidy_df=tidy_df)
    assert fig is not None
    assert len(fig.data) >= 1
    assert fig.data[0].type == "heatmap"
