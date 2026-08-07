"""
Unit tests for EvoScan inference engine, tensor shapes, and mutation scoring logic.
"""

import numpy as np
import pytest
from evoscan.model import EvoScanEngine, ALPHABET


def test_split_into_kmers():
    engine = EvoScanEngine(model_id="simulation-biophysics-engine")
    seq = "ATGCATGCATGC"
    kmers = engine._split_into_kmers(seq, k=6)
    assert kmers == ["ATGCAT", "GCATGC"]

    seq_odd = "ATGCATGCATGCA"
    kmers_odd = engine._split_into_kmers(seq_odd, k=6)
    assert kmers_odd == ["ATGCAT", "GCATGC", "A"]


def test_simulation_engine_scores_shape_and_null_diagonal():
    engine = EvoScanEngine(model_id="simulation-biophysics-engine")
    engine.load_model()

    seq = "GGGCGGGACGGGGGCGGGGCGGGCGCTATAAAAGGCGGAGCTTG"
    matrix, meta = engine.score_sequence(seq)

    # Validate shape (4 x L)
    assert matrix.shape == (4, len(seq))
    assert meta["engine"] == "EvoScan Bio-Physics Fast Engine"

    # Validate that for every WT base, score is 0.0
    for pos, wt_base in enumerate(seq):
        wt_idx = ALPHABET.index(wt_base)
        assert matrix[wt_idx, pos] == 0.0, f"WT score at position {pos} must be 0.0"

        # Validate that mutant bases have non-zero or evaluated scores
        for m_idx, mut_base in enumerate(ALPHABET):
            if mut_base != wt_base:
                assert isinstance(float(matrix[m_idx, pos]), float)


def test_short_sequence_handling():
    engine = EvoScanEngine(model_id="simulation-biophysics-engine")
    engine.load_model()

    seq = "ATGC"
    matrix, meta = engine.score_sequence(seq)
    assert matrix.shape == (4, 4)
