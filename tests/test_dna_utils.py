"""
Unit tests for EvoScan DNA utilities, FASTA parser, and statistical metrics.
"""

import numpy as np
import pytest
from evoscan.utils import (
    clean_and_validate_dna,
    parse_fasta,
    compute_sequence_stats,
    format_results_to_dataframe,
    classify_mutation_effect,
)


def test_clean_and_validate_dna_standard():
    raw = "  atgc \n ATGC  "
    seq, valid, msg, warnings = clean_and_validate_dna(raw)
    assert valid is True
    assert seq == "ATGCATGC"
    assert len(seq) == 8


def test_clean_and_validate_dna_fasta():
    raw = ">Beta_Globin_Promoter Homo sapiens\nGGGCGGGAC\nGGGGGCGGGGCGGGCGCTATAAAAGGCGGAGCTTG\n"
    seq, valid, msg, warnings = clean_and_validate_dna(raw)
    assert valid is True
    assert seq.startswith("GGGCGGGAC")
    assert any("FASTA" in w for w in warnings)


def test_clean_and_validate_dna_invalid_characters():
    raw = "ATGCZ123XYZ"
    seq, valid, msg, warnings = clean_and_validate_dna(raw)
    assert valid is False
    assert "Caratteri non validi" in msg
    assert "X" in msg or "Z" in msg


def test_parse_fasta_multi():
    fasta = """>seq1 Header 1
ATGCATGC
>seq2 Header 2
CGTACGTA
"""
    records = parse_fasta(fasta)
    assert len(records) == 2
    assert "seq1 Header 1" in records
    assert records["seq1 Header 1"] == "ATGCATGC"
    assert records["seq2 Header 2"] == "CGTACGTA"


def test_compute_sequence_stats():
    seq = "CGCGATAT"
    stats = compute_sequence_stats(seq)
    assert stats["length"] == 8
    # 4 GC out of 8 = 50%
    assert stats["gc_content_pct"] == 50.0
    # CG appears 2 times
    assert stats["cpg_count"] == 2
    assert stats["counts"]["A"] == 2
    assert stats["counts"]["C"] == 2
    assert stats["counts"]["G"] == 2
    assert stats["counts"]["T"] == 2


def test_format_results_to_dataframe():
    seq = "ATGC"
    # Dummy score matrix (4 x 4) for sequence A, T, G, C
    # Rows: 0=A, 1=C, 2=G, 3=T
    scores = np.array(
        [
            [0.0, -1.2, -0.8, -2.1],  # A
            [-1.5, -0.9, -1.8, 0.0],  # C (WT at pos 3)
            [-0.9, -1.1, 0.0, -1.3],  # G (WT at pos 2)
            [-2.4, 0.0, -1.4, -0.7],  # T (WT at pos 1)
        ],
        dtype=np.float32,
    )

    wide_df, tidy_df = format_results_to_dataframe(scores, seq)
    assert wide_df.shape == (4, 4)
    assert list(wide_df.columns) == ["1-A", "2-T", "3-G", "4-C"]

    assert len(tidy_df) == 16
    assert "Mutation" in tidy_df.columns
    assert "Effect" in tidy_df.columns

    # Verify WT diagonal is recognized
    wt_rows = tidy_df[tidy_df["Is_WildType"]]
    assert len(wt_rows) == 4
    for _, row in wt_rows.iterrows():
        assert row["Delta_Score_LLR"] == 0.0


def test_classify_mutation_effect():
    assert "Wild-Type" in classify_mutation_effect(0.0, "A", "A")
    assert "Distruttiva" in classify_mutation_effect(-2.5, "A", "G")
    assert "Moderatamente" in classify_mutation_effect(-1.0, "A", "G")
    assert "Tollerata" in classify_mutation_effect(0.05, "A", "G")
    assert "Arricchita" in classify_mutation_effect(1.5, "A", "G")
