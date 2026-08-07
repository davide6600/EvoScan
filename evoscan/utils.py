"""
EvoScan Utility Functions: DNA Sequence Cleaning, FASTA Parsing, Bio-Statistics, and Data Formatting.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

VALID_BASES = {"A", "C", "G", "T"}
VALID_BASES_EXTENDED = {"A", "C", "G", "T", "N"}
ALPHABET = ["A", "C", "G", "T"]


def clean_and_validate_dna(
    raw_input: str, allow_n: bool = False
) -> Tuple[str, bool, str, List[str]]:
    """
    Cleans raw DNA text or FASTA input and validates nucleotide alphabet.

    Args:
        raw_input: Raw string (may contain FASTA headers, spaces, numbers, newlines).
        allow_n: If True, allows degenerate nucleotide 'N'.

    Returns:
        Tuple of (cleaned_sequence, is_valid, status_message, list_of_warnings).
    """
    if not raw_input or not raw_input.strip():
        return "", False, "Please enter a valid DNA sequence.", []

    warnings: List[str] = []
    lines = raw_input.strip().splitlines()
    seq_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            warnings.append(f"FASTA header detected and stripped: {stripped[:50]}...")
            continue
        # Remove numbers and whitespace (e.g. NCBI format: ' 1 atcgatcg 60 ')
        cleaned_line = re.sub(r"[\s\d_.-]+", "", stripped).upper()
        seq_lines.append(cleaned_line)

    cleaned_sequence = "".join(seq_lines)

    if len(cleaned_sequence) == 0:
        return "", False, "No nucleotide bases found after cleaning input text.", warnings

    valid_set = VALID_BASES_EXTENDED if allow_n else VALID_BASES
    invalid_chars = sorted(list(set(cleaned_sequence) - valid_set))

    if invalid_chars:
        sample_invalid = ", ".join(f"'{c}'" for c in invalid_chars[:5])
        return (
            cleaned_sequence,
            False,
            f"Invalid characters detected in DNA sequence: {sample_invalid}. "
            f"Allowed alphabet: {', '.join(sorted(valid_set))}.",
            warnings,
        )

    if len(cleaned_sequence) < 6:
        warnings.append(
            "Warning: Sequence is shorter than 6 nucleotides (1 k-mer). "
            "Genomic 6-mer foundation models perform best with sequences of at least 6 bases."
        )

    return cleaned_sequence, True, "Valid sequence ready for analysis.", warnings


def parse_fasta(fasta_text: str) -> Dict[str, str]:
    """
    Parses a multi-record or single-record FASTA formatted string.

    Args:
        fasta_text: Raw FASTA string.

    Returns:
        Dictionary mapping header names to cleaned DNA sequences.
    """
    records: Dict[str, str] = {}
    current_header = "Sequence_1"
    current_seq_parts: List[str] = []

    for line in fasta_text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if current_seq_parts:
                records[current_header] = "".join(current_seq_parts)
                current_seq_parts = []
            current_header = stripped[1:].strip() or f"Sequence_{len(records) + 1}"
        else:
            cleaned = re.sub(r"[\s\d_.-]+", "", stripped).upper()
            current_seq_parts.append(cleaned)

    if current_seq_parts:
        records[current_header] = "".join(current_seq_parts)

    return records


def compute_sequence_stats(sequence: str) -> Dict[str, Any]:
    """
    Calculates comprehensive biological and compositional metrics for a DNA sequence.

    Args:
        sequence: Cleaned DNA sequence.

    Returns:
        Dictionary with statistical metrics (GC%, CpG count, length, base counts).
    """
    length = len(sequence)
    if length == 0:
        return {
            "length": 0,
            "gc_content_pct": 0.0,
            "at_gc_ratio": 0.0,
            "cpg_count": 0,
            "cpg_oe_ratio": 0.0,
            "counts": {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0},
            "frequencies": {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0, "N": 0.0},
        }

    counts = {
        "A": sequence.count("A"),
        "C": sequence.count("C"),
        "G": sequence.count("G"),
        "T": sequence.count("T"),
        "N": sequence.count("N"),
    }
    freqs = {base: count / length for base, count in counts.items()}

    gc_count = counts["C"] + counts["G"]
    gc_pct = (gc_count / length) * 100.0 if length > 0 else 0.0

    at_count = counts["A"] + counts["T"]
    at_gc_ratio = at_count / gc_count if gc_count > 0 else float("inf")

    # CpG dinucleotide count & Observed/Expected ratio
    cpg_count = len(re.findall(r"(?=CG)", sequence))
    c_count = counts["C"]
    g_count = counts["G"]
    expected_cpg = (c_count * g_count) / length if length > 0 else 0
    cpg_oe_ratio = (cpg_count / expected_cpg) if expected_cpg > 0 else 0.0

    purines = counts["A"] + counts["G"]
    pyrimidines = counts["C"] + counts["T"]
    pur_pyr_ratio = purines / pyrimidines if pyrimidines > 0 else float("inf")

    return {
        "length": length,
        "gc_content_pct": round(gc_pct, 2),
        "at_gc_ratio": round(at_gc_ratio, 3),
        "purine_pyrimidine_ratio": round(pur_pyr_ratio, 3),
        "cpg_count": cpg_count,
        "cpg_oe_ratio": round(cpg_oe_ratio, 3),
        "counts": counts,
        "frequencies": {k: round(v, 4) for k, v in freqs.items()},
    }


def classify_mutation_effect(score: float, wt_base: str, mut_base: str) -> str:
    """
    Classifies the functional biological impact based on delta log-likelihood ratio (dLLR).
    """
    if wt_base == mut_base:
        return "Wild-Type (Neutral)"
    if score <= -2.0:
        return "Highly Disruptive / Deleterious"
    if score <= -0.75:
        return "Moderately Deleterious"
    if score < -0.2:
        return "Slightly Deleterious"
    if score <= 0.2:
        return "Tolerated / Neutral"
    if score <= 1.0:
        return "Slightly Enriched"
    return "Highly Enriched / Favorable"


def format_results_to_dataframe(
    score_matrix: np.ndarray, sequence: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Formats the raw mutation score matrix (4 x L) into:
    1. A wide DataFrame formatted for Heatmaps (rows=A,C,G,T; cols=1-A, 2-C...).
    2. A tidy long-format DataFrame with metadata for filtering, plotting, and CSV export.

    Args:
        score_matrix: Numpy array of shape (4, len(sequence)).
        sequence: The original wild-type DNA string.

    Returns:
        Tuple of (wide_matrix_df, tidy_long_df).
    """
    length = len(sequence)
    if score_matrix.shape != (4, length):
        raise ValueError(
            f"Shape mismatch: score_matrix shape {score_matrix.shape} "
            f"does not match expected (4, {length})."
        )

    # Column labels with position and wild-type base: '1-A', '2-C', etc.
    col_labels = [f"{i+1}-{sequence[i]}" for i in range(length)]
    wide_df = pd.DataFrame(score_matrix, index=ALPHABET, columns=col_labels)

    # Build tidy long DataFrame
    rows = []
    for pos_idx in range(length):
        wt_base = sequence[pos_idx]
        pos_num = pos_idx + 1
        for base_idx, mut_base in enumerate(ALPHABET):
            score = float(score_matrix[base_idx, pos_idx])
            mutation_name = f"{wt_base}{pos_num}{mut_base}"
            is_wt = wt_base == mut_base
            classification = classify_mutation_effect(score, wt_base, mut_base)

            rows.append(
                {
                    "Position": pos_num,
                    "WT_Base": wt_base,
                    "Mutant_Base": mut_base,
                    "Mutation": mutation_name,
                    "Delta_Score_LLR": round(score, 4),
                    "Is_WildType": is_wt,
                    "Effect": classification,
                    "Disruption_Level": abs(score) if not is_wt else 0.0,
                }
            )

    tidy_df = pd.DataFrame(rows)
    return wide_df, tidy_df
