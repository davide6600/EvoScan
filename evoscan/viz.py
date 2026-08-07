"""
EvoScan Visualization Engine: Interactive Plotly Saturation Mutagenesis Heatmaps,
Positional Mutational Profiles, and Substitution Matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

COLORSCALES = {
    "RdBu_r (Biomolecular Standard)": "RdBu_r",
    "Spectral_r (High Contrast)": "Spectral_r",
    "Viridis (Perceptually Uniform)": "Viridis",
    "Plasma (High Dynamic Range)": "Plasma",
    "Picnic (Subtle Divergent)": "Picnic",
}


def create_saturation_heatmap(
    wide_df: pd.DataFrame,
    sequence: str,
    colorscale: str = "RdBu_r",
    show_values: bool = False,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> go.Figure:
    """
    Builds the flagship 2D interactive Saturation Mutagenesis Heatmap.

    Args:
        wide_df: DataFrame with rows ['A', 'C', 'G', 'T'] and columns ['1-A', '2-C', ...].
        sequence: Clean wild-type DNA string.
        colorscale: Plotly continuous color scale name.
        show_values: If True, overlays numeric scores directly onto cells.
        z_min: Minimum color range clamp.
        z_max: Maximum color range clamp.

    Returns:
        Plotly Figure.
    """
    length = len(sequence)
    z_values = wide_df.values  # Shape: (4, L)
    y_labels = list(wide_df.index)
    x_labels = list(wide_df.columns)

    # Compute symmetric or robust color bounds centered at 0
    min_val = float(np.min(z_values)) if z_min is None else z_min
    max_val = float(np.max(z_values)) if z_max is None else z_max
    abs_limit = max(abs(min_val), abs(max_val), 1.5)

    # Build rich hover text matrix
    hover_text = []
    for y_idx, mut_base in enumerate(y_labels):
        row_hover = []
        for x_idx, col_name in enumerate(x_labels):
            pos_num = x_idx + 1
            wt_base = sequence[x_idx]
            score = z_values[y_idx, x_idx]
            is_wt = wt_base == mut_base

            if is_wt:
                status = "🟢 <b>Wild-Type (Riferimento)</b>"
            elif score <= -2.0:
                status = "🔴 <b>Altamente Distruttiva (Deleteria)</b>"
            elif score <= -0.75:
                status = "🟠 <b>Moderatamente Deleteria</b>"
            elif score <= 0.2:
                status = "⚪ <b>Tollerata / Neutra</b>"
            else:
                status = "🔵 <b>Arricchita / Favorevole</b>"

            text = (
                f"<b>Locus:</b> Posizione {pos_num} ({wt_base})<br>"
                f"<b>Mutazione:</b> {wt_base}{pos_num}{mut_base}<br>"
                f"<b>ΔLLR Score:</b> {score:+.3f}<br>"
                f"<b>Impatto:</b> {status}"
            )
            row_hover.append(text)
        hover_text.append(row_hover)

    fig = go.Figure()

    # Main Heatmap Trace
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=hover_text,
            hoverinfo="text",
            colorscale=colorscale,
            zmid=0.0,
            zmin=-abs_limit if z_min is None else z_min,
            zmax=abs_limit * 0.6 if z_max is None else z_max,
            colorbar=dict(
                title=dict(
                    text="<b>Δ Log-Likelihood Ratio (LLR)</b>",
                    side="right",
                    font=dict(size=13, color="#E2E8F0"),
                ),
                tickfont=dict(size=11, color="#CBD5E1"),
                len=0.9,
                thickness=18,
                outlinewidth=0,
            ),
        )
    )

    # Add markers for Wild-Type reference diagonal
    wt_x = []
    wt_y = []
    for idx, wt_base in enumerate(sequence):
        wt_x.append(x_labels[idx])
        wt_y.append(wt_base)

    fig.add_trace(
        go.Scatter(
            x=wt_x,
            y=wt_y,
            mode="markers",
            marker=dict(
                symbol="circle",
                size=7,
                color="rgba(255, 255, 255, 0.9)",
                line=dict(color="#0F172A", width=1.5),
            ),
            name="Wild-Type Reference (WT)",
            hoverinfo="skip",
        )
    )

    # Optional text annotations on cells
    if show_values and length <= 60:
        annotations = []
        for y_idx, mut_base in enumerate(y_labels):
            for x_idx, col_name in enumerate(x_labels):
                val = z_values[y_idx, x_idx]
                txt = "WT" if sequence[x_idx] == mut_base else f"{val:+.1f}"
                annotations.append(
                    dict(
                        x=col_name,
                        y=mut_base,
                        text=txt,
                        showarrow=False,
                        font=dict(
                            size=9,
                            color="#FFFFFF" if abs(val) > 1.0 else "#0F172A",
                        ),
                    )
                )
        fig.update_layout(annotations=annotations)

    # Configure modern aesthetic styling
    fig.update_layout(
        title=dict(
            text="<b>Mappa di Mutagenesi a Saturazione (Deep Mutational Scanning)</b>",
            font=dict(size=18, color="#F8FAFC", family="Inter, system-ui, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(
                text="<b>Posizione e Nucleotide Wild-Type</b>",
                font=dict(size=13, color="#CBD5E1"),
            ),
            tickangle=-45 if length > 25 else 0,
            tickfont=dict(size=10, color="#94A3B8"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="<b>Nucleotide Mutato</b>",
                font=dict(size=13, color="#CBD5E1"),
            ),
            tickfont=dict(size=13, color="#F1F5F9", family="monospace"),
            categoryorder="array",
            categoryarray=["T", "G", "C", "A"],  # standard top-to-bottom layout
            showgrid=False,
            zeroline=False,
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(30, 41, 59, 0.8)",
        margin=dict(l=60, r=40, t=60, b=70),
        height=380,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94A3B8", size=11),
        ),
    )

    return fig


def create_position_sensitivity_plot(
    tidy_df: pd.DataFrame, sequence: str
) -> go.Figure:
    """
    Creates a 1D positional vulnerability profile plot (max deleterious score & mean sensitivity).
    """
    # Filter out wild-type self-comparisons
    mutants = tidy_df[~tidy_df["Is_WildType"]].copy()

    # Group by position to compute stats
    pos_stats = (
        mutants.groupby("Position")
        .agg(
            WT_Base=("WT_Base", "first"),
            Min_Score=("Delta_Score_LLR", "min"),  # most deleterious mutation
            Mean_Score=("Delta_Score_LLR", "mean"),
            Max_Score=("Delta_Score_LLR", "max"),
        )
        .reset_index()
    )

    pos_stats["Locus"] = [f"{p}-{pos_stats.loc[i, 'WT_Base']}" for i, p in enumerate(pos_stats["Position"])]
    pos_stats["Max_Disruption"] = np.abs(pos_stats["Min_Score"])

    fig = go.Figure()

    # Area plot for max disruption
    fig.add_trace(
        go.Bar(
            x=pos_stats["Locus"],
            y=pos_stats["Max_Disruption"],
            name="Vulnerabilità Massima (|Min ΔLLR|)",
            marker=dict(
                color=pos_stats["Max_Disruption"],
                colorscale="Reds",
                showscale=False,
            ),
            hovertemplate="<b>Locus:</b> %{x}<br><b>Vulnerabilità Max:</b> %{y:.3f}<extra></extra>",
        )
    )

    # Line for mean score
    fig.add_trace(
        go.Scatter(
            x=pos_stats["Locus"],
            y=np.abs(pos_stats["Mean_Score"]),
            mode="lines+markers",
            name="Impatto Medio Deleterio",
            line=dict(color="#38BDF8", width=2.5),
            marker=dict(size=5, color="#0284C7"),
            hovertemplate="<b>Locus:</b> %{x}<br><b>Impatto Medio:</b> %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Profilo di Sensibilità Mutazionale per Posizione (Hotspot Vulnerability)</b>",
            font=dict(size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            title=dict(text="<b>Posizione Nucleotidica</b>", font=dict(color="#CBD5E1")),
            tickfont=dict(size=10, color="#94A3B8"),
            tickangle=-45 if len(sequence) > 25 else 0,
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(text="<b>Grado di Vulnerabilità (|ΔLLR|)</b>", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#94A3B8"),
            gridcolor="rgba(255, 255, 255, 0.1)",
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(30, 41, 59, 0.8)",
        margin=dict(l=60, r=40, t=50, b=60),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94A3B8"),
        ),
    )

    return fig


def create_score_distribution_plot(tidy_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive histogram showing the distribution of mutation scores.
    """
    mutants = tidy_df[~tidy_df["Is_WildType"]].copy()

    fig = px.histogram(
        mutants,
        x="Delta_Score_LLR",
        nbins=30,
        color="Effect",
        color_discrete_map={
            "Fortemente Distruttiva / Deleteria": "#EF4444",
            "Moderatamente Deleteria": "#F97316",
            "Lievemente Sfavorevole": "#FBBF24",
            "Tollerata / Neutra": "#94A3B8",
            "Lievemente Arricchita": "#38BDF8",
            "Fortemente Arricchita / Favorevole": "#3B82F6",
        },
        labels={"Delta_Score_LLR": "Punteggio ΔLLR (Log-Likelihood Ratio)"},
    )

    fig.update_layout(
        title=dict(
            text="<b>Distribuzione degli Score di Impatto Mutazionale</b>",
            font=dict(size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            title=dict(text="<b>ΔLLR Score</b>", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#94A3B8"),
            gridcolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            title=dict(text="<b>Conteggio Mutazioni</b>", font=dict(color="#CBD5E1")),
            tickfont=dict(color="#94A3B8"),
            gridcolor="rgba(255, 255, 255, 0.1)",
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(30, 41, 59, 0.8)",
        margin=dict(l=60, r=40, t=50, b=60),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#94A3B8", size=10),
        ),
    )

    return fig


def create_substitution_matrix_plot(tidy_df: pd.DataFrame) -> go.Figure:
    """
    Creates a 4x4 matrix plot showing average delta-score for every WT -> Mutant base transition.
    """
    mutants = tidy_df[~tidy_df["Is_WildType"]]
    pivot_sub = (
        mutants.groupby(["WT_Base", "Mutant_Base"])["Delta_Score_LLR"]
        .mean()
        .unstack(fill_value=0.0)
    )

    for b in ["A", "C", "G", "T"]:
        if b in pivot_sub.index and b in pivot_sub.columns:
            pivot_sub.loc[b, b] = 0.0

    pivot_sub = pivot_sub.reindex(index=["A", "C", "G", "T"], columns=["A", "C", "G", "T"])

    fig = px.imshow(
        pivot_sub,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0.0,
        labels=dict(x="Nucleotide Mutato", y="Nucleotide Wild-Type", color="Media ΔLLR"),
    )

    fig.update_layout(
        title=dict(
            text="<b>Matrice di Sostituzione Media (WT ➔ Mutante)</b>",
            font=dict(size=15, color="#F8FAFC"),
        ),
        xaxis=dict(tickfont=dict(size=12, color="#CBD5E1")),
        yaxis=dict(tickfont=dict(size=12, color="#CBD5E1")),
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        plot_bgcolor="rgba(30, 41, 59, 0.8)",
        margin=dict(l=50, r=40, t=50, b=50),
        height=300,
    )
    return fig
