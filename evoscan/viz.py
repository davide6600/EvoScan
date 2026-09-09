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
    "YlOrRd (Pathogenicity Hotspots)": "YlOrRd",
    "Inferno (Deep Mutational Scan)": "Inferno",
}


def create_saturation_heatmap(
    wide_df: pd.DataFrame,
    sequence: str,
    colorscale: str = "RdBu_r",
    show_values: bool = False,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    is_phred: bool = False,
    metric_label: str = "Δ Log-Likelihood Ratio (LLR)",
    title: str = "Saturation Mutagenesis Heatmap (Deep Mutational Scanning)",
    tidy_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Builds the flagship 2D interactive Saturation Mutagenesis Heatmap.
    Supports both bidirectional ΔLLR (Nucleotide Transformer) and
    calibrated positive Phred scores (Google DeepMind AlphaGenome Atlas).

    Args:
        wide_df: DataFrame with rows ['A', 'C', 'G', 'T'] and columns ['1-A', '2-C', ...].
        sequence: Clean wild-type DNA string.
        colorscale: Plotly continuous color scale name.
        show_values: If True, overlays numeric scores directly onto cells.
        z_min: Minimum color range clamp.
        z_max: Maximum color range clamp.
        is_phred: If True, uses Phred pathogenicity thresholds (0 to 70).
        metric_label: Label displayed on the colorbar and hover tooltip.
        title: Title of the heatmap.
        tidy_df: Optional tidy DataFrame containing additional variant metadata.

    Returns:
        Plotly Figure.
    """
    length = len(sequence)
    z_values = wide_df.values  # Shape: (4, L)
    y_labels = list(wide_df.index)
    x_labels = list(wide_df.columns)

    min_val = float(np.min(z_values)) if z_min is None else z_min
    max_val = float(np.max(z_values)) if z_max is None else z_max

    # Build lookup map from tidy_df if provided
    var_meta_lookup = {}
    if tidy_df is not None and not tidy_df.empty:
        for _, row in tidy_df.iterrows():
            pos = int(row.get("local_pos", row.get("Position", 1)))
            alt = str(row.get("alt", row.get("Mutant_Base", "")))
            key = (pos, alt)
            var_meta_lookup[key] = row

    # Build rich hover text matrix
    hover_text = []
    for y_idx, mut_base in enumerate(y_labels):
        row_hover = []
        for x_idx, col_name in enumerate(x_labels):
            pos_num = x_idx + 1
            wt_base = sequence[x_idx]
            score = float(z_values[y_idx, x_idx])
            is_wt = (wt_base == mut_base)

            extra_meta = var_meta_lookup.get((pos_num, mut_base), None)

            if is_phred:
                # AlphaGenome AVI Phred calibration
                if is_wt:
                    status = "🟢 <b>Wild-Type Reference (Baseline)</b>"
                elif score >= 30.0:
                    status = "🔴 <b>Extremely Disruptive (Top 0.1% Pathogenic)</b>"
                elif score >= 20.0:
                    status = "🟠 <b>High Disruption (Top 1% Disruptive)</b>"
                elif score >= 10.0:
                    status = "🟡 <b>Moderate Disruption (Top 10%)</b>"
                elif score >= 5.0:
                    status = "🔵 <b>Mild Impact</b>"
                else:
                    status = "⚪ <b>Tolerated / Neutral</b>"

                text = (
                    f"<b>Locus:</b> Position {pos_num} ({wt_base})<br>"
                    f"<b>Mutation:</b> {wt_base}{pos_num}{mut_base}<br>"
                    f"<b>{metric_label}:</b> {score:.2f}<br>"
                    f"<b>Impact Tier:</b> {status}"
                )
                if extra_meta is not None:
                    top_mod = extra_meta.get("top_modality")
                    pct = extra_meta.get("top_percentile")
                    if top_mod and top_mod != "Unassigned":
                        text += f"<br><b>Top Modality:</b> {top_mod}"
                    if pct is not None and not is_wt:
                        text += f"<br><b>Genome Percentile:</b> Top {pct:.2f}%"
            else:
                # Nucleotide Transformer ΔLLR
                if is_wt:
                    status = "🟢 <b>Wild-Type (Reference)</b>"
                elif score <= -2.0:
                    status = "🔴 <b>Highly Disruptive (Deleterious)</b>"
                elif score <= -0.75:
                    status = "🟠 <b>Moderately Deleterious</b>"
                elif score <= 0.2:
                    status = "⚪ <b>Tolerated / Neutral</b>"
                else:
                    status = "🔵 <b>Enriched / Favorable</b>"

                text = (
                    f"<b>Locus:</b> Position {pos_num} ({wt_base})<br>"
                    f"<b>Mutation:</b> {wt_base}{pos_num}{mut_base}<br>"
                    f"<b>{metric_label}:</b> {score:+.3f}<br>"
                    f"<b>Impact:</b> {status}"
                )

            row_hover.append(text)
        hover_text.append(row_hover)

    fig = go.Figure()

    # Color scaling configuration
    if is_phred:
        # Sequential scale starting at 0
        actual_scale = colorscale if colorscale not in ["RdBu_r", "Picnic"] else "YlOrRd"
        heatmap_args = dict(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=hover_text,
            hoverinfo="text",
            colorscale=actual_scale,
            zmin=0.0 if z_min is None else z_min,
            zmax=max(35.0, max_val) if z_max is None else z_max,
        )
    else:
        # Divergent scale centered at 0
        abs_limit = max(abs(min_val), abs(max_val), 1.5)
        heatmap_args = dict(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=hover_text,
            hoverinfo="text",
            colorscale=colorscale,
            zmid=0.0,
            zmin=-abs_limit if z_min is None else z_min,
            zmax=abs_limit * 0.6 if z_max is None else z_max,
        )

    # Main Heatmap Trace
    fig.add_trace(
        go.Heatmap(
            **heatmap_args,
            colorbar=dict(
                title=dict(
                    text=f"<b>{metric_label}</b>",
                    side="right",
                    font=dict(size=12, color="#E2E8F0"),
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
                val = float(z_values[y_idx, x_idx])
                if sequence[x_idx] == mut_base:
                    txt = "WT"
                elif is_phred:
                    txt = f"{val:.1f}"
                else:
                    txt = f"{val:+.1f}"

                annotations.append(
                    dict(
                        x=col_name,
                        y=mut_base,
                        text=txt,
                        showarrow=False,
                        font=dict(
                            size=9,
                            color="#FFFFFF" if (val > 15.0 if is_phred else abs(val) > 1.0) else "#0F172A",
                        ),
                    )
                )
        fig.update_layout(annotations=annotations)

    # Configure modern aesthetic styling
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color="#F8FAFC", family="Inter, system-ui, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(
                text="<b>Position & Wild-Type Base</b>",
                font=dict(size=13, color="#CBD5E1"),
            ),
            tickangle=-45 if length > 25 else 0,
            tickfont=dict(size=10, color="#94A3B8"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="<b>Mutated Nucleotide</b>",
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
    tidy_df: pd.DataFrame,
    sequence: str,
    is_phred: bool = False,
    metric_label: str = "ΔLLR",
) -> go.Figure:
    """
    Creates a 1D positional vulnerability profile plot.
    Supports both ΔLLR (Nucleotide Transformer) and Phred / Modality scores (AlphaGenome Atlas).
    """
    df = tidy_df.copy()

    # Standardize column names if needed
    pos_col = "local_pos" if "local_pos" in df.columns else ("Position" if "Position" in df.columns else "position")
    wt_col = "ref" if "ref" in df.columns else "WT_Base"
    mut_col = "alt" if "alt" in df.columns else "Mutant_Base"
    score_col = "avi_phred" if ("avi_phred" in df.columns and is_phred) else ("score" if "score" in df.columns else "Delta_Score_LLR")

    # Filter out wild-type self-comparisons
    if "Is_WildType" in df.columns:
        mutants = df[~df["Is_WildType"]].copy()
    else:
        mutants = df[df[wt_col] != df[mut_col]].copy()

    if mutants.empty:
        mutants = df.copy()

    # Group by position to compute stats
    pos_stats = (
        mutants.groupby(pos_col)
        .agg(
            WT_Base=(wt_col, "first"),
            Min_Score=(score_col, "min"),
            Mean_Score=(score_col, "mean"),
            Max_Score=(score_col, "max"),
        )
        .reset_index()
    )

    pos_stats["Locus"] = [f"{p}-{pos_stats.loc[i, 'WT_Base']}" for i, p in enumerate(pos_stats[pos_col])]

    if is_phred:
        # For Phred: higher score = higher pathogenic impact
        pos_stats["Peak_Impact"] = pos_stats["Max_Score"]
        pos_stats["Avg_Impact"] = pos_stats["Mean_Score"]
        bar_title = "Peak Vulnerability (Max Phred)"
        line_title = "Mean Vulnerability (Mean Phred)"
        y_title = "<b>Vulnerability Degree (AVI Phred -10 log10 P)</b>"
    else:
        # For ΔLLR: negative score = deleterious
        pos_stats["Peak_Impact"] = np.abs(pos_stats["Min_Score"])
        pos_stats["Avg_Impact"] = np.abs(pos_stats["Mean_Score"])
        bar_title = "Maximum Vulnerability (|Min ΔLLR|)"
        line_title = "Mean Deleterious Impact"
        y_title = "<b>Vulnerability Degree (|ΔLLR|)</b>"

    fig = go.Figure()

    # Bar plot for peak vulnerability
    fig.add_trace(
        go.Bar(
            x=pos_stats["Locus"],
            y=pos_stats["Peak_Impact"],
            name=bar_title,
            marker=dict(
                color=pos_stats["Peak_Impact"],
                colorscale="Reds",
                showscale=False,
            ),
            hovertemplate="<b>Locus:</b> %{x}<br><b>Peak Vulnerability:</b> %{y:.2f}<extra></extra>",
        )
    )

    # Line for mean score
    fig.add_trace(
        go.Scatter(
            x=pos_stats["Locus"],
            y=pos_stats["Avg_Impact"],
            mode="lines+markers",
            name=line_title,
            line=dict(color="#38BDF8", width=2.5),
            marker=dict(size=5, color="#0284C7"),
            hovertemplate="<b>Locus:</b> %{x}<br><b>Mean Impact:</b> %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Positional Mutational Vulnerability Profile (Hotspot Vulnerability)</b>",
            font=dict(size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            title=dict(text="<b>Nucleotide Position</b>", font=dict(color="#CBD5E1")),
            tickfont=dict(size=10, color="#94A3B8"),
            tickangle=-45 if len(sequence) > 25 else 0,
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color="#CBD5E1")),
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


def create_score_distribution_plot(
    tidy_df: pd.DataFrame,
    is_phred: bool = False,
) -> go.Figure:
    """
    Creates an interactive histogram showing the distribution of mutation scores.
    """
    df = tidy_df.copy()

    # Identify columns
    wt_col = "ref" if "ref" in df.columns else "WT_Base"
    mut_col = "alt" if "alt" in df.columns else "Mutant_Base"
    score_col = "avi_phred" if ("avi_phred" in df.columns and is_phred) else ("score" if "score" in df.columns else "Delta_Score_LLR")

    if "Is_WildType" in df.columns:
        mutants = df[~df["Is_WildType"]].copy()
    else:
        mutants = df[df[wt_col] != df[mut_col]].copy()

    if mutants.empty:
        mutants = df.copy()

    if is_phred:
        # Classify into Phred impact tiers for coloring
        def phred_tier(val: float) -> str:
            if val >= 30.0:
                return "Extreme Disruption (≥30 Phred)"
            elif val >= 20.0:
                return "High Impact (20-30 Phred)"
            elif val >= 10.0:
                return "Moderate Impact (10-20 Phred)"
            elif val >= 5.0:
                return "Mild Impact (5-10 Phred)"
            return "Tolerated / Neutral (<5 Phred)"

        mutants["Impact_Tier"] = mutants[score_col].apply(phred_tier)

        color_map = {
            "Extreme Disruption (≥30 Phred)": "#EF4444",
            "High Impact (20-30 Phred)": "#F97316",
            "Moderate Impact (10-20 Phred)": "#FBBF24",
            "Mild Impact (5-10 Phred)": "#38BDF8",
            "Tolerated / Neutral (<5 Phred)": "#94A3B8",
        }

        fig = px.histogram(
            mutants,
            x=score_col,
            nbins=30,
            color="Impact_Tier",
            color_discrete_map=color_map,
            labels={score_col: "AlphaGenome AVI Score (Phred -10 log10 P)"},
        )
        x_title = "<b>AlphaGenome AVI Score (Phred)</b>"
    else:
        color_map = {
            "Highly Disruptive / Deleterious": "#EF4444",
            "Moderately Deleterious": "#F97316",
            "Slightly Deleterious": "#FBBF24",
            "Tolerated / Neutral": "#94A3B8",
            "Slightly Enriched": "#38BDF8",
            "Highly Enriched / Favorable": "#3B82F6",
        }
        fig = px.histogram(
            mutants,
            x=score_col,
            nbins=30,
            color="Effect" if "Effect" in mutants.columns else None,
            color_discrete_map=color_map,
            labels={score_col: "ΔLLR Score (Log-Likelihood Ratio)"},
        )
        x_title = "<b>ΔLLR Score (Log-Likelihood Ratio)</b>"

    fig.update_layout(
        title=dict(
            text="<b>Distribution of Mutational Impact Scores</b>",
            font=dict(size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="#CBD5E1")),
            tickfont=dict(color="#94A3B8"),
            gridcolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            title=dict(text="<b>Mutation Count</b>", font=dict(color="#CBD5E1")),
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


def create_substitution_matrix_plot(
    tidy_df: pd.DataFrame,
    is_phred: bool = False,
) -> go.Figure:
    """
    Creates a 4x4 matrix plot showing average delta-score or Phred score for every WT -> Mutant base transition.
    """
    df = tidy_df.copy()

    wt_col = "ref" if "ref" in df.columns else "WT_Base"
    mut_col = "alt" if "alt" in df.columns else "Mutant_Base"
    score_col = "avi_phred" if ("avi_phred" in df.columns and is_phred) else ("score" if "score" in df.columns else "Delta_Score_LLR")

    if "Is_WildType" in df.columns:
        mutants = df[~df["Is_WildType"]].copy()
    else:
        mutants = df[df[wt_col] != df[mut_col]].copy()

    if mutants.empty:
        mutants = df.copy()

    pivot_sub = (
        mutants.groupby([wt_col, mut_col])[score_col]
        .mean()
        .unstack(fill_value=0.0)
    )

    for b in ["A", "C", "G", "T"]:
        if b in pivot_sub.index and b in pivot_sub.columns:
            pivot_sub.loc[b, b] = 0.0

    pivot_sub = pivot_sub.reindex(index=["A", "C", "G", "T"], columns=["A", "C", "G", "T"])

    if is_phred:
        colorscale = "YlOrRd"
        color_label = "Mean AVI Phred"
        imshow_args = dict(
            color_continuous_scale=colorscale,
        )
    else:
        colorscale = "RdBu_r"
        color_label = "Mean ΔLLR"
        imshow_args = dict(
            color_continuous_scale=colorscale,
            color_continuous_midpoint=0.0,
        )

    fig = px.imshow(
        pivot_sub,
        text_auto=".2f",
        labels=dict(x="Mutated Nucleotide", y="Wild-Type Nucleotide", color=color_label),
        **imshow_args,
    )

    fig.update_layout(
        title=dict(
            text="<b>Mean Substitution Matrix (WT ➔ Mutant)</b>",
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
