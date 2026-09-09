"""
EvoScan: Zero-Shot DNA Mutation Map Visualizer (Saturation Mutagenesis Heatmap)
Dual-Engine Platform:
1. Google DeepMind AlphaGenome Atlas (Human Genomic Loci GRCh38 via Cloud gRPC Engine).
2. Nucleotide Transformer & Genomic Foundation Models (Custom / Synthetic DNA Sequences).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from evoscan.utils import (
    clean_and_validate_dna,
    parse_fasta,
    compute_sequence_stats,
    format_results_to_dataframe,
)
from evoscan.model import EvoScanEngine, AVAILABLE_MODELS, DEFAULT_MODEL_ID
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
    create_substitution_matrix_plot,
    COLORSCALES,
)

# Page configuration
st.set_page_config(
    page_title="EvoScan - DNA Mutation Heatmap Visualizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Scientific CSS Design
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Header Gradient */
.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-bottom: 1.2rem;
    font-weight: 400;
}

/* Metric Cards */
.metric-card {
    background: rgba(125, 135, 155, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(8px);
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.45rem;
    font-weight: 700;
}

.metric-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.15rem;
}

/* Badges */
.badge-pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

.badge-blue {
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.badge-purple {
    background: rgba(192, 132, 252, 0.15);
    color: #C084FC;
    border: 1px solid rgba(192, 132, 252, 0.3);
}

.badge-green {
    background: rgba(74, 222, 128, 0.15);
    color: #4ADE80;
    border: 1px solid rgba(74, 222, 128, 0.3);
}

.badge-orange {
    background: rgba(251, 146, 60, 0.15);
    color: #FB923C;
    border: 1px solid rgba(251, 146, 60, 0.3);
}

/* Callout Box */
.callout-box {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1.2rem;
}

/* Button enhancements */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_sample_presets() -> Dict[str, Any]:
    """Loads biological preset sequences from JSON for the local engine."""
    preset_path = Path(__file__).parent / "sample_data" / "sequences.json"
    if preset_path.exists():
        with open(preset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=False)
def get_inference_engine(model_id: str) -> EvoScanEngine:
    """Cached loader for the local neural genomic model engine."""
    engine = EvoScanEngine(model_id=model_id)
    engine.load_model()
    return engine


def math_chunks(length: int) -> int:
    return (length + 5) // 6


def main():
    # --- Sidebar Configuration ---
    st.sidebar.markdown("## 🧬 **EvoScan Studio**")
    st.sidebar.markdown("In Silico Saturation Mutagenesis & Deep Mutational Scanning Platform.")

    # Engine Selection
    engine_mode = st.sidebar.radio(
        "Genomic Scoring Engine",
        [
            "🧬 Google DeepMind AlphaGenome Atlas",
            "🧪 Nucleotide Transformer & Local Models",
        ],
        index=0,
        help="Choose between scoring human genomic loci (GRCh38) via Google AlphaGenome Atlas (Cloud gRPC) "
             "or arbitrary/synthetic DNA sequences via local Nucleotide Transformer.",
    )

    st.sidebar.markdown("---")

    # =========================================================================
    # BRANCH 1: GOOGLE DEEPMIND ALPHAGENOME ATLAS ENGINE
    # =========================================================================
    if "AlphaGenome" in engine_mode:
        st.sidebar.markdown("### ☁️ **AlphaGenome Atlas Settings**")

        # Personal API Key Input (EMPTY BY DEFAULT)
        env_key = os.environ.get("ALPHAGENOME_API_KEY", "")
        default_sidebar_key = ""

        user_api_key = st.sidebar.text_input(
            "Personal DeepMind API Key",
            value=default_sidebar_key,
            type="password",
            placeholder="Paste your ALPHAGENOME_API_KEY here...",
            help="Enter your personal Google DeepMind API key. The key is never stored or shared.",
        )

        active_api_key = user_api_key.strip() if user_api_key.strip() else env_key.strip()

        if not active_api_key:
            st.sidebar.warning(
                "⚠️ **API Key Not Configured**\n\n"
                "A personal API key (free for Research Use Only) is required to query AlphaGenome Atlas.\n\n"
                "👉 [**Request your API Key**](https://deepmind.google.com/science/alphagenome/api)\n\n"
                "See guide: `ALPHAGENOME_API_GUIDE.md`"
            )
        else:
            st.sidebar.success("🔑 **API Key Active** (Personal / Session)")

        # Modality Selection
        st.sidebar.markdown("### 🎛️ **Functional Modality Layer**")
        modality_keys = list(MODALITY_OPTIONS.keys())
        modality_labels = [MODALITY_OPTIONS[k]["name"] for k in modality_keys]

        selected_mod_idx = st.sidebar.selectbox(
            "Scoring Layer",
            range(len(modality_keys)),
            format_func=lambda i: modality_labels[i],
            index=0,
            help="Select the biological or epigenomic metric to score for each candidate mutation.",
        )
        selected_modality = modality_keys[selected_mod_idx]
        st.sidebar.caption(f"ℹ️ {MODALITY_OPTIONS[selected_modality]['description']}")

        # Genomic Preset Selection
        st.sidebar.markdown("### 📚 **Predefined Clinical & Biological Loci**")
        preset_names = list(GENOMIC_PRESETS.keys()) + ["-- Custom Genomic Coordinates --"]
        selected_preset_name = st.sidebar.selectbox(
            "Select GRCh38 Clinical Locus",
            preset_names,
            index=0,
        )

        # Coordinate string
        if selected_preset_name in GENOMIC_PRESETS:
            default_region = GENOMIC_PRESETS[selected_preset_name]["region"]
            preset_info = GENOMIC_PRESETS[selected_preset_name]
        else:
            default_region = "chr11:5225720-5225780"
            preset_info = None

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 **Visualization Settings**")
        selected_colorscale = st.sidebar.selectbox(
            "Heatmap Color Palette",
            list(COLORSCALES.keys()),
            index=5,  # Default to YlOrRd for Phred
        )
        actual_colorscale = COLORSCALES[selected_colorscale]

        show_overlay_values = st.sidebar.checkbox(
            "Overlay numeric scores on cells",
            value=False,
            help="Displays numeric Phred or ΔLLR values directly inside each heatmap cell.",
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
            <div style='font-size:0.75rem; color:#64748B;'>
                <b>EvoScan v1.2.0</b> • DeepMind AlphaGenome Atlas Engine<br>
                <a href='https://deepmind.google.com/science/alphagenome/atlas' target='_blank' style='color:#38BDF8;'>Atlas Web Portal</a> • 
                <a href='https://deepmind.google.com/science/alphagenome/api' target='_blank' style='color:#C084FC;'>API Registration</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Main Header AlphaGenome
        col_h1, col_h2 = st.columns([0.75, 0.25])
        with col_h1:
            st.markdown("<div class='hero-title'>🧬 EvoScan × AlphaGenome Atlas</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='hero-subtitle'>"
                "High-resolution zero-shot saturation mutagenesis across human GRCh38 loci, "
                "powered by Google DeepMind's 1-Mb context, 9,440-track foundation model."
                "</div>",
                unsafe_allow_html=True,
            )
        with col_h2:
            st.markdown(
                """
                <div style='text-align: right; padding-top: 0.5rem;'>
                    <span class='badge-pill badge-purple'>AlphaGenome Atlas</span>
                    <span class='badge-pill badge-blue'>1-Mb Context</span>
                    <span class='badge-pill badge-orange'>AVI Phred 0-70</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Region Input and Locus Viewer
        st.markdown("### 1. Genomic Locus Selection (GRCh38)")

        col_reg1, col_reg2 = st.columns([0.7, 0.3])
        with col_reg1:
            region_str = st.text_input(
                "Genomic Coordinates (format chr:start-end):",
                value=default_region,
                placeholder="e.g. chr11:5225720-5225780 or chr17:7676080-7676150",
                help="Enter 1-based coordinates on the GRCh38 reference genome (recommended max window: 1,000 bp).",
            )
            if preset_info:
                st.info(
                    f"🧬 **Gene:** `{preset_info['gene']}` | **Region:** `{preset_info['region']}`\n\n"
                    f"💡 {preset_info['description']}",
                    icon="ℹ️",
                )

        with col_reg2:
            # Parse region for deep link button
            try:
                p_chrom, p_start, p_end = parse_genomic_region(region_str)
                locus_link = f"https://deepmind.google.com/science/alphagenome/atlas?q={p_chrom}%3A{p_start}-{p_end}&m=locus"
                st.markdown("<br>", unsafe_allow_html=True)
                st.link_button(
                    "🔗 Explore in DeepMind Atlas",
                    url=locus_link,
                    help="Opens the official Google DeepMind viewer for this genomic interval.",
                    use_container_width=True,
                )
            except Exception:
                p_chrom, p_start, p_end = None, None, None

        # Parse & Validate Coordinates
        region_valid = False
        chrom, start, end = "", 0, 0
        try:
            chrom, start, end = parse_genomic_region(region_str)
            region_valid = True
            width = end - start + 1
        except Exception as e:
            st.error(f"Coordinate error: {str(e)}")
            width = 0

        # Status cards for region
        if region_valid:
            col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
            with col_sc1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Chromosome</div>
                        <div class='metric-value'>{chrom}</div>
                        <div class='metric-sub'>GRCh38 Reference</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_sc2:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Interval Length</div>
                        <div class='metric-value'>{width:,} bp</div>
                        <div class='metric-sub'>Analyzed Window</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_sc3:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Total SNVs Scored</div>
                        <div class='metric-value'>{width * 3:,}</div>
                        <div class='metric-sub'>3 substitutions per base</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_sc4:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Active Layer</div>
                        <div class='metric-value' style='font-size:1.1rem;'>{selected_modality}</div>
                        <div class='metric-sub'>{MODALITY_OPTIONS[selected_modality]['category']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Trigger Button
        run_ag_col1, run_ag_col2 = st.columns([0.35, 0.65])
        with run_ag_col1:
            run_ag = st.button(
                "🚀 Run AlphaGenome Atlas Scan (Cloud gRPC)",
                type="primary",
                use_container_width=True,
                disabled=not region_valid,
            )

        # API Key warning block if user tries to run without key
        if run_ag and not active_api_key:
            st.error(
                "❌ **Missing API Key:** To query Google DeepMind AlphaGenome Atlas cloud servers, "
                "please enter your personal API key in the left sidebar.\n\n"
                "👉 [**Click here to register and generate your personal API Key**](https://deepmind.google.com/science/alphagenome/api)\n\n"
                "The key is free for non-commercial academic research (RUO)."
            )
            return

        # Execution logic
        if run_ag and region_valid and active_api_key:
            with st.spinner("Connecting to Google DeepMind AlphaGenome Atlas gRPC cluster..."):
                t0 = time.time()
                try:
                    engine = AlphaGenomeEngine(api_key=active_api_key)
                    score_matrix, ref_seq, tidy_df, meta = engine.score_genomic_interval(
                        chrom=chrom,
                        start_1_based=start,
                        end_1_based=end,
                        modality=selected_modality,
                    )
                    elapsed = time.time() - t0

                    is_phred = (selected_modality == "AVI_SCORE")
                    wide_df, _ = format_results_to_dataframe(
                        score_matrix=score_matrix,
                        sequence=ref_seq,
                        is_phred=is_phred,
                        metric_name="Score",
                    )

                    stats = compute_sequence_stats(ref_seq)

                    st.session_state["alphagenome_results"] = {
                        "wide_df": wide_df,
                        "tidy_df": tidy_df,
                        "score_matrix": score_matrix,
                        "sequence": ref_seq,
                        "stats": stats,
                        "meta": meta,
                        "elapsed": elapsed,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "modality": selected_modality,
                    }
                    st.success(
                        f"✅ AlphaGenome Atlas processed {meta['variants_count']:,} variants "
                        f"in **{elapsed:.2f} s** via gRPC! Peak Phred score: **{meta['max_phred']:.2f}**."
                    )
                except Exception as e:
                    st.error(f"Error querying AlphaGenome Atlas: {str(e)}")
                    return

        # Display AlphaGenome Results
        if "alphagenome_results" in st.session_state:
            res = st.session_state["alphagenome_results"]
            if res["chrom"] == chrom and res["start"] == start and res["end"] == end:
                wide_df = res["wide_df"]
                tidy_df = res["tidy_df"]
                ref_seq = res["sequence"]
                is_phred = (res["modality"] == "AVI_SCORE")
                metric_name = "AVI Phred Score (-10 log10 P)" if is_phred else f"{res['modality']} Impact"

                st.markdown("---")
                st.markdown("### 2. Interactive Results & Deep Mutational Scanning Heatmap")

                tab_heat, tab_prof, tab_dist, tab_table, tab_export, tab_guide = st.tabs(
                    [
                        "📊 Saturation Heatmap",
                        "📈 Positional Vulnerability Profile",
                        "🔄 Substitution Matrix & Distribution",
                        "📋 Variant Catalog & Deep-Links",
                        "📥 Data Export & Download",
                        "📖 Methodological Guide & API Key",
                    ]
                )

                with tab_heat:
                    st.markdown(
                        f"2D Saturation Mutagenesis Heatmap for **{chrom}:{start}-{end}** ({len(ref_seq)} bp). "
                        f"Displaying: **{MODALITY_OPTIONS[res['modality']]['name']}**. "
                        "White markers denote the wild-type reference sequence."
                    )
                    fig_heat = create_saturation_heatmap(
                        wide_df=wide_df,
                        sequence=ref_seq,
                        colorscale=actual_colorscale,
                        show_values=show_overlay_values,
                        is_phred=is_phred,
                        metric_label=metric_name,
                        title=f"AlphaGenome Saturation Mutagenesis: {chrom}:{start}-{end} ({res['modality']})",
                        tidy_df=tidy_df,
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                    st.caption(
                        "💡 **Phred Interpretation Guide (0–70):** "
                        "0 = Wild-type / Neutral; "
                        "5–10 = Mild Impact; "
                        "10–20 = Moderate Impact (Top 10%); "
                        "20–30 = High Impact (Top 1%); "
                        "≥30 = Extremely Disruptive (Top 0.1% genome-wide pathogenic tier)."
                    )

                with tab_prof:
                    st.markdown(
                        "This plot identifies **hyper-vulnerable regulatory hotspots**: "
                        "loci where nucleotide substitutions cause maximal disruption to the model's prediction."
                    )
                    fig_prof = create_position_sensitivity_plot(
                        tidy_df=tidy_df,
                        sequence=ref_seq,
                        is_phred=is_phred,
                        metric_label=metric_name,
                    )
                    st.plotly_chart(fig_prof, use_container_width=True)

                    if not tidy_df.empty:
                        score_col = "avi_phred" if is_phred else "score"
                        top_vulnerable = (
                            tidy_df.groupby("local_pos")
                            .agg(
                                WT=("ref", "first"),
                                Genomic_Pos=("position", "first"),
                                Max_Score=(score_col, "max"),
                                Top_Mod=("top_modality", "first"),
                            )
                            .sort_values("Max_Score", ascending=False)
                            .head(3)
                        )

                        st.markdown("#### 🚨 Top 3 Hyper-Vulnerable Hotspots:")
                        cols_top = st.columns(3)
                        for idx, (lpos, row) in enumerate(top_vulnerable.iterrows()):
                            with cols_top[idx]:
                                st.markdown(
                                    f"""
                                    <div class='metric-card' style='border-color: rgba(239, 68, 68, 0.4);'>
                                        <div class='metric-label' style='color:#EF4444;'>Position {lpos} ({row['WT']}) • {chrom}:{row['Genomic_Pos']}</div>
                                        <div class='metric-value'>{row['Max_Score']:.2f}</div>
                                        <div class='metric-sub'>Peak {metric_name} | {row['Top_Mod']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                with tab_dist:
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        fig_dist = create_score_distribution_plot(tidy_df, is_phred=is_phred)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    with col_d2:
                        fig_sub = create_substitution_matrix_plot(tidy_df, is_phred=is_phred)
                        st.plotly_chart(fig_sub, use_container_width=True)

                with tab_table:
                    st.markdown("#### 📋 Variant Catalog with Deep-Links to DeepMind Atlas")
                    st.caption("Each variant can be directly explored inside Google DeepMind's multi-omic interactive viewer with one click.")

                    col_f1, col_f2 = st.columns([0.4, 0.6])
                    with col_f1:
                        min_phred = st.slider(
                            "Filter by Minimum Phred Score:",
                            min_value=0.0,
                            max_value=float(tidy_df["avi_phred"].max()) if not tidy_df.empty else 50.0,
                            value=5.0,
                            step=1.0,
                        )
                    with col_f2:
                        search_var = st.text_input(
                            "Search variant or position (e.g. 'T>A' or '5225730'):",
                            placeholder="Search...",
                        )

                    display_table = tidy_df.copy()
                    if "avi_phred" in display_table.columns:
                        display_table = display_table[display_table["avi_phred"] >= min_phred]
                    if search_var:
                        display_table = display_table[
                            display_table["variant"].str.contains(search_var, case=False)
                            | display_table["position"].astype(str).str.contains(search_var)
                            | display_table["mutation"].str.contains(search_var, case=False)
                        ]

                    st.dataframe(
                        display_table,
                        column_config={
                            "atlas_url": st.column_config.LinkColumn(
                                "AlphaGenome Atlas",
                                display_text="🔗 Explore in Atlas",
                                help="Opens the official AlphaGenome Atlas variant page on DeepMind's web portal",
                            ),
                            "avi_phred": st.column_config.NumberColumn(
                                "AVI Phred",
                                format="%.2f",
                            ),
                            "score": st.column_config.NumberColumn(
                                "Modal Score",
                                format="%.3f",
                            ),
                            "top_percentile": st.column_config.NumberColumn(
                                "Genome Top %",
                                format="%.3f%%",
                            ),
                        },
                        use_container_width=True,
                        height=380,
                    )

                with tab_export:
                    st.markdown("#### 📥 Export Results in Standard Bioinformatics Formats")
                    col_ex1, col_ex2 = st.columns(2)
                    with col_ex1:
                        csv_wide = wide_df.to_csv()
                        st.download_button(
                            label="📄 Download 2D Heatmap Matrix (Wide CSV)",
                            data=csv_wide,
                            file_name=f"alphagenome_wide_{chrom}_{start}_{end}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with col_ex2:
                        csv_tidy = tidy_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Full Tidy Dataset (Long CSV)",
                            data=csv_tidy,
                            file_name=f"alphagenome_tidy_{chrom}_{start}_{end}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                with tab_guide:
                    st.markdown(
                        r"""
                        ### 🧬 Google DeepMind AlphaGenome Atlas & EvoScan

                        #### 1. AlphaGenome Foundation Model Architecture
                        - **1-Megabase (1 Mb) Context Window:** Captures long-range chromatin interactions and distal regulatory elements (enhancers, silencers, CTCF insulators).
                        - **Single-Base Pair Resolution (1 bp):** Predicts across 9,440 parallel biological tracks spanning 18 functional modalities (splicing, RNA-seq, DNase I, ATAC-seq, ChIP-TF, histone marks, 3D contacts).
                        - **Phred Score Calibration (0–70):**
                          $$\text{Phred} = -10 \log_{10}(1 - Q)$$
                          where $Q$ is the empirical genome-wide quantile score calculated across all ~9 billion human SNVs.
                          - $\ge 10$: Top 10% (moderate impact)
                          - $\ge 20$: Top 1% (high impact)
                          - $\ge 30$: Top 0.1% (extremely disruptive / pathogenic tier)

                        #### 2. How to Obtain a Personal API Key
                        1. Visit the official portal: [**Google DeepMind AlphaGenome API**](https://deepmind.google.com/science/alphagenome/api).
                        2. Sign in with your Google account.
                        3. Accept the **Research Use Only (RUO)** terms of service.
                        4. Copy your API key and paste it into EvoScan's left sidebar, or save it to `~/.env` as `ALPHAGENOME_API_KEY=your_key`.
                        5. Consult [ALPHAGENOME_API_GUIDE.md](ALPHAGENOME_API_GUIDE.md) in this repository for full details.
                        """
                    )

    # =========================================================================
    # BRANCH 2: NUCLEOTIDE TRANSFORMER & LOCAL HEURISTIC ENGINE
    # =========================================================================
    else:
        st.sidebar.markdown("### 🧪 **Local Foundation Model Settings**")

        model_choices = list(AVAILABLE_MODELS.keys())
        model_labels = [AVAILABLE_MODELS[m]["name"] for m in model_choices]
        selected_model_idx = st.sidebar.selectbox(
            "Select Model",
            range(len(model_choices)),
            format_func=lambda i: model_labels[i],
            index=0,
            help="Choose the Transformers model to use for logit extraction.",
        )
        selected_model_id = model_choices[selected_model_idx]
        st.sidebar.caption(f"ℹ️ {AVAILABLE_MODELS[selected_model_id]['description']}")

        scoring_mode = st.sidebar.radio(
            "Scoring Mode",
            ["marginal", "masked"],
            format_func=lambda m: (
                "⚡ Fast Marginal (1-Pass Rapid)"
                if m == "marginal"
                else "🔬 Masked Marginal Scan (Iterative MLM)"
            ),
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 **Visualization Settings**")

        selected_colorscale = st.sidebar.selectbox(
            "Heatmap Color Palette",
            list(COLORSCALES.keys()),
            index=0,
        )
        actual_colorscale = COLORSCALES[selected_colorscale]

        show_overlay_values = st.sidebar.checkbox(
            "Overlay numeric scores on cells",
            value=False,
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📚 **Biological Presets (Sequences)**")
        presets = load_sample_presets()
        preset_names = ["-- Select an example --"] + list(presets.keys())
        selected_preset = st.sidebar.selectbox(
            "Load Example Sequence", preset_names, index=1 if presets else 0
        )

        st.sidebar.markdown("---")
        import torch

        device_str = "🚀 CUDA (GPU)" if torch.cuda.is_available() else "💻 CPU Host"
        st.sidebar.markdown(f"**Active Hardware:** `{device_str}`")

        # Main Header Local Mode
        col_h1, col_h2 = st.columns([0.8, 0.2])
        with col_h1:
            st.markdown("<div class='hero-title'>🧬 EvoScan (Local Engine)</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='hero-subtitle'>"
                "Zero-shot saturation mutagenesis across arbitrary and synthetic DNA sequences "
                "powered by genomic foundation models (Nucleotide Transformer)."
                "</div>",
                unsafe_allow_html=True,
            )
        with col_h2:
            st.markdown(
                """
                <div style='text-align: right; padding-top: 0.5rem;'>
                    <span class='badge-pill badge-blue'>Nucleotide Transformer</span>
                    <span class='badge-pill badge-purple'>Zero-Shot DMS</span>
                    <span class='badge-pill badge-green'>Streamlit + Plotly</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Sequence Input Section
        st.markdown("### 1. DNA Sequence Input")

        default_seq = ""
        preset_desc = ""
        if selected_preset != "-- Select an example --" and selected_preset in presets:
            default_seq = presets[selected_preset]["sequence"]
            preset_desc = (
                f"**{selected_preset}** ({presets[selected_preset]['organism']} - "
                f"{presets[selected_preset]['type']}): {presets[selected_preset]['description']}"
            )

        col_in1, col_in2 = st.columns([0.7, 0.3])

        with col_in1:
            raw_seq_input = st.text_area(
                "Enter DNA sequence (plain text or FASTA format with '>' header):",
                value=default_seq,
                height=130,
                placeholder="e.g. >Promoter\nGGGCGGGACGGGGGCGGGGCGGGCGCTATAAAAGGCGGAGCTTG",
            )
            if preset_desc:
                st.info(preset_desc, icon="💡")

        with col_in2:
            uploaded_file = st.file_uploader(
                "Or upload FASTA file (.fasta, .fa, .txt):",
                type=["fasta", "fa", "txt", "fna"],
            )
            if uploaded_file is not None:
                file_content = uploaded_file.getvalue().decode("utf-8")
                fasta_records = parse_fasta(file_content)
                if fasta_records:
                    first_key = list(fasta_records.keys())[0]
                    raw_seq_input = fasta_records[first_key]
                    st.success(f"Loaded FASTA record: `{first_key}` ({len(raw_seq_input)} bp)")

        cleaned_seq, is_valid, val_msg, warnings = clean_and_validate_dna(raw_seq_input)

        for w in warnings:
            st.warning(w, icon="⚠️")

        if not is_valid and raw_seq_input.strip():
            st.error(val_msg, icon="❌")

        if cleaned_seq and is_valid:
            stats = compute_sequence_stats(cleaned_seq)
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Sequence Length</div>
                        <div class='metric-value'>{stats['length']} bp</div>
                        <div class='metric-sub'>{math_chunks(stats['length'])} 6-mer tokens</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s2:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>GC Content</div>
                        <div class='metric-value'>{stats['gc_content_pct']}%</div>
                        <div class='metric-sub'>A+T: {stats['counts']['A'] + stats['counts']['T']} | G+C: {stats['counts']['G'] + stats['counts']['C']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s3:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>CpG Dinucleotides</div>
                        <div class='metric-value'>{stats['cpg_count']} sites</div>
                        <div class='metric-sub'>Obs/Exp Ratio: {stats['cpg_oe_ratio']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s4:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Purine/Pyrimidine Ratio</div>
                        <div class='metric-value'>{stats['purine_pyrimidine_ratio']}</div>
                        <div class='metric-sub'>Pu (A+G) / Py (C+T)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        run_col1, run_col2 = st.columns([0.35, 0.65])
        with run_col1:
            run_analysis = st.button(
                "🧬 Run Zero-Shot DMS (Analyze Mutations)",
                type="primary",
                use_container_width=True,
                disabled=not (cleaned_seq and is_valid),
            )

        if run_analysis and cleaned_seq and is_valid:
            with st.spinner("Initializing genomic model and extracting logits..."):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(p: float, msg: str):
                    progress_bar.progress(p)
                    status_text.markdown(f"*{msg}*")

                t0 = time.time()
                engine = get_inference_engine(selected_model_id)

                try:
                    score_matrix, meta = engine.score_sequence(
                        sequence=cleaned_seq,
                        mode=scoring_mode,
                        progress_callback=update_progress,
                    )
                    elapsed = time.time() - t0

                    progress_bar.empty()
                    status_text.empty()

                    wide_df, tidy_df = format_results_to_dataframe(
                        score_matrix=score_matrix, sequence=cleaned_seq
                    )

                    st.session_state["results"] = {
                        "wide_df": wide_df,
                        "tidy_df": tidy_df,
                        "score_matrix": score_matrix,
                        "sequence": cleaned_seq,
                        "stats": stats,
                        "meta": meta,
                        "elapsed": elapsed,
                        "model_id": selected_model_id,
                    }
                    st.success(
                        f"✅ Deep Mutational Scanning completed in **{elapsed:.2f} s** "
                        f"across {len(cleaned_seq)} positions ({len(cleaned_seq) * 4} variants computed)!"
                    )
                except Exception as e:
                    st.error(f"Error during model inference: {str(e)}")
                    return

        if "results" in st.session_state and st.session_state["results"]["sequence"] == cleaned_seq:
            res = st.session_state["results"]
            wide_df = res["wide_df"]
            tidy_df = res["tidy_df"]
            seq = res["sequence"]

            st.markdown("---")
            st.markdown("### 2. Interactive Results & Deep Mutational Scanning Heatmap")

            tab_heat, tab_profile, tab_dist, tab_table, tab_export, tab_guide = st.tabs(
                [
                    "📊 Saturation Heatmap",
                    "📈 Positional Vulnerability Profile",
                    "🔄 Substitution Matrix & Distribution",
                    "📋 Variant Table & Hotspot Filter",
                    "📥 Data Export & Download",
                    "📖 Methodological Guide",
                ]
            )

            with tab_heat:
                fig_heat = create_saturation_heatmap(
                    wide_df=wide_df,
                    sequence=seq,
                    colorscale=actual_colorscale,
                    show_values=show_overlay_values,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            with tab_profile:
                fig_prof = create_position_sensitivity_plot(tidy_df=tidy_df, sequence=seq)
                st.plotly_chart(fig_prof, use_container_width=True)

                mutants = tidy_df[~tidy_df["Is_WildType"]]
                top_vulnerable = (
                    mutants.groupby("Position")
                    .agg(
                        WT=("WT_Base", "first"),
                        Max_Impact=("Delta_Score_LLR", "min"),
                    )
                    .sort_values("Max_Impact", ascending=True)
                    .head(3)
                )

                st.markdown("#### 🚨 Top 3 Hyper-Sensitive Regulatory Loci:")
                cols_top = st.columns(3)
                for idx, (pos, row) in enumerate(top_vulnerable.iterrows()):
                    with cols_top[idx]:
                        st.markdown(
                            f"""
                            <div class='metric-card' style='border-color: rgba(239, 68, 68, 0.4);'>
                                <div class='metric-label' style='color:#EF4444;'>Position {pos} ({row['WT']})</div>
                                <div class='metric-value'>{row['Max_Impact']:.3f}</div>
                                <div class='metric-sub'>Peak vulnerability ΔLLR</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with tab_dist:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_dist = create_score_distribution_plot(tidy_df)
                    st.plotly_chart(fig_dist, use_container_width=True)
                with col_d2:
                    fig_sub = create_substitution_matrix_plot(tidy_df)
                    st.plotly_chart(fig_sub, use_container_width=True)

            with tab_table:
                col_f1, col_f2 = st.columns([0.4, 0.6])
                with col_f1:
                    effect_filter = st.multiselect(
                        "Filter by Functional Effect:",
                        options=list(tidy_df["Effect"].unique()),
                        default=[
                            "Highly Disruptive / Deleterious",
                            "Moderately Deleterious",
                        ],
                    )
                with col_f2:
                    search_mut = st.text_input(
                        "Search specific mutation (e.g. 'A10G' or position '15'):",
                        placeholder="Search...",
                    )

                filtered_df = tidy_df.copy()
                if effect_filter:
                    filtered_df = filtered_df[filtered_df["Effect"].isin(effect_filter)]
                if search_mut:
                    filtered_df = filtered_df[
                        filtered_df["Mutation"].str.contains(search_mut, case=False)
                        | filtered_df["Position"].astype(str).str.contains(search_mut)
                    ]

                st.dataframe(
                    filtered_df.sort_values("Delta_Score_LLR", ascending=True),
                    use_container_width=True,
                    height=340,
                )

            with tab_export:
                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    csv_wide = wide_df.to_csv()
                    st.download_button(
                        label="📄 Download 2D Heatmap Matrix (Wide CSV)",
                        data=csv_wide,
                        file_name="evoscan_mutation_matrix_wide.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_ex2:
                    csv_tidy = tidy_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Full Tidy Dataset (Long CSV)",
                        data=csv_tidy,
                        file_name="evoscan_mutation_scores_long.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            with tab_guide:
                st.markdown(
                    r"""
                    ### 🧬 EvoScan Theoretical Foundations & Methodology

                    #### 1. In Silico Saturation Mutagenesis
                    Experimental **Deep Mutational Scanning (DMS)** systematically synthesizes and functionally assays all possible single-nucleotide variants. 
                    `EvoScan` performs this process *in silico* in **zero-shot** mode, leveraging biological representations learned by genomic foundation models.

                    #### 2. Mathematical Formulation of Mutation Scores ($\Delta\text{LLR}$)
                    $$\Delta \text{LLR}(j, m) = \log P_\mathcal{M}(x_j = m \mid \mathbf{x}_{\setminus j}) - \log P_\mathcal{M}(x_j = x_j^{\text{WT}} \mid \mathbf{x}_{\setminus j})$$
                    - **$m = x_j^{\text{WT}}$:** $\Delta \text{LLR} = 0.0$ (neutral baseline reference).
                    - **$\Delta \text{LLR} \ll 0$:** Variant strongly disfavored by natural genomic distribution (deleterious / disruptive effect).
                    - **$\Delta \text{LLR} \approx 0$:** Neutral or evolutionarily tolerated variant.
                    - **$\Delta \text{LLR} > 0$:** Variant enriched or favored in local sequence context.
                    """
                )


if __name__ == "__main__":
    main()
