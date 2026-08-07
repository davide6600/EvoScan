"""
EvoScan: Zero-Shot DNA Mutation Map Visualizer (Saturation Mutagenesis Heatmap)
Powered by Genomic Foundation Models (Nucleotide Transformer), PyTorch & Streamlit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any

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

/* DNA Monospace Box */
.dna-box {
    font-family: 'Fira Code', monospace;
    letter-spacing: 0.1em;
    word-break: break-all;
    background: #0F172A;
    border: 1px solid #334155;
    padding: 0.75rem;
    border-radius: 8px;
    color: #38BDF8;
    font-size: 0.85rem;
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
    """Loads biological preset sequences from JSON."""
    preset_path = Path(__file__).parent / "sample_data" / "sequences.json"
    if preset_path.exists():
        with open(preset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=False)
def get_inference_engine(model_id: str) -> EvoScanEngine:
    """Cached loader for the neural genomic model engine."""
    engine = EvoScanEngine(model_id=model_id)
    engine.load_model()
    return engine


def main():
    # --- Sidebar Configuration ---
    st.sidebar.markdown("## 🧬 **EvoScan Config**")
    st.sidebar.markdown("Genomic foundation model & visualization settings.")

    # Model Selection
    model_choices = list(AVAILABLE_MODELS.keys())
    model_labels = [AVAILABLE_MODELS[m]["name"] for m in model_choices]
    selected_model_idx = st.sidebar.selectbox(
        "Genomic Model (Foundation Model)",
        range(len(model_choices)),
        format_func=lambda i: model_labels[i],
        index=0,
        help="Select the Transformers genomic foundation model to use for logit scoring.",
    )
    selected_model_id = model_choices[selected_model_idx]

    # Display short description
    st.sidebar.caption(f"ℹ️ {AVAILABLE_MODELS[selected_model_id]['description']}")

    # Scoring Mode
    scoring_mode = st.sidebar.radio(
        "Scoring Mode",
        ["marginal", "masked"],
        format_func=lambda m: (
            "⚡ Fast Marginal (1-Pass Rapid)"
            if m == "marginal"
            else "🔬 Masked Marginal Scan (Rigorous MLM)"
        ),
        help="Fast Marginal computes all mutation scores in a single forward pass; Masked performs iterative per-k-mer MLM masking.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 **Visualization Options**")

    selected_colorscale = st.sidebar.selectbox(
        "Heatmap Color Palette",
        list(COLORSCALES.keys()),
        index=0,
    )
    actual_colorscale = COLORSCALES[selected_colorscale]

    show_overlay_values = st.sidebar.checkbox(
        "Show numeric scores on cells",
        value=False,
        help="Display numeric ΔLLR values directly inside each heatmap cell.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 **Biological Presets**")
    presets = load_sample_presets()
    preset_names = ["-- Select an example --"] + list(presets.keys())
    selected_preset = st.sidebar.selectbox(
        "Load Reference Sequence", preset_names, index=1 if presets else 0
    )

    # Device Status Indicator
    st.sidebar.markdown("---")
    import torch

    device_str = "🚀 CUDA (GPU)" if torch.cuda.is_available() else "💻 CPU Host"
    st.sidebar.markdown(f"**Active Hardware:** `{device_str}`")
    st.sidebar.markdown(
        "<div style='font-size:0.75rem; color:#64748B;'>EvoScan v1.0.0 • Open-Source MIT</div>",
        unsafe_allow_html=True,
    )

    # --- Main Header ---
    col_h1, col_h2 = st.columns([0.8, 0.2])
    with col_h1:
        st.markdown("<div class='hero-title'>🧬 EvoScan</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-subtitle'>"
            "Zero-Shot DNA Saturation Mutagenesis Heatmap & Deep Mutational Scanning Visualizer "
            "powered by Genomic Foundation Models."
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

    # --- Sequence Input Section ---
    st.markdown("### 1. DNA Sequence Input")

    # Handle preset population
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
            "Enter DNA sequence (raw text or FASTA format with '>' header):",
            value=default_seq,
            height=130,
            placeholder="e.g. >Gene_Promoter\nGGGCGGGACGGGGGCGGGGCGGGCGCTATAAAAGGCGGAGCTTG",
            help="Supports nucleotide bases A, C, G, T. Spaces, numbers, and FASTA headers are automatically stripped.",
        )
        if preset_desc:
            st.info(preset_desc, icon="💡")

    with col_in2:
        uploaded_file = st.file_uploader(
            "Or upload FASTA file (.fasta, .fa, .txt):",
            type=["fasta", "fa", "txt", "fna"],
            help="Upload a FASTA file from your computer.",
        )
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            fasta_records = parse_fasta(file_content)
            if fasta_records:
                first_key = list(fasta_records.keys())[0]
                raw_seq_input = fasta_records[first_key]
                st.success(f"Loaded FASTA record: `{first_key}` ({len(raw_seq_input)} bp)")

    # Validate input sequence
    cleaned_seq, is_valid, val_msg, warnings = clean_and_validate_dna(raw_seq_input)

    for w in warnings:
        st.warning(w, icon="⚠️")

    if not is_valid and raw_seq_input.strip():
        st.error(val_msg, icon="❌")

    # Sequence stats live bar
    if cleaned_seq and is_valid:
        stats = compute_sequence_stats(cleaned_seq)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Sequence Length</div>
                    <div class='metric-value'>{stats['length']} bp</div>
                    <div class='metric-sub'>{math_chunks(stats['length'])} 6-mer k-mers</div>
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

    # --- Analysis Trigger ---
    run_col1, run_col2 = st.columns([0.35, 0.65])
    with run_col1:
        run_analysis = st.button(
            "🧬 Run Zero-Shot DMS (Analyze Mutations)",
            type="primary",
            use_container_width=True,
            disabled=not (cleaned_seq and is_valid),
        )

    # Execution State
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
                    f"✅ Deep Mutational Scanning completed successfully in **{elapsed:.2f} s** "
                    f"across {len(cleaned_seq)} nucleotide positions ({len(cleaned_seq) * 4} variants computed)!"
                )
            except Exception as e:
                st.error(f"Error during model inference: {str(e)}")
                return

    # --- Display Results ---
    if "results" in st.session_state and st.session_state["results"]["sequence"] == cleaned_seq:
        res = st.session_state["results"]
        wide_df = res["wide_df"]
        tidy_df = res["tidy_df"]
        seq = res["sequence"]

        st.markdown("---")
        st.markdown("### 2. Interactive Results & Deep Mutational Scanning Heatmaps")

        # Tabs Layout
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
            st.markdown(
                r"The 2D heatmap displays the estimated functional impact ($\Delta\text{LLR}$) for every single nucleotide substitution. "
                r"White markers indicate the Wild-Type reference sequence ($\Delta\text{LLR} = 0.0$)."
            )
            fig_heat = create_saturation_heatmap(
                wide_df=wide_df,
                sequence=seq,
                colorscale=actual_colorscale,
                show_values=show_overlay_values,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.caption(
                "💡 **Reading Guide:** "
                "Strongly negative values (red) indicate deleterious/disruptive mutations altering genomic context likelihood. "
                "Values near zero denote well-tolerated variants. Zoom, pan, and hover over any cell for detailed metrics."
            )

        with tab_profile:
            st.markdown(
                "This plot highlights **hyper-vulnerable loci**: positions where any nucleotide substitution "
                "causes a sharp decrease in log-likelihood (e.g., invariant core bases in promoters or splice sites)."
            )
            fig_prof = create_position_sensitivity_plot(tidy_df=tidy_df, sequence=seq)
            st.plotly_chart(fig_prof, use_container_width=True)

            # Highlight Top 3 Most Sensitive Positions
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
            st.markdown("#### Explore & Filter Predicted Variants")
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
            st.markdown("#### 📥 Export Results in Open Data Formats")
            st.markdown(
                "Download full Deep Mutational Scanning results as CSV files for downstream pipelines in Python, R/Bioconductor, or command-line bioinformatics workflows."
            )

            col_ex1, col_ex2 = st.columns(2)

            with col_ex1:
                # Wide format CSV
                csv_wide = wide_df.to_csv()
                st.download_button(
                    label="📄 Download 2D Heatmap Matrix (Wide CSV)",
                    data=csv_wide,
                    file_name="evoscan_mutation_matrix_wide.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("4 x L matrix format (rows: A, C, G, T; columns: 1-A, 2-C...).")

            with col_ex2:
                # Tidy long format CSV
                csv_tidy = tidy_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Full Tidy Dataset (Long CSV)",
                    data=csv_tidy,
                    file_name="evoscan_mutation_scores_long.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption(
                    "Tidy tabular format with columns: Position, WT_Base, Mutant_Base, Mutation, Score, Effect."
                )

        with tab_guide:
            st.markdown(
                r"""
                ### 🧬 Theoretical Foundations & EvoScan Methodology

                #### 1. What is In Silico Saturation Mutagenesis?
                Experimental **Deep Mutational Scanning (DMS)** systematically synthesizes and functionally assays all possible single-nucleotide variants across a DNA target sequence. 
                `EvoScan` conducts this process *in silico* in **zero-shot** mode, leveraging the deep evolutionary and biological representations learned by Genomic Foundation Models (*Genomic Foundation Models*).

                #### 2. Mathematical Formulation of Mutation Scores ($\Delta\text{LLR}$)
                Given a genomic language model $\mathcal{M}$ and a wild-type sequence $\mathbf{x} = (x_1, x_2, \dots, x_L)$, for each position $j \in \{1, \dots, L\}$ and each mutant nucleotide $m \in \{A, C, G, T\}$:

                $$\Delta \text{LLR}(j, m) = \log P_\mathcal{M}(x_j = m \mid \mathbf{x}_{\setminus j}) - \log P_\mathcal{M}(x_j = x_j^{\text{WT}} \mid \mathbf{x}_{\setminus j})$$

                - **If $m = x_j^{\text{WT}}$:** $\Delta \text{LLR} = 0.0$ (neutral baseline reference).
                - **If $\Delta \text{LLR} \ll 0$:** The mutation is strongly disfavored by natural genomic distributions (deleterious / disruptive effect).
                - **If $\Delta \text{LLR} \approx 0$:** The variant is neutral or evolutionarily tolerated.
                - **If $\Delta \text{LLR} > 0$:** The variant is enriched or preferred in the local sequence context.

                #### 3. Handling 6-mer Tokenization
                Models in the `Nucleotide Transformer` family use a vocabulary of $4^6 = 4096$ non-overlapping 6-mers. 
                `EvoScan` maps each nucleotide $j$ to the containing k-mer token $T_k = \mathbf{x}[6k : 6k+6]$, substitutes the base at offset $(j \bmod 6)$ to construct the mutant k-mer $T_k^{(m)}$, and extracts the exact logits from the model output tensor without shape mismatch errors.
                """
            )


def math_chunks(length: int) -> int:
    return (length + 5) // 6


if __name__ == "__main__":
    main()
