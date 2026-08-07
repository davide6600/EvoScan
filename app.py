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
    st.sidebar.markdown("Parametri del modello fondazionale e visualizzazione.")

    # Model Selection
    model_choices = list(AVAILABLE_MODELS.keys())
    model_labels = [AVAILABLE_MODELS[m]["name"] for m in model_choices]
    selected_model_idx = st.sidebar.selectbox(
        "Modello Genomico (Foundation Model)",
        range(len(model_choices)),
        format_func=lambda i: model_labels[i],
        index=0,
        help="Seleziona il modello genomico Transformers da usare per il calcolo dei logit.",
    )
    selected_model_id = model_choices[selected_model_idx]

    # Display short description
    st.sidebar.caption(f"ℹ️ {AVAILABLE_MODELS[selected_model_id]['description']}")

    # Scoring Mode
    scoring_mode = st.sidebar.radio(
        "Modalità di Scoring",
        ["marginal", "masked"],
        format_func=lambda m: (
            "⚡ Fast Marginal (1-Pass Rapido)"
            if m == "marginal"
            else "🔬 Masked Marginal Scan (MLM Rigoroso)"
        ),
        help="Fast Marginal calcola tutti i punteggi in un singolo forward pass; Masked effettua la mascheratura iterativa per ogni k-mero.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 **Opzioni Grafiche**")

    selected_colorscale = st.sidebar.selectbox(
        "Palette Cromatica Heatmap",
        list(COLORSCALES.keys()),
        index=0,
    )
    actual_colorscale = COLORSCALES[selected_colorscale]

    show_overlay_values = st.sidebar.checkbox(
        "Mostra valori numerici nelle celle",
        value=False,
        help="Attiva la visualizzazione diretta dei punteggi numerici all'interno delle caselle della mappa.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 **Esempi Biologici Predefiniti**")
    presets = load_sample_presets()
    preset_names = ["-- Seleziona un esempio --"] + list(presets.keys())
    selected_preset = st.sidebar.selectbox(
        "Carica Sequenza di Riferimento", preset_names, index=1 if presets else 0
    )

    # Device Status Indicator
    st.sidebar.markdown("---")
    import torch

    device_str = "🚀 CUDA (GPU)" if torch.cuda.is_available() else "💻 CPU Host"
    st.sidebar.markdown(f"**Hardware attivo:** `{device_str}`")
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
    st.markdown("### 1. Inserimento Sequenza di DNA")

    # Handle preset population
    default_seq = ""
    preset_desc = ""
    if selected_preset != "-- Seleziona un esempio --" and selected_preset in presets:
        default_seq = presets[selected_preset]["sequence"]
        preset_desc = (
            f"**{selected_preset}** ({presets[selected_preset]['organism']} - "
            f"{presets[selected_preset]['type']}): {presets[selected_preset]['description']}"
        )

    col_in1, col_in2 = st.columns([0.7, 0.3])

    with col_in1:
        raw_seq_input = st.text_area(
            "Inserisci sequenza di DNA (formato testo grezzo o FASTA con header '>'):",
            value=default_seq,
            height=130,
            placeholder="es. >Gene_Promoter\nGGGCGGGACGGGGGCGGGGCGGGCGCTATAAAAGGCGGAGCTTG",
            help="Supporta sequenze di nucleotidi A, C, G, T. Gli spazi, i numeri e gli header FASTA vengono puliti automaticamente.",
        )
        if preset_desc:
            st.info(preset_desc, icon="💡")

    with col_in2:
        uploaded_file = st.file_uploader(
            "Oppure carica file FASTA (.fasta, .fa, .txt):",
            type=["fasta", "fa", "txt", "fna"],
            help="Carica un file FASTA dal tuo computer.",
        )
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            fasta_records = parse_fasta(file_content)
            if fasta_records:
                first_key = list(fasta_records.keys())[0]
                raw_seq_input = fasta_records[first_key]
                st.success(f"Caricato record FASTA: `{first_key}` ({len(raw_seq_input)} bp)")

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
                    <div class='metric-label'>Lunghezza Sequenza</div>
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
                    <div class='metric-label'>Contenuto GC</div>
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
                    <div class='metric-label'>Dinucleotidi CpG</div>
                    <div class='metric-value'>{stats['cpg_count']} siti</div>
                    <div class='metric-sub'>Obs/Exp Ratio: {stats['cpg_oe_ratio']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_s4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Rapporto Purine/Pirimidine</div>
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
            "🧬 Analizza Mutazioni (Run Zero-Shot DMS)",
            type="primary",
            use_container_width=True,
            disabled=not (cleaned_seq and is_valid),
        )

    # Execution State
    if run_analysis and cleaned_seq and is_valid:
        with st.spinner("Inizializzazione del modello genomico ed estrazione dei logits..."):
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
                    f"✅ Analisi di Deep Mutational Scanning completata con successo in **{elapsed:.2f} s** "
                    f"su {len(cleaned_seq)} posizioni nucleotidiche ({len(cleaned_seq) * 4} varianti calcolate)!"
                )
            except Exception as e:
                st.error(f"Errore durante l'inferenza del modello: {str(e)}")
                return

    # --- Display Results ---
    if "results" in st.session_state and st.session_state["results"]["sequence"] == cleaned_seq:
        res = st.session_state["results"]
        wide_df = res["wide_df"]
        tidy_df = res["tidy_df"]
        seq = res["sequence"]

        st.markdown("---")
        st.markdown("### 2. Risultati e Mappe di Calore Interattive")

        # Tabs Layout
        tab_heat, tab_profile, tab_dist, tab_table, tab_export, tab_guide = st.tabs(
            [
                "📊 Heatmap di Saturazione",
                "📈 Profilo di Sensibilità",
                "🔄 Matrice di Sostituzione & Distribuzione",
                "📋 Tabella Varianti & Filtro Hotspot",
                "📥 Esportazione & Download",
                "📖 Guida Metodologica",
            ]
        )

        with tab_heat:
            st.markdown(
                r"La mappa di calore 2D mostra l'impatto stimato ($\Delta\text{LLR}$) di ogni singola sostituzione nucleotidica. "
                r"I punti bianchi indicano la sequenza Wild-Type di riferimento ($\Delta\text{LLR} = 0.0$)."
            )
            fig_heat = create_saturation_heatmap(
                wide_df=wide_df,
                sequence=seq,
                colorscale=actual_colorscale,
                show_values=show_overlay_values,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.caption(
                "💡 **Guida alla lettura:** "
                "Valori fortemente negativi (rosso) indicano mutazioni deleterie/distruttive che alterano la probabilità del contesto genomico. "
                "Valori prossimi a zero indicano varianti ben tollerate. Puoi effettuare zoom, pan e hover su ogni singola cella."
            )

        with tab_profile:
            st.markdown(
                "Questo grafico evidenzia i **locus iper-vulnerabili**: posizioni in cui qualsiasi sostituzione "
                "provoca un crollo della log-likelihood (es. nucleotidi invarianti in promotori o siti di splicing)."
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

            st.markdown("#### 🚨 Top 3 Locus ad Altissima Sensibilità Funzionale:")
            cols_top = st.columns(3)
            for idx, (pos, row) in enumerate(top_vulnerable.iterrows()):
                with cols_top[idx]:
                    st.markdown(
                        f"""
                        <div class='metric-card' style='border-color: rgba(239, 68, 68, 0.4);'>
                            <div class='metric-label' style='color:#EF4444;'>Posizione {pos} ({row['WT']})</div>
                            <div class='metric-value'>{row['Max_Impact']:.3f}</div>
                            <div class='metric-sub'>Picco di vulnerabilità massima ΔLLR</div>
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
            st.markdown("#### Esplora e Filtra le Varianti Predette")
            col_f1, col_f2 = st.columns([0.4, 0.6])
            with col_f1:
                effect_filter = st.multiselect(
                    "Filtra per Classificazione Effetto:",
                    options=list(tidy_df["Effect"].unique()),
                    default=[
                        "Fortemente Distruttiva / Deleteria",
                        "Moderatamente Deleteria",
                    ],
                )
            with col_f2:
                search_mut = st.text_input(
                    "Cerca mutazione specifica (es. 'A10G' o posizione '15'):",
                    placeholder="Cerca...",
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
            st.markdown("#### 📥 Download Risultati in Formato Aperto")
            st.markdown(
                "Scarica i risultati completi del Deep Mutational Scanning in formato CSV per ulteriori analisi in Python, R o pipeline bioinformatiche."
            )

            col_ex1, col_ex2 = st.columns(2)

            with col_ex1:
                # Wide format CSV
                csv_wide = wide_df.to_csv()
                st.download_button(
                    label="📄 Scarica Matrice Heatmap 2D (Wide CSV)",
                    data=csv_wide,
                    file_name="evoscan_mutation_matrix_wide.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("Formato matrice 4 x L (righe: A,C,G,T; colonne: 1-A, 2-C...).")

            with col_ex2:
                # Tidy long format CSV
                csv_tidy = tidy_df.to_csv(index=False)
                st.download_button(
                    label="📊 Scarica Dataset Tidy Completo (Long CSV)",
                    data=csv_tidy,
                    file_name="evoscan_mutation_scores_long.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption(
                    "Formato tabellare tidy con colonne: Position, WT_Base, Mutant_Base, Mutation, Score, Effect."
                )

        with tab_guide:
            st.markdown(
                r"""
                ### 🧬 Fondamenti Teorici e Metodologia di EvoScan

                #### 1. Che cos'è la Saturation Mutagenesis In Silico?
                Il **Deep Mutational Scanning (DMS)** sperimentale consiste nel sintetizzare e saggiare funzionalmente tutte le possibili mutazioni a singolo nucleotide di una sequenza di DNA. 
                `EvoScan` esegue questa procedura *in silico* in modalità **zero-shot**, sfruttando la conoscenza biologica ed evolutiva appresa dai Modelli Fondazionali Genomici (*Genomic Foundation Models*).

                #### 2. Definizione Matematica dello Score di Mutazione ($\Delta\text{LLR}$)
                Dato un modello linguistico genomico $\mathcal{M}$ e una sequenza wild-type $\mathbf{x} = (x_1, x_2, \dots, x_L)$, per ogni posizione $j \in \{1, \dots, L\}$ e ogni nucleotide mutato $m \in \{A, C, G, T\}$:

                $$\Delta \text{LLR}(j, m) = \log P_\mathcal{M}(x_j = m \mid \mathbf{x}_{\setminus j}) - \log P_\mathcal{M}(x_j = x_j^{\text{WT}} \mid \mathbf{x}_{\setminus j})$$

                - **Se $m = x_j^{\text{WT}}$:** $\Delta \text{LLR} = 0.0$ (riferimento neutro).
                - **Se $\Delta \text{LLR} \ll 0$:** La mutazione è altamente sfavorita dalla distribuzione probabilistica naturale (effetto distruttivo/deleterio).
                - **Se $\Delta \text{LLR} \approx 0$:** La mutazione è considerata neutra o tollerata dalla selezione evolutiva.
                - **Se $\Delta \text{LLR} > 0$:** La variante è arricchita o preferita nel contesto genomico locale.

                #### 3. Gestione della Tokenizzazione a 6-Meri
                I modelli della famiglia `Nucleotide Transformer` utilizzano un vocabolario di $4^6 = 4096$ k-meri non sovrapposti. 
                `EvoScan` mappa in modo esatto ciascun nucleotide $j$ al k-mero contenente $T_k = \mathbf{x}[6k : 6k+6]$, sostituisce il nucleotide all'offset $(j \bmod 6)$ generando il k-mero mutante $T_k^{(m)}$, ed estrae i corrispondenti logits dal tensore di output evitando ogni tipo di shape mismatch.
                """
            )


def math_chunks(length: int) -> int:
    return (length + 5) // 6


if __name__ == "__main__":
    main()
