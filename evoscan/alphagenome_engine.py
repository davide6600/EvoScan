"""
EvoScan AlphaGenome Engine: Cloud gRPC Integration with Google DeepMind AlphaGenome Atlas.
Provides real-time multi-omic saturation mutagenesis, AVI Phred calibration,
and 18-modality biological feature attribution for human genomic loci (GRCh38).
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import quote

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALPHABET = ["A", "C", "G", "T"]

# Atlas 18 Biological Feature Modalities & Display Labels
MODALITY_OPTIONS: Dict[str, Dict[str, str]] = {
    "AVI_SCORE": {
        "name": "🔥 Global AVI Impact (Calibrated Phred 0-70)",
        "description": "Composite AlphaGenome Variant Impact score calibrated on genome-wide SNV quantiles.",
        "category": "Global Impact",
    },
    "MERGED_SPLICING": {
        "name": "✂️ Splicing Disruption (Sites & Junctions)",
        "description": "Disruption of canonical 5'/3' splice sites, site usage shifts, and novel cryptic junctions.",
        "category": "Splicing",
    },
    "MAX_ABS_RNA_SEQ": {
        "name": "📊 RNA-seq Expression Impact (log2 Fold Change)",
        "description": "Quantitative disruption or elevation of steady-state transcript abundance.",
        "category": "Transcription",
    },
    "MAX_ABS_DNASE": {
        "name": "🔓 DNase I Chromatin Accessibility",
        "description": "Loss or gain of DNase I hypersensitive regulatory chromatin sites.",
        "category": "Chromatin",
    },
    "MAX_ABS_ATAC": {
        "name": "🧬 ATAC-seq Open Chromatin Accessibility",
        "description": "Transposase-accessible chromatin peak alterations across regulatory elements.",
        "category": "Chromatin",
    },
    "MAX_ABS_CHIP_TF": {
        "name": "🧲 Transcription Factor Binding (ChIP-TF)",
        "description": "Creation or destruction of transcription factor binding motifs (CTCF, GATA, CEBP, etc.).",
        "category": "Regulatory",
    },
    "MAX_ABS_CHIP_HISTONE": {
        "name": "🏷️ Histone Modifications (H3K27ac / H3K4me3)",
        "description": "Epigenetic chromatin mark remodeling at active enhancers and promoters.",
        "category": "Epigenetics",
    },
    "ALPHAMISSENSE": {
        "name": "💊 AlphaMissense Coding Impact",
        "description": "DeepMind AlphaMissense predicted pathogenicity for missense amino acid substitutions.",
        "category": "Protein Impact",
    },
    "CACTUS_241_WAY": {
        "name": "🐾 Cactus Multi-Species Mammalian Conservation",
        "description": "Phylogenetic sequence constraint across 241 mammalian genomes.",
        "category": "Conservation",
    },
    "PHASTCONS_470_WAY": {
        "name": "🐟 PhastCons 470-Vertebrate Conservation",
        "description": "Evolutionary conservation score across 470 vertebrate species.",
        "category": "Conservation",
    },
}

# Curated Biological & Clinical Genomic Presets
GENOMIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "HBB Promoter (Beta-Globin TATA & CACC Core)": {
        "region": "chr11:5225720-5225780",
        "chrom": "chr11",
        "start": 5225720,
        "end": 5225780,
        "gene": "HBB",
        "description": "Critical promoter controlling hemoglobin subunit beta expression. Mutations cause beta-thalassemia.",
    },
    "TP53 Exon 5 (Tumor Suppressor DNA-Binding Core)": {
        "region": "chr17:7676080-7676150",
        "chrom": "chr17",
        "start": 7676080,
        "end": 7676150,
        "gene": "TP53",
        "description": "Hyper-mutated DNA-binding hotspot in human carcinomas (Li-Fraumeni syndrome & somatic cancer).",
    },
    "BRCA1 Exon 11 Splice Donor Site": {
        "region": "chr17:43071060-43071130",
        "chrom": "chr17",
        "start": 43071060,
        "end": 43071130,
        "gene": "BRCA1",
        "description": "Canonical splice donor junction. Disruption causes aberrant exon skipping and hereditary breast/ovarian cancer.",
    },
    "CAPN3 Exon 10 Splice Donor (Limb-Girdle Muscular Dystrophy)": {
        "region": "chr15:42387790-42387850",
        "chrom": "chr15",
        "start": 42387790,
        "end": 42387850,
        "gene": "CAPN3",
        "description": "Pathogenic splice donor variant site (c.1332+1G>A / G>C) driving calpainopathy.",
    },
    "APOA1 TATA Promoter (Hypoalphalipoproteinemia)": {
        "region": "chr11:116837630-116837690",
        "chrom": "chr11",
        "start": 116837630,
        "end": 116837690,
        "gene": "APOA1",
        "description": "TATA box promoter controlling HDL cholesterol transcript abundance in liver and cardiac tissue.",
    },
    "BTK Promoter (Agammaglobulinemia SPI1/PU.1 Motif)": {
        "region": "chrX:101386210-101386270",
        "chrom": "chrX",
        "start": 101386210,
        "end": 101386270,
        "gene": "BTK",
        "description": "SPI1 regulatory binding motif driving Bruton tyrosine kinase transcription in B-cells.",
    },
}


def parse_genomic_region(region_str: str, max_window: int = 1000) -> Tuple[str, int, int]:
    """
    Parses and validates a genomic region string in 'chr:start-end' format.

    Args:
        region_str: String like 'chr11:5225720-5225780' or '11:5,225,720-5,225,780'.
        max_window: Maximum allowed window width (default 1000 bp).

    Returns:
        Tuple of (chrom, start_1_based, end_1_based).
    """
    cleaned = region_str.strip().replace(",", "")
    match = re.match(r"^(?:chr)?([0-9A-Za-z]+):(\d+)-(\d+)$", cleaned, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Formato regione non valido: '{region_str}'. "
            "Usa il formato 'chr:inizio-fine' (es. 'chr11:5225720-5225780')."
        )

    chrom_raw, start_str, end_str = match.groups()
    chrom = chrom_raw if chrom_raw.lower().startswith("chr") else f"chr{chrom_raw}"
    start = int(start_str)
    end = int(end_str)

    if start <= 0 or end <= 0:
        raise ValueError("Le coordinate genomiche devono essere numeri interi positivi (>0).")
    if end < start:
        raise ValueError(f"La coordinata di fine ({end}) non può essere minore di quella d'inizio ({start}).")

    width = end - start + 1
    if width > max_window:
        raise ValueError(
            f"Finestra genomica di {width:,} bp supera il limite interattivo consentito ({max_window:,} bp). "
            f"Riduci l'intervallo per garantire un'elaborazione in tempo reale (<2s)."
        )

    return chrom, start, end


def format_atlas_url(chrom: str, pos: int, ref: str, alt: str) -> str:
    """
    Constructs a clickable deep-link to the official Google DeepMind AlphaGenome Atlas.
    """
    var_str = f"{chrom}:{pos}:{ref}>{alt}"
    encoded_q = quote(var_str, safe="")
    return (
        f"https://deepmind.google.com/science/alphagenome/atlas"
        f"?q={encoded_q}&m=variant&lItems=avi,section:RNA_SEQ,section:DNASE,section:CHIP_TF"
    )


class AlphaGenomeEngine:
    """
    EvoScan Cloud Engine powered by Google DeepMind AlphaGenome Atlas gRPC Service.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Resolve API key priority: Explicit argument -> Environment variable -> ~/.env
        self.api_key = api_key or os.environ.get("ALPHAGENOME_API_KEY", "")

        if not self.api_key:
            # Check ~/.env
            home_env = os.path.expanduser("~/.env")
            if os.path.exists(home_env):
                try:
                    with open(home_env, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("ALPHAGENOME_API_KEY="):
                                self.api_key = line.split("=", 1)[1].strip().strip('"\'')
                                break
                except Exception:
                    pass

        self.client = None
        self.configured = bool(self.api_key and self.api_key.strip())
        self._init_error: Optional[str] = None

        if self.configured:
            self._initialize_client()

    def _initialize_client(self) -> bool:
        """Initializes the gRPC client via alphagenome.atlas.atlas."""
        try:
            from alphagenome.atlas import atlas
            self.client = atlas.create(self.api_key.strip())
            self._init_error = None
            return True
        except Exception as e:
            self._init_error = str(e)
            self.client = None
            logger.error(f"AlphaGenome client initialization failed: {e}")
            return False

    def validate_connection(self) -> Tuple[bool, str]:
        """
        Performs a lightweight probe query to verify API key and gRPC connectivity.
        """
        if not self.configured or not self.client:
            return False, "Chiave API AlphaGenome non configurata. Inserisci la tua API key personale."

        try:
            from alphagenome.data import genome
            test_var = genome.Variant("chr11", 5225488, "A", "T")
            res = self.client.query_variant(test_var, requested_scorers=["AVI_SCORE"])
            if res and "AVI_SCORE" in res:
                return True, "Connessione stabilita con successo ai server Google DeepMind AlphaGenome Atlas."
            return False, "Nessun dato restituito dal server AlphaGenome."
        except Exception as e:
            err_msg = str(e)
            if "UNAUTHENTICATED" in err_msg or "API_KEY_INVALID" in err_msg:
                return False, "Chiave API non valida o non autorizzata. Verifica la tua chiave su Google DeepMind."
            return False, f"Errore di connessione gRPC: {err_msg}"

    def score_genomic_interval(
        self,
        chrom: str,
        start_1_based: int,
        end_1_based: int,
        modality: str = "AVI_SCORE",
    ) -> Tuple[np.ndarray, str, pd.DataFrame, Dict[str, Any]]:
        """
        Queries precomputed dense saturation mutagenesis over a human genomic interval.

        Args:
            chrom: Chromosome string (e.g. 'chr11').
            start_1_based: 1-based start coordinate (inclusive).
            end_1_based: 1-based end coordinate (inclusive).
            modality: One of MODALITY_OPTIONS keys (e.g. 'AVI_SCORE', 'MERGED_SPLICING', etc.).

        Returns:
            Tuple of:
              - score_matrix: np.ndarray of shape (4, L) matching ALPHABET ['A', 'C', 'G', 'T'].
              - ref_sequence: Wild-type DNA sequence string of length L.
              - tidy_df: DataFrame with all SNVs, Phred scores, top modalities, and Atlas URLs.
              - metadata: Dictionary containing timing, coverage, and interval stats.
        """
        if not self.configured or not self.client:
            raise ValueError(
                "Impossibile eseguire la scansione: Chiave API AlphaGenome mancante. "
                "Configura la tua API key personale nella barra laterale."
            )

        from alphagenome.data import genome

        # Convert 1-based closed to 0-based half-open interval for the API
        start_0 = start_1_based - 1
        end_0 = end_1_based
        length = end_1_based - start_1_based + 1

        interval = genome.Interval(chromosome=chrom, start=start_0, end=end_0)

        # Query gRPC Atlas service for dense precomputed scores
        requested_scorers = ["AVI_SCORE", "AVI_SCORE_FEATURE_IMPORTANCE"]
        res = self.client.query_interval(
            interval,
            requested_scorers=requested_scorers,
            progress_bar=False,
        )

        if "AVI_SCORE" not in res:
            raise RuntimeError("La risposta del server AlphaGenome Atlas non contiene i punteggi 'AVI_SCORE'.")

        avi_adata = res["AVI_SCORE"]
        fi_adata = res.get("AVI_SCORE_FEATURE_IMPORTANCE")

        variants: List[genome.Variant] = avi_adata.obs["variant"].tolist()
        if not variants:
            raise RuntimeError(f"Nessuna variante annotata per l'intervallo {chrom}:{start_1_based}-{end_1_based}.")

        # Reconstruct wild-type reference sequence across positions
        ref_dict: Dict[int, str] = {}
        for var in variants:
            ref_dict[var.position] = var.reference_bases

        # Ensure all positions are covered
        missing_positions = [pos for pos in range(start_1_based, end_1_based + 1) if pos not in ref_dict]
        if missing_positions:
            logger.warning(f"Posizioni mancanti nell'intervallo: {missing_positions[:5]}")

        ref_chars = [ref_dict.get(pos, "N") for pos in range(start_1_based, end_1_based + 1)]
        ref_sequence = "".join(ref_chars)

        # Build 4 x L score matrix
        score_matrix = np.zeros((4, length), dtype=np.float32)

        # Extract quantile layers
        quantiles = (
            avi_adata.layers["quantiles"]
            if (hasattr(avi_adata, "layers") and avi_adata.layers and "quantiles" in avi_adata.layers)
            else avi_adata.X
        )

        # Feature importance matrix and column names
        feat_names: List[str] = []
        if fi_adata is not None:
            if hasattr(fi_adata, "var") and fi_adata.var is not None and "name" in fi_adata.var.columns:
                feat_names = fi_adata.var["name"].tolist()
            elif hasattr(fi_adata, "var_names"):
                feat_names = list(fi_adata.var_names)

        modality_col_idx: Optional[int] = None
        if modality != "AVI_SCORE" and modality in feat_names:
            modality_col_idx = feat_names.index(modality)

        records: List[Dict[str, Any]] = []

        for idx, var in enumerate(variants):
            pos = var.position
            local_pos_idx = pos - start_1_based
            if local_pos_idx < 0 or local_pos_idx >= length:
                continue

            ref_base = var.reference_bases
            alt_base = var.alternate_bases
            if alt_base not in ALPHABET:
                continue

            alt_row_idx = ALPHABET.index(alt_base)

            # Compute calibrated Phred score
            q_val = quantiles[idx][0] if hasattr(quantiles[idx], "__getitem__") else quantiles[idx]
            q_float = float(q_val)
            tail = max(1e-7, 1.0 - q_float)
            phred = -10.0 * math.log10(tail)

            # Extract feature importance and top modality
            top_mod = "Unassigned"
            top_val = 0.0
            feat_imp_dict: Dict[str, float] = {}

            if fi_adata is not None and idx < fi_adata.X.shape[0]:
                fi_row = fi_adata.X[idx]
                if len(feat_names) == len(fi_row):
                    feat_imp_dict = {f: float(fi_row[j]) for j, f in enumerate(feat_names)}
                    top_idx = int(np.argmax(fi_row))
                    top_mod = feat_names[top_idx]
                    top_val = float(fi_row[top_idx])

            # Select target matrix value depending on active modality
            if modality == "AVI_SCORE":
                cell_value = float(phred)
            elif modality_col_idx is not None and fi_adata is not None:
                cell_value = float(fi_adata.X[idx, modality_col_idx])
            else:
                cell_value = float(phred)

            score_matrix[alt_row_idx, local_pos_idx] = cell_value

            atlas_url = format_atlas_url(chrom, pos, ref_base, alt_base)
            percentile = tail * 100.0

            records.append({
                "variant": str(var),
                "position": pos,
                "local_pos": local_pos_idx + 1,
                "ref": ref_base,
                "alt": alt_base,
                "mutation": f"{ref_base}{local_pos_idx + 1}{alt_base}",
                "score": cell_value,
                "avi_phred": phred,
                "top_percentile": percentile,
                "top_modality": top_mod,
                "top_feature_importance": top_val,
                "atlas_url": atlas_url,
            })

        # Reference alleles remain 0.0 in score_matrix by definition of wild-type
        tidy_df = pd.DataFrame(records)
        if not tidy_df.empty:
            tidy_df = tidy_df.sort_values("avi_phred", ascending=False).reset_index(drop=True)

        metadata = {
            "chromosome": chrom,
            "start": start_1_based,
            "end": end_1_based,
            "length": length,
            "variants_count": len(variants),
            "modality": modality,
            "modality_name": MODALITY_OPTIONS.get(modality, {}).get("name", modality),
            "max_phred": float(tidy_df["avi_phred"].max()) if not tidy_df.empty else 0.0,
            "mean_phred": float(tidy_df["avi_phred"].mean()) if not tidy_df.empty else 0.0,
        }

        return score_matrix, ref_sequence, tidy_df, metadata
