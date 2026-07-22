from __future__ import annotations

"""Gene-level variant scoring.

Variant annotations are aggregated by gene. Each SnpEff impact class contributes
at most once to the final presence-based VariantScore, while raw counts are kept
for reporting.
"""
import numpy as np
import pandas as pd

from src.config import VariantScoringConfig


# -----------------------------------------------------------------------------
# Score schema
# -----------------------------------------------------------------------------
IMPACT_LEVELS = ["HIGH", "MODERATE", "LOW", "MODIFIER"]
OUTPUT_COLUMNS = [
    "GeneID",
    "VariantScore",
    "N_Variants",
    "N_HIGH",
    "N_MODERATE",
    "N_LOW",
    "N_MODIFIER",
]


# -----------------------------------------------------------------------------
# Input preparation and aggregation
# -----------------------------------------------------------------------------
def _prepare_input_table(variant_gene_table: pd.DataFrame) -> pd.DataFrame:
    if variant_gene_table is None or variant_gene_table.empty:
        return pd.DataFrame()

    prepared_table = variant_gene_table.copy()
    if "GeneID" not in prepared_table.columns:
        raise ValueError("score_variants_by_gene requires a 'GeneID' column.")

    prepared_table["GeneID"] = prepared_table["GeneID"].astype(str).str.strip()
    prepared_table = prepared_table[prepared_table["GeneID"].ne("")].copy()
    return prepared_table


def _normalize_impact_series(variant_table: pd.DataFrame) -> pd.Series:
    if "Impact" not in variant_table.columns:
        return pd.Series("", index=variant_table.index, dtype=str)
    return variant_table["Impact"].fillna("").astype(str).str.upper()


def _add_impact_indicator_columns(variant_table: pd.DataFrame) -> pd.DataFrame:
    prepared_table = variant_table.copy()
    impact_series = _normalize_impact_series(prepared_table)

    for impact_level in IMPACT_LEVELS:
        indicator_column = f"_is_{impact_level.lower()}"
        prepared_table[indicator_column] = (impact_series == impact_level).astype(int)

    return prepared_table


def _aggregate_variant_counts_by_gene(variant_table: pd.DataFrame) -> pd.DataFrame:
    grouped_table = variant_table.groupby("GeneID", dropna=False)

    gene_summary_table = pd.DataFrame({"GeneID": grouped_table.size().index})
    gene_summary_table["N_Variants"] = grouped_table.size().values.astype(int)
    gene_summary_table["N_HIGH"] = grouped_table["_is_high"].sum().values.astype(int)
    gene_summary_table["N_MODERATE"] = grouped_table["_is_moderate"].sum().values.astype(int)
    gene_summary_table["N_LOW"] = grouped_table["_is_low"].sum().values.astype(int)
    gene_summary_table["N_MODIFIER"] = grouped_table["_is_modifier"].sum().values.astype(int)
    return gene_summary_table


# -----------------------------------------------------------------------------
# Presence-based score calculation
# -----------------------------------------------------------------------------
def _compute_variant_score(
    gene_summary_table: pd.DataFrame,
    config: VariantScoringConfig,
) -> pd.Series:
    has_high_variant = (gene_summary_table["N_HIGH"] > 0).astype(float)
    has_moderate_variant = (gene_summary_table["N_MODERATE"] > 0).astype(float)
    has_low_variant = (gene_summary_table["N_LOW"] > 0).astype(float)
    has_modifier_variant = (gene_summary_table["N_MODIFIER"] > 0).astype(float)

    variant_score = (
        config.high_weight * has_high_variant
        + config.moderate_weight * has_moderate_variant
        + config.low_weight * has_low_variant
        + config.modifier_weight * has_modifier_variant
    )
    return np.minimum(variant_score, config.max_score).astype(float)


# -----------------------------------------------------------------------------
# Public scoring function
# -----------------------------------------------------------------------------
def score_variants_by_gene(
    variant_gene_table: pd.DataFrame,
    config: VariantScoringConfig | None = None,
) -> pd.DataFrame:
    """
    Aggregate variant-level annotations into gene-level summary statistics.

    Each impact class is counted for reporting, but contributes to VariantScore
    only once per gene through a presence-based weight.
    """
    scoring_config = config or VariantScoringConfig()
    scoring_config.validate()

    prepared_table = _prepare_input_table(variant_gene_table)
    if prepared_table.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    prepared_table = _add_impact_indicator_columns(prepared_table)
    gene_summary_table = _aggregate_variant_counts_by_gene(prepared_table)
    gene_summary_table["VariantScore"] = _compute_variant_score(
        gene_summary_table,
        scoring_config,
    )

    return gene_summary_table[OUTPUT_COLUMNS]
