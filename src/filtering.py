from __future__ import annotations

"""Variant filtering utilities.

Functions in this module select variants by genomic interval, normalize genotype
strings, and retain fixed homozygous differences between two parental samples.
"""
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
HOMOZYGOUS_GENOTYPES = {"0/0", "1/1"}


# -----------------------------------------------------------------------------
# Internal validation and preparation
# -----------------------------------------------------------------------------
def _validate_required_columns(
    table: pd.DataFrame,
    required_columns: list[str],
    function_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"{function_name} is missing required columns: {missing_columns}")


def _prepare_interval_columns(
    variant_table: pd.DataFrame,
    chrom_col: str,
    pos_col: str,
) -> pd.DataFrame:
    prepared = variant_table.copy()
    prepared[chrom_col] = prepared[chrom_col].astype(str).str.strip()
    prepared[pos_col] = pd.to_numeric(prepared[pos_col], errors="coerce")
    prepared = prepared.dropna(subset=[pos_col]).copy()
    return prepared


# -----------------------------------------------------------------------------
# Public filtering functions
# -----------------------------------------------------------------------------
def filter_by_interval(
    df: pd.DataFrame,
    chrom: str,
    start: int,
    end: int,
    chrom_col: str = "CHROM",
    pos_col: str = "POS",
) -> pd.DataFrame:
    """
    Filter variants within a genomic interval.
    """
    _validate_required_columns(df, [chrom_col, pos_col], "filter_by_interval")

    prepared_table = _prepare_interval_columns(df, chrom_col=chrom_col, pos_col=pos_col)
    chrom_label = str(chrom).strip()
    interval_start = int(start)
    interval_end = int(end)

    interval_mask = (
        (prepared_table[chrom_col] == chrom_label)
        & (prepared_table[pos_col] >= interval_start)
        & (prepared_table[pos_col] <= interval_end)
    )
    return prepared_table.loc[interval_mask].copy()


def normalize_genotype(value: str) -> str:
    """
    Normalize genotype strings into a comparable format.

    Examples
    --------
    '0' -> '0/0'
    '1' -> '1/1'
    '0|0' -> '0/0'
    '1/1:35,12:99' -> '1/1'
    '.' -> './.'
    """

    genotype_text = "" if value is None else str(value).strip()

    if genotype_text in {"", "nan", "NaN", "."}:
        return "./."

    if ":" in genotype_text:
        genotype_text = genotype_text.split(":", 1)[0]

    genotype_text = genotype_text.replace("|", "/")

    # 단일 allele 표기 보정
    if genotype_text == "0":
        return "0/0"

    if genotype_text == "1":
        return "1/1"

    return genotype_text


def _build_fixed_difference_mask(
    genotype_a: pd.Series,
    genotype_b: pd.Series,
) -> pd.Series:
    return (
        genotype_a.isin(HOMOZYGOUS_GENOTYPES)
        & genotype_b.isin(HOMOZYGOUS_GENOTYPES)
        & (genotype_a != genotype_b)
    )


def filter_fixed_difference_variants(
    df: pd.DataFrame,
    sample1_col: str,
    sample2_col: str,
) -> pd.DataFrame:
    """
    Retain variants showing fixed homozygous differences between two samples.
    """
    _validate_required_columns(
        df,
        [sample1_col, sample2_col],
        "filter_fixed_difference_variants",
    )

    genotype_table = df.copy()
    genotype_a = genotype_table[sample1_col].map(normalize_genotype)
    genotype_b = genotype_table[sample2_col].map(normalize_genotype)

    fixed_difference_mask = _build_fixed_difference_mask(genotype_a, genotype_b)
    return genotype_table.loc[fixed_difference_mask].copy()


filter_fixed_diff = filter_fixed_difference_variants
