from __future__ import annotations

"""SnpEff ANN parsing utilities.

This module expands comma-separated ANN records into one row per annotation and
extracts normalized gene identifiers, impact classes, and effect terms.
"""
import re
from typing import Any, Iterator

import pandas as pd


# -----------------------------------------------------------------------------
# ANN schema
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS = ["CHROM", "POS", "REF", "ALT", "ANN"]
OUTPUT_COLUMNS = ["CHROM", "POS", "REF", "ALT", "GeneID", "Impact", "Effect"]


# -----------------------------------------------------------------------------
# Gene identifier and ANN-field helpers
# -----------------------------------------------------------------------------
def normalize_gene_id(value: str) -> str:
    """
    Normalize a gene identifier from SnpEff ANN fields.

    This function applies lightweight cleanup only and avoids
    species-specific assumptions so the module remains reusable.
    """
    gene_id = str(value or "").strip()
    if not gene_id:
        return ""

    gene_id = re.sub(r"^(GENE_|gene:|Gene:)", "", gene_id)
    gene_id = re.sub(r"(\.\d+)$", "", gene_id)
    gene_id = re.sub(r"(-T\d+|-t\d+|-mRNA.*|-transcript.*)$", "", gene_id)
    return gene_id.strip()


def _validate_required_columns(annotation_table: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in annotation_table.columns]
    if missing_columns:
        raise ValueError(f"explode_ann is missing required columns: {missing_columns}")


def _pick_gene_id(annotation_fields: list[str]) -> str:
    """
    Select the most appropriate gene identifier from ANN fields.

    Typical SnpEff ANN order:
    Allele | Annotation | Annotation_Impact | Gene_Name | Gene_ID | ...
    """
    gene_name = annotation_fields[3].strip() if len(annotation_fields) > 3 and annotation_fields[3] else ""
    gene_id = annotation_fields[4].strip() if len(annotation_fields) > 4 and annotation_fields[4] else ""
    return normalize_gene_id(gene_id or gene_name)


def _safe_position(value: Any) -> int | str:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def _iter_annotation_entries(annotation_text: str) -> Iterator[list[str]]:
    for entry in annotation_text.split(","):
        cleaned_entry = str(entry).strip()
        if cleaned_entry:
            yield cleaned_entry.split("|")


def _build_annotation_row(record: Any, annotation_fields: list[str]) -> dict[str, Any] | None:
    effect = annotation_fields[1].strip() if len(annotation_fields) > 1 else ""
    impact = annotation_fields[2].strip() if len(annotation_fields) > 2 else ""
    gene_id = _pick_gene_id(annotation_fields)

    if not gene_id:
        return None

    return {
        "CHROM": str(getattr(record, "CHROM")),
        "POS": _safe_position(getattr(record, "POS")),
        "REF": str(getattr(record, "REF")),
        "ALT": str(getattr(record, "ALT")),
        "GeneID": gene_id,
        "Impact": impact,
        "Effect": effect,
    }


# -----------------------------------------------------------------------------
# Public parser
# -----------------------------------------------------------------------------
def explode_ann(annotation_table: pd.DataFrame) -> pd.DataFrame:
    """
    Expand SnpEff ANN annotations into one row per annotation entry.
    """
    _validate_required_columns(annotation_table)

    if annotation_table.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    expanded_rows: list[dict[str, Any]] = []

    for record in annotation_table.itertuples(index=False):
        raw_ann_value = getattr(record, "ANN", "")
        ann_text = "" if pd.isna(raw_ann_value) else str(raw_ann_value).strip()

        if not ann_text or ann_text.lower() == "nan":
            continue

        for annotation_fields in _iter_annotation_entries(ann_text):
            row = _build_annotation_row(record, annotation_fields)
            if row is not None:
                expanded_rows.append(row)

    return pd.DataFrame(expanded_rows, columns=OUTPUT_COLUMNS)
