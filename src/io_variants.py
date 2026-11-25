from __future__ import annotations

"""Input helpers for annotated variant tables.

The loader accepts tab-separated variant files, standardizes annotation columns,
infers two parental genotype columns, and expands a wildcard genotype header
such as GEN[*].GT to match the genotype columns in the data.
"""

import csv
from pathlib import Path
from typing import Sequence

import pandas as pd


# -----------------------------------------------------------------------------
# Input schema
# -----------------------------------------------------------------------------
REQUIRED_BASE_COLUMNS = ["CHROM", "POS", "REF", "ALT"]
ANNOTATION_COLUMN_CANDIDATES = [
    "ANN",
    "Annotation",
    "SnpEff_ANN",
    "EFF",
    "Effect",
]


# -----------------------------------------------------------------------------
# Header inspection
# -----------------------------------------------------------------------------
def _find_annotation_column(columns: Sequence[str]) -> str | None:
    """Return the name of the first recognized annotation column."""
    for col in columns:
        if str(col).strip() in ANNOTATION_COLUMN_CANDIDATES:
            return str(col).strip()
    return None


def _read_header_and_max_width(path: Path) -> tuple[list[str], int]:
    """Read the TSV header and find the widest row in the file."""
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:
        reader = csv.reader(f, delimiter="\t")

        header_fields = next((row for row in reader if row), None)
        if header_fields is None:
            raise ValueError(f"Input file is empty: {path}")

        header_fields = [str(col).strip() for col in header_fields]
        max_width = len(header_fields)

        for row in reader:
            if row:
                max_width = max(max_width, len(row))

    return header_fields, max_width


def _expand_wildcard_gt_header(
    header_fields: Sequence[str],
    max_data_width: int,
) -> list[str] | None:
    """Expand a single GEN[*].GT header to match the data's genotype columns.

    For example, this header:

        CHROM POS REF ALT GEN[*].GT ANN

    with rows containing two genotype values becomes:

        CHROM POS REF ALT Sample1_GT Sample2_GT ANN
    """
    header = [str(col).strip() for col in header_fields]

    placeholder_indices = [
        i
        for i, col in enumerate(header)
        if col.strip().lower() in {"gen[*].gt", "gt"}
    ]

    if len(placeholder_indices) != 1:
        return None

    gt_idx = placeholder_indices[0]

    # The wildcard represents at least one genotype column. If data rows are
    # wider than the header, the extra fields belong to additional genotypes.
    genotype_count = max(1, max_data_width - len(header) + 1)

    genotype_columns = [
        f"Sample{i}_GT"
        for i in range(1, genotype_count + 1)
    ]

    return (
        header[:gt_idx]
        + genotype_columns
        + header[gt_idx + 1:]
    )


# -----------------------------------------------------------------------------
# Column standardization and sample inference
# -----------------------------------------------------------------------------
def _standardize_annotation_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a recognized annotation column to ANN, or add an empty ANN."""
    annotation_column = next(
        (
            col
            for col in ANNOTATION_COLUMN_CANDIDATES
            if col in df.columns
        ),
        None,
    )

    if annotation_column and annotation_column != "ANN":
        df = df.rename(columns={annotation_column: "ANN"})

    if "ANN" not in df.columns:
        df["ANN"] = ""

    return df


def _infer_sample_columns(df: pd.DataFrame) -> list[str]:
    """Infer the first two parental genotype columns."""
    base_columns = set(REQUIRED_BASE_COLUMNS + ["ANN"])

    sample_columns = [
        str(col).strip()
        for col in df.columns
        if col not in base_columns
        and not str(col).startswith("Unnamed:")
    ]

    if len(sample_columns) >= 2:
        return sample_columns[:2]

    unnamed_columns = [
        str(col)
        for col in df.columns
        if str(col).startswith("Unnamed:")
    ]

    if len(sample_columns) == 1 and unnamed_columns:
        repaired_sample_columns = [sample_columns[0]]

        for col in unnamed_columns:
            series = df[col].fillna("").astype(str).str.strip()

            if series.ne("").any():
                repaired_sample_columns.append(col)

            if len(repaired_sample_columns) >= 2:
                return repaired_sample_columns[:2]

    raise ValueError(
        "Could not infer two sample columns. "
        "The input must contain two parental genotype columns. "
        f"Found non-base columns: {sample_columns}"
    )


# -----------------------------------------------------------------------------
# Public loader
# -----------------------------------------------------------------------------
def load_variants_tsv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a tab-separated variant table and return its two genotype columns."""
    path = Path(path)

    header_fields, max_data_width = _read_header_and_max_width(path)
    expanded_header = _expand_wildcard_gt_header(
        header_fields,
        max_data_width,
    )

    if expanded_header is not None:
        # The source header contains a wildcard such as GEN[*].GT. Replace it
        # with one column name per genotype value found in the data.
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            low_memory=False,
            header=0,
            names=expanded_header,
        )
    else:
        # Standard TSV with explicit column names.
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            low_memory=False,
        )

    df.columns = [str(col).strip() for col in df.columns]

    # Handle the existing special input format.
    if list(df.columns[:8]) == [
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
    ]:
        df = pd.DataFrame(
            {
                "CHROM": df["#CHROM"],
                "POS": df["POS"],
                "REF": df["ID"],
                "ALT": df["REF"],
                "Sample1_GT": df["ALT"],
                "Sample2_GT": df["QUAL"],
                "ANN": df["FILTER"],
            }
        )

    # Standard VCF-style chromosome column.
    elif "#CHROM" in df.columns:
        df = df.rename(columns={"#CHROM": "CHROM"})

    # Standardize annotation column name.
    df = _standardize_annotation_column(df)

    missing_columns = [
        col
        for col in REQUIRED_BASE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input variant table is missing required columns: "
            f"{missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    df["POS"] = df["POS"].astype(str)

    sample_columns = _infer_sample_columns(df)

    print("Variant columns:", df.columns.tolist())
    print("Sample columns:", sample_columns)

    return df, sample_columns