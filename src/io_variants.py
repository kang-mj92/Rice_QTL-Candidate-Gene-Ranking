from __future__ import annotations

"""Input helpers for annotated variant tables.

The loader accepts tab-separated variant files, standardizes annotation columns,
infers two parental genotype columns, and repairs one known malformed-header case.
"""
from pathlib import Path
from typing import Sequence

import pandas as pd


# -----------------------------------------------------------------------------
# Input schema
# -----------------------------------------------------------------------------
REQUIRED_BASE_COLUMNS = ["CHROM", "POS", "REF", "ALT"]
ANNOTATION_COLUMN_CANDIDATES = ["ANN", "Annotation", "SnpEff_ANN", "EFF", "Effect"]


# -----------------------------------------------------------------------------
# Header inspection and repair
# -----------------------------------------------------------------------------
def _find_annotation_column(columns: Sequence[str]) -> str | None:
    for col in ANNOTATION_COLUMN_CANDIDATES:
        if col in columns:
            return col
    return None


def _read_first_nonempty_lines(path: Path, n: int = 5) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                lines.append(line.rstrip("\n\r"))
            if len(lines) >= n:
                break
    return lines


def _count_fields(line: str) -> int:
    return len(str(line).split("\t"))


def _looks_like_single_gt_header_misalignment(
    header_fields: Sequence[str],
    data_lines: Sequence[str],
) -> bool:
    ann_col = _find_annotation_column(header_fields)
    if ann_col is None:
        return False

    base_plus_ann = set(REQUIRED_BASE_COLUMNS + [ann_col])
    non_base_cols = [col for col in header_fields if col not in base_plus_ann]
    if len(non_base_cols) != 1:
        return False

    only_sample_col = str(non_base_cols[0]).strip().lower()
    if not (only_sample_col in {"gen[*].gt", "gt"} or only_sample_col.endswith(".gt")):
        return False

    header_width = len(header_fields)
    data_widths = [_count_fields(line) for line in data_lines if str(line).strip()]
    if not data_widths:
        return False

    extra_one_count = sum(width == header_width + 1 for width in data_widths)
    return extra_one_count >= max(1, len(data_widths) // 2)


def _fix_misaligned_header(header_fields: Sequence[str]) -> list[str]:
    header_fields = list(header_fields)

    ann_col = _find_annotation_column(header_fields)
    if ann_col is None:
        raise ValueError("Could not repair malformed header because no annotation column was found.")

    base_plus_ann = set(REQUIRED_BASE_COLUMNS + [ann_col])
    non_base_cols = [col for col in header_fields if col not in base_plus_ann]

    if len(non_base_cols) != 1:
        raise ValueError(
            "Could not repair malformed header because exactly one sample-like column "
            f"was not found. Found: {non_base_cols}"
        )

    sample_idx = header_fields.index(non_base_cols[0])

    fixed = header_fields.copy()
    fixed[sample_idx] = "Sample1_GT"
    fixed.insert(sample_idx + 1, "Sample2_GT")
    return fixed


# -----------------------------------------------------------------------------
# Column standardization and sample inference
# -----------------------------------------------------------------------------
def _standardize_annotation_column(df: pd.DataFrame) -> pd.DataFrame:
    annotation_column = next(
        (col for col in ANNOTATION_COLUMN_CANDIDATES if col in df.columns),
        None,
    )

    if annotation_column and annotation_column != "ANN":
        df = df.rename(columns={annotation_column: "ANN"})

    if "ANN" not in df.columns:
        df["ANN"] = ""

    return df


def _infer_sample_columns(df: pd.DataFrame) -> list[str]:
    base_columns = set(REQUIRED_BASE_COLUMNS + ["ANN"])
    sample_columns = [
        str(col).strip()
        for col in df.columns
        if col not in base_columns and not str(col).startswith("Unnamed:")
    ]

    if len(sample_columns) >= 2:
        return sample_columns[:2]

    unnamed_columns = [str(col) for col in df.columns if str(col).startswith("Unnamed:")]

    if len(sample_columns) == 1 and unnamed_columns:
        repaired_sample_columns: list[str] = [sample_columns[0]]
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
    path = Path(path)

    preview_lines = _read_first_nonempty_lines(path, n=5)
    if not preview_lines:
        raise ValueError(f"Input file is empty: {path}")

    header_fields = preview_lines[0].split("\t")
    data_preview_lines = preview_lines[1:]

    if _looks_like_single_gt_header_misalignment(header_fields, data_preview_lines):
        fixed_header = _fix_misaligned_header(header_fields)

        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            low_memory=False,
            header=None,
            skiprows=1,
        )

        if df.shape[1] != len(fixed_header):
            raise ValueError(
                "Header repair was attempted, but the repaired column count does not match "
                f"the data width. Repaired header has {len(fixed_header)} columns, "
                f"but data has {df.shape[1]} columns."
            )

        df.columns = fixed_header
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)

    df.columns = [str(col).strip() for col in df.columns]

    missing_columns = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Input variant table is missing required columns: "
            f"{missing_columns}. Found columns: {list(df.columns)[:30]}"
        )

    df = _standardize_annotation_column(df)
    df["POS"] = df["POS"].astype(str)

    sample_columns = _infer_sample_columns(df)
    return df, sample_columns
