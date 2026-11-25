# -*- coding: utf-8 -*-
"""Create candidate-gene ranking figures for each QTL.

This script generates:
1. Top-N horizontal bar plots of candidate-gene scores.
2. QTL interval scatter plots using physical gene positions.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CANDIDATE_SHEETS = (
    "ALL_CANDIDATES",
    "TOP_PRIORITY",
    "TOP_KNOWN_ALL",
    "TOP_NOVEL_ALL",
)

SCORE_COLUMNS = (
    "PriorityScore",
    "PriorityBaseScore",
    "Score_known",
    "Score_novel",
    "Score_known_lit",
    "Score_novel_lit",
    "FinalScore",
    "Score",
)

SYMBOL_COLUMNS = (
    "RAP-DB Gene Symbol Synonym(s)",
    "CGSNL Gene Symbol",
    "Oryzabase Gene Symbol Synonym(s)",
)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _sanitize_filename(value: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", str(value))
    return re.sub(r"_+", "_", text).strip("_")


def _pick_candidate_sheet(xlsx: Path) -> str:
    available = pd.ExcelFile(xlsx).sheet_names

    for sheet in CANDIDATE_SHEETS:
        if sheet in available:
            return sheet

    raise ValueError(
        "No candidate-ranking sheet was found. Expected one of: "
        f"{list(CANDIDATE_SHEETS)}"
    )


def _resolve_score_column(table: pd.DataFrame) -> str:
    for column in SCORE_COLUMNS:
        if column in table.columns:
            return column

    raise ValueError(
        "No supported ranking-score column was found. Expected one of: "
        f"{list(SCORE_COLUMNS)}"
    )


def _first_valid_symbol(row: pd.Series) -> str:
    for column in SYMBOL_COLUMNS:
        if column not in row.index or pd.isna(row[column]):
            continue

        value = str(row[column]).strip()
        if value and value.lower() != "nan":
            return value.split(",")[0].split(";")[0].strip()

    return ""


def _normalize_chromosome(value: str) -> str:
    match = re.search(r"(\d+)", str(value).strip())
    if not match:
        return str(value).strip()

    return f"chr{int(match.group(1)):02d}"


def _load_gene_positions(gff3_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    with gff3_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            seqid, _, feature, start, end, _, _, _, attributes = fields
            if feature.lower() not in {"gene", "locus"}:
                continue

            gene_id = ""
            for key in ("ID", "Name", "gene_id", "locus_id"):
                match = re.search(rf"(?:^|;){key}=([^;]+)", attributes)
                if match:
                    gene_id = match.group(1).strip()
                    break

            if not gene_id:
                match = re.search(r"(Os\d{2}g\d{7})", attributes)
                gene_id = match.group(1) if match else ""

            if not gene_id:
                continue

            try:
                start_bp = int(start)
                end_bp = int(end)
            except ValueError:
                continue

            rows.append(
                {
                    "GeneID": gene_id,
                    "CHR_GFF": _normalize_chromosome(seqid),
                    "START_GFF": start_bp,
                    "END_GFF": end_bp,
                    "MID_GFF": (start_bp + end_bp) / 2.0,
                }
            )

    positions = pd.DataFrame(rows)
    if positions.empty:
        raise ValueError(f"No gene coordinates were parsed from: {gff3_path}")

    return positions.drop_duplicates(subset=["GeneID"])


def _bbox_overlap_area(box_a, box_b) -> float:
    """Return the intersection area of two rendered text bounding boxes."""
    width = max(0.0, min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0))
    height = max(0.0, min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0))
    return width * height


def _add_top5_symbol_labels(
    axis,
    figure,
    top5: pd.DataFrame,
    score_column: str,
) -> None:
    """Place up to five gene symbols inside the axes without label overlap."""
    if "Symbol" not in top5.columns or top5.empty:
        return

    label_rows = (
        top5.sort_values(score_column, ascending=False)
        .drop_duplicates(subset=["GeneID"])
        .head(5)
    )

    # Nearby offsets are tried first. More distant positions are considered
    # only when labels would otherwise overlap.
    candidate_offsets = [
        (8, 8),
        (8, -12),
        (-8, 8),
        (-8, -12),
        (0, 18),
        (0, -22),
        (16, 12),
        (-16, 12),
        (16, -18),
        (-16, -18),
        (24, 8),
        (-24, 8),
        (24, -12),
        (-24, -12),
        (0, 28),
        (0, -32),
        (32, 16),
        (-32, 16),
        (32, -22),
        (-32, -22),
    ]

    placed_boxes = []

    for _, row in label_rows.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol or symbol.lower() == "nan":
            continue

        point_x = float(row["MID_GFF"])
        point_y = float(row[score_column])

        selected_offset = None
        best_fallback = None

        for dx, dy in candidate_offsets:
            distance = float(np.hypot(dx, dy))
            use_connector = distance > 13

            arrowprops = (
                {
                    "arrowstyle": "-",
                    "color": "0.35",
                    "linewidth": 0.7,
                    "shrinkA": 2,
                    "shrinkB": 4,
                }
                if use_connector
                else None
            )

            trial = axis.annotate(
                symbol,
                xy=(point_x, point_y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=9,
                fontweight="bold",
                zorder=6,
                annotation_clip=False,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
                arrowprops=arrowprops,
            )

            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()

            # Measure only the label's background/text box, not its connector.
            text_box = trial.get_bbox_patch().get_window_extent(renderer)
            axes_box = axis.get_window_extent(renderer)

            inside_axes = (
                text_box.x0 >= axes_box.x0
                and text_box.x1 <= axes_box.x1
                and text_box.y0 >= axes_box.y0
                and text_box.y1 <= axes_box.y1
            )

            overlaps = sum(
                _bbox_overlap_area(text_box, existing_box) > 0
                for existing_box in placed_boxes
            )
            overlap_area = sum(
                _bbox_overlap_area(text_box, existing_box)
                for existing_box in placed_boxes
            )

            trial.remove()

            if inside_axes and overlaps == 0:
                selected_offset = (dx, dy)
                break

            if inside_axes:
                # Retain the in-bounds position with the least overlap as a
                # fallback in case every candidate position is occupied.
                fallback_score = (overlaps, overlap_area, distance)
                if best_fallback is None or fallback_score < best_fallback[0]:
                    best_fallback = (fallback_score, (dx, dy))

        if selected_offset is None and best_fallback is not None:
            selected_offset = best_fallback[1]

        # If the axes are too narrow for the label, skip it rather than
        # allowing the text to spill outside the graph.
        if selected_offset is None:
            continue

        dx, dy = selected_offset
        use_connector = float(np.hypot(dx, dy)) > 13

        arrowprops = (
            {
                "arrowstyle": "-",
                "color": "0.35",
                "linewidth": 0.7,
                "shrinkA": 2,
                "shrinkB": 4,
            }
            if use_connector
            else None
        )

        annotation = axis.annotate(
            symbol,
            xy=(point_x, point_y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=9,
            fontweight="bold",
            zorder=6,
            annotation_clip=False,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            arrowprops=arrowprops,
        )

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        placed_boxes.append(
            annotation.get_bbox_patch().get_window_extent(renderer).expanded(
                1.04,
                1.10,
            )
        )


# -----------------------------------------------------------------------------
# 1. Load candidate-ranking results
# -----------------------------------------------------------------------------

def load_candidate_ranking(
    xlsx: str | Path,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load the candidate-ranking table and identify its score column."""
    xlsx = Path(xlsx)
    if not xlsx.exists():
        raise FileNotFoundError(f"Ranking workbook was not found: {xlsx}")

    sheet = sheet_name or _pick_candidate_sheet(xlsx)
    table = pd.read_excel(xlsx, sheet_name=sheet, dtype=str)
    table.columns = [str(column).strip() for column in table.columns]

    required = ["QTL", "GeneID", "START", "END"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet}' is missing required columns: {missing}")

    score_column = _resolve_score_column(table)
    table[score_column] = pd.to_numeric(table[score_column], errors="coerce")
    table["START"] = pd.to_numeric(table["START"], errors="coerce")
    table["END"] = pd.to_numeric(table["END"], errors="coerce")

    if "PEAK" in table.columns:
        table["PEAK"] = pd.to_numeric(table["PEAK"], errors="coerce")

    table["QTL"] = table["QTL"].astype(str).str.strip()
    table["GeneID"] = table["GeneID"].astype(str).str.strip()
    table = table.dropna(subset=[score_column, "START", "END"]).copy()
    table = table[(table["QTL"] != "") & (table["GeneID"] != "")].copy()

    return table, sheet, score_column


# -----------------------------------------------------------------------------
# 2. Attach representative RAP-DB gene symbols
# -----------------------------------------------------------------------------

def attach_gene_symbols(
    candidate_table: pd.DataFrame,
    rapdb_tsv: str | Path,
) -> pd.DataFrame:
    """Attach one representative gene symbol to each RAP-DB locus ID."""
    if "Symbol" in candidate_table.columns and candidate_table["Symbol"].notna().any():
        return candidate_table.copy()

    rapdb_tsv = Path(rapdb_tsv)
    if not rapdb_tsv.exists():
        raise FileNotFoundError(
            f"RAP-DB annotation table was not found: {rapdb_tsv}"
        )

    annotation = pd.read_csv(
        rapdb_tsv,
        sep="\t",
        dtype=str,
        low_memory=False,
    )
    annotation.columns = [str(column).strip() for column in annotation.columns]

    if "Locus_ID" not in annotation.columns:
        raise ValueError(
            "RAP-DB annotation table must contain a 'Locus_ID' column."
        )

    annotation["GeneID"] = annotation["Locus_ID"].astype(str).str.strip()
    annotation["Symbol"] = annotation.apply(_first_valid_symbol, axis=1)
    annotation["Symbol"] = annotation["Symbol"].replace(
        {"": np.nan, "nan": np.nan}
    )
    annotation = annotation[["GeneID", "Symbol"]].drop_duplicates(
        subset=["GeneID"]
    )

    output = candidate_table.copy()
    output["GeneID"] = output["GeneID"].astype(str).str.strip()
    return output.merge(annotation, on="GeneID", how="left")


# -----------------------------------------------------------------------------
# 3. Top-N horizontal bar plot
# -----------------------------------------------------------------------------

def gene_ranking_top20_bar(
    qtl_table: pd.DataFrame,
    score_column: str,
    output_png: str | Path,
    top_n: int = 20,
) -> Path:
    """Draw the top-ranked candidate genes as a horizontal bar plot."""
    if qtl_table.empty:
        raise ValueError("Cannot draw a bar plot from an empty QTL table.")

    ranked = (
        qtl_table.dropna(subset=[score_column])
        .sort_values(score_column, ascending=False)
        .drop_duplicates(subset=["GeneID"])
        .head(int(top_n))
        .iloc[::-1]
        .copy()
    )

    values = ranked[score_column].astype(float).to_numpy()
    gene_ids = ranked["GeneID"].astype(str).tolist()

    symbols = (
        ranked["Symbol"].fillna("").astype(str).tolist()
        if "Symbol" in ranked.columns
        else [""] * len(ranked)
    )

    y_positions = np.arange(len(ranked))
    figure, axis = plt.subplots(figsize=(7, 6))

    axis.barh(y_positions, values)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(gene_ids, fontsize=13)

    maximum = float(np.max(values)) if len(values) else 1.0
    text_padding = maximum * 0.02 if maximum > 0 else 0.02

    for y, value, symbol in zip(y_positions, values, symbols):
        if symbol and symbol.lower() != "nan":
            axis.text(
                value + text_padding,
                y,
                symbol,
                va="center",
                ha="left",
                fontsize=15,
                fontweight="bold",
            )

    axis.set_xlim(0, maximum * 1.18 if maximum > 0 else 1.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(figure)

    return output_png


# -----------------------------------------------------------------------------
# 4. QTL interval scatter plot
# -----------------------------------------------------------------------------

def gene_interval_scatter(
    qtl_table: pd.DataFrame,
    score_column: str,
    output_png: str | Path,
) -> Path:
    """Draw candidate scores across one physical QTL interval."""
    if qtl_table.empty:
        raise ValueError("Cannot draw an interval scatter plot from an empty QTL table.")
    if "MID_GFF" not in qtl_table.columns:
        raise ValueError("The QTL table must contain 'MID_GFF' gene coordinates.")

    table = qtl_table.copy()
    table["MID_GFF"] = pd.to_numeric(table["MID_GFF"], errors="coerce")
    table[score_column] = pd.to_numeric(table[score_column], errors="coerce")
    table = table.dropna(subset=["MID_GFF", score_column]).copy()

    if table.empty:
        raise ValueError(
            "No genes with valid GFF3 coordinates were available for plotting."
        )

    top5_ids = (
        table.sort_values(score_column, ascending=False)["GeneID"]
        .astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )
    table["is_top5"] = table["GeneID"].astype(str).isin(top5_ids)

    ordinary = table[~table["is_top5"]]
    top5 = table[table["is_top5"]]

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.scatter(
        ordinary["MID_GFF"],
        ordinary[score_column],
        s=28,
        alpha=0.65,
        color="black",
        zorder=2,
    )
    axis.scatter(
        top5["MID_GFF"],
        top5[score_column],
        s=68,
        alpha=0.98,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
    )

    peak_values = (
        pd.to_numeric(table["PEAK"], errors="coerce").dropna()
        if "PEAK" in table.columns
        else pd.Series(dtype=float)
    )
    peak_bp = float(peak_values.iloc[0]) if len(peak_values) else np.nan

    if np.isfinite(peak_bp):
        axis.axvline(
            peak_bp,
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.95,
            zorder=1,
        )

    start_values = pd.to_numeric(table["START"], errors="coerce").dropna()
    end_values = pd.to_numeric(table["END"], errors="coerce").dropna()

    start_bp = (
        float(start_values.iloc[0])
        if len(start_values)
        else float(table["MID_GFF"].min())
    )
    end_bp = (
        float(end_values.iloc[0])
        if len(end_values)
        else float(table["MID_GFF"].max())
    )

    x_min = min(start_bp, float(table["MID_GFF"].min()))
    x_max = max(end_bp, float(table["MID_GFF"].max()))
    x_padding = (x_max - x_min) * 0.04 if x_max > x_min else 1.0
    axis.set_xlim(x_min - x_padding, x_max + x_padding)

    chromosome = ""
    if "CHR_GFF" in table.columns and table["CHR_GFF"].notna().any():
        chromosome = str(table["CHR_GFF"].dropna().iloc[0])
    elif "CHR" in table.columns and table["CHR"].notna().any():
        chromosome = str(table["CHR"].dropna().iloc[0])

    chromosome_label = chromosome.replace("chr", "Chr")
    axis.xaxis.set_major_locator(MultipleLocator(5e5))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x / 1e6:.1f}")
    )
    axis.set_xlabel(
        f"Physical position on {chromosome_label} (Mb)",
        fontsize=12,
    )
    axis.set_ylabel("Integrated score of candidate genes", fontsize=12)

    y_min = float(table[score_column].min())
    y_max = float(table[score_column].max())
    y_padding = (y_max - y_min) * 0.15 if y_max > y_min else 0.05
    axis.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
            alpha=0.65,
            label="Candidate genes",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="tab:blue",
            markeredgecolor="black",
            markersize=8,
            label="Top 5 Candidate Genes",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="QTL peak",
        ),
    ]

    axis.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.3),
        borderaxespad=0,
    )

    # Finalize the axes size before placing labels so their positions are
    # checked against the actual graph area.
    figure.tight_layout(rect=[0, 0, 0.80, 1])
    figure.canvas.draw()
    _add_top5_symbol_labels(axis, figure, top5, score_column)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(figure)

    return output_png


# -----------------------------------------------------------------------------
# 5. Generate both figure types for every QTL
# -----------------------------------------------------------------------------

def make_ranking_figures(
    xlsx: str | Path,
    rapdb_tsv: str | Path,
    gff3_path: str | Path,
    output_dir: str | Path,
    qtl_order: list[str] | None = None,
    sheet_name: str | None = None,
    top_n: int = 20,
) -> list[Path]:
    """Generate a top-N bar plot and interval scatter plot for each QTL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_table, source_sheet, score_column = load_candidate_ranking(
        xlsx=xlsx,
        sheet_name=sheet_name,
    )
    candidate_table = attach_gene_symbols(candidate_table, rapdb_tsv)

    positions = _load_gene_positions(Path(gff3_path))
    candidate_table = candidate_table.merge(
        positions,
        on="GeneID",
        how="left",
    )

    if qtl_order is None:
        qtl_order = (
            candidate_table["QTL"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
    else:
        qtl_order = [
            str(qtl).strip()
            for qtl in qtl_order
            if str(qtl).strip()
        ]

    created_files: list[Path] = []

    for qtl in qtl_order:
        qtl_table = candidate_table[
            candidate_table["QTL"].astype(str) == qtl
        ].copy()

        if qtl_table.empty:
            print(f"[Skipped] No candidate rows were found for QTL: {qtl}")
            continue

        filename = _sanitize_filename(qtl)
        bar_path = output_dir / f"{filename}_top{int(top_n)}_bar.png"
        scatter_path = output_dir / f"{filename}_interval_scatter.png"

        created_files.append(
            gene_ranking_top20_bar(
                qtl_table=qtl_table,
                score_column=score_column,
                output_png=bar_path,
                top_n=top_n,
            )
        )

        created_files.append(
            gene_interval_scatter(
                qtl_table=qtl_table,
                score_column=score_column,
                output_png=scatter_path,
            )
        )

    print(f"[Ranking] Source sheet: {source_sheet}")
    print(f"[Ranking] Score column: {score_column}")
    print(f"[Ranking] Figures created: {len(created_files)}")

    return created_files


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = (
        SCRIPT_DIR.parent
        if SCRIPT_DIR.name.lower() == "src"
        else SCRIPT_DIR
    )

    RESULTS_XLSX = PROJECT_ROOT / "output" / "results.xlsx"
    RAPDB_TSV = PROJECT_ROOT / "data" / "rapdb.tsv"
    GFF3 = PROJECT_ROOT / "data" / "IRGSP-1.0_locus.gff"
    FIGURE_DIR = PROJECT_ROOT / "output" / "figures_ranking"

    make_ranking_figures(
        xlsx=RESULTS_XLSX,
        rapdb_tsv=RAPDB_TSV,
        gff3_path=GFF3,
        output_dir=FIGURE_DIR,
    )