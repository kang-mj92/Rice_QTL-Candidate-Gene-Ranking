# -*- coding: utf-8 -*-
"""
QTL candidate gene ranking pipeline
===================================

This script prioritizes candidate genes within QTL intervals by integrating:

1. variant impact,
2. distance from the QTL peak,
3. annotation-based trait relevance,
4. semantic similarity, and
5. PubMed-derived literature evidence.

Expected project layout
-----------------------
project_root/
├── data/
│   ├── variants.tsv
│   ├── rapdb.tsv
│   └── qtl_runs.xlsx
├── src/
│   ├── config.py
│   ├── filtering.py
│   ├── io_variants.py
│   ├── parse_ann.py
│   ├── pubmed_client.py
│   ├── scoring_variant.py
│   └── trait_profile_mining.py
├── cache/
├── output/
└── run_qtl_candidate_ranking.py

Environment variables
---------------------
NCBI_EMAIL
    Contact email recommended by NCBI E-utilities.
NCBI_API_KEY
    Optional API key. When present, the request rate is increased.

Notes
-----
- Input file names and score weights are examples and can be changed in the
  configuration dataclasses below.
- API credentials and personal paths are never hard-coded.
- Curated PMIDs are used for post hoc retrieval evaluation, not as search terms.
"""
from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import VariantScoringConfig
from src.filtering import filter_by_interval, filter_fixed_difference_variants
from src.io_variants import load_variants_tsv
from src.parse_ann import explode_ann
from src.pubmed_client import pubmed_search_with_cache
from src.scoring_variant import score_variants_by_gene
from src.trait_profile_mining import rapdb_global_profile
from src.plot_ranking import make_ranking_figures


# ============================================================================
# 1. Configuration and data containers
# ============================================================================

@dataclass(frozen=True)
class PathSettings:
    root_dir: Path
    data_dir: Path
    output_dir: Path
    cache_dir: Path
    variant_tsv: Path
    rapdb_tsv: Path
    gff3_path: Path
    run_workbook: Path
    run_sheet: str
    output_workbook: Path
    raw_candidates_tsv: Path
    pubmed_cache_dir: Path
    embedding_cache_dir: Path


@dataclass(frozen=True)
class ApiSettings:
    ncbi_email: str = ""
    ncbi_api_key: Optional[str] = None
    ncbi_requests_per_second: float = 3.0


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_seed: int = 300
    document_batch_size: int = 64
    text_batch_size: int = 32
    default_embedding_dim: int = 384


@dataclass(frozen=True)
class RankingLimits:
    pubmed_retmax: int = 20
    top_pubmed_genes: int = 50
    top_articles_per_gene: int = 20
    top_articles_for_support: int = 10
    top_priority_per_qtl: int = 20


@dataclass(frozen=True)
class WeightSettings:
    literature_embed: float = 0.4
    literature_keyword: float = 0.4
    literature_mesh: float = 0.15
    literature_pubtype: float = 0.05

    curated_pmid: float = 0.45
    curated_oryzabase: float = 0.35
    curated_text_hit: float = 0.20


    gene_functional_hit: float = 0.60
    gene_embedding_similarity: float = 0.40


    base_variant: float = 0.30
    base_distance: float = 0.30
    base_trait_relatedness: float = 0.40

    final_variant: float = 0.25
    final_distance: float = 0.25
    final_trait_relatedness: float = 0.25
    final_literature: float = 0.25


@dataclass(frozen=True)
class PipelineSettings:
    paths: PathSettings
    api: ApiSettings
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    limits: RankingLimits = field(default_factory=RankingLimits)
    weights: WeightSettings = field(default_factory=WeightSettings)
    debug_pubmed: bool = False
    variant_scoring_config: VariantScoringConfig = field(default_factory=VariantScoringConfig)


@dataclass
class RapdbResources:
    annotation_df: pd.DataFrame
    annotation_texts: List[str]
    doc_embeddings: np.ndarray
    gene_id_column: str
    gene_to_row_indices: Dict[str, List[int]]
    gene_text_by_gene: Dict[str, str]
    gene_query_info_by_gene: Dict[str, dict]
    embedding_dim: int


@dataclass
class VariantResources:
    variant_df: pd.DataFrame
    parent_a_genotype_col: str
    parent_b_genotype_col: str
    chrom_style: str


@dataclass(frozen=True)
class QtlRunSpec:
    qtl_id: str
    trait_name: str
    chromosome: str
    interval_start: int
    interval_end: int
    peak_position: int


@dataclass
class QtlRankingResult:
    ranking_df: pd.DataFrame
    top_gene_ids: List[str]
    trait_profile_keywords: List[str]


def build_default_settings(root_dir: Optional[Path] = None) -> PipelineSettings:
    """Create portable project-relative paths and API settings."""
    root_dir = Path(root_dir or Path(__file__).resolve().parent)
    data_dir = root_dir / "data"
    output_dir = root_dir / "output"
    cache_dir = root_dir / "cache"

    api_key = os.getenv("NCBI_API_KEY", "").strip() or None
    requests_per_second = 10.0 if api_key else 3.0

    path_settings = PathSettings(
        root_dir=root_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        variant_tsv=data_dir / "variants.tsv",
        rapdb_tsv=data_dir / "rapdb.tsv",
        gff3_path=data_dir / "IRGSP-1.0_locus.gff",
        run_workbook=data_dir / "qtl_runs.xlsx",
        run_sheet="Runs",
        output_workbook=output_dir / "results.xlsx",
        raw_candidates_tsv=output_dir / "all_candidates_raw.tsv",
        pubmed_cache_dir=cache_dir / "pubmed",
        embedding_cache_dir=cache_dir / "embedding",
    )
    api_settings = ApiSettings(
        ncbi_email=os.getenv("NCBI_EMAIL", "").strip(),
        ncbi_api_key=api_key,
        ncbi_requests_per_second=requests_per_second,
    )
    return PipelineSettings(paths=path_settings, api=api_settings)


DEFAULT_SETTINGS = build_default_settings()


# ============================================================================
# 2. Shared constants and low-level utilities
# ============================================================================

IMPACT_PRIORITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}
REQUIRED_RUN_COLUMNS = ["QTL", "TRAIT", "CHR", "START", "END", "PEAK"]
QTL_GROUP_COLUMNS = ["QTL", "TRAIT", "CHR", "START", "END", "PEAK"]
QTL_GENE_GROUP_COLUMNS = QTL_GROUP_COLUMNS + ["GeneID"]


STOP_WORDS = {
    "protein", "proteins", "binding", "domain", "family", "putative", "predicted",
    "unknown", "uncharacterized", "hypothetical", "conserved", "expressed", "isoform", "fragment",
    "factor", "like", "related", "homolog", "subunit", "complex", "component", "member",
    "containing", "contains", "similar", "partial", "novel",
    "gene", "genes", "locus", "transcript", "transcripts", "mrna", "cdna", "genome", "genomic", "chromosome",
    "plant", "plants", "rice", "oryza", "sativa",
    "retrotransposon", "transposon", "transposable", "element", "gag", "pol", "integrase", "reverse", "transcriptase",
    "promoter", "sequence", "region", "signal", "peptide", "chloroplast", "mitochondrial",
    "tata", "tat", "bacterial", "eukaryotic",
}

GENERIC_STOP = set("""
protein proteins gene genes family domain member like related homolog
putative predicted unknown hypothetical uncharacterized similar contains containing
factor binding subunit complex component response level
transport transporter ligase kinase phosphatase synthase enzyme
plant plants rice oryza sativa oryzabase cgsnl rap rapdb synonym synonyms symbol name
description comment go interpro literature pmidm pmid
subcellular location transcript evidence orf curation
""".split())

DESC_BAD = set("""
protein proteins gene genes family domain member
putative predicted unknown hypothetical uncharacterized
similar contains containing
factor binding subunit complex component
enzyme activity process response
""".split())

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]

def _to01(x, x_max=10.0) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x / float(x_max)))

def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def detect_chrom_style(df, chrom_col="CHROM"):
    vals = df[chrom_col].astype(str).dropna().unique()[:300]
    return "chr" if any(v.lower().startswith("chr") for v in vals) else "num"

def normalize_chr(x, style):
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    n = int(m.group(1))
    return f"chr{n:02d}" if style == "chr" else str(n)

def trait_match_score(text: str, keywords: List[str]) -> int:
    if not text:
        return 0
    t = text.lower()
    score = 0
    for k in (keywords or [])[:30]:
        kk = str(k).lower().strip()
        if kk and kk in t:
            score += 1
    return score

def _extract_pmids(text: str) -> List[str]:
    if not text:
        return []
    return sorted(set(re.findall(r"\b\d{5,9}\b", str(text))))

def _uniq_keep(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        x = _norm_space(x)
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _safe_sheet_name(prefix: str, qtl: str) -> str:
    q = re.sub(r"[\\/*?:\[\]]", "_", str(qtl or "").strip())
    q = q[:25] if q else "NA"
    return f"{prefix}_{q}"[:31]

# ============================================================================
# 3. RAP-DB annotation parsing and gene-query construction
# ============================================================================

def detect_geneid_col(rap: pd.DataFrame) -> str:
    if "Locus_ID" in rap.columns:
        return "Locus_ID"
    for c in rap.columns:
        if c.lower() in ["geneid", "gene_id", "locus", "locus_id", "rapdb", "rapdb_id"]:
            return c
    return rap.columns[0]

def _build_rapdb_text_column(rap: pd.DataFrame) -> pd.Series:
    text_cols = [
        "Description",
        "RAP-DB Gene Symbol Synonym(s)",
        "RAP-DB Gene Name Synonym(s)",
        "CGSNL Gene Symbol",
        "CGSNL Gene Name",
        "Oryzabase Gene Symbol Synonym(s)",
        "Oryzabase Gene Name Synonym(s)",
        "Oryzabase Trait Gene ID",
        "InterPro",
        "GO",
        "Comment",
        "Literature PMID",
    ]
    for c in text_cols:
        if c not in rap.columns:
            rap[c] = ""
    return rap[text_cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()

def build_gene_to_rows(rap: pd.DataFrame, gene_col: str) -> Dict[str, List[int]]:
    gene_to_rows: Dict[str, List[int]] = {}
    for idx, gid in enumerate(rap[gene_col].astype(str).fillna("")):
        gid = gid.strip()
        if not gid or gid.lower() == "nan":
            continue
        gene_to_rows.setdefault(gid, []).append(idx)
    return gene_to_rows

def build_gene_text_map(rap: pd.DataFrame, gene_col: str) -> Dict[str, str]:
    tmp = rap[[gene_col, "TEXT"]].copy()
    tmp[gene_col] = tmp[gene_col].astype(str).str.strip()
    tmp["TEXT"] = tmp["TEXT"].fillna("").astype(str)
    tmp = tmp[tmp[gene_col].ne("")]
    return (
        tmp.groupby(gene_col, dropna=False)["TEXT"]
        .apply(lambda s: " | ".join([x for x in s.tolist() if x]))
        .to_dict()
    )

def build_gene_alias_map(rap_tsv: Path) -> dict:
    rap = pd.read_csv(rap_tsv, sep="\t", dtype=str, low_memory=False)

    if "Locus_ID" in rap.columns:
        gene_col = "Locus_ID"
    else:
        gene_col = rap.columns[0]

    def _norm_val(v: str, cap: int) -> str:
        v = "" if v is None else str(v)
        if v.lower() == "nan":
            return ""
        v = v.replace("\r\n", "\n").replace("\r", "\n")
        v = v.replace("\n", " \\n ")
        v = re.sub(r"\s+", " ", v).strip()
        if len(v) > cap:
            v = v[:cap] + "…"
        return v

    SYMBOL_COLS = [
        ("CGSNL_SYM", "CGSNL Gene Symbol"),
        ("RAP_SYM", "RAP-DB Gene Symbol Synonym(s)"),
        ("ORY_SYM", "Oryzabase Gene Symbol Synonym(s)"),
    ]
    DESC_COLS = [
        ("DESC", "Description"),
        ("CGSNL_NM", "CGSNL Gene Name"),
        ("RAP_NM", "RAP-DB Gene Name Synonym(s)"),
        ("ORY_NM", "Oryzabase Gene Name Synonym(s)"),
        ("GO", "GO"),
        ("IPR", "InterPro"),
        ("CMT", "Comment"),
        ("PMID", "Literature PMID"),
        ("ORY_TRAIT", "Oryzabase Trait Gene ID"),
        ("LOC", "Subcellular location"),
    ]

    for _, c in SYMBOL_COLS + DESC_COLS:
        if c not in rap.columns:
            rap[c] = ""

    out: Dict[str, dict] = {}
    for _, row in rap.iterrows():
        gid = str(row.get(gene_col, "")).strip()
        if not gid or gid.lower() == "nan":
            continue

        sym_parts = []
        for lab, c in SYMBOL_COLS:
            v = _norm_val(row.get(c, ""), cap=220)
            if v:
                sym_parts.append(f"{lab}={v}")

        desc_parts = []
        for lab, c in DESC_COLS:
            v = _norm_val(row.get(c, ""), cap=260)
            if v:
                desc_parts.append(f"{lab}={v}")

        sym_text = " | ".join(sym_parts)[:900]
        desc_text = " | ".join(desc_parts)[:1400]
        all_text = " | ".join([t for t in [sym_text, desc_text] if t])[:1800]

        out[gid] = {
            "SYMBOL_TEXT": sym_text,
            "DESC_TEXT": desc_text,
            "ALL_TEXT": all_text,
        }
    return out

def _strip_os_prefix_symbol(sym: str) -> Optional[str]:
    s = _norm_space(sym)
    if not s:
        return None
    if s.lower().startswith("os") and len(s) > 2:
        t = s[2:]
        if re.search(r"\d", t) and len(t) >= 3:
            return t
    return None

def is_symbol_like(tok: str) -> bool:
    t = (tok or "").strip()
    if len(t) < 2 or len(t) > 15:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*", t):
        return False
    tl = t.lower()
    if tl in GENERIC_STOP:
        return False
    if sum(ch.isalpha() for ch in t) < 2:
        return False
    if re.fullmatch(r"[A-Z]{4,}", t):
        return False
    if tl in {"oryzabase", "cgsnl", "rapdb", "synonym", "symbol", "name"}:
        return False
    return True

def build_gene_query_info_map(rap: pd.DataFrame, gene_col: str) -> Dict[str, dict]:
    """
    Build gene-specific anchors for the two PubMed retrieval layers.

    - DirectSymbols: rice gene symbols used as written.
    - ExactPhrases: specific phrases derived from gene names/descriptions.
    - OrthologSymbols: symbols with the rice-specific ``Os`` prefix removed.
    """
    info = {}

    cols_needed = [
        gene_col,
        "RAP-DB Gene Symbol Synonym(s)",
        "CGSNL Gene Symbol",
        "Oryzabase Gene Symbol Synonym(s)",
        "Description",
        "RAP-DB Gene Name Synonym(s)",
        "CGSNL Gene Name",
        "Oryzabase Gene Name Synonym(s)",
        "Literature PMID",
    ]
    for c in cols_needed:
        if c not in rap.columns:
            rap[c] = ""

    grouped = rap.groupby(gene_col, dropna=False)

    for gid, sub in grouped:
        gid = str(gid).strip()
        if not gid or gid.lower() == "nan":
            continue

        symbol_fields = [
            "CGSNL Gene Symbol",
            "RAP-DB Gene Symbol Synonym(s)",
            "Oryzabase Gene Symbol Synonym(s)",
        ]
        name_fields = [
            "Description",
            "CGSNL Gene Name",
            "RAP-DB Gene Name Synonym(s)",
            "Oryzabase Gene Name Synonym(s)",
        ]

        direct_symbols: List[str] = []
        exact_phrases: List[str] = []
        curated_pmids: List[str] = []

        for c in symbol_fields:
            for raw in sub[c].fillna("").astype(str).tolist():
                parts = re.split(r"[;,/|\n]+", raw)
                for p in parts:
                    p = _norm_space(p)
                    if not p or p.lower() == "nan":
                        continue
                    if len(p) > 20:
                        continue
                    if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*", p):
                        direct_symbols.append(p)

        for c in name_fields:
            for raw in sub[c].fillna("").astype(str).tolist():
                s = _norm_space(raw)
                if not s or s.lower() == "nan":
                    continue
                exact_phrases.append(s)

        for raw in sub["Literature PMID"].fillna("").astype(str).tolist():
            curated_pmids.extend(_extract_pmids(raw))

        direct_symbols = _uniq_keep(direct_symbols)
        exact_phrases = _uniq_keep(exact_phrases)
        curated_pmids = _uniq_keep(curated_pmids)

        ortholog_symbols = _uniq_keep([
            x for x in [_strip_os_prefix_symbol(s) for s in direct_symbols] if x
        ])


        exact_keep = []
        for p in exact_phrases:
            pl = p.lower()
            if len(p) < 5:
                continue
            if pl in DESC_BAD:
                continue
            if re.search(r"\b\d+\b", p) or any(ds.lower() in pl for ds in direct_symbols[:3]):
                exact_keep.append(p)
        exact_keep = _uniq_keep(exact_keep)[:4]

        info[gid] = {
            "GeneID": gid,
            "DirectSymbols": direct_symbols[:6],
            "ExactPhrases": exact_keep[:4],
            "CuratedPMIDs": curated_pmids,
            "OrthologSymbols": ortholog_symbols[:4],
        }

    return info

def summarize_curated_pmid_retrieval(curated_pmids: List[str], records: List[dict]) -> Dict[str, object]:
    curated = sorted({str(x).strip() for x in (curated_pmids or []) if str(x).strip()})
    retrieved = []
    seen = set()
    for rec in (records or []):
        pmid = str((rec or {}).get("pmid", "")).strip()
        if pmid and pmid not in seen:
            retrieved.append(pmid)
            seen.add(pmid)

    matched = sorted(set(curated) & set(retrieved))
    total = len(curated)
    matched_n = len(matched)
    recall = float(matched_n / total) if total else np.nan

    return {
        "CuratedPMIDTotal": int(total),
        "RetrievedPMIDCount": int(len(retrieved)),
        "RetrievedCuratedPMIDCount": int(matched_n),
        "RetrievedCuratedPMIDs": ";".join(matched),
        "CuratedPMIDRecall": float(recall) if total else np.nan,
    }

def make_direct_rice_term(gid: str, qinfo: dict) -> Tuple[str, List[str]]:
    anchors = []
    seen = set()

    def _add(x):
        x = _norm_space(x)
        if not x:
            return
        if x not in seen:
            anchors.append(x)
            seen.add(x)

    _add(gid)
    for s in (qinfo.get("DirectSymbols") or [])[:3]:
        if is_symbol_like(s):
            _add(s)
    for p in (qinfo.get("ExactPhrases") or [])[:2]:
        if len(p) >= 5:
            _add(p)

    gene_or = " OR ".join([f'"{x}"[Title/Abstract]' for x in anchors])
    term = (
        f"({gene_or}) AND "
        f"(\"Oryza sativa\"[MeSH Terms] OR rice[Title/Abstract] OR "
        f"\"Oryza sativa\"[Title/Abstract] OR Oryza[Title/Abstract])"
    )
    return term, anchors

def make_ortholog_term(qinfo: dict) -> Tuple[str, List[str]]:
    syms = [s for s in (qinfo.get("OrthologSymbols") or []) if is_symbol_like(s)]
    syms = _uniq_keep(syms)[:3]
    if not syms:
        return "", []

    sym_or = " OR ".join([f'"{s}"[Title/Abstract]' for s in syms])
    plant_filter = (
        "\"Plants\"[MeSH Terms] OR plant[Title/Abstract] OR plants[Title/Abstract] OR "
        "Arabidopsis[Title/Abstract] OR Zea[Title/Abstract] OR maize[Title/Abstract] OR "
        "Triticum[Title/Abstract] OR wheat[Title/Abstract] OR Hordeum[Title/Abstract] OR "
        "barley[Title/Abstract] OR Sorghum[Title/Abstract] OR Oryza[Title/Abstract] OR rice[Title/Abstract]"
    )
    return f"({sym_or}) AND ({plant_filter})", syms

# ============================================================================
# 4. Functional relevance and article-level scoring helpers
# ============================================================================

def functional_score_from_map(
    genes_df: pd.DataFrame,
    trait_keywords: List[str],
    gene_text_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Fraction of trait-profile keywords found in each gene annotation.
    """
    kws = []
    seen = set()
    for k in trait_keywords or []:
        kk = str(k).strip().lower()
        if not kk:
            continue
        if kk in STOP_WORDS or kk in GENERIC_STOP:
            continue
        if kk not in seen:
            kws.append(kk)
            seen.add(kk)

    rows = []
    for gid in genes_df["GeneID"].astype(str).tolist():
        txt = str(gene_text_map.get(gid, "")).lower()
        hits = []
        if txt and kws:
            for kw in kws:
                if kw in txt:
                    hits.append(kw)

        n_hits = len(hits)
        fscore = min(1.0, n_hits / 8.0)
        show_hits = ";".join(hits[:8])
        rows.append((gid, float(fscore), show_hits))

    return pd.DataFrame(rows, columns=["GeneID", "FunctionalKeywordHit", "FuncMatches"])

def _count_pmids(text: str) -> int:
    if not text:
        return 0
    return len(set(re.findall(r"\b\d{5,9}\b", str(text))))

def mesh_species_boost(mesh_terms: List[str]) -> float:
    if not mesh_terms:
        return 0.0
    s = " | ".join([str(m).lower() for m in mesh_terms])
    if "oryza sativa" in s:
        return 1.0
    if re.search(r"\boryza\b", s):
        return 1.0
    if "plants" in s or "plant" in s:
        return 0.5
    return 0.0

def pubtype_weight(pub_types: List[str]) -> float:
    if not pub_types:
        return 0.7
    pt = " | ".join([str(p).lower() for p in pub_types])
    if "editorial" in pt or "comment" in pt or "news" in pt or "letter" in pt:
        return 0.2
    if "review" in pt or "systematic review" in pt or "meta-analysis" in pt:
        return 0.6
    return 1.0

def has_plant_signal(title: str, abstract: str, mesh_terms: List[str]) -> bool:
    txt = ((title or "") + " " + (abstract or "")).lower()
    mesh = " | ".join([str(m).lower() for m in (mesh_terms or [])])
    if "plants" in mesh or "plant" in mesh:
        return True
    for term in ["arabidopsis", "zea", "maize", "triticum", "wheat", "hordeum", "barley", "sorghum", "oryza", "rice", "plant", "plants"]:
        if term in txt:
            return True
    return False


# ============================================================================
# 5. Trait profiles, curated evidence, and embedding caches
# ============================================================================

def get_trait_profile(
    trait_name: str,
    rapdb_df: pd.DataFrame,
    rapdb_texts: List[str],
    rapdb_doc_embeddings: np.ndarray,
    embedder,
    settings: PipelineSettings,
) -> Tuple[List[str], dict]:
    """
    Build a trait keyword profile with the shared RAPDB annotations.
    """
    profile_kwargs = dict(
        trait=trait_name,
        rapdb_tsv=str(settings.paths.rapdb_tsv),
        top_k=60,
        min_count=3,
        use_semantic_seed=True,
        top_k_seed=settings.embedding.top_k_seed,
        embed_model=settings.embedding.model_name,
        embed_cache_dir=str(settings.paths.embedding_cache_dir),
    )
    try:
        return rapdb_global_profile(
            **profile_kwargs,
            rap_df=rapdb_df,
            rap_texts=rapdb_texts,
            doc_emb=rapdb_doc_embeddings,
            embedder=embedder,
        )
    except TypeError:
        return rapdb_global_profile(**profile_kwargs)


def build_curated_evidence_map(
    rapdb_df: pd.DataFrame,
    gene_id_column: str,
    trait_keywords: Optional[List[str]] = None,
    weight_settings: WeightSettings = DEFAULT_SETTINGS.weights,
) -> Dict[str, dict]:
    """
    Build gene-level curated evidence from RAPDB and Oryzabase-related columns.
    """
    working_df = rapdb_df.copy()

    pmid_column = "Literature PMID" if "Literature PMID" in working_df.columns else None
    oryzabase_trait_column = "Oryzabase Trait Gene ID" if "Oryzabase Trait Gene ID" in working_df.columns else None

    text_columns = [
        "Description",
        "CGSNL Gene Symbol",
        "CGSNL Gene Name",
        "RAP-DB Gene Symbol Synonym(s)",
        "RAP-DB Gene Name Synonym(s)",
        "Oryzabase Gene Symbol Synonym(s)",
        "Oryzabase Gene Name Synonym(s)",
        "Comment",
        "GO",
        "InterPro",
        "Oryzabase Trait Gene ID",
        "Literature PMID",
    ]
    for column_name in text_columns:
        if column_name not in working_df.columns:
            working_df[column_name] = ""

    working_df["_CURATED_TEXT"] = (
        working_df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )

    normalized_keywords: List[str] = []
    seen_keywords: set[str] = set()
    for keyword in (trait_keywords or []):
        normalized_keyword = str(keyword).strip().lower()
        if not normalized_keyword or normalized_keyword in STOP_WORDS or normalized_keyword in seen_keywords:
            continue
        normalized_keywords.append(normalized_keyword)
        seen_keywords.add(normalized_keyword)

    curated_by_gene: Dict[str, dict] = {}
    for gene_id, gene_rows in working_df.groupby(gene_id_column, dropna=False):
        normalized_gene_id = str(gene_id).strip()
        if not normalized_gene_id or normalized_gene_id.lower() == "nan":
            continue

        pmid_text = (
            " | ".join(gene_rows[pmid_column].fillna("").astype(str).tolist())
            if pmid_column else ""
        )
        oryzabase_text = (
            " | ".join(gene_rows[oryzabase_trait_column].fillna("").astype(str).tolist())
            if oryzabase_trait_column else ""
        )
        curated_text = " | ".join(gene_rows["_CURATED_TEXT"].fillna("").astype(str).tolist())

        pmid_count = _count_pmids(pmid_text)
        has_pmid = float(pmid_count > 0)
        has_oryzabase_trait = float(bool(str(oryzabase_text).strip()) and str(oryzabase_text).strip().lower() != "nan")

        matched_keywords: List[str] = []
        if curated_text and normalized_keywords:
            for keyword in normalized_keywords:
                if keyword in curated_text:
                    matched_keywords.append(keyword)

        trait_hit_count = len(matched_keywords)
        trait_hit_score = min(1.0, trait_hit_count / 5.0)

        curated_score = (
            weight_settings.curated_pmid * has_pmid
            + weight_settings.curated_oryzabase * has_oryzabase_trait
            + weight_settings.curated_text_hit * trait_hit_score
        )

        curated_by_gene[normalized_gene_id] = {
            "RAP_PMID_Count": int(pmid_count),
            "RAP_HasPMID": has_pmid,
            "RAP_HasOryzabaseTrait": has_oryzabase_trait,
            "RAP_TraitTextHitsN": int(trait_hit_count),
            "RAP_TraitTextHits": ";".join(matched_keywords[:10]),
            "CuratedEvidenceScore": float(curated_score),
        }
    return curated_by_gene


def load_or_build_doc_embeddings(
    rapdb_texts: List[str],
    settings: PipelineSettings,
    embedder,
) -> np.ndarray:
    cache_dir = settings.paths.embedding_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not rapdb_texts:
        return np.zeros((0, settings.embedding.default_embedding_dim), dtype=np.float32)

    cache_key = _hash_key(
        settings.embedding.model_name,
        str(len(rapdb_texts)),
        rapdb_texts[0][:200],
        rapdb_texts[-1][:200],
    )
    cache_path = cache_dir / f"rapdb_docemb_{cache_key}.npy"

    if cache_path.exists():
        return np.load(str(cache_path))

    doc_embeddings = embedder.encode(
        rapdb_texts,
        batch_size=settings.embedding.document_batch_size,
        show_progress_bar=True,
    )
    doc_embeddings = _l2_normalize(np.array(doc_embeddings, dtype=np.float32))
    np.save(str(cache_path), doc_embeddings)
    return doc_embeddings


def build_text_embeddings_batch(
    embedder,
    texts: List[str],
    settings: PipelineSettings,
    batch_size: Optional[int] = None,
    embedding_dim: Optional[int] = None,
) -> np.ndarray:
    batch_size = batch_size or settings.embedding.text_batch_size
    embedding_dim = embedding_dim or settings.embedding.default_embedding_dim

    if not texts:
        return np.zeros((0, embedding_dim), dtype=np.float32)

    encoded = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return _l2_normalize(np.array(encoded, dtype=np.float32))


def text_vs_trait_similarity_batch(text_embeddings: np.ndarray, trait_vector: np.ndarray) -> List[float]:
    """Return 0-1 scaled cosine similarities between text embeddings and a trait vector."""
    if text_embeddings.size == 0:
        return []

    similarity = (text_embeddings @ trait_vector.T).ravel()
    similarity_01 = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)
    return [float(value) for value in similarity_01]


def literature_score_from_precomputed(
    *,
    title: str,
    abstract: str,
    profile_keywords: List[str],
    embed01: float,
    mesh_terms: Optional[List[str]] = None,
    pub_types: Optional[List[str]] = None,
    rice_mode: bool = False,
    weight_settings: WeightSettings = DEFAULT_SETTINGS.weights,
) -> Dict[str, float]:
    combined_text = ((title or "") + " " + (abstract or "")).strip()

    keyword_hits = trait_match_score(combined_text, profile_keywords)
    keyword_score = float(_to01(keyword_hits, x_max=8.0))

    mesh_list = mesh_terms or []
    pubtype_list = pub_types or []

    mesh_score = float(mesh_species_boost(mesh_list))
    pubtype_score = float(pubtype_weight(pubtype_list))

    trait_relevance = (
        weight_settings.literature_embed * embed01
        + weight_settings.literature_keyword * keyword_score
        + weight_settings.literature_mesh * mesh_score
        + weight_settings.literature_pubtype * pubtype_score
    )

    return {
        "KeywordHits": float(keyword_hits),
        "KeywordScore": float(keyword_score),
        "EmbedRel": float(embed01),
        "MeshBoost": float(mesh_score),
        "PubTypeW": float(pubtype_score),
        "TraitRelevance": float(trait_relevance),
    }


# ============================================================================
# 6. Reusable trait-level caches
# ============================================================================

class TraitCaches:
    """
    Cache repeated trait-level computations.
    """
    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.profile_cache: Dict[str, List[str]] = {}
        self.trait_vec_cache: Dict[str, np.ndarray] = {}
        self.curated_cache: Dict[str, Dict[str, dict]] = {}

    def get_profile(
        self,
        trait_name: str,
        rapdb_df: pd.DataFrame,
        rapdb_texts: List[str],
        rapdb_doc_embeddings: np.ndarray,
        embedder,
    ) -> List[str]:
        trait_key = str(trait_name).strip().lower()
        if trait_key in self.profile_cache:
            return self.profile_cache[trait_key]

        profile_keywords, profile_meta = get_trait_profile(
            trait_name=trait_name,
            rapdb_df=rapdb_df,
            rapdb_texts=rapdb_texts,
            rapdb_doc_embeddings=rapdb_doc_embeddings,
            embedder=embedder,
            settings=self.settings,
        )
        print("PROFILE_META:", profile_meta)
        self.profile_cache[trait_key] = profile_keywords
        return profile_keywords

    def get_trait_vec(self, trait_name: str, embedder) -> np.ndarray:
        trait_key = str(trait_name).strip().lower()
        if trait_key in self.trait_vec_cache:
            return self.trait_vec_cache[trait_key]

        trait_vector = embedder.encode([trait_name], batch_size=1, show_progress_bar=False)
        trait_vector = _l2_normalize(np.array(trait_vector, dtype=np.float32))
        self.trait_vec_cache[trait_key] = trait_vector
        return trait_vector

    def get_curated_map(
        self,
        trait_name: str,
        rapdb_df: pd.DataFrame,
        gene_id_column: str,
    ) -> Dict[str, dict]:
        trait_key = str(trait_name).strip().lower()
        if trait_key in self.curated_cache:
            return self.curated_cache[trait_key]

        profile_keywords = self.profile_cache.get(trait_key, [])
        curated_map = build_curated_evidence_map(
            rapdb_df=rapdb_df,
            gene_id_column=gene_id_column,
            trait_keywords=profile_keywords,
            weight_settings=self.settings.weights,
        )
        self.curated_cache[trait_key] = curated_map
        return curated_map


# ============================================================================
# 7. Input validation and resource loading
# ============================================================================

def ensure_pipeline_io(settings: PipelineSettings) -> None:
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.pubmed_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    for required_path in [
        settings.paths.variant_tsv,
        settings.paths.rapdb_tsv,
        settings.paths.gff3_path,
        settings.paths.run_workbook,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing input: {required_path}")

    if not settings.api.ncbi_email:
        print("WARNING: NCBI_EMAIL is empty; NCBI recommends providing a contact email.")


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required. Install it with: pip install sentence-transformers") from exc
    return SentenceTransformer(model_name)


def load_rapdb_resources(settings: PipelineSettings, embedder) -> RapdbResources:
    print("Loading RAPDB...")
    rapdb_df = pd.read_csv(settings.paths.rapdb_tsv, sep="\t", dtype=str, low_memory=False)
    rapdb_df["TEXT"] = _build_rapdb_text_column(rapdb_df)
    rapdb_texts = rapdb_df["TEXT"].fillna("").astype(str).tolist()

    print("Building/Loading shared RAPDB doc embeddings...")
    rapdb_doc_embeddings = load_or_build_doc_embeddings(
        rapdb_texts=rapdb_texts,
        settings=settings,
        embedder=embedder,
    )

    gene_id_column = detect_geneid_col(rapdb_df)
    gene_to_row_indices = build_gene_to_rows(rapdb_df, gene_id_column)
    gene_text_by_gene = build_gene_text_map(rapdb_df, gene_id_column)
    gene_query_info_by_gene = build_gene_query_info_map(rapdb_df, gene_id_column)


    embedding_dim = (
        int(rapdb_doc_embeddings.shape[1])
        if len(rapdb_doc_embeddings.shape) == 2
        else settings.embedding.default_embedding_dim
    )

    return RapdbResources(
        annotation_df=rapdb_df,
        annotation_texts=rapdb_texts,
        doc_embeddings=rapdb_doc_embeddings,
        gene_id_column=gene_id_column,
        gene_to_row_indices=gene_to_row_indices,
        gene_text_by_gene=gene_text_by_gene,
        gene_query_info_by_gene=gene_query_info_by_gene,
        embedding_dim=embedding_dim,
    )


def load_variant_resources(settings: PipelineSettings) -> VariantResources:
    variant_df, genotype_columns = load_variants_tsv(str(settings.paths.variant_tsv))
    parent_a_genotype_col, parent_b_genotype_col = genotype_columns[0], genotype_columns[1]
    chrom_style = detect_chrom_style(variant_df)
    print("Detected variants CHROM style:", chrom_style)

    return VariantResources(
        variant_df=variant_df,
        parent_a_genotype_col=parent_a_genotype_col,
        parent_b_genotype_col=parent_b_genotype_col,
        chrom_style=chrom_style,
    )


def load_qtl_run_specs(settings: PipelineSettings, chrom_style: str) -> List[QtlRunSpec]:
    run_df = pd.read_excel(settings.paths.run_workbook, sheet_name=settings.paths.run_sheet, dtype=str)
    missing_columns = [column_name for column_name in REQUIRED_RUN_COLUMNS if column_name not in run_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in {settings.paths.run_workbook}/{settings.paths.run_sheet}: {missing_columns}"
        )

    for numeric_column in ["START", "END", "PEAK"]:
        run_df[numeric_column] = pd.to_numeric(run_df[numeric_column], errors="raise").astype(int)

    run_specs: List[QtlRunSpec] = []
    for _, row in run_df.iterrows():
        run_specs.append(
            QtlRunSpec(
                qtl_id=str(row["QTL"]).strip(),
                trait_name=str(row["TRAIT"]).strip(),
                chromosome=normalize_chr(row["CHR"], chrom_style),
                interval_start=int(row["START"]),
                interval_end=int(row["END"]),
                peak_position=int(row["PEAK"]),
            )
        )
    return run_specs


# ============================================================================
# 8. Variant filtering and initial gene ranking
# ============================================================================

def normalize_gene_ids(variant_gene_df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = variant_gene_df.copy()
    normalized_df["GeneID"] = (
        normalized_df["GeneID"].astype(str)
        .str.replace("gene:", "", regex=False)
        .str.replace(r"^GENE_", "", regex=True)
        .str.replace("-.*", "", regex=True)
        .str.replace(r"^Os(\d\d)t", lambda match: f"Os{match.group(1)}g", regex=True)
    )
    return normalized_df


def deduplicate_gene_variant_annotations(variant_gene_df: pd.DataFrame) -> pd.DataFrame:
    deduplicated_df = variant_gene_df.copy()
    deduplicated_df["ImpactPriority"] = (
        deduplicated_df.get("Impact", pd.Series(index=deduplicated_df.index, dtype=str))
        .astype(str)
        .str.upper()
        .map(IMPACT_PRIORITY)
        .fillna(0)
    )
    deduplicated_df = (
        deduplicated_df.sort_values(
            ["CHROM", "POS", "GeneID", "ImpactPriority"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(subset=["CHROM", "POS", "GeneID"], keep="first")
        .drop(columns=["ImpactPriority"])
        .reset_index(drop=True)
    )
    return deduplicated_df


def build_qtl_variant_gene_table(
    variant_df: pd.DataFrame,
    qtl_run: QtlRunSpec,
    parent_a_genotype_col: str,
    parent_b_genotype_col: str,
) -> pd.DataFrame:
    interval_variant_df = filter_by_interval(
        variant_df,
        qtl_run.chromosome,
        qtl_run.interval_start,
        qtl_run.interval_end,
    )
    fixed_difference_df = filter_fixed_difference_variants(
        interval_variant_df,
        parent_a_genotype_col,
        parent_b_genotype_col,
    )
    if fixed_difference_df.empty:
        return pd.DataFrame()

    exploded_annotation_df = explode_ann(fixed_difference_df)
    if exploded_annotation_df.empty:
        return pd.DataFrame()

    normalized_gene_df = normalize_gene_ids(exploded_annotation_df)
    return deduplicate_gene_variant_annotations(normalized_gene_df)


def compute_gene_distance_scores(variant_gene_df: pd.DataFrame, qtl_run: QtlRunSpec) -> pd.DataFrame:
    interval_length = max(1, int(qtl_run.interval_end) - int(qtl_run.interval_start))
    distance_df = variant_gene_df.copy()
    distance_df["PeakDistance"] = (distance_df["POS"].astype(int) - int(qtl_run.peak_position)).abs()

    gene_distance_df = distance_df.groupby("GeneID", as_index=False)["PeakDistance"].min()
    gene_distance_df["DistanceScore"] = (1 - gene_distance_df["PeakDistance"] / interval_length).clip(0, 1)
    return gene_distance_df[["GeneID", "DistanceScore"]]


def compute_gene_embedding_scores(
    gene_ids: List[str],
    trait_vector: np.ndarray,
    rapdb_doc_embeddings: np.ndarray,
    gene_to_row_indices: Dict[str, List[int]],
) -> pd.DataFrame:
    row_similarity_scores = (rapdb_doc_embeddings @ trait_vector.T).ravel()
    embedding_rows: List[Tuple[str, float]] = []

    for gene_id in gene_ids:
        row_indices = gene_to_row_indices.get(gene_id, [])
        embedding_score = float(np.max(row_similarity_scores[row_indices])) if row_indices else 0.0
        embedding_rows.append((gene_id, embedding_score))

    return pd.DataFrame(embedding_rows, columns=["GeneID", "EmbedScore"])


def build_curated_metadata_df(curated_metadata_by_gene: Dict[str, dict]) -> pd.DataFrame:
    curated_metadata_df = pd.DataFrame(
        [{"GeneID": gene_id, **metadata} for gene_id, metadata in curated_metadata_by_gene.items()]
    )
    if curated_metadata_df.empty:
        curated_metadata_df = pd.DataFrame(
            columns=[
                "GeneID",
                "RAP_PMID_Count",
                "RAP_HasPMID",
                "RAP_HasOryzabaseTrait",
                "RAP_TraitTextHitsN",
                "RAP_TraitTextHits",
                "CuratedEvidenceScore",
            ]
        )
    return curated_metadata_df


def fill_curated_metadata_defaults(ranking_df: pd.DataFrame) -> pd.DataFrame:
    filled_df = ranking_df.copy()
    for column_name, default_value in [
        ("RAP_PMID_Count", 0),
        ("RAP_HasPMID", 0.0),
        ("RAP_HasOryzabaseTrait", 0.0),
        ("RAP_TraitTextHitsN", 0),
    ]:
        filled_df[column_name] = pd.to_numeric(
            filled_df.get(column_name, default_value), errors="coerce"
        ).fillna(default_value)

    filled_df["RAP_TraitTextHits"] = filled_df.get("RAP_TraitTextHits", "").fillna("")
    return filled_df.drop(columns=["CuratedEvidenceScore"], errors="ignore")


def add_trait_relatedness_columns(
    ranking_df: pd.DataFrame,
    weight_settings: WeightSettings,
) -> pd.DataFrame:
    scored_df = ranking_df.copy()
    scored_df["VariantScore"] = pd.to_numeric(scored_df["VariantScore"], errors="coerce").fillna(0.0).clip(0, 1)
    scored_df["DistanceScore"] = pd.to_numeric(scored_df["DistanceScore"], errors="coerce").fillna(0.0).clip(0, 1)
    scored_df["FunctionalKeywordHit"] = (
        pd.to_numeric(scored_df["FunctionalKeywordHit"], errors="coerce").fillna(0.0).clip(0, 1)
    )
    scored_df["EmbedScore"] = pd.to_numeric(scored_df["EmbedScore"], errors="coerce").fillna(0.0)
    scored_df["FunctionalSemanticSimilarity"] = np.clip(scored_df["EmbedScore"], 0.0, 1.0)
    scored_df["TraitRelatednessScore"] = (
        weight_settings.gene_functional_hit * scored_df["FunctionalKeywordHit"]
        + weight_settings.gene_embedding_similarity * scored_df["FunctionalSemanticSimilarity"]
    ).clip(0.0, 1.0)
    return scored_df


def add_base_priority_score(
    ranking_df: pd.DataFrame,
    weight_settings: WeightSettings,
) -> pd.DataFrame:
    base_score_df = ranking_df.copy()
    base_score_df["PriorityBaseScore"] = (
        weight_settings.base_variant * base_score_df["VariantScore"]
        + weight_settings.base_distance * base_score_df["DistanceScore"]
        + weight_settings.base_trait_relatedness * base_score_df["TraitRelatednessScore"]
    )
    return base_score_df


def add_qtl_metadata(ranking_df: pd.DataFrame, qtl_run: QtlRunSpec) -> pd.DataFrame:
    metadata_df = ranking_df.copy()
    metadata_df["QTL"] = qtl_run.qtl_id
    metadata_df["TRAIT"] = qtl_run.trait_name
    metadata_df["CHR"] = qtl_run.chromosome
    metadata_df["START"] = int(qtl_run.interval_start)
    metadata_df["END"] = int(qtl_run.interval_end)
    metadata_df["PEAK"] = int(qtl_run.peak_position)
    return metadata_df


def select_top_gene_ids_for_pubmed(
    ranking_df: pd.DataFrame,
    top_pubmed_genes: int,
) -> List[str]:
    return (
        ranking_df.sort_values("PriorityBaseScore", ascending=False)
        .head(top_pubmed_genes)["GeneID"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def run_one_qtl_rank(
    qtl_run: QtlRunSpec,
    variant_resources: VariantResources,
    rapdb_resources: RapdbResources,
    embedder,
    trait_caches: TraitCaches,
    settings: PipelineSettings,
) -> QtlRankingResult:
    variant_gene_df = build_qtl_variant_gene_table(
        variant_df=variant_resources.variant_df,
        qtl_run=qtl_run,
        parent_a_genotype_col=variant_resources.parent_a_genotype_col,
        parent_b_genotype_col=variant_resources.parent_b_genotype_col,
    )
    if variant_gene_df.empty:
        return QtlRankingResult(pd.DataFrame(), [], [])

    variant_score_df = score_variants_by_gene(
        variant_gene_df,
        config=settings.variant_scoring_config,
    )
    if variant_score_df.empty:
        return QtlRankingResult(pd.DataFrame(), [], [])

    gene_distance_df = compute_gene_distance_scores(variant_gene_df, qtl_run)
    rank_df = variant_score_df.merge(gene_distance_df, on="GeneID", how="left")
    rank_df["DistanceScore"] = rank_df["DistanceScore"].fillna(0)

    trait_profile_keywords = trait_caches.get_profile(
        trait_name=qtl_run.trait_name,
        rapdb_df=rapdb_resources.annotation_df,
        rapdb_texts=rapdb_resources.annotation_texts,
        rapdb_doc_embeddings=rapdb_resources.doc_embeddings,
        embedder=embedder,
    )
    curated_metadata_by_gene = trait_caches.get_curated_map(
        trait_name=qtl_run.trait_name,
        rapdb_df=rapdb_resources.annotation_df,
        gene_id_column=rapdb_resources.gene_id_column,
    )

    functional_score_df = functional_score_from_map(
        genes_df=variant_score_df[["GeneID"]],
        trait_keywords=trait_profile_keywords,
        gene_text_map=rapdb_resources.gene_text_by_gene,
    )
    rank_df = rank_df.merge(functional_score_df, on="GeneID", how="left")
    rank_df["FunctionalKeywordHit"] = rank_df["FunctionalKeywordHit"].fillna(0)
    rank_df["FuncMatches"] = rank_df["FuncMatches"].fillna("")

    trait_vector = trait_caches.get_trait_vec(qtl_run.trait_name, embedder)
    gene_embedding_df = compute_gene_embedding_scores(
        gene_ids=variant_score_df["GeneID"].astype(str).tolist(),
        trait_vector=trait_vector,
        rapdb_doc_embeddings=rapdb_resources.doc_embeddings,
        gene_to_row_indices=rapdb_resources.gene_to_row_indices,
    )
    rank_df = rank_df.merge(gene_embedding_df, on="GeneID", how="left")

    curated_metadata_df = build_curated_metadata_df(curated_metadata_by_gene)
    rank_df = rank_df.merge(curated_metadata_df, on="GeneID", how="left")
    rank_df = fill_curated_metadata_defaults(rank_df)
    rank_df = add_trait_relatedness_columns(rank_df, settings.weights)
    rank_df = add_base_priority_score(rank_df, settings.weights)
    rank_df = add_qtl_metadata(rank_df, qtl_run)

    top_gene_ids = select_top_gene_ids_for_pubmed(
        ranking_df=rank_df,
        top_pubmed_genes=settings.limits.top_pubmed_genes,
    )
    return QtlRankingResult(rank_df, top_gene_ids, trait_profile_keywords)


# ============================================================================
# 9. PubMed retrieval and article scoring
# ============================================================================

def empty_query_info(gene_id: str) -> Dict[str, Any]:
    return {
        "GeneID": gene_id,
        "DirectSymbols": [],
        "ExactPhrases": [],
        "CuratedPMIDs": [],
        "OrthologSymbols": [],
    }


def fetch_pubmed_payload(search_term: str, settings: PipelineSettings) -> Dict[str, Any]:
    if not search_term:
        return {"term": "", "records": [], "pmids": []}
    return pubmed_search_with_cache(
        term=search_term,
        retmax=settings.limits.pubmed_retmax,
        cache_dir=str(settings.paths.pubmed_cache_dir),
        email=settings.api.ncbi_email,
        api_key=settings.api.ncbi_api_key,
        rps=settings.api.ncbi_requests_per_second,
    )


def deduplicate_pubmed_records(*record_groups: List[dict]) -> List[dict]:
    merged_records: List[dict] = []
    seen_pmids: set[str] = set()

    for record_group in record_groups:
        for record in record_group:
            pmid = str(record.get("pmid", "")).strip()
            if pmid and pmid in seen_pmids:
                continue
            if pmid:
                seen_pmids.add(pmid)
            merged_records.append(record)
    return merged_records


def query_pubmed_layers_for_gene(
    gene_id: str,
    gene_query_info: Dict[str, Any],
    settings: PipelineSettings,
) -> Tuple[Dict[str, Dict[str, Any]], List[dict], Dict[str, Any], Dict[str, List[str]]]:
    direct_term, direct_anchors = make_direct_rice_term(gene_id, gene_query_info)
    direct_payload = fetch_pubmed_payload(direct_term, settings)
    direct_records = direct_payload.get("records") or []

    ortholog_term, ortholog_anchors = make_ortholog_term(gene_query_info)
    ortholog_payload = fetch_pubmed_payload(ortholog_term, settings) if ortholog_term else {"term": "", "records": [], "pmids": []}
    ortholog_records = ortholog_payload.get("records") or []

    merged_records: List[dict] = []
    seen_pmids: set[str] = set()

    for record in direct_records:
        record_copy = dict(record)
        record_copy["RetrievalLayer"] = "direct_rice"
        pmid = str(record_copy.get("pmid", "")).strip()
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        merged_records.append(record_copy)

    for record in ortholog_records:
        record_copy = dict(record)
        record_copy["RetrievalLayer"] = "ortholog"
        pmid = str(record_copy.get("pmid", "")).strip()
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        merged_records.append(record_copy)

    curated_pmids = gene_query_info.get("CuratedPMIDs") or []
    pmid_eval = summarize_curated_pmid_retrieval(curated_pmids, merged_records)

    query_payloads = {
        "direct": direct_payload,
        "ortholog": ortholog_payload,
    }
    query_anchors = {
        "direct": direct_anchors,
        "ortholog": ortholog_anchors,
    }
    return query_payloads, merged_records, pmid_eval, query_anchors


def build_gene_pmid_eval_row(qtl_run: QtlRunSpec, gene_id: str, pmid_eval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "QTL": qtl_run.qtl_id,
        "TRAIT": qtl_run.trait_name,
        "CHR": qtl_run.chromosome,
        "START": qtl_run.interval_start,
        "END": qtl_run.interval_end,
        "PEAK": qtl_run.peak_position,
        "GeneID": gene_id,
        **pmid_eval,
    }


def score_pubmed_records_for_gene(
    qtl_run: QtlRunSpec,
    gene_id: str,
    merged_records: List[dict],
    gene_query_info: Dict[str, Any],
    curated_pmids: List[str],
    query_payloads: Dict[str, Dict[str, Any]],
    pmid_eval: Dict[str, Any],
    trait_profile_keywords: List[str],
    trait_vector: np.ndarray,
    embedder,
    embedding_dim: int,
    settings: PipelineSettings,
) -> Tuple[List[dict], List[dict]]:
    texts = [
        ((record.get("title", "") or "") + " " + (record.get("abstract", "") or "")).strip()
        for record in merged_records
    ]
    text_embeddings = build_text_embeddings_batch(
        embedder=embedder,
        texts=texts,
        settings=settings,
        batch_size=settings.embedding.text_batch_size,
        embedding_dim=embedding_dim,
    )
    embed_scores = text_vs_trait_similarity_batch(text_embeddings, trait_vector) if texts else []

    scored_records: List[Tuple[float, dict, dict]] = []
    for record, embed01 in zip(merged_records, embed_scores):
        title = record.get("title", "") or ""
        abstract = record.get("abstract", "") or ""
        mesh_terms = record.get("mesh_terms", []) or []
        pub_types = record.get("pub_types", []) or []

        trait_score = literature_score_from_precomputed(
            title=title,
            abstract=abstract,
            profile_keywords=trait_profile_keywords,
            embed01=embed01,
            mesh_terms=mesh_terms,
            pub_types=pub_types,
            rice_mode=False,
            weight_settings=settings.weights,
        )

        enriched_record = {
            **record,
            "ArticleFinalScore": float(trait_score["TraitRelevance"]),
            "BestEvidenceLayer": record.get("RetrievalLayer", ""),
        }
        scored_records.append((trait_score["TraitRelevance"], enriched_record, trait_score))

    scored_records.sort(key=lambda item: (item[0], item[1].get("pmid", "")), reverse=True)

    evidence_rows: List[dict] = []
    layer_score_rows: List[dict] = []
    for _, enriched_record, trait_score in scored_records[: settings.limits.top_articles_per_gene]:
        pmid = str(enriched_record.get("pmid", "")).strip()
        retrieval_layer = enriched_record.get("RetrievalLayer", "")

        evidence_rows.append({
            "QTL": qtl_run.qtl_id,
            "TRAIT": qtl_run.trait_name,
            "CHR": qtl_run.chromosome,
            "START": qtl_run.interval_start,
            "END": qtl_run.interval_end,
            "PEAK": qtl_run.peak_position,
            "GeneID": gene_id,
            "RetrievalLayer": retrieval_layer,
            "CuratedPMIDQuery": "",
            "DirectRiceQuery": query_payloads["direct"].get("term", ""),
            "OrthologQuery": query_payloads["ortholog"].get("term", ""),
            "CuratedPMIDTotal": pmid_eval["CuratedPMIDTotal"],
            "RetrievedPMIDCount": pmid_eval["RetrievedPMIDCount"],
            "RetrievedCuratedPMIDCount": pmid_eval["RetrievedCuratedPMIDCount"],
            "RetrievedCuratedPMIDs": pmid_eval["RetrievedCuratedPMIDs"],
            "CuratedPMIDRecall": pmid_eval["CuratedPMIDRecall"],
            "PMID": pmid,
            "Title": enriched_record.get("title", "") or "",
            "Abstract": enriched_record.get("abstract", "") or "",
            "Journal": enriched_record.get("source", "") or "",
            "PubDate": enriched_record.get("pubdate", "") or "",
            "Authors": enriched_record.get("authors", "") or "",
            "Language": enriched_record.get("language", "") or "",
            "TraitScore": int(trait_score["KeywordHits"]),
            "KeywordScore": trait_score["KeywordScore"],
            "EmbedRel": trait_score["EmbedRel"],
            "MeshBoost": trait_score["MeshBoost"],
            "PubTypeW": trait_score["PubTypeW"],
            "TraitRelevance": trait_score["TraitRelevance"],
            "BestEvidenceLayer": retrieval_layer,
            "ArticleFinalScore": trait_score["TraitRelevance"],
            "PubTypes": ";".join(enriched_record.get("pub_types", []) or []),
            "MeSH": ";".join(enriched_record.get("mesh_terms", []) or []),
        })

        layer_score_rows.append({
            "QTL": qtl_run.qtl_id,
            "TRAIT": qtl_run.trait_name,
            "CHR": qtl_run.chromosome,
            "START": qtl_run.interval_start,
            "END": qtl_run.interval_end,
            "PEAK": qtl_run.peak_position,
            "GeneID": gene_id,
            "RetrievalLayer": retrieval_layer,
            "ArticleFinalScore": trait_score["TraitRelevance"],
        })
    return evidence_rows, layer_score_rows


def collect_pubmed_evidence_for_qtl(
    qtl_run: QtlRunSpec,
    top_gene_ids: List[str],
    rapdb_resources: RapdbResources,
    trait_profile_keywords: List[str],
    embedder,
    trait_caches: TraitCaches,
    settings: PipelineSettings,
) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    trait_vector = trait_caches.get_trait_vec(qtl_run.trait_name, embedder)
    evidence_rows: List[dict] = []
    layer_score_rows: List[dict] = []
    pmid_eval_rows: List[dict] = []

    for gene_id in top_gene_ids:
        gene_query_info = rapdb_resources.gene_query_info_by_gene.get(gene_id, empty_query_info(gene_id))
        curated_pmids = gene_query_info.get("CuratedPMIDs") or []

        query_payloads, merged_records, pmid_eval, query_anchors = query_pubmed_layers_for_gene(
            gene_id=gene_id,
            gene_query_info=gene_query_info,
            settings=settings,
        )
        pmid_eval_rows.append(build_gene_pmid_eval_row(qtl_run, gene_id, pmid_eval))

        if settings.debug_pubmed:
            print(
                f"  PubMed {gene_id} direct={len(query_payloads['direct'].get('records') or [])} "
                f"ortho={len(query_payloads['ortholog'].get('records') or [])} "
                f"curated_total={pmid_eval['CuratedPMIDTotal']} "
                f"curated_retrieved={pmid_eval['RetrievedCuratedPMIDCount']} "
                f"direct_anchors={query_anchors['direct']} "
                f"ortho_anchors={query_anchors['ortholog']} "
            )

        gene_evidence_rows, gene_layer_rows = score_pubmed_records_for_gene(
            qtl_run=qtl_run,
            gene_id=gene_id,
            merged_records=merged_records,
            gene_query_info=gene_query_info,
            curated_pmids=curated_pmids,
            query_payloads=query_payloads,
            pmid_eval=pmid_eval,
            trait_profile_keywords=trait_profile_keywords,
            trait_vector=trait_vector,
            embedder=embedder,
            embedding_dim=rapdb_resources.embedding_dim,
            settings=settings,
        )
        evidence_rows.extend(gene_evidence_rows)
        layer_score_rows.extend(gene_layer_rows)

    evidence_df = pd.DataFrame(evidence_rows) if evidence_rows else pd.DataFrame()
    return evidence_df, layer_score_rows, pmid_eval_rows


# ============================================================================
# 10. Literature aggregation and final priority scoring
# ============================================================================

def top_n_mean(
    value_df: pd.DataFrame,
    score_column: str,
    output_column: str,
    top_n: int,
    retrieval_layer: Optional[str] = None,
) -> pd.DataFrame:
    working_df = value_df.copy()
    if retrieval_layer is not None and "RetrievalLayer" in working_df.columns:
        working_df = working_df[working_df["RetrievalLayer"].eq(retrieval_layer)]

    if working_df.empty:
        return pd.DataFrame(columns=QTL_GENE_GROUP_COLUMNS + [output_column])

    return (
        working_df.sort_values(score_column, ascending=False)
        .groupby(QTL_GENE_GROUP_COLUMNS, dropna=False)
        .head(top_n)
        .groupby(QTL_GENE_GROUP_COLUMNS, dropna=False, as_index=False)[score_column]
        .mean()
        .rename(columns={score_column: output_column})
    )


def minmax_normalize_within_qtl(result_df: pd.DataFrame, input_column: str, output_column: str) -> pd.DataFrame:
    normalized_df = result_df.copy()
    normalized_df[output_column] = normalized_df.groupby(QTL_GROUP_COLUMNS)[input_column].transform(
        lambda series: (series - series.min()) / (series.max() - series.min() + 1e-9)
    ).fillna(0)
    return normalized_df


def merge_pmid_eval_summary(ranking_df: pd.DataFrame, pmid_eval_rows: List[dict]) -> pd.DataFrame:
    merged_df = ranking_df.copy()
    if pmid_eval_rows:
        pmid_eval_df = pd.DataFrame(pmid_eval_rows)
        return merged_df.merge(pmid_eval_df, on=QTL_GENE_GROUP_COLUMNS, how="left")

    merged_df["CuratedPMIDTotal"] = 0
    merged_df["RetrievedPMIDCount"] = 0
    merged_df["RetrievedCuratedPMIDCount"] = 0
    merged_df["RetrievedCuratedPMIDs"] = ""
    merged_df["CuratedPMIDRecall"] = np.nan
    return merged_df


def add_literature_support_scores(
    ranking_df: pd.DataFrame,
    layer_score_rows: List[dict],
    pmid_eval_rows: List[dict],
    settings: PipelineSettings,
) -> pd.DataFrame:
    supported_df = ranking_df.copy()

    if layer_score_rows:
        layer_score_df = pd.DataFrame(layer_score_rows)

        literature_support_df = top_n_mean(
            layer_score_df,
            "ArticleFinalScore",
            "LiteratureScore",
            settings.limits.top_articles_for_support,
        )
        supported_df = supported_df.merge(
            literature_support_df,
            on=QTL_GENE_GROUP_COLUMNS,
            how="left",
        )
    else:
        supported_df["LiteratureScore"] = 0.0

    supported_df = merge_pmid_eval_summary(supported_df, pmid_eval_rows)

    numeric_columns = [
        "LiteratureScore",
        "CuratedPMIDTotal",
        "RetrievedPMIDCount",
        "RetrievedCuratedPMIDCount",
        "CuratedPMIDRecall",
    ]
    for column_name in numeric_columns:
        supported_df[column_name] = pd.to_numeric(
            supported_df.get(column_name, 0), errors="coerce"
        )

    supported_df["LiteratureScore"] = supported_df["LiteratureScore"].fillna(0.0)

    for column_name in ["CuratedPMIDTotal", "RetrievedPMIDCount", "RetrievedCuratedPMIDCount"]:
        supported_df[column_name] = supported_df[column_name].fillna(0).astype(int)

    supported_df["RetrievedCuratedPMIDs"] = supported_df.get("RetrievedCuratedPMIDs", "").fillna("")

    # Normalize literature evidence within each QTL before final-score integration.
    supported_df = minmax_normalize_within_qtl(
        supported_df,
        "LiteratureScore",
        "LitSupportScore",
    )

    return supported_df


def add_final_priority_score(ranking_df: pd.DataFrame, weight_settings: WeightSettings) -> pd.DataFrame:
    final_df = ranking_df.copy()
    final_df["PriorityScore"] = (
        weight_settings.final_variant * final_df["VariantScore"]
        + weight_settings.final_distance * final_df["DistanceScore"]
        + weight_settings.final_trait_relatedness * final_df["TraitRelatednessScore"]
        + weight_settings.final_literature * final_df["LitSupportScore"]
    )
    return final_df


# ============================================================================
# 11. Output assembly and export
# ============================================================================

def combine_qtl_outputs(evidence_dfs: List[pd.DataFrame], ranking_dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    evidence_all_df = pd.concat(evidence_dfs, ignore_index=True) if evidence_dfs else pd.DataFrame()
    all_candidates_df = pd.concat(ranking_dfs, ignore_index=True) if ranking_dfs else pd.DataFrame()
    return evidence_all_df, all_candidates_df


def build_top_priority_table(all_candidates_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if all_candidates_df.empty or "PriorityScore" not in all_candidates_df.columns:
        return pd.DataFrame()

    ordered_df = all_candidates_df.sort_values(["QTL", "PriorityScore"], ascending=[True, False]).reset_index(drop=True)
    return (
        ordered_df.groupby(QTL_GROUP_COLUMNS, dropna=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def sort_evidence_table(evidence_df: pd.DataFrame) -> pd.DataFrame:
    if evidence_df.empty:
        return evidence_df
    sort_columns = [column_name for column_name in ["QTL", "GeneID", "ArticleFinalScore"] if column_name in evidence_df.columns]
    if not sort_columns:
        return evidence_df
    ascending = [True, True, False][: len(sort_columns)]
    return evidence_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def save_pipeline_outputs(
    evidence_df: pd.DataFrame,
    all_candidates_df: pd.DataFrame,
    top_priority_df: pd.DataFrame,
    settings: PipelineSettings,
) -> None:
    if all_candidates_df.empty and evidence_df.empty:
        print("\nNo QTL produced results; nothing saved.")
        return

    all_candidates_to_save = all_candidates_df.copy()
    evidence_to_save = sort_evidence_table(evidence_df)

    if not all_candidates_to_save.empty:
        all_candidates_to_save = all_candidates_to_save.sort_values(
            ["QTL", "PriorityScore"], ascending=[True, False]
        ).reset_index(drop=True)
        all_candidates_to_save.to_csv(settings.paths.raw_candidates_tsv, sep="\t", index=False)

    with pd.ExcelWriter(settings.paths.output_workbook, engine="openpyxl") as writer:
        if not evidence_to_save.empty:
            evidence_to_save.to_excel(writer, sheet_name="PUBMED_EVIDENCE", index=False)
        if not all_candidates_to_save.empty:
            all_candidates_to_save.to_excel(writer, sheet_name="ALL_CANDIDATES", index=False)
        if not top_priority_df.empty:
            top_priority_df.to_excel(writer, sheet_name="TOP_PRIORITY", index=False)

        if not all_candidates_to_save.empty and "QTL" in all_candidates_to_save.columns:
            for qtl_id, qtl_df in all_candidates_to_save.groupby("QTL", sort=True, dropna=False):
                qtl_df = qtl_df.sort_values("PriorityScore", ascending=False).reset_index(drop=True)
                qtl_df.to_excel(writer, sheet_name=_safe_sheet_name("QTL", qtl_id), index=False)

        if not evidence_to_save.empty and "QTL" in evidence_to_save.columns:
            for qtl_id, qtl_df in evidence_to_save.groupby("QTL", sort=True, dropna=False):
                sort_columns = [column_name for column_name in ["GeneID", "ArticleFinalScore"] if column_name in qtl_df.columns]
                if sort_columns:
                    qtl_df = qtl_df.sort_values(sort_columns, ascending=[True, False][: len(sort_columns)]).reset_index(drop=True)
                qtl_df.to_excel(writer, sheet_name=_safe_sheet_name("EVID", qtl_id), index=False)

        if not top_priority_df.empty and "QTL" in top_priority_df.columns:
            for qtl_id, qtl_df in top_priority_df.groupby("QTL", sort=True, dropna=False):
                qtl_df = qtl_df.sort_values("PriorityScore", ascending=False).reset_index(drop=True)
                qtl_df.to_excel(writer, sheet_name=_safe_sheet_name("TOP", qtl_id), index=False)

    print(f"\nSaved: {settings.paths.output_workbook}")
    print(f"Saved raw TSV: {settings.paths.raw_candidates_tsv}")


# ============================================================================
# 12. Ranking figure generation
# ============================================================================

def generate_ranking_figures(settings: PipelineSettings) -> List[Path]:
    """Generate QTL-level Top-20 bar plots and interval scatter plots."""
    figure_dir = settings.paths.output_dir / "figures_ranking"

    created_files = make_ranking_figures(
        xlsx=settings.paths.output_workbook,
        rapdb_tsv=settings.paths.rapdb_tsv,
        gff3_path=settings.paths.gff3_path,
        output_dir=figure_dir,
        top_n=settings.limits.top_priority_per_qtl,
    )

    print(f"Saved ranking figures: {figure_dir}")
    return created_files


# ============================================================================
# 13. Pipeline orchestration and command-line entry point
# ============================================================================

def run_pipeline(settings: PipelineSettings = DEFAULT_SETTINGS) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_pipeline_io(settings)

    print("Loading embedding model...")
    embedder = load_sentence_transformer(settings.embedding.model_name)

    rapdb_resources = load_rapdb_resources(settings, embedder)
    variant_resources = load_variant_resources(settings)
    qtl_run_specs = load_qtl_run_specs(settings, variant_resources.chrom_style)
    trait_caches = TraitCaches(settings)

    all_evidence_dfs: List[pd.DataFrame] = []
    all_candidate_dfs: List[pd.DataFrame] = []

    for run_index, qtl_run in enumerate(qtl_run_specs):
        print(
            f"\n=== RUN {run_index}: {qtl_run.trait_name} "
            f"{qtl_run.chromosome}:{qtl_run.interval_start}-{qtl_run.interval_end} "
            f"PEAK={qtl_run.peak_position} ==="
        )

        qtl_result = run_one_qtl_rank(
            qtl_run=qtl_run,
            variant_resources=variant_resources,
            rapdb_resources=rapdb_resources,
            embedder=embedder,
            trait_caches=trait_caches,
            settings=settings,
        )
        if qtl_result.ranking_df.empty:
            print("  -> No ranking results (empty after filtering/ANN).")
            continue

        evidence_df, layer_score_rows, pmid_eval_rows = collect_pubmed_evidence_for_qtl(
            qtl_run=qtl_run,
            top_gene_ids=qtl_result.top_gene_ids,
            rapdb_resources=rapdb_resources,
            trait_profile_keywords=qtl_result.trait_profile_keywords,
            embedder=embedder,
            trait_caches=trait_caches,
            settings=settings,
        )
        if not evidence_df.empty:
            all_evidence_dfs.append(evidence_df)

        ranked_with_literature_df = add_literature_support_scores(
            ranking_df=qtl_result.ranking_df,
            layer_score_rows=layer_score_rows,
            pmid_eval_rows=pmid_eval_rows,
            settings=settings,
        )
        ranked_with_literature_df = add_final_priority_score(
            ranked_with_literature_df,
            settings.weights,
        )
        all_candidate_dfs.append(ranked_with_literature_df)

    evidence_all_df, all_candidates_df = combine_qtl_outputs(all_evidence_dfs, all_candidate_dfs)
    top_priority_df = build_top_priority_table(
        all_candidates_df=all_candidates_df,
        top_n=settings.limits.top_priority_per_qtl,
    )
    save_pipeline_outputs(
        evidence_df=evidence_all_df,
        all_candidates_df=all_candidates_df,
        top_priority_df=top_priority_df,
        settings=settings,
    )

    if settings.paths.output_workbook.exists():
        generate_ranking_figures(settings)

    return evidence_all_df, all_candidates_df, top_priority_df


def main(settings: PipelineSettings = DEFAULT_SETTINGS) -> None:
    run_pipeline(settings)


if __name__ == "__main__":
    main()
