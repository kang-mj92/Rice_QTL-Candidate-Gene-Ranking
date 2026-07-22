from __future__ import annotations

"""Trait-profile mining from RAP-DB-compatible annotations.

The module builds trait-relevant term profiles using keyword or semantic seed
selection, frequency-based term mining, optional semantic refinement, and local
embedding caches. LLM-related arguments are retained only for backward
compatibility and are not executed.
"""
import hashlib
import os
import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from src.trait_profile import expand_terms_for_trait, normalize_trait


# -----------------------------------------------------------------------------
# Annotation text sources and token filters
# -----------------------------------------------------------------------------
TEXT_SOURCE_COLUMNS = [
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
]

TOKEN_STOPWORDS = set(
    """
    a an and are as at be by for from has have in into is it its of on or that the to with without
    protein proteins family domain putative similar hypothetical isoform like uncharacterized
    expressed predicted contains containing involved localization process processes
    """.split()
)

DOMAIN_BAN_TERMS = {"rice", "oryza", "locus", "gene", "transcript"}


# -----------------------------------------------------------------------------
# Text and embedding helpers
# -----------------------------------------------------------------------------
def _tokenize_annotation_text(text: str) -> list[str]:
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9+\-_/ ]+", " ", normalized)

    tokens: list[str] = []
    for token in normalized.split():
        if len(token) < 3:
            continue
        if token in TOKEN_STOPWORDS or token in DOMAIN_BAN_TERMS:
            continue
        if re.fullmatch(r"\d{7}", token):
            continue
        tokens.append(token)

    return tokens


def _build_reference_text_column(reference_table: pd.DataFrame) -> pd.Series:
    parts = []
    for column_name in TEXT_SOURCE_COLUMNS:
        if column_name in reference_table.columns:
            parts.append(reference_table[column_name].fillna("").astype(str))
        else:
            parts.append(pd.Series("", index=reference_table.index, dtype=str))

    return pd.concat(parts, axis=1).agg(" | ".join, axis=1).str.lower()


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-12
    return array / norms


def _hash_key(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8", errors="ignore"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def _get_embedder(model_name: str, embedder=None):
    if embedder is not None:
        return embedder

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _build_embedding_cache_path(
    texts: list[str],
    model_name: str,
    cache_dir: Optional[str],
    cache_prefix: str,
) -> Optional[str]:
    if not cache_dir:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    preview_first = texts[0][:200] if texts else ""
    preview_last = texts[-1][:200] if texts else ""
    cache_key = _hash_key(model_name, str(len(texts)), preview_first, preview_last)
    return os.path.join(cache_dir, f"{cache_prefix}_{cache_key}.npy")


def _load_or_build_embeddings(
    texts: list[str],
    model_name: str,
    cache_dir: Optional[str] = None,
    cache_prefix: str = "embeddings",
    batch_size: int = 64,
    show_progress_bar: bool = False,
    embedder=None,
) -> np.ndarray:
    cache_path = _build_embedding_cache_path(
        texts=texts,
        model_name=model_name,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
    )

    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    model = _get_embedder(model_name=model_name, embedder=embedder)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    embeddings = _l2_normalize(np.array(embeddings, dtype=np.float32))

    if cache_path:
        np.save(cache_path, embeddings)

    return embeddings


# -----------------------------------------------------------------------------
# RAP-DB reference preparation and seed selection
# -----------------------------------------------------------------------------
def _load_reference_table(
    rapdb_tsv: str,
    rap_df: Optional[pd.DataFrame],
    rap_texts: Optional[list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    if rap_df is None:
        reference_table = pd.read_csv(rapdb_tsv, sep="\t", dtype=str, low_memory=False)
    else:
        reference_table = rap_df.copy()

    if rap_texts is None:
        reference_table["TEXT"] = _build_reference_text_column(reference_table)
        reference_texts = reference_table["TEXT"].tolist()
    else:
        reference_texts = [str(text) for text in rap_texts]
        reference_table["TEXT"] = pd.Series(reference_texts, index=reference_table.index)

    return reference_table, reference_texts


def _select_seed_rows_semantic(
    reference_table: pd.DataFrame,
    reference_doc_embeddings: np.ndarray,
    normalized_trait: str,
    model_name: str,
    top_k_seed: int,
    embedder=None,
) -> pd.DataFrame:
    model = _get_embedder(model_name=model_name, embedder=embedder)
    trait_embedding = model.encode([normalized_trait])
    trait_embedding = _l2_normalize(np.array(trait_embedding, dtype=np.float32))

    similarity_scores = (reference_doc_embeddings @ trait_embedding.T).ravel()
    k = int(min(max(1, top_k_seed), len(reference_table)))
    top_indices = np.argsort(-similarity_scores)[:k]

    seed_table = reference_table.iloc[top_indices].copy()
    seed_table["SEM_SCORE"] = similarity_scores[top_indices]
    return seed_table.sort_values("SEM_SCORE", ascending=False)


def _select_seed_rows_keyword(
    reference_table: pd.DataFrame,
    normalized_trait: str,
) -> pd.DataFrame:
    expanded_terms = expand_terms_for_trait(normalized_trait)
    match_mask = pd.Series(False, index=reference_table.index)

    for term in expanded_terms:
        match_mask = match_mask | reference_table["TEXT"].str.contains(re.escape(term), na=False)

    seed_table = reference_table[match_mask].copy()

    if len(seed_table) >= 30:
        return seed_table

    for partial_term in normalized_trait.split():
        if len(partial_term) >= 3:
            match_mask = match_mask | reference_table["TEXT"].str.contains(
                re.escape(partial_term),
                na=False,
            )

    return reference_table[match_mask].copy()


# -----------------------------------------------------------------------------
# Candidate-term mining and semantic refinement
# -----------------------------------------------------------------------------
def _mine_candidate_terms(
    seed_table: pd.DataFrame,
    min_count: int,
    top_k: int,
) -> list[str]:
    token_counter: Counter[str] = Counter()
    for text in seed_table["TEXT"].tolist():
        token_counter.update(_tokenize_annotation_text(text))

    token_counts = [
        (token, count)
        for token, count in token_counter.items()
        if count >= min_count
    ]
    token_counts.sort(key=lambda item: item[1], reverse=True)
    return [token for token, _ in token_counts[:top_k]]


def _canonicalize_term(term: str) -> str:
    normalized = str(term).strip().lower()
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    simple_map = {
        "photoperiodic": "photoperiod",
        "day length": "daylength",
        "flowering": "flowering",
        "heading": "heading",
        "circadian": "circadian",
    }
    return simple_map.get(normalized, normalized)


def _semantic_refine_terms(
    trait: str,
    candidates: list[str],
    model_name: str,
    cache_dir: Optional[str] = None,
    embedder=None,
    min_sim: float = 0.20,
    top_k: Optional[int] = None,
) -> dict:
    if not candidates:
        return {"kept": [], "scores": [], "synonym_map": {}}

    normalized_trait = normalize_trait(trait)
    raw_terms = [str(value).strip().lower() for value in candidates if str(value).strip()]
    canonical_terms = [_canonicalize_term(value) for value in raw_terms]

    synonym_map: dict[str, str] = {}
    unique_terms: list[str] = []
    seen_terms: set[str] = set()

    for raw_term, canonical_term in zip(raw_terms, canonical_terms):
        synonym_map[raw_term] = canonical_term
        if canonical_term not in seen_terms:
            unique_terms.append(canonical_term)
            seen_terms.add(canonical_term)

    if not unique_terms:
        return {"kept": [], "scores": [], "synonym_map": synonym_map}

    model = _get_embedder(model_name=model_name, embedder=embedder)
    term_embeddings = _load_or_build_embeddings(
        texts=unique_terms,
        model_name=model_name,
        cache_dir=cache_dir,
        cache_prefix="termemb",
        batch_size=128,
        show_progress_bar=False,
        embedder=model,
    )
    trait_embedding = model.encode([normalized_trait])
    trait_embedding = _l2_normalize(np.array(trait_embedding, dtype=np.float32))

    similarities = (term_embeddings @ trait_embedding.T).ravel()

    scored_terms = [
        {
            "term": term,
            "canonical": term,
            "similarity": float(similarity),
        }
        for term, similarity in zip(unique_terms, similarities)
    ]
    scored_terms.sort(key=lambda item: item["similarity"], reverse=True)

    kept_terms = [
        item["term"]
        for item in scored_terms
        if item["similarity"] >= float(min_sim)
    ]
    if top_k is not None:
        kept_terms = kept_terms[: int(top_k)]

    return {
        "kept": kept_terms,
        "scores": scored_terms,
        "synonym_map": synonym_map,
    }


def _merge_profile_terms(
    normalized_trait: str,
    expanded_terms: list[str],
    mined_terms: list[str],
) -> list[str]:
    merged_terms: list[str] = []
    seen_terms: set[str] = set()

    for term in [normalized_trait] + expanded_terms + mined_terms:
        canonical_term = _canonicalize_term(term)
        if canonical_term and canonical_term not in seen_terms:
            merged_terms.append(canonical_term)
            seen_terms.add(canonical_term)

    return merged_terms


# -----------------------------------------------------------------------------
# Public trait-profile mining API
# -----------------------------------------------------------------------------
def rapdb_global_profile(
    trait: str,
    rapdb_tsv: str,
    top_k: int = 60,
    min_count: int = 3,
    use_semantic_seed: bool = True,
    top_k_seed: int = 300,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_cache_dir: Optional[str] = None,
    use_llm_refine: bool = False,
    ollama_model: str = "llama3.1:8b",
    llm_cluster_only: Optional[str] = None,
    use_semantic_refine: bool = True,
    refine_min_sim: float = 0.25,
    refine_top_k: Optional[int] = None,
    rap_df: Optional[pd.DataFrame] = None,
    rap_texts: Optional[list[str]] = None,
    doc_emb: Optional[np.ndarray] = None,
    embedder=None,
):
    """
    Build a trait-relevant annotation term profile from a RAP-DB-compatible table.

    LLM-related parameters are retained only for backward compatibility.
    """
    normalized_trait = normalize_trait(trait)
    reference_table, reference_texts = _load_reference_table(
        rapdb_tsv=rapdb_tsv,
        rap_df=rap_df,
        rap_texts=rap_texts,
    )

    model = _get_embedder(model_name=embed_model, embedder=embedder)

    if use_semantic_seed:
        seed_method = "semantic"
        reference_doc_embeddings = doc_emb
        if reference_doc_embeddings is None:
            reference_doc_embeddings = _load_or_build_embeddings(
                texts=reference_texts,
                model_name=embed_model,
                cache_dir=embed_cache_dir,
                cache_prefix="reference_docemb",
                batch_size=64,
                show_progress_bar=True,
                embedder=model,
            )
        seed_table = _select_seed_rows_semantic(
            reference_table=reference_table,
            reference_doc_embeddings=reference_doc_embeddings,
            normalized_trait=normalized_trait,
            model_name=embed_model,
            top_k_seed=top_k_seed,
            embedder=model,
        )
    else:
        seed_method = "keyword"
        seed_table = _select_seed_rows_keyword(
            reference_table=reference_table,
            normalized_trait=normalized_trait,
        )

    if "TEXT" not in seed_table.columns:
        seed_table["TEXT"] = _build_reference_text_column(seed_table)

    mined_terms = _mine_candidate_terms(
        seed_table=seed_table,
        min_count=min_count,
        top_k=top_k,
    )

    refine_info = None
    refined_terms = None
    if use_semantic_refine and mined_terms:
        try:
            refine_info = _semantic_refine_terms(
                trait=normalized_trait,
                candidates=mined_terms,
                model_name=embed_model,
                cache_dir=embed_cache_dir,
                embedder=model,
                min_sim=refine_min_sim,
                top_k=refine_top_k,
            )
            candidate_refined_terms = refine_info.get("kept")
            if isinstance(candidate_refined_terms, list) and candidate_refined_terms:
                refined_terms = candidate_refined_terms
        except Exception as exc:
            refine_info = {"error": str(exc)}

    final_tail_terms = refined_terms if refined_terms is not None else mined_terms
    expanded_terms = expand_terms_for_trait(normalized_trait)
    merged_terms = _merge_profile_terms(
        normalized_trait=normalized_trait,
        expanded_terms=expanded_terms,
        mined_terms=final_tail_terms,
    )

    metadata = {
        "trait": normalized_trait,
        "seed_rows": int(len(seed_table)),
        "seed_method": seed_method,
        "top_k_seed": int(top_k_seed) if use_semantic_seed else None,
        "embed_model": embed_model if use_semantic_seed or use_semantic_refine else None,
        "embed_cache_dir": embed_cache_dir if use_semantic_seed or use_semantic_refine else None,
        "top_k": int(top_k),
        "min_count": int(min_count),
        "use_llm_refine": bool(use_llm_refine),
        "ollama_model": None,
        "llm_cluster_only": None,
        "llm_info": None,
        "use_semantic_refine": bool(use_semantic_refine),
        "refine_min_sim": float(refine_min_sim),
        "refine_top_k": int(refine_top_k) if refine_top_k is not None else None,
        "refine_info": refine_info,
        "used_refined_terms": bool(refined_terms is not None),
        "shared_rap_df": bool(rap_df is not None),
        "shared_rap_texts": bool(rap_texts is not None),
        "shared_doc_emb": bool(doc_emb is not None),
        "shared_embedder": bool(embedder is not None),
    }
    return merged_terms, metadata
