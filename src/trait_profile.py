from __future__ import annotations

"""Trait-name normalization and predefined keyword expansion.

Canonical trait labels and compact keyword dictionaries provide deterministic
terms for annotation matching and literature retrieval.
"""
import re
from typing import Iterable


# -----------------------------------------------------------------------------
# Canonical trait names
# -----------------------------------------------------------------------------
TRAIT_ALIAS_MAP = {
    "heading date": "flowering time",
    "hd": "flowering time",
    "flowering": "flowering time",
    "days to heading": "flowering time",
    "yield": "grain yield",
    "grain yield": "grain yield",
    "culm length": "plant height",
    "plant height": "plant height",
    "panicle number": "tiller number",
    "tiller number": "tiller number",
    "seed number": "grain number",
    "grain number": "grain number",
    "grain weight": "grain weight",
    "tgw": "grain weight",
}


# -----------------------------------------------------------------------------
# Predefined trait keywords
# -----------------------------------------------------------------------------
TRAIT_KEYWORD_MAP = {
    "flowering time": [
        "heading",
        "flowering",
        "photoperiod",
        "daylength",
        "circadian",
        "floral transition",
        "reproductive timing",
    ],
    "grain yield": [
        "yield",
        "grain",
        "panicle",
        "seed",
        "spikelet",
        "productivity",
    ],
    "plant height": [
        "height",
        "culm",
        "internode",
        "elongation",
        "stem growth",
    ],
    "tiller number": [
        "tiller",
        "branch",
        "axillary bud",
        "shoot branching",
    ],
    "grain weight": [
        "grain weight",
        "seed weight",
        "endosperm",
        "grain filling",
    ],
    "grain number": [
        "grain number",
        "spikelet number",
        "panicle branch",
        "seed set",
    ],
}


# -----------------------------------------------------------------------------
# Internal normalization helpers
# -----------------------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()



def _normalize_trait_key(trait: str) -> str:
    normalized = _normalize_whitespace(trait).lower()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized



def _deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    unique_values: list[str] = []
    seen_values: set[str] = set()

    for value in values:
        normalized_value = _normalize_whitespace(value)
        if not normalized_value:
            continue
        if normalized_value not in seen_values:
            unique_values.append(normalized_value)
            seen_values.add(normalized_value)

    return unique_values



# -----------------------------------------------------------------------------
# Public trait-profile functions
# -----------------------------------------------------------------------------
def normalize_trait(trait: str) -> str:
    """
    Normalize a trait label to a canonical representation.

    Examples
    --------
    'HD' -> 'flowering time'
    'culm-length' -> 'plant height'
    """
    if not trait:
        return ""

    trait_key = _normalize_trait_key(trait)
    return TRAIT_ALIAS_MAP.get(trait_key, trait_key)



def get_trait_keywords(trait: str) -> list[str]:
    """
    Return canonical trait keywords for annotation and literature mining.
    """
    canonical_trait = normalize_trait(trait)
    keywords = TRAIT_KEYWORD_MAP.get(canonical_trait, [])
    return _deduplicate_preserve_order(keywords)



def expand_terms_for_trait(trait: str) -> list[str]:
    """
    Return a merged trait term list including the canonical trait label and
    its predefined related keywords.
    """
    canonical_trait = normalize_trait(trait)
    if not canonical_trait:
        return []

    return _deduplicate_preserve_order([canonical_trait, *get_trait_keywords(canonical_trait)])
