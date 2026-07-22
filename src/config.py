from __future__ import annotations

"""Configuration models for the QTL candidate-gene ranking pipeline.

This module contains validated, immutable settings used by downstream scoring
functions. Keeping score weights here makes the ranking logic easier to audit
and reproduce.
"""
# -----------------------------------------------------------------------------
# Variant scoring configuration
# -----------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass(frozen=True)
class VariantScoringConfig:
    """
    Configuration for simple gene-level variant scoring.

    Each impact class contributes at most once per gene based on presence:
    - HIGH
    - MODERATE
    - LOW
    - MODIFIER

    The final VariantScore is the sum of present impact-class weights,
    clipped to ``max_score``.
    """

    high_weight: float = 0.600
    moderate_weight: float = 0.390
    low_weight: float = 0.009
    modifier_weight: float = 0.001

    max_score: float = 1.0

    def validate(self) -> None:
        weights = [
            self.high_weight,
            self.moderate_weight,
            self.low_weight,
            self.modifier_weight,
        ]
        if any(weight < 0 for weight in weights):
            raise ValueError("Impact weights must be non-negative.")
        if self.max_score <= 0:
            raise ValueError("max_score must be positive.")
