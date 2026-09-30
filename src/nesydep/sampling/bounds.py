"""Sampling lower bounds (paper §7.1, Theorems 1 & 2).

Faithful port of the prototype
``code-for-CFD/.../supplemental/sampling_bound/sampling_bound.py``.

Notation (mirrors the paper):
  * ``p_hat``     -- estimated normalised support frequency of the rarest
                     candidate pattern.
  * ``eps``       -- additive error bound ε.
  * ``eta``       -- recall η (confidence the sample is representative).
  * ``lam``       -- Chernoff ratio λ.
  * ``delta_hat`` -- normalised support threshold δ̂.
"""
from __future__ import annotations

import math

import pandas as pd


def variance_augmented_bound(p_hat: float, eps: float, eta: float) -> int:
    """Variance-augmented Hoeffding-Bernstein bound (Theorem 1).

    N = p̂(1-p̂) · ln(2/(1-η)) / (2 ε²)
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1
    num = p_hat * (1.0 - p_hat) * math.log(2.0 / (1.0 - eta))
    den = 2.0 * eps * eps
    return max(1, int(math.ceil(num / den)))


def chebyshev_bound(p_hat: float, eps: float, eta: float) -> int:
    """Chebyshev lower bound (comparison baseline)."""
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1
    return max(1, int(math.ceil(p_hat * (1.0 - p_hat) / (eps * eps * (1.0 - eta)))))


def bennett_bound(p_hat: float, eps: float, eta: float) -> int:
    """Bennett lower bound (comparison baseline)."""
    if p_hat <= 0.0:
        return 1
    num = (p_hat + eps) * math.log1p(eps / p_hat) - eps
    den = math.log(2.0 / (1.0 - eta))
    if num <= 0.0 or den <= 0.0:
        return 1
    return max(1, int(math.ceil(num / den)))


def hybrid_chernoff_bound(eps: float, eta: float) -> int:
    """Hybrid Chernoff bound (comparison baseline).

    N = (3/2) · ln(2/(1-η)) / ε²
    """
    return max(1, int(math.ceil(1.5 * math.log(2.0 / (1.0 - eta)) / (eps * eps))))


def _regime_boundaries(lam: float) -> tuple[float, float]:
    """Regime boundaries p̂₁ = (1−√(1−24λ))/2, p̂₂ = (1+√(1−24λ))/2."""
    disc = 1.0 - 24.0 * lam
    if disc < 0.0:
        return float("inf"), float("inf")
    sqrt_disc = math.sqrt(disc)
    return (1.0 - sqrt_disc) / 2.0, (1.0 + sqrt_disc) / 2.0


def theorem2_bound(p_hat: float, eps: float, eta: float, lam: float) -> int:
    """Theorem 2: the piecewise tightest sampling lower bound used by RepSampler.

    * regime A (variance-augmented):  p̂ ≤ p̂₁ or p̂ > p̂₂, with 1/(24λ) ≥ 1
    * regime B (hybrid Chernoff):     p̂ ∈ (p̂₁, p̂₂]
    * regime C (scaled variance):     1/(24λ) < 1
    """
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1
    inv_term = 1.0 / (24.0 * lam) if lam > 0 else float("inf")
    p1, p2 = _regime_boundaries(lam)

    if inv_term < 1.0:  # regime C
        num = p_hat * (1.0 - p_hat) * math.log(2.0 / (1.0 - eta))
        den = 2.0 * eps * eps * (1.0 - inv_term)
        return max(1, int(math.ceil(num / den)))
    if p1 < p_hat <= p2:  # regime B
        return hybrid_chernoff_bound(eps, eta)
    return variance_augmented_bound(p_hat, eps, eta)  # regime A


def p_hat_estimate(df: pd.DataFrame, delta_hat: float) -> float:
    """Estimate p̂ as the normalised support of the rarest *frequent* value.

    For each column, take the relative frequency of its least frequent value
    that still clears the absolute support implied by δ̂; return the minimum
    across columns (the worst-case pattern the bound is most sensitive to).
    """
    n = len(df)
    if n == 0:
        return delta_hat
    abs_support = max(1, int(math.ceil(delta_hat * n)))
    worst = delta_hat
    for col in df.columns:
        counts = df[col].astype(str).value_counts()
        freq = counts[counts >= abs_support]
        if freq.empty:
            continue
        rel = freq.min() / n
        if rel < worst:
            worst = rel
    return float(min(0.5, max(1e-6, worst)))


def compare_bounds(p_hat: float, eps: float, eta: float, lam: float) -> dict[str, int]:
    """All four bounds plus the Theorem-2 selection at once."""
    return {
        "variance_augmented": variance_augmented_bound(p_hat, eps, eta),
        "chebyshev": chebyshev_bound(p_hat, eps, eta),
        "bennett": bennett_bound(p_hat, eps, eta),
        "hybrid_chernoff": hybrid_chernoff_bound(eps, eta),
        "theorem2": theorem2_bound(p_hat, eps, eta, lam),
    }
