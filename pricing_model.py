
"""
Flex Value Pricing Model (EEM -> EDG)

Core idea:
- Start from a distribution of flex value (EUR) per contract / per period.
- Produce risk measures (VaR, ES), economic capital (EC), and risk-adjusted prices using multiple methods:
  1) Cost-of-Capital (CoC) on Economic Capital (VaR/ES)
  2) "PPT method" (95%-Mean gap * scaling * diversification * hurdle)
  3) Certainty Equivalent (CARA / CRRA utility)
  4) Simple revenue share with downside protection (floor/participation)

Designed to be used from Streamlit (app.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np


RiskMeasure = Literal["var", "es"]  # ES = Expected Shortfall (a.k.a. CVaR)
TailSide = Literal["downside", "upside"]


@dataclass(frozen=True)
class RiskStats:
    mean: float
    std: float
    q: float
    var: float
    es: float
    alpha: float
    side: TailSide
    n: int


def _validate_samples(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError("Need at least 100 finite samples to compute stable risk measures.")
    return x


def quantile(x: np.ndarray, alpha: float, side: TailSide = "downside") -> float:
    """
    Returns the alpha-quantile of the distribution.
    For downside risk, alpha is typically small (e.g., 0.05) meaning 'bad' low outcomes.
    For upside tail, alpha is high (e.g., 0.95).
    """
    x = _validate_samples(x)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    q = np.quantile(x, alpha if side == "downside" else alpha)
    return float(q)


def var_es(
    x: np.ndarray,
    alpha: float = 0.05,
    side: TailSide = "downside",
) -> RiskStats:
    """
    Compute VaR and ES on the chosen tail side.

    Interpretation:
    - Downside: VaR_alpha = quantile at alpha (e.g., 5th percentile of value)
               ES_alpha = average of outcomes <= VaR (worse than VaR)
    - Upside:   VaR_alpha = quantile at alpha (e.g., 95th percentile if alpha=0.95)
               ES_alpha = average of outcomes >= VaR

    NOTE: Many risk teams define VaR in terms of *loss* rather than *value*.
          Here we compute on *value*. Convert to loss if desired.
    """
    x = _validate_samples(x)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    q = float(np.quantile(x, alpha))
    if side == "downside":
        tail = x[x <= q]
        # If extremely small tail, fall back to a few worst samples
        if tail.size < 5:
            tail = np.sort(x)[: max(5, int(0.01 * x.size))]
        es = float(tail.mean())
        var_val = q
    else:
        tail = x[x >= q]
        if tail.size < 5:
            tail = np.sort(x)[-max(5, int(0.01 * x.size)):]
        es = float(tail.mean())
        var_val = q

    return RiskStats(
        mean=float(x.mean()),
        std=float(x.std(ddof=1)),
        q=q,
        var=var_val,
        es=es,
        alpha=float(alpha),
        side=side,
        n=int(x.size),
    )


def economic_capital(
    x: np.ndarray,
    alpha: float = 0.05,
    measure: RiskMeasure = "es",
    side: TailSide = "downside",
) -> Tuple[float, RiskStats]:
    """
    Economic Capital (EC) = Mean - TailMetric for downside on VALUE distribution.
    - If value outcomes are worse when smaller, then Mean - VaR (or Mean - ES) is capital.
    - This is the "gap" to cover adverse outcomes.

    For upside side, you can define EC differently; by default we still return Mean - metric,
    but usually EC is a downside concept.
    """
    stats = var_es(x, alpha=alpha, side=side)
    metric = stats.es if measure == "es" else stats.var
    ec = stats.mean - metric
    return float(ec), stats


def price_cost_of_capital(
    x: np.ndarray,
    alpha: float = 0.05,
    measure: RiskMeasure = "es",
    hurdle_rate: float = 0.1888,
    diversification_factor: float = 1.0,
    side: TailSide = "downside",
) -> Dict[str, float]:
    """
    Price = Mean - hurdle_rate * diversified_EC
    diversified_EC = EC * diversification_factor
    """
    ec, stats = economic_capital(x, alpha=alpha, measure=measure, side=side)
    dec = ec * diversification_factor
    deduction = hurdle_rate * dec
    price = stats.mean - deduction
    return {
        "mean": stats.mean,
        "std": stats.std,
        "alpha": alpha,
        "metric": stats.es if measure == "es" else stats.var,
        "ec": ec,
        "diversified_ec": dec,
        "hurdle_rate": hurdle_rate,
        "risk_deduction": deduction,
        "price": price,
    }


def price_ppt_method(
    x: np.ndarray,
    quantile_level: float = 0.95,
    scaling_factor: float = 3.0,
    diversification_factor: float = 0.8,
    hurdle_rate: float = 0.1888,
    # In the PPT example, "95%-risk - mean" looks like a positive number.
    # That is consistent if "risk" is a LOSS quantile, or if they actually mean (Mean - P5) on value.
    # We'll implement the defensible VALUE version: unexpected_risk = Mean - P(1-quantile_level).
    # quantile_level=0.95 -> downside alpha = 0.05
) -> Dict[str, float]:
    """
    Implements the PPT logic in a consistent way for value distributions:

    unexpected_risk = Mean - P5(value)  (if quantile_level=0.95)
    risk_deduction = unexpected_risk * scaling_factor * diversification_factor * hurdle_rate
    price = Mean - risk_deduction
    """
    x = _validate_samples(x)
    mean = float(x.mean())
    alpha = 1.0 - float(quantile_level)
    p_bad = float(np.quantile(x, alpha))
    unexpected_risk = mean - p_bad
    tail_adj = unexpected_risk * scaling_factor
    diversified = tail_adj * diversification_factor
    deduction = diversified * hurdle_rate
    price = mean - deduction
    return {
        "mean": mean,
        "p_bad": p_bad,
        "alpha": alpha,
        "unexpected_risk": unexpected_risk,
        "scaling_factor": scaling_factor,
        "diversification_factor": diversification_factor,
        "hurdle_rate": hurdle_rate,
        "risk_deduction": deduction,
        "price": price,
    }


def price_certainty_equivalent_cara(
    x: np.ndarray,
    risk_aversion: float = 0.05,
) -> Dict[str, float]:
    """
    Certainty Equivalent under CARA (exponential) utility:
      U(w) = -exp(-a w)
      CE = -(1/a) * log( E[exp(-a w)] )
    with a=risk_aversion.

    Larger a => more risk-averse => lower price.
    """
    x = _validate_samples(x)
    a = float(risk_aversion)
    if a <= 0:
        raise ValueError("risk_aversion must be > 0")
    m = np.max(-a * x)  # log-sum-exp stabilization
    ce = -(1.0 / a) * (np.log(np.mean(np.exp(-a * x - m))) + m)
    return {"mean": float(x.mean()), "ce": float(ce), "risk_aversion": a, "price": float(ce)}


def price_certainty_equivalent_crra(
    x: np.ndarray,
    gamma: float = 2.0,
    shift: float = 0.0,
) -> Dict[str, float]:
    """
    CRRA utility requires positive wealth. For flex value, we use a shift:
      w = x + shift, must be > 0.
    Utility:
      U(w) = (w^(1-gamma) - 1)/(1-gamma) for gamma != 1
    CE is w_ce such that U(w_ce) = E[U(w)], then price = w_ce - shift.

    This is more "finance-y" but needs careful shift selection.
    """
    x = _validate_samples(x)
    g = float(gamma)
    sh = float(shift)
    w = x + sh
    if np.any(w <= 0):
        raise ValueError("CRRA requires x + shift > 0 for all samples. Increase shift.")
    if abs(g - 1.0) < 1e-9:
        # log utility
        eu = float(np.mean(np.log(w)))
        w_ce = float(np.exp(eu))
    else:
        eu = float(np.mean((np.power(w, 1.0 - g) - 1.0) / (1.0 - g)))
        w_ce = float(np.power((eu * (1.0 - g) + 1.0), 1.0 / (1.0 - g)))
    price = w_ce - sh
    return {"mean": float(x.mean()), "ce": float(price), "gamma": g, "shift": sh, "price": float(price)}


def price_revenue_share(
    x: np.ndarray,
    share_to_edg: float = 0.7,
    floor: float = 0.0,
    cap: Optional[float] = None,
) -> Dict[str, float]:
    """
    A commercial alternative:
      EDG receives a fraction of realized flex value with a floor and optional cap.
    We translate that into an "expected price" (expected payout) to use as transfer price.

    payout = clip(share*x, floor, cap)
    """
    x = _validate_samples(x)
    s = float(share_to_edg)
    if not (0 <= s <= 1):
        raise ValueError("share_to_edg must be in [0,1].")
    payout = s * x
    payout = np.maximum(payout, float(floor))
    if cap is not None:
        payout = np.minimum(payout, float(cap))
    return {
        "mean": float(x.mean()),
        "share": s,
        "floor": float(floor),
        "cap": float(cap) if cap is not None else np.nan,
        "expected_payout": float(payout.mean()),
        "price": float(payout.mean()),
    }


def generate_flex_value_samples(
    n: int = 20000,
    mean: float = 50.0,
    vol: float = 30.0,
    dist: Literal["normal", "student_t", "lognormal", "mixture"] = "mixture",
    df: int = 5,
    downside_jump_prob: float = 0.03,
    downside_jump_size: float = 80.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate synthetic flex value samples for experimentation.
    'mixture' gives fat tails and occasional downside jumps.
    """
    rng = np.random.default_rng(seed)
    n = int(n)
    if n < 1000:
        n = 1000

    if dist == "normal":
        x = rng.normal(loc=mean, scale=vol, size=n)
    elif dist == "student_t":
        # scaled t to approximate mean/vol
        t = rng.standard_t(df=df, size=n)
        x = mean + vol * t / np.sqrt(df / (df - 2)) if df > 2 else mean + vol * t
    elif dist == "lognormal":
        # choose parameters to match approximate mean/vol
        # if X ~ logN(mu,s^2): mean = exp(mu+s^2/2), var=(exp(s^2)-1)exp(2mu+s^2)
        # solve roughly with numeric approach
        target_mean, target_std = mean, vol
        s2 = np.log(1 + (target_std**2)/(target_mean**2 + 1e-9))
        s = np.sqrt(max(s2, 1e-9))
        mu = np.log(max(target_mean, 1e-9)) - 0.5*s2
        x = rng.lognormal(mean=mu, sigma=s, size=n)
        # allow negatives by shifting down a bit? keep as-is (mostly positive)
        x = x - (np.mean(x) - mean)
    elif dist == "mixture":
        # base: mildly fat-tailed
        base = rng.standard_t(df=max(df, 3), size=n)
        base = mean + (vol * 0.8) * base / np.sqrt(df / (df - 2))
        # jumps: rare downside shocks (e.g., operational failure / extreme prices)
        jumps = rng.random(n) < downside_jump_prob
        shock = rng.exponential(scale=downside_jump_size, size=n)
        x = base - jumps * shock
    else:
        raise ValueError("Unknown dist.")
    return x


def summarize(x: np.ndarray) -> Dict[str, float]:
    x = _validate_samples(x)
    return {
        "n": float(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
