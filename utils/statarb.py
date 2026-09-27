"""Classical statistical-arbitrage pair construction (CODE, not the LLM).

This module replaces the previous low-correlation pair selection. The goal is a
pair with a *strong, stable, mean-reverting relationship*, scored by:

  * Cointegration (Engle-Granger)        -> 30%
  * Spread stationarity (ADF on spread)  -> 20%
  * Mean-reversion quality (half-life)   -> 20%
  * Correlation stability (rolling 1Y)   -> 15%
  * Trading opportunity (historical)     -> 15%

Correlation is used for STABILITY, not as "lower is better": classical pair
trading needs a stable, moderate-to-high relationship. Very low correlation,
unstable rolling correlation, non-stationary spreads, cointegration p>0.05,
half-life outside ~[2,60] days, too few opportunities, and recent structural
breaks are all penalised.

All metrics are computed here; the LLM only ever receives the text summaries
produced by ``summarize_pair_text``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint, adfuller
except Exception as e:  # pragma: no cover
    raise ImportError(
        "statarb pair construction requires statsmodels. Install with "
        "`pip install statsmodels`."
    ) from e

ROLL = 252          # ~1 trading year
HALF_LIFE_MIN = 2
HALF_LIFE_MAX = 60

# composite weights (must sum to 1.0)
W_COINT = 0.30
W_STATIONARY = 0.20
W_MEANREV = 0.20
W_STABILITY = 0.15
W_OPPORTUNITY = 0.15


@dataclass
class StatArbPair:
    pair: Tuple[str, str]
    n_obs: int
    # A. correlation
    corr_10y: float
    corr_5y: float
    corr_1y: float
    rolling_corr_mean: float
    rolling_corr_std: float
    # B. cointegration (Engle-Granger)
    coint_pvalue: float
    coint_stat: float
    # C. stationarity (ADF on spread)
    adf_pvalue: float
    adf_stat: float
    # D. hedge ratio
    hedge_ratio: float
    alpha: float
    # E. spread metrics
    spread_mean: float
    spread_std: float
    current_z: float
    spread_vol: float
    z_p05: float
    z_p50: float
    z_p95: float
    # F. mean reversion
    half_life: float
    half_life_ok: bool
    # G. trading opportunity
    n_entry_1: int
    n_entry_2: int
    hit_rate: float
    hist_sharpe: float
    win_rate: float
    # H. structural stability
    rolling_corr_stability: float
    spread_vol_stability: float
    breakdown_flag: bool
    # composite + sub-scores
    composite: float = 0.0
    sub_scores: Dict[str, float] = field(default_factory=dict)
    # estimation provenance (Step 2: log-price construction)
    spread_definition: str = "log(A) - beta * log(B)"
    cointegration_input: str = "log_prices"
    adf_input: str = "log_spread"
    half_life_input: str = "log_spread"

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["pair"] = list(self.pair)
        return d


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def _ols_hedge_ratio(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """OLS: a = alpha + beta * b. Returns (beta, alpha)."""
    x = b.values.astype(float)
    y = a.values.astype(float)
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1]), float(coef[0])


def _half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    if len(s) < 5:
        return float("nan")
    lag = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    A = np.column_stack([np.ones(len(lag)), lag.values])
    coef, *_ = np.linalg.lstsq(A, delta.values, rcond=None)
    b = float(coef[1])
    if b >= 0:
        return float("inf")
    return float(-math.log(2) / b)


def _spread_backtest(z: np.ndarray, dspread: np.ndarray, spread_std: float,
                     entry: float = 1.0, exit_: float = 0.5) -> Dict[str, float]:
    """Simple historical z-reversion backtest on the spread.

    Returns hit_rate, sharpe (annualised), win_rate, and #entries (>=1, >=2).
    PnL is normalised by the spread standard deviation.
    """
    n = len(z)
    pos = 0
    daily: List[float] = []
    trade_pnls: List[float] = []
    cur_trade = 0.0
    for t in range(1, n):
        # realised PnL from yesterday's position (normalised by spread std)
        step = pos * (dspread[t] / spread_std if spread_std > 1e-12 else 0.0)
        daily.append(step)
        cur_trade += step
        # update position based on yesterday's z (mean-reversion rules)
        zt = z[t - 1]
        if pos == 0:
            if zt >= entry:
                pos = -1  # spread high -> bet it falls
            elif zt <= -entry:
                pos = +1  # spread low -> bet it rises
        elif abs(zt) <= exit_:        # close the trade near equilibrium
            trade_pnls.append(cur_trade)
            cur_trade = 0.0
            pos = 0

    daily_arr = np.array(daily) if daily else np.array([0.0])
    sharpe = (float(daily_arr.mean() / daily_arr.std() * math.sqrt(252))
              if daily_arr.std() > 1e-12 else 0.0)
    wins = sum(1 for p in trade_pnls if p > 0)
    win_rate = (wins / len(trade_pnls)) if trade_pnls else 0.0
    entries_1 = int(np.sum((z[:-1] < 1) & (z[1:] >= 1)) + np.sum((z[:-1] > -1) & (z[1:] <= -1)))
    entries_2 = int(np.sum((z[:-1] < 2) & (z[1:] >= 2)) + np.sum((z[:-1] > -2) & (z[1:] <= -2)))
    # hit rate: of |z|>=1 events, fraction that revert toward 0 within HALF_LIFE_MAX days
    hit = _revert_hit_rate(z, entry=1.0, horizon=int(HALF_LIFE_MAX))
    return {
        "hit_rate": hit, "sharpe": sharpe, "win_rate": win_rate,
        "n_entry_1": entries_1, "n_entry_2": entries_2,
    }


def _revert_hit_rate(z: np.ndarray, entry: float, horizon: int) -> float:
    idx = np.where(np.abs(z) >= entry)[0]
    if len(idx) == 0:
        return 0.0
    hits = 0
    total = 0
    for i in idx:
        if i + 1 >= len(z):
            continue
        total += 1
        window = z[i + 1: i + 1 + horizon]
        if len(window) and np.any(np.sign(window) != np.sign(z[i])) or (len(window) and np.min(np.abs(window)) <= 0.5):
            hits += 1
    return (hits / total) if total else 0.0


def _safe(x: float, default: float = 0.0) -> float:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(xf) or math.isinf(xf):
        return default
    return xf


# --------------------------------------------------------------------------- #
# per-pair metrics
# --------------------------------------------------------------------------- #
def compute_statarb_metrics(price_df: pd.DataFrame, a: str, b: str) -> Optional[StatArbPair]:
    sub = price_df[[a, b]].dropna()
    sub = sub[(sub[a] > 0) & (sub[b] > 0)]
    if len(sub) < 60:
        return None
    pa, pb = sub[a].astype(float), sub[b].astype(float)

    # Correlations are reported on raw prices (level co-movement).
    corr_10y = _safe(pa.corr(pb))
    corr_5y = _safe(pa.iloc[-1260:].corr(pb.iloc[-1260:]))
    corr_1y = _safe(pa.iloc[-ROLL:].corr(pb.iloc[-ROLL:]))

    # Step 2: cointegration / hedge ratio / spread / ADF / half-life use LOG prices.
    la, lb = np.log(pa), np.log(pb)
    roll = la.rolling(ROLL).corr(lb).dropna()
    rolling_corr_mean = _safe(roll.mean(), corr_10y)
    rolling_corr_std = _safe(roll.std(), 0.0)

    # cointegration (Engle-Granger) on log prices
    try:
        coint_stat, coint_p, _ = coint(la, lb)
    except Exception:
        coint_stat, coint_p = 0.0, 1.0
    coint_stat, coint_p = _safe(coint_stat), _safe(coint_p, 1.0)

    # hedge ratio + log spread:  log(A) = alpha + beta*log(B);  spread = log(A) - beta*log(B) - alpha
    beta, alpha = _ols_hedge_ratio(la, lb)
    spread = la - (beta * lb + alpha)
    spread_mean = _safe(spread.mean())
    spread_std = _safe(spread.std(), 1.0)
    spread_vol = _safe(spread.diff().std())

    # ADF on spread
    try:
        try:
            adf_res = adfuller(spread.values, autolag="AIC", result_object=False)
        except TypeError:
            # statsmodels < 0.15 does not expose result_object.
            adf_res = adfuller(spread.values, autolag="AIC")
        adf_stat, adf_p = _safe(adf_res[0]), _safe(adf_res[1], 1.0)
    except Exception:
        adf_stat, adf_p = 0.0, 1.0

    z = ((spread - spread_mean) / spread_std).values if spread_std > 1e-12 else np.zeros(len(spread))
    current_z = _safe(z[-1])
    z_p05, z_p50, z_p95 = (_safe(np.percentile(z, 5)), _safe(np.percentile(z, 50)),
                           _safe(np.percentile(z, 95)))

    half_life = _half_life(spread)
    hl_ok = (not math.isinf(half_life)) and (HALF_LIFE_MIN <= half_life <= HALF_LIFE_MAX) \
        if not math.isnan(half_life) else False

    bt = _spread_backtest(z, spread.diff().fillna(0.0).values, spread_std)

    # structural stability
    rolling_corr_stability = _safe(1.0 - min(rolling_corr_std / 0.3, 1.0))
    recent_vol = _safe(spread.diff().iloc[-ROLL:].std(), spread_vol)
    spread_vol_stability = _safe(1.0 - min(abs(recent_vol - spread_vol) / (spread_vol + 1e-9), 1.0))
    recent_corr = corr_1y
    breakdown_flag = bool(
        (recent_corr < 0.5 * max(rolling_corr_mean, 1e-9) and rolling_corr_mean > 0.2)
        or (recent_vol > 2.0 * spread_vol)
        or (coint_p > 0.10 and adf_p > 0.10)
    )

    pc = StatArbPair(
        pair=(a, b), n_obs=int(len(sub)),
        corr_10y=corr_10y, corr_5y=corr_5y, corr_1y=corr_1y,
        rolling_corr_mean=rolling_corr_mean, rolling_corr_std=rolling_corr_std,
        coint_pvalue=coint_p, coint_stat=coint_stat,
        adf_pvalue=adf_p, adf_stat=adf_stat,
        hedge_ratio=_safe(beta, 1.0), alpha=_safe(alpha),
        spread_mean=spread_mean, spread_std=spread_std, current_z=current_z,
        spread_vol=spread_vol, z_p05=z_p05, z_p50=z_p50, z_p95=z_p95,
        half_life=_safe(half_life, default=999.0), half_life_ok=hl_ok,
        n_entry_1=int(bt["n_entry_1"]), n_entry_2=int(bt["n_entry_2"]),
        hit_rate=_safe(bt["hit_rate"]), hist_sharpe=_safe(bt["sharpe"]),
        win_rate=_safe(bt["win_rate"]),
        rolling_corr_stability=rolling_corr_stability,
        spread_vol_stability=spread_vol_stability,
        breakdown_flag=breakdown_flag,
    )
    _score(pc)
    return pc


# --------------------------------------------------------------------------- #
# composite scoring
# --------------------------------------------------------------------------- #
def _pval_score(p: float) -> float:
    # p<=0.01 -> 1 ; p>=0.10 -> 0 (linear between)
    return float(min(1.0, max(0.0, (0.10 - p) / 0.09)))


def _half_life_score(hl: float) -> float:
    if hl is None or math.isnan(hl) or math.isinf(hl):
        return 0.0
    if hl < HALF_LIFE_MIN or hl > HALF_LIFE_MAX:
        return 0.0
    if hl < 5:
        return 0.5 + 0.5 * (hl - HALF_LIFE_MIN) / (5 - HALF_LIFE_MIN)
    if hl <= 20:
        return 1.0
    return max(0.3, 1.0 - 0.7 * (hl - 20) / (HALF_LIFE_MAX - 20))


def _score(pc: StatArbPair) -> None:
    coint_s = _pval_score(pc.coint_pvalue)
    adf_s = _pval_score(pc.adf_pvalue)
    mr_s = 0.6 * _half_life_score(pc.half_life) + 0.4 * pc.hit_rate
    # stability: moderate-to-high stable correlation
    stab_s = (max(0.0, min(1.0, (pc.rolling_corr_mean - 0.3) / 0.6))
              * max(0.0, min(1.0, 1.0 - pc.rolling_corr_std / 0.3)))
    years = max(pc.n_obs / 252.0, 1e-9)
    opp_s = (0.4 * min(1.0, pc.n_entry_1 / (years * 4))
             + 0.3 * min(1.0, max(0.0, (pc.hist_sharpe + 0.5) / 2.0))
             + 0.3 * pc.win_rate)
    composite = (W_COINT * coint_s + W_STATIONARY * adf_s + W_MEANREV * mr_s
                 + W_STABILITY * stab_s + W_OPPORTUNITY * opp_s)

    # Low-/negative-correlation penalty: classical pair trading needs a genuine
    # positive co-movement relationship. Pairs that are "cointegrated" only
    # because one leg is near-stationary (correlation ~0 or negative) are
    # spurious and effectively disqualified (floor ~0.05).
    corr_penalty = float(min(1.0, max(0.05, pc.corr_10y / 0.6)))
    composite *= corr_penalty

    if pc.breakdown_flag:
        composite *= 0.5  # structural break penalty
    pc.sub_scores = {
        "cointegration": round(coint_s, 4), "stationarity": round(adf_s, 4),
        "mean_reversion": round(mr_s, 4), "correlation_stability": round(stab_s, 4),
        "trading_opportunity": round(opp_s, 4),
        "correlation_penalty": round(corr_penalty, 4),
    }
    pc.composite = round(float(composite), 4)


def rank_statarb_pairs(price_df: pd.DataFrame, symbols: Sequence[str],
                       top_k: int = 10) -> List[StatArbPair]:
    """Compute metrics for every pair and rank by composite score (desc)."""
    out: List[StatArbPair] = []
    for a, b in itertools.combinations(symbols, 2):
        if a not in price_df.columns or b not in price_df.columns:
            continue
        pc = compute_statarb_metrics(price_df, a, b)
        if pc is not None:
            out.append(pc)
    out.sort(key=lambda p: p.composite, reverse=True)
    return out[:top_k] if top_k else out


def current_zscore(pc: StatArbPair, price_a: float, price_b: float) -> float:
    # spread is defined on LOG prices: log(A) - beta*log(B) - alpha
    if pc.spread_std <= 1e-12 or price_a <= 0 or price_b <= 0:
        return 0.0
    spread = math.log(price_a) - (pc.hedge_ratio * math.log(price_b) + pc.alpha)
    return _safe((spread - pc.spread_mean) / pc.spread_std)


# --------------------------------------------------------------------------- #
# code-based news summarisation (Step 7)
# --------------------------------------------------------------------------- #
_POS = ("beat", "bullish", "surge", "growth", "upgrade", "record", "gain", "strong",
        "rally", "positive", "outperform", "raise")
_NEG = ("miss", "bearish", "plunge", "downgrade", "lawsuit", "decline", "weak",
        "fall", "cut", "negative", "drop", "probe", "warn", "recall")
_EVENT_KEYS = {
    "earnings": ("earnings", "eps", "quarter", "guidance", "revenue", "results"),
    "regulatory": ("regulat", "antitrust", "ftc", "sec ", "lawsuit", "probe", "fine", "investigat"),
    "m&a": ("acqui", "merger", "buyout", "takeover", "deal to buy", "acquisition"),
    "analyst": ("upgrade", "downgrade", "price target", "initiat", "rating", "analyst"),
}

# CRITICAL relationship-breaking events (Step 3 news-break-score). Deliberately
# NARROW and rare: only events that actually invalidate a leg justify an IMMEDIATE
# BREAK before the scheduled validation day. Routine M&A *activity* by a company
# (e.g. "X announced an acquisition of Y") is NOT critical -- only the leg itself
# being acquired/delisted/bankrupt/halted/charged-with-fraud counts.
_CRITICAL_EVENT_KWS = (
    "acquired by", "to be acquired", "agrees to be acquired", "agreed to be acquired",
    "agreed to acquire it", "buyout offer for", "take private",
    "files for bankruptcy", "filed for bankruptcy", "chapter 11", "chapter 7",
    "insolvenc", "going concern doubt",
    "will be delisted", "delisting", "delisted from",
    "trading halt", "halted trading", "shares halted",
    "accounting fraud", "accounting scandal", "ponzi", "embezzl",
    "sec charges", "charged with fraud", "financial restatement",
)


def detect_critical_event(items: Sequence[str]) -> Optional[str]:
    """Return the first CRITICAL relationship-breaking event phrase, else None."""
    text = " ".join(items).lower()
    for kw in _CRITICAL_EVENT_KWS:
        if kw in text:
            return kw
    return None


def summarize_stock_news(items: Sequence[str]) -> Dict[str, object]:
    """Code-based per-stock news summary for the past week (sentiment + events).

    Adds a ``critical`` flag (Step 3): True when a relationship-destroying event
    (merger/acquisition, bankruptcy, delisting, fraud, trading halt, ...) is
    present. This flag is consumed ONLY by the relationship monitor; pair selection
    ignores it, so selection behaviour is unchanged.
    """
    if not items:
        return {"sentiment": "neutral", "events": [], "risk": "low",
                "major_event": "none", "n_items": 0, "critical": False,
                "critical_event": None}
    text = " ".join(items).lower()
    pos = sum(text.count(w) for w in _POS)
    neg = sum(text.count(w) for w in _NEG)
    sentiment = "positive" if pos > neg * 1.2 else ("negative" if neg > pos * 1.2 else "neutral")
    events = [k for k, kws in _EVENT_KEYS.items() if any(kw in text for kw in kws)]
    critical_event = detect_critical_event(items)
    risk = "high" if ("regulatory" in events or (neg >= 3 and sentiment == "negative")) else (
        "medium" if (events and sentiment != "positive") else "low")
    major = events[0] if events else "none"
    return {"sentiment": sentiment, "events": events, "risk": risk,
            "major_event": major, "n_items": len(items),
            "critical": bool(critical_event), "critical_event": critical_event}


def pair_news_risk(a_sum: Dict[str, object], b_sum: Dict[str, object]) -> str:
    """Relationship-level risk from the two legs' news."""
    risky_events = {"regulatory", "m&a"}
    for s in (a_sum, b_sum):
        if risky_events & set(s.get("events", [])):
            return "high"  # structural change can break the historical relationship
    sents = {a_sum.get("sentiment"), b_sum.get("sentiment")}
    if sents == {"positive", "negative"}:
        return "medium"  # divergent shocks
    if a_sum.get("risk") == "high" or b_sum.get("risk") == "high":
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# textual summary for the LLM (Step 8) -- NO raw arrays
# --------------------------------------------------------------------------- #
def _quality_label(pc: StatArbPair) -> str:
    if pc.composite >= 0.7 and not pc.breakdown_flag:
        return "High"
    if pc.composite >= 0.45:
        return "Medium"
    return "Low"


def _mr_label(pc: StatArbPair) -> str:
    s = pc.sub_scores.get("mean_reversion", 0.0)
    return "Strong" if s >= 0.7 else ("Moderate" if s >= 0.4 else "Weak")


def _stability_label(pc: StatArbPair) -> str:
    s = pc.rolling_corr_stability
    return "High" if s >= 0.7 else ("Moderate" if s >= 0.4 else "Low")


# --------------------------------------------------------------------------- #
# Sector / theme map (for reflection resemblance + "same sector" detection)
# --------------------------------------------------------------------------- #
SECTORS: Dict[str, str] = {
    "AAPL": "consumer_tech_hardware",
    "ADBE": "software",
    "AMZN": "ecommerce_cloud",
    "BMRN": "biotech",
    "CRM": "software",
    "GOOGL": "internet_software",
    "META": "internet_social",
    "MSFT": "software_cloud",
    "NVDA": "semiconductors",
    "TSLA": "auto_ev",
}


def pair_theme(a: str, b: str) -> str:
    """A coarse theme label for a pair (used for reflection resemblance)."""
    sa = SECTORS.get(a.upper(), "other")
    sb = SECTORS.get(b.upper(), "other")
    if sa == sb:
        return sa
    return f"mixed:{'/'.join(sorted((sa, sb)))}"


def summarize_pair_text(
    pc: StatArbPair, rank: int,
    news_a: Optional[Dict[str, object]] = None,
    news_b: Optional[Dict[str, object]] = None,
    current_z: Optional[float] = None,
) -> Dict[str, object]:
    """Return a dict of scalar metrics + labels + a formatted text block.

    Contains NO raw price/return series, NO raw correlation matrix -- only
    summarised numbers and text the LLM can reason over.
    """
    a, b = pc.pair
    cz = pc.current_z if current_z is None else current_z
    na = news_a or {"sentiment": "neutral", "events": [], "risk": "low", "major_event": "none"}
    nb = news_b or {"sentiment": "neutral", "events": [], "risk": "low", "major_event": "none"}
    rel_risk = pair_news_risk(na, nb)
    text = (
        f"Pair Candidate #{rank}\n"
        f"Stocks: {a} / {b}\n"
        f"Composite Score: {pc.composite:.2f}\n"
        f"10Y Correlation: {pc.corr_10y:.2f}\n"
        f"5Y Correlation: {pc.corr_5y:.2f}\n"
        f"1Y Correlation: {pc.corr_1y:.2f}\n"
        f"Rolling Correlation Stability: {_stability_label(pc)}\n"
        f"Cointegration p-value: {pc.coint_pvalue:.3f}\n"
        f"ADF p-value: {pc.adf_pvalue:.3f}\n"
        f"Hedge Ratio: {pc.hedge_ratio:.2f}\n"
        f"Spread Half-Life: {pc.half_life:.1f} trading days "
        f"({'reasonable' if pc.half_life_ok else 'out of range'})\n"
        f"Mean Reversion Quality: {_mr_label(pc)}\n"
        f"Current Spread Z: {cz:.2f}\n"
        f"Historical Opportunities: |z|>=1: {pc.n_entry_1}, |z|>=2: {pc.n_entry_2}\n"
        f"Historical Spread Sharpe: {pc.hist_sharpe:.2f}\n"
        f"Historical Win Rate: {pc.win_rate * 100:.0f}%\n"
        f"Relationship Breakdown: {'DETECTED' if pc.breakdown_flag else 'None detected'}\n"
        f"Past-Week News:\n"
        f"  {a}: {na['sentiment']} sentiment; event: {na.get('major_event','none')}; risk: {na['risk']}.\n"
        f"  {b}: {nb['sentiment']} sentiment; event: {nb.get('major_event','none')}; risk: {nb['risk']}.\n"
        f"Pair-Level Risk: {rel_risk}\n"
        f"Pair Quality: {_quality_label(pc)}"
    )
    return {
        "rank": rank,
        "pair": [a, b],
        "composite_score": pc.composite,
        "corr_10y": round(pc.corr_10y, 3),
        "corr_5y": round(pc.corr_5y, 3),
        "corr_1y": round(pc.corr_1y, 3),
        "rolling_corr_stability": _stability_label(pc),
        "cointegration_pvalue": round(pc.coint_pvalue, 4),
        "adf_pvalue": round(pc.adf_pvalue, 4),
        "hedge_ratio": round(pc.hedge_ratio, 3),
        "half_life_days": round(pc.half_life, 2),
        "half_life_ok": pc.half_life_ok,
        "mean_reversion_quality": _mr_label(pc),
        "current_z": round(cz, 3),
        "opportunities_z1": pc.n_entry_1,
        "opportunities_z2": pc.n_entry_2,
        "historical_sharpe": round(pc.hist_sharpe, 3),
        "historical_win_rate": round(pc.win_rate, 3),
        "relationship_breakdown": pc.breakdown_flag,
        "news_summary": {a: na, b: nb, "pair_level_risk": rel_risk},
        "pair_quality": _quality_label(pc),
        "text": text,
    }


# =========================================================================== #
# Multi-Horizon Pair Selection (Steps 3, 4, 5, 9)
# =========================================================================== #
HORIZON_ORDER: Tuple[str, ...] = ("1Y", "3Y", "5Y", "10Y", "20Y")
FALLBACK_HORIZON = "FALLBACK"


@dataclass
class MultiHorizonProfile:
    """A pair's stat-arb metrics across every horizon + derived relationship view.

    ``horizons`` maps a horizon label ("1Y".."20Y") to its ``StatArbPair`` (a
    horizon is omitted when there is insufficient overlapping data). The derived
    fields summarise how the relationship behaves SHORT vs LONG term and whether
    it is strengthening, weakening, or stable -- this is the Multi-Horizon
    Relationship Profile that GPT-4.1-mini reasons over (Step 5).
    """
    pair: Tuple[str, str]
    theme: str
    horizons: Dict[str, StatArbPair] = field(default_factory=dict)
    short_term_relationship_strength: float = 0.0
    medium_term_relationship_strength: float = 0.0
    long_term_relationship_strength: float = 0.0
    horizon_consistency: float = 0.0
    relationship_stability_across_horizons: float = 0.0
    relationship_trend: str = "inconsistent"
    current_z: float = 0.0
    # reflection-adjusted scoring (Step 9; filled by apply_reflection_adjustment)
    base_score: float = 0.0
    success_bonus: float = 0.0
    failure_penalty: float = 0.0
    adjusted_score: float = 0.0
    reflection_adjustment_reason: str = "no reflection memory applied"

    def horizon_metrics(self, h: str) -> Optional[StatArbPair]:
        return self.horizons.get(h)

    def to_dict(self) -> Dict[str, object]:
        return {
            "pair": list(self.pair),
            "theme": self.theme,
            "multi_horizon_metrics": {h: pc.to_dict() for h, pc in self.horizons.items()},
            "multi_horizon_relationship_profile": {
                "short_term_relationship_strength": round(self.short_term_relationship_strength, 4),
                "medium_term_relationship_strength": round(self.medium_term_relationship_strength, 4),
                "long_term_relationship_strength": round(self.long_term_relationship_strength, 4),
                "horizon_consistency": round(self.horizon_consistency, 4),
                "relationship_stability_across_horizons": round(self.relationship_stability_across_horizons, 4),
                "relationship_trend": self.relationship_trend,
            },
            "current_z": round(self.current_z, 4),
            "base_score": round(self.base_score, 4),
            "success_bonus": round(self.success_bonus, 4),
            "failure_penalty": round(self.failure_penalty, 4),
            "adjusted_score": round(self.adjusted_score, 4),
            "reflection_adjustment_reason": self.reflection_adjustment_reason,
        }


def _pstd(vals: Sequence[float]) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def _classify_trend(short: Optional[float], medium: Optional[float],
                    long: Optional[float], hi: float = 0.5) -> str:
    s, l = short, long
    if s is None and medium is not None:
        s = medium
    if l is None and medium is not None:
        l = medium
    if s is None or l is None:
        return "inconsistent"
    if s >= hi and l >= hi and abs(s - l) < 0.15:
        return "stable across horizons"
    if s > l + 0.15:
        return "strengthening recently"
    if l > s + 0.15:
        return "weakening recently"
    if s >= hi and l < 0.30:
        return "short-term only"
    if l >= hi and s < 0.30:
        return "long-term only"
    return "inconsistent"


def compute_multihorizon_profile(
    horizon_dfs: Dict[str, pd.DataFrame], a: str, b: str,
    current_prices: Optional[Dict[str, float]] = None,
) -> Optional[MultiHorizonProfile]:
    """Compute per-horizon stat-arb metrics + the derived relationship profile."""
    horizons: Dict[str, StatArbPair] = {}
    for h in HORIZON_ORDER:
        df = horizon_dfs.get(h)
        if df is None or a not in df.columns or b not in df.columns:
            continue
        pc = compute_statarb_metrics(df, a, b)
        if pc is not None:
            horizons[h] = pc
    if not horizons:
        return None

    def comp(h: str) -> Optional[float]:
        return horizons[h].composite if h in horizons else None

    short = comp("1Y")
    medium = _mean([comp("3Y"), comp("5Y")])
    long = _mean([comp("10Y"), comp("20Y")])
    comps = [pc.composite for pc in horizons.values()]
    corrs = [pc.corr_10y for pc in horizons.values()]  # per-horizon full correlation
    mean_comp = (sum(comps) / len(comps)) if comps else 0.0
    horizon_consistency = max(0.0, min(1.0, 1.0 - _pstd(comps) / (mean_comp + 1e-9)))
    stability = max(0.0, min(1.0, 1.0 - _pstd(corrs) / 0.30))

    # current z preference: shortest available horizon (most recent regime)
    shortest = next((h for h in HORIZON_ORDER if h in horizons), None)
    cz = horizons[shortest].current_z if shortest else 0.0
    if current_prices and a in current_prices and b in current_prices and shortest:
        cz = current_zscore(horizons[shortest], current_prices[a], current_prices[b])

    prof = MultiHorizonProfile(
        pair=(a, b), theme=pair_theme(a, b), horizons=horizons,
        short_term_relationship_strength=_safe(short, 0.0),
        medium_term_relationship_strength=_safe(medium, 0.0),
        long_term_relationship_strength=_safe(long, 0.0),
        horizon_consistency=horizon_consistency,
        relationship_stability_across_horizons=stability,
        relationship_trend=_classify_trend(short, medium, long),
        current_z=_safe(cz, 0.0),
    )
    # base_score = best composite across horizons (representative statistical quality)
    prof.base_score = round(max(comps), 4)
    prof.adjusted_score = prof.base_score
    return prof


# --------------------------------------------------------------------------- #
# Horizon qualification + priority selection (Step 4)
# --------------------------------------------------------------------------- #
def horizon_qualifies(
    pc: StatArbPair, *, coint_pmax: float, adf_pmax: float,
    half_life_min: float, half_life_max: float, composite_min: float,
) -> Tuple[bool, List[str]]:
    """Return (qualified, failed_thresholds) for a single horizon's metrics."""
    failed: List[str] = []
    if not (pc.coint_pvalue <= coint_pmax):
        failed.append(f"cointegration p={pc.coint_pvalue:.3f} > {coint_pmax:.3f}")
    if not (pc.adf_pvalue <= adf_pmax):
        failed.append(f"ADF p={pc.adf_pvalue:.3f} > {adf_pmax:.3f}")
    if not (half_life_min <= pc.half_life <= half_life_max):
        failed.append(f"half-life {pc.half_life:.1f}d outside [{half_life_min:.0f},{half_life_max:.0f}]")
    if not (pc.composite >= composite_min):
        failed.append(f"composite {pc.composite:.3f} < {composite_min:.3f}")
    if pc.breakdown_flag:
        failed.append("active structural breakdown")
    return (len(failed) == 0), failed


def discover_pairs_coint10y(
    profiles: Sequence[MultiHorizonProfile], top_k: int = 5, *,
    top_n: int = 20, half_life_min: float = 2.0, half_life_max: float = 60.0,
    coint_pmax: float = 0.05, adf_pmax: float = 0.05,
    return_diagnostics: bool = False,
):
    """Stage 1 Pair Discovery (rule-based, deterministic).

    Pipeline (causal 10Y horizon metrics already on each profile):
        Top-``top_n`` (=20) by 10Y cointegration p-value (ascending)
        -> require cointegration and ADF p-values <= 0.05 and
           ``half_life_min <= half_life <= half_life_max`` (default [2, 60])
        -> Top-``top_k`` (=5), PRESERVING the cointegration p-value ordering.

    If FEWER than ``top_k`` pairs pass the qualification gate, all passing pairs are
    used (no backfill with failing pairs). If ZERO pairs pass, fall back to the original
    Top-``top_k`` by cointegration p-value and flag ``fallback_used=True``.

    Returns ``[(profile, coint_pvalue), ...]``. With ``return_diagnostics=True`` also
    returns a diagnostics dict (total candidates, the Top-20 with half-lives, number
    passing the filter, the final pool, and the fallback flag).
    """
    ranked: List[Tuple[MultiHorizonProfile, float, "StatArbPair"]] = []
    for prof in profiles:
        # Pair selection must never substitute a 1Y/3Y/5Y/20Y profile for 10Y.
        pc = prof.horizon_metrics("10Y")
        if pc is not None:
            ranked.append((prof, float(pc.coint_pvalue), pc))
    ranked.sort(key=lambda x: x[1])      # strongest cointegration (lowest p) first
    top20 = ranked[:top_n]

    def _qualifies(pc) -> bool:
        hl = pc.half_life
        return (
            pc.coint_pvalue <= coint_pmax
            and pc.adf_pvalue <= adf_pmax
            and (hl == hl)
            and not math.isinf(hl)
            and (half_life_min <= hl <= half_life_max)
        )

    passed = [(prof, p) for (prof, p, pc) in top20 if _qualifies(pc)]
    fallback_used = False
    if passed:
        selected = passed[:top_k]
    else:
        fallback_used = True
        selected = [(prof, p) for (prof, p, pc) in top20[:top_k]]

    if return_diagnostics:
        diag = {
            "total_candidates": len(ranked),
            "top20": [{"pair": "/".join(prof.pair), "coint_pvalue": round(p, 6),
                       "adf_pvalue": round(float(pc.adf_pvalue), 6),
                       "half_life": round(float(pc.half_life), 2)}
                      for (prof, p, pc) in top20],
            "n_qualified": len(passed),
            # Kept for compatibility with existing diagnostic readers.
            "n_pass_halflife": len(passed),
            "final_pool": ["/".join(prof.pair) for (prof, _) in selected],
            "fallback_used": fallback_used,
        }
        return selected, diag
    return selected


def select_horizon(
    profiles: Sequence[MultiHorizonProfile], *, coint_pmax: float, adf_pmax: float,
    half_life_min: float, half_life_max: float, composite_min: float,
) -> Tuple[str, List[MultiHorizonProfile], bool, Dict[str, List[str]]]:
    """Apply the 1Y>3Y>5Y>10Y>20Y>fallback priority rule (Step 4).

    Returns ``(selection_horizon, eligible_profiles, qualified, failed_by_pair)``.
    ``eligible_profiles`` is ranked by adjusted_score (desc). When NO horizon has
    a qualified pair, ``selection_horizon='FALLBACK'``, ``qualified=False`` and
    every profile is eligible (forced selection -- never NO_TRADE).
    """
    thr = dict(coint_pmax=coint_pmax, adf_pmax=adf_pmax, half_life_min=half_life_min,
               half_life_max=half_life_max, composite_min=composite_min)
    for h in HORIZON_ORDER:
        eligible: List[MultiHorizonProfile] = []
        for prof in profiles:
            pc = prof.horizon_metrics(h)
            if pc is None:
                continue
            ok, _ = horizon_qualifies(pc, **thr)
            if ok:
                eligible.append(prof)
        if eligible:
            eligible.sort(key=lambda p: p.adjusted_score, reverse=True)
            return h, eligible, True, {}

    # FALLBACK: nothing qualifies anywhere -> still pick the best available.
    failed_by_pair: Dict[str, List[str]] = {}
    for prof in profiles:
        # explain failure against the pair's best (highest-composite) horizon
        best_h = max(prof.horizons, key=lambda hh: prof.horizons[hh].composite)
        _, failed = horizon_qualifies(prof.horizons[best_h], **thr)
        failed_by_pair["/".join(prof.pair)] = [f"[{best_h}] " + f for f in failed]
    ranked = sorted(profiles, key=lambda p: p.adjusted_score, reverse=True)
    return FALLBACK_HORIZON, list(ranked), False, failed_by_pair


# --------------------------------------------------------------------------- #
# Reflection-adjusted scoring (Step 9): base -> +success_bonus -> -failure_penalty
# --------------------------------------------------------------------------- #
def _resemblance(prof: MultiHorizonProfile, entry: Dict[str, object]) -> Tuple[float, List[str]]:
    """Similarity weight (0..1-ish) of a candidate to a remembered pair entry."""
    score, why = 0.0, []
    stocks = {prof.pair[0].upper(), prof.pair[1].upper()}
    e_stocks = {str(s).upper() for s in entry.get("stocks", [])}
    if stocks == e_stocks:
        score += 1.0
        why.append("same exact pair")
    elif stocks & e_stocks:
        score += 0.4
        why.append("shares a leg")
    if entry.get("theme") and entry.get("theme") == prof.theme:
        score += 0.5
        why.append(f"same theme ({prof.theme})")
    # similar half-life bucket (use representative 10Y/long horizon if present)
    rep = prof.horizon_metrics("10Y") or next(iter(prof.horizons.values()))
    e_hl = entry.get("half_life")
    if e_hl is not None and rep is not None and rep.half_life < 1e6:
        if abs(rep.half_life - float(e_hl)) <= 10:
            score += 0.3
            why.append("similar half-life")
    # similar cointegration strength bucket
    e_coint = entry.get("coint_pvalue")
    if e_coint is not None and rep is not None:
        same_bucket = (rep.coint_pvalue <= 0.05) == (float(e_coint) <= 0.05)
        if same_bucket:
            score += 0.2
            why.append("similar cointegration strength")
    return score, why


def apply_reflection_adjustment(profiles: Sequence[MultiHorizonProfile], pair_memory) -> None:
    """Set base/success_bonus/failure_penalty/adjusted_score on each profile (Step 9).

    ``pair_memory`` is duck-typed: it may expose ``success_pairs`` and
    ``failed_pairs`` (lists of dicts with stocks/theme/half_life/coint_pvalue).
    Resemblance to a SUCCESS adds a bonus; resemblance to a FAILURE subtracts a
    penalty. ``adjusted_score`` is what ranking uses.
    """
    successes = list(getattr(pair_memory, "success_pairs", []) or []) if pair_memory else []
    failures = list(getattr(pair_memory, "failed_pairs", []) or []) if pair_memory else []
    for prof in profiles:
        bonus, penalty = 0.0, 0.0
        reasons: List[str] = []
        for e in successes:
            r, why = _resemblance(prof, e)
            if r > 0:
                bonus += 0.10 * min(r, 1.5)
                reasons.append("success<" + ",".join(why) + ">")
        for e in failures:
            r, why = _resemblance(prof, e)
            if r > 0:
                penalty += 0.12 * min(r, 1.5)
                reasons.append("failure<" + ",".join(why) + ">")
        bonus = round(min(bonus, 0.30), 4)
        penalty = round(min(penalty, 0.40), 4)
        prof.success_bonus = bonus
        prof.failure_penalty = penalty
        prof.adjusted_score = round(max(0.0, prof.base_score + bonus - penalty), 4)
        prof.reflection_adjustment_reason = (
            "; ".join(reasons) if reasons else "no resemblance to remembered pairs")


# --------------------------------------------------------------------------- #
# Textual multi-horizon summary for the LLM (Step 5/6) -- NO raw arrays
# --------------------------------------------------------------------------- #
def _strength_label(x: float) -> str:
    return "Strong" if x >= 0.6 else ("Moderate" if x >= 0.35 else "Weak")


def summarize_multihorizon_text(
    prof: MultiHorizonProfile, rank: int, *,
    news_a: Optional[Dict[str, object]] = None,
    news_b: Optional[Dict[str, object]] = None,
    selection_horizon: Optional[str] = None,
    qualified: Optional[bool] = None,
    failed_thresholds: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Return scalar multi-horizon metrics + a formatted text block (no raw series)."""
    a, b = prof.pair
    na = news_a or {"sentiment": "neutral", "events": [], "risk": "low", "major_event": "none"}
    nb = news_b or {"sentiment": "neutral", "events": [], "risk": "low", "major_event": "none"}
    rel_risk = pair_news_risk(na, nb)

    per_horizon_lines = []
    per_horizon_metrics: Dict[str, Dict[str, object]] = {}
    for h in HORIZON_ORDER:
        pc = prof.horizon_metrics(h)
        if pc is None:
            continue
        per_horizon_lines.append(
            f"  {h}: corr={pc.corr_10y:.2f}, coint_p={pc.coint_pvalue:.3f}, "
            f"adf_p={pc.adf_pvalue:.3f}, half_life={pc.half_life:.1f}d, "
            f"|z|>=1 opps={pc.n_entry_1}, composite={pc.composite:.2f}")
        per_horizon_metrics[h] = {
            "pearson_correlation": round(pc.corr_10y, 3),
            "rolling_corr_mean": round(pc.rolling_corr_mean, 3),
            "rolling_corr_std": round(pc.rolling_corr_std, 3),
            "cointegration_pvalue": round(pc.coint_pvalue, 4),
            "cointegration_stat": round(pc.coint_stat, 3),
            "hedge_ratio": round(pc.hedge_ratio, 3),
            "adf_pvalue": round(pc.adf_pvalue, 4),
            "adf_stat": round(pc.adf_stat, 3),
            "spread_mean": round(pc.spread_mean, 4),
            "spread_std": round(pc.spread_std, 4),
            "current_z": round(pc.current_z, 3),
            "half_life_days": round(pc.half_life, 2),
            "opportunities_z1": pc.n_entry_1,
            "opportunities_z2": pc.n_entry_2,
            "historical_sharpe": round(pc.hist_sharpe, 3),
            "historical_win_rate": round(pc.win_rate, 3),
            "structural_break_flag": pc.breakdown_flag,
        }

    text = (
        f"Pair Candidate #{rank}\n"
        f"Stocks: {a} / {b}  (theme: {prof.theme})\n"
        f"Selection Horizon: {selection_horizon or '(n/a)'}  | Qualified: {qualified}\n"
        f"Adjusted Score: {prof.adjusted_score:.2f} "
        f"(base {prof.base_score:.2f} + bonus {prof.success_bonus:.2f} "
        f"- penalty {prof.failure_penalty:.2f})\n"
        f"Multi-Horizon Metrics:\n" + "\n".join(per_horizon_lines) + "\n"
        f"Short-Term Relationship Strength: {_strength_label(prof.short_term_relationship_strength)} "
        f"({prof.short_term_relationship_strength:.2f})\n"
        f"Medium-Term Relationship Strength: {_strength_label(prof.medium_term_relationship_strength)} "
        f"({prof.medium_term_relationship_strength:.2f})\n"
        f"Long-Term Relationship Strength: {_strength_label(prof.long_term_relationship_strength)} "
        f"({prof.long_term_relationship_strength:.2f})\n"
        f"Horizon Consistency: {prof.horizon_consistency:.2f}\n"
        f"Relationship Stability Across Horizons: {prof.relationship_stability_across_horizons:.2f}\n"
        f"Relationship Trend: {prof.relationship_trend}\n"
        f"Current Spread Z: {prof.current_z:.2f}\n"
        f"Reflection Adjustment: {prof.reflection_adjustment_reason}\n"
        f"Past-Week News:\n"
        f"  {a}: {na['sentiment']} sentiment; event: {na.get('major_event','none')}; risk: {na['risk']}.\n"
        f"  {b}: {nb['sentiment']} sentiment; event: {nb.get('major_event','none')}; risk: {nb['risk']}.\n"
        f"Pair-Level Risk: {rel_risk}\n"
        + (f"Failed Thresholds: {', '.join(failed_thresholds)}\n" if failed_thresholds else "")
    )
    return {
        "rank": rank,
        "pair": [a, b],
        "theme": prof.theme,
        "selection_horizon": selection_horizon,
        "qualified": qualified,
        "failed_thresholds": failed_thresholds or [],
        "base_score": prof.base_score,
        "success_bonus": prof.success_bonus,
        "failure_penalty": prof.failure_penalty,
        "adjusted_score": prof.adjusted_score,
        "reflection_adjustment_reason": prof.reflection_adjustment_reason,
        "current_z": round(prof.current_z, 3),
        "relationship_trend": prof.relationship_trend,
        "short_term_relationship_strength": round(prof.short_term_relationship_strength, 3),
        "medium_term_relationship_strength": round(prof.medium_term_relationship_strength, 3),
        "long_term_relationship_strength": round(prof.long_term_relationship_strength, 3),
        "horizon_consistency": round(prof.horizon_consistency, 3),
        "relationship_stability_across_horizons": round(prof.relationship_stability_across_horizons, 3),
        "multi_horizon_metrics": per_horizon_metrics,
        "news_summary": {a: na, b: nb, "pair_level_risk": rel_risk},
        "text": text,
    }


def rank_multihorizon_profiles(
    horizon_dfs: Dict[str, pd.DataFrame], symbols: Sequence[str],
    current_prices: Optional[Dict[str, float]] = None,
) -> List[MultiHorizonProfile]:
    """Build a MultiHorizonProfile for every pair (unranked; scoring applied later)."""
    out: List[MultiHorizonProfile] = []
    for a, b in itertools.combinations(symbols, 2):
        prof = compute_multihorizon_profile(horizon_dfs, a, b, current_prices)
        if prof is not None:
            out.append(prof)
    out.sort(key=lambda p: p.base_score, reverse=True)
    return out
