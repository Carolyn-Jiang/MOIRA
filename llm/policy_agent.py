"""Policy agent: explicit 3-action decisions + reflection-based improvement.

This module implements the reasoning layer of the redesigned framework. It
supports two interchangeable backends:

  * ``offline`` (default): a deterministic, reproducible rule-based policy. Used
    for tests and for fully offline experiment runs. Decisions are a pure
    function of (signal_z, recent 3-day PnL, news sentiment, policy params), so
    runs are bit-for-bit reproducible given a fixed seed/dataset.

  * ``open_source``: an open-weight LLM served through a Chat Completions endpoint
    that receives CODE-COMPUTED summaries (never raw time series) plus the
    persistent POLICY / REFLECTION / LESSONS LEARNED context, and returns one of
    LONG_SHORT / SHORT_LONG / FLAT. Temperature is never set (provider default).

Persistent prompt memory:
    The agent keeps a ``Policy`` object whose text representation embeds three
    persistent sections -- POLICY, REFLECTION, LESSONS LEARNED -- that are
    injected into every future decision prompt.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from strategies.types import Action, PositionState, action_from_label
from utils.statarb import pair_theme
from llm.errors import InferenceError

_ACTION_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "action_selection.txt")
_DIRECTION_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "direction_discovery.txt")
_PAIR_SELECT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "pair_selection_discovery.txt")
# Trading universe used by the reflection filter to reject ticker-specific content.
_DEFAULT_UNIVERSE = ("AAPL", "ADBE", "AMZN", "BMRN", "CRM", "GOOGL", "META", "MSFT", "NVDA", "TSLA")


def load_base_action_prompt() -> str:
    """Canonical base action prompt from llm/prompts/action_selection.txt (Step 1/8)."""
    try:
        with open(_ACTION_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:  # pragma: no cover - prompt file is part of the repo
        return ("You are a market-neutral pair trader. Choose exactly one action from "
                "LONG_SHORT, SHORT_LONG, FLAT (no HOLD). Prefer a directional view; "
                "use FLAT only as a last resort. BREAK forces FLAT.\n"
                'Return ONLY JSON: {"action":"LONG_SHORT|SHORT_LONG|FLAT","rationale":"..."}')


def load_pair_selection_discovery_prompt() -> str:
    """Pair-Selection prompt for the discovery flow (llm/prompts/pair_selection_discovery.txt).

    The LLM chooses ONE pair from the causal Top-5 candidate pool (it does not choose an
    action -- that is the separate Direction-Discovery step)."""
    try:
        with open(_PAIR_SELECT_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:  # pragma: no cover - prompt file is part of the repo
        return ("PAIR SELECTION TASK\nChoose ONE pair from the candidate pool. "
                'Return ONLY JSON: {"selected_pair":"AAA/BBB","pair_selection_reason":"..."}')


def load_direction_discovery_prompt() -> str:
    """Canonical Direction-Discovery base prompt (llm/prompts/direction_discovery.txt).

    Stage-2 instruction-first prompt: pair selection is already done; the LLM picks ONE
    of the Top-5 pairs + an action (LONG_SHORT/SHORT_LONG/FLAT) + a reason, reasoning on
    RELATIVE news and the spread state.
    """
    try:
        with open(_DIRECTION_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:  # pragma: no cover - prompt file is part of the repo
        return ("Pick ONE of the Top-5 candidate pairs and one action "
                "(LONG_SHORT/SHORT_LONG/FLAT). Compare Asset A vs B (relative news) and "
                "combine with the spread z-score. Return ONLY JSON: "
                '{"selected_pair":["A","B"],"action":"LONG_SHORT|SHORT_LONG|FLAT","reason":"..."}')


# --------------------------------------------------------------------------- #
# Reflection filter: only TRANSFERABLE decision principles may enter the long-term
# prompt. Reject any lesson that memorizes specifics (tickers, pairs, asset names,
# dates, or specific LONG_SHORT/SHORT_LONG recommendations / pair preferences).
# --------------------------------------------------------------------------- #
_DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
_PAIR_RE = re.compile(r"\b[A-Z]{1,6}\s*/\s*[A-Z]{1,6}\b")


def sanitize_reflection(text: str, symbols: Optional[Sequence[str]] = None) -> bool:
    """Return True iff ``text`` is a TRANSFERABLE decision principle that is safe to add
    to the long-term prompt; False if it memorizes specifics.

    Rejects content with: a date, a ``X/Y`` pair, any ticker in the universe, an asset
    preference (prefer/avoid/outperform tied to a name), or a specific directional
    recommendation (ALWAYS ... LONG_SHORT/SHORT_LONG, or a direction bound to a pair).
    """
    t = str(text or "")
    up = t.upper()
    if _DATE_RE.search(t) or re.search(r"\bstep\s+(?:\d+|XXX)\b", t, re.IGNORECASE):
        return False
    if _PAIR_RE.search(up):
        return False
    syms = [str(s).upper() for s in (symbols or _DEFAULT_UNIVERSE)]
    if any(re.search(r"\b" + re.escape(s) + r"\b", up) for s in syms):
        return False
    # "always LONG_SHORT", "LONG_SHORT pair X", memorized directional recommendation
    if re.search(r"\bALWAYS\b.*\b(LONG_SHORT|SHORT_LONG)\b", up) or \
            re.search(r"\b(LONG_SHORT|SHORT_LONG)\b.*\b(PAIR|ASSET)\b", up):
        return False
    # asset/pair preferences ("prefer pair X", "avoid pair Y", "X usually outperforms Y")
    if re.search(r"\b(PREFER|AVOID)\b.*\b(PAIR|ASSET)\b", up) or "OUTPERFORM" in up:
        return False
    return True


def filter_reflection_lessons(lessons: Sequence[str],
                              symbols: Optional[Sequence[str]] = None) -> List[str]:
    """Keep only the lessons that pass ``sanitize_reflection`` (transferable principles)."""
    return [l for l in lessons if sanitize_reflection(l, symbols)]


def build_spread_state(rank: int, pc) -> Dict[str, object]:
    """Spread-state features for one candidate pair (from its causal StatArbPair).

    Includes cointegration rank + p-value, hedge ratio beta, current spread, historical
    spread mean/std, current z-score, half-life, and distance-from-equilibrium (z, in
    standard deviations). No raw price series.
    """
    mu = float(getattr(pc, "spread_mean", 0.0))
    sd = float(getattr(pc, "spread_std", 1.0)) or 1.0
    z = float(getattr(pc, "current_z", 0.0))
    cur = mu + z * sd                     # current spread reconstructed from z
    return {
        "cointegration_rank": int(rank),
        "cointegration_pvalue": round(float(getattr(pc, "coint_pvalue", 1.0)), 4),
        "hedge_ratio_beta": round(float(getattr(pc, "hedge_ratio", 1.0)), 3),
        "current_spread": round(cur, 4),
        "historical_spread_mean": round(mu, 4),
        "historical_spread_std": round(sd, 4),
        "current_zscore": round(z, 3),
        "half_life_days": round(float(getattr(pc, "half_life", float("nan"))), 1),
        "distance_from_equilibrium_sigma": round(z, 3),
    }


_SENT_SCORE = {"strong_positive": 2.0, "positive": 1.0, "neutral": 0.0,
               "negative": -1.0, "strong_negative": -2.0}


def _relative_news(a: str, na: Dict[str, object], b: str, nb: Dict[str, object]) -> Dict[str, object]:
    """Pair-centric RELATIVE comparison (A vs B), not independent per-asset analysis."""
    risk = {"low": 0.0, "medium": 0.5, "high": 1.0}
    sa = _SENT_SCORE.get(str(na.get("sentiment")), 0.0) - risk.get(str(na.get("risk")), 0.0)
    sb = _SENT_SCORE.get(str(nb.get("sentiment")), 0.0) - risk.get(str(nb.get("risk")), 0.0)
    diff = sa - sb
    if abs(diff) < 0.5:
        verdict = "similar -> no clear relative edge (FLAT may be appropriate)"
        likely = "none"
    elif diff > 0:
        verdict = f"{a} has the stronger relative news -> {a} likely outperforms {b}"
        likely = a
    else:
        verdict = f"{b} has the stronger relative news -> {b} likely outperforms {a}"
        likely = b
    return {"relative_news_score_A_minus_B": round(diff, 2),
            "likely_outperformer": likely, "summary": verdict}


@dataclass
class DirectionDecision:
    """Stage-2 output: chosen pair + action + reason (+ structured failure flags).

    ``failed`` is True ONLY when the action is a fallback caused by an error (parse
    failure, invalid action, invalid pair, or an LLM exception) -- never when the model
    genuinely chooses FLAT. ``failure_type`` names the cause when ``failed`` is True.
    """
    selected_pair: Tuple[str, str]
    action: Action
    reason: str
    model: str = "offline-deterministic"
    raw_response: str = ""
    failed: bool = False
    failure_type: Optional[str] = None


@dataclass
class PairSelectionMemory:
    """Stage-1 PAIR QUALITY memory -- evaluates SPREAD MEAN-REVERSION QUALITY ONLY.

    Allowed inputs: cointegration behaviour, spread behaviour, mean-reversion success/
    failure, spread breakdown, cointegration deterioration. FORBIDDEN: LLM actions,
    LONG_SHORT/SHORT_LONG/FLAT decisions, news interpretation, trading PnL. Persists
    across training rounds and is NEVER modified by Direction memory.
    """
    QUALITIES = ("mean_reversion_success", "mean_reversion_failure",
                 "spread_breakdown", "cointegration_deterioration")
    records: List[Dict[str, object]] = field(default_factory=list)

    def add(self, *, pair: Tuple[str, str], quality: str, note: str = "",
            coint_pvalue: Optional[float] = None, half_life: Optional[float] = None) -> None:
        if quality not in self.QUALITIES:
            raise ValueError(f"PairSelectionMemory quality must be one of {self.QUALITIES}")
        self.records.append({
            "pair": "/".join(pair), "quality": quality, "note": note,
            "coint_pvalue": coint_pvalue, "half_life": half_life})

    def render(self) -> str:
        if not self.records:
            return "PAIR QUALITY MEMORY (spread mean-reversion):\n- (none yet)\n"
        lines = "\n".join(f"- {r['pair']}: {r['quality']}"
                          f"{(' (' + r['note'] + ')') if r['note'] else ''}"
                          for r in self.records[-12:])
        return "PAIR QUALITY MEMORY (spread mean-reversion only):\n" + lines + "\n"

    def has_content(self) -> bool:
        return bool(self.records)


@dataclass
class DirectionMemory:
    """Stage-2 DECISION QUALITY memory -- training-only; CLEARED at the end of each round.

    Stores per decision: selected pair, action, reason, realized next-period PnL. Used
    only to generate the round reflection. NEVER modifies Pair Selection memory.
    """
    records: List[Dict[str, object]] = field(default_factory=list)

    def add_decision(self, *, selected_pair: str, action: str, reason: str,
                     decision_index: Optional[int] = None,
                     decision_date: Optional[str] = None,
                     lifecycle_state: Optional[str] = None,
                     original_action: Optional[str] = None) -> int:
        """Append a decision record (one per trading day under daily Direction Discovery).

        ``action`` is the FINAL executed action; ``original_action`` is the model's raw
        choice before any FLAT-override. ``decision_index``/``decision_date`` anchor the
        record so realized PnL is attributed by date/index, not a fixed-stride block."""
        self.records.append({"selected_pair": selected_pair, "action": action,
                             "reason": reason, "realized_pnl": None,
                             "decision_index": decision_index,
                             "decision_date": decision_date,
                             "lifecycle_state": lifecycle_state,
                             "original_action": original_action,
                             "final_action": action})
        return len(self.records) - 1

    def set_realized_pnl(self, idx: int, pnl: float) -> None:
        if 0 <= idx < len(self.records):
            self.records[idx]["realized_pnl"] = float(pnl)

    def set_counterfactual(self, idx: int, *, actual_pnl, cf, best, right,
                           sentence, prices=None, deltas=None) -> None:
        """Attach the daily counterfactual analysis to record ``idx`` (Steps 1/2)."""
        if not (0 <= idx < len(self.records)):
            return
        r = self.records[idx]
        r["realized_pnl"] = (None if actual_pnl is None else float(actual_pnl))
        r["actual_pnl"] = r["realized_pnl"]
        r["cf_long_short"] = float(cf.get("LONG_SHORT", 0.0))
        r["cf_short_long"] = float(cf.get("SHORT_LONG", 0.0))
        r["cf_flat"] = float(cf.get("FLAT", 0.0))
        r["best_action"] = best
        r["right_or_wrong"] = "RIGHT" if right else "WRONG"
        r["memory_sentence"] = sentence
        if prices is not None:
            r["price_stock1"], r["price_stock2"] = prices
        if deltas is not None:
            r["delta_stock1"], r["delta_stock2"] = deltas

    def recent(self, n: int = 3) -> List[Dict[str, object]]:
        """The last ``n`` FINALIZED daily records (those with a counterfactual), as
        compact summaries for the daily Direction-Discovery prompt (Step 3)."""
        done = [r for r in self.records if r.get("right_or_wrong") is not None]
        out = []
        for r in done[-n:]:
            out.append({
                "date": r.get("decision_date"), "action": r.get("action"),
                "reason": r.get("reason"), "actual_pnl": r.get("actual_pnl"),
                "cf_long_short": r.get("cf_long_short"),
                "cf_short_long": r.get("cf_short_long"),
                "cf_flat": r.get("cf_flat"), "best_action": r.get("best_action"),
                "right_or_wrong": r.get("right_or_wrong"),
                "memory_sentence": r.get("memory_sentence")})
        return out

    def clear(self) -> None:
        self.records.clear()

    def render(self) -> str:
        if not self.records:
            return "DIRECTION DECISIONS (this round):\n- (none yet)\n"
        lines = "\n".join(
            f"- {r['selected_pair']} {r['action']} pnl="
            f"{(round(r['realized_pnl'], 4) if r['realized_pnl'] is not None else 'pending')}"
            for r in self.records[-20:])
        return "DIRECTION DECISIONS (this round):\n" + lines + "\n"


# --------------------------------------------------------------------------- #
# Policy object (the thing that gets versioned, accepted, rolled back)
# --------------------------------------------------------------------------- #
@dataclass
class Policy:
    # The initial policy starts deliberately CONSERVATIVE (high entry threshold ->
    # trades rarely) so the reflection-driven search has room to learn to trade.
    version: int = 1
    entry_z: float = 1.5          # |z| beyond which we take a directional pair position
    exit_band: float = 0.5        # |z| inside which we go FLAT
    # (risk_aversion removed: it was behaviorally inert -- not consumed by action
    #  selection, pair ranking/selection, pair switching, or position sizing.)
    # --- pair-selection / NO_TRADE / break thresholds (tracked for convergence) ---
    coint_pmax: float = 0.05          # hard max cointegration p-value
    adf_pmax: float = 0.05            # hard max ADF p-value on the log spread
    half_life_min: float = 2.0        # hard min half-life (trading days)
    half_life_max: float = 60.0       # hard max half-life (trading days)
    no_trade_threshold: float = 0.0   # min composite score to allow a trade (NO_TRADE gate)
    relationship_break_threshold: float = 0.5  # break confidence above which we switch
    policy_text: str = (
        "Enter a mean-reversion pair position when the log-spread z-score is "
        "extreme; go FLAT near equilibrium."
    )
    reflection_text: str = "(no reflection yet)"
    lessons: List[str] = field(default_factory=list)

    # parameters tracked by the policy-parameter convergence test
    def tracked_params(self) -> Dict[str, float]:
        return {
            "entry_z": self.entry_z,
            "exit_band": self.exit_band,
            "coint_pmax": self.coint_pmax,
            "adf_pmax": self.adf_pmax,
            "half_life_min": self.half_life_min,
            "half_life_max": self.half_life_max,
            "no_trade_threshold": self.no_trade_threshold,
            "relationship_break_threshold": self.relationship_break_threshold,
        }

    def render_context(self) -> str:
        """Persistent prompt context injected into every decision prompt."""
        lessons = "\n".join(f"- {l}" for l in self.lessons) or "- (none yet)"
        return (
            "POLICY:\n"
            f"{self.policy_text}\n"
            f"Parameters: entry_z={self.entry_z:.2f}, exit_band={self.exit_band:.2f}\n"
            "END POLICY\n\n"
            "REFLECTION:\n"
            f"{self.reflection_text}\n"
            "END REFLECTION\n\n"
            "LESSONS LEARNED:\n"
            f"{lessons}\n"
            "END LESSONS\n"
        )

    def copy(self) -> "Policy":
        return Policy(
            version=self.version,
            entry_z=self.entry_z,
            exit_band=self.exit_band,
            coint_pmax=self.coint_pmax,
            adf_pmax=self.adf_pmax,
            half_life_min=self.half_life_min,
            half_life_max=self.half_life_max,
            no_trade_threshold=self.no_trade_threshold,
            relationship_break_threshold=self.relationship_break_threshold,
            policy_text=self.policy_text,
            reflection_text=self.reflection_text,
            lessons=list(self.lessons),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Decision result
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    action: Action
    rationale: str
    stance: str  # "aggressive" | "defensive" | "neutral"


NO_TRADE = "NO_TRADE"


@dataclass
class PairChoice:
    """Result of a pair-selection decision (used for selection logging).

    Pair selection ALWAYS returns a pair now (Step 1): ``pair`` is never None and
    ``no_trade`` is retained only as a permanently-False deprecated field so older
    log consumers keep working. When no candidate qualifies, ``qualified`` is False
    and ``forced_pair_selection`` is True (the closest-to-qualified pair is chosen).
    """
    pair: Optional[Tuple[str, str]]
    reasoning: str
    confidence: float
    risks: List[str]
    model: str
    prompt: str
    raw_response: str
    no_trade: bool = False               # deprecated: selection never returns NO_TRADE
    pair_reflection_used: bool = False
    # --- multi-horizon selection (Steps 2-6) ---
    selection_horizon: str = "10Y"
    qualified: bool = True
    forced_pair_selection: bool = False
    failed_thresholds: List[str] = field(default_factory=list)
    multi_horizon_profile_used: bool = False
    # --- reflection-adjusted scoring (Step 9) ---
    base_score: float = 0.0
    success_bonus: float = 0.0
    failure_penalty: float = 0.0
    adjusted_score: float = 0.0
    reflection_adjustment_reason: str = ""
    success_memory_used: bool = False
    failure_memory_used: bool = False


# Relationship-status state machine: 3-level PUBLIC design HEALTHY / WARNING / BREAK.
# The previous DANGER layer was removed; serious pre-break stress is carried by the
# internal ``pre_break_risk`` flag (HIGH) on a WARNING (preserving the force-FLAT
# safety) and is never exposed as a public status.
HEALTHY, WARNING, BREAK = "HEALTHY", "WARNING", "BREAK"
# INTERNAL-ONLY statistical condition (statistics alone, before confirmation). It is
# NOT a public lifecycle status and never appears in relationship_status / news_risk.
BREAK_CANDIDATE = "BREAK_CANDIDATE"
_RISK_BY_STATUS = {HEALTHY: "LOW", WARNING: "MEDIUM", BREAK: "CRITICAL"}

# Step 5/6: 10%-STRICTER statistical BOUNDARIES on the raw break score (this replaces
# the old 2/4/6 boundaries together with the 1.10 score scaling). Same metrics, same
# definitions (z-score / ADF / cointegration / half-life / correlation / spread-vol);
# only the DECISION boundaries are 10% tighter, so the pair reaches each level sooner:
#   stat_score >= 1.8 -> WARNING
#   stat_score >= 3.6 -> BREAK_CANDIDATE (internal; statistics-alone, pre-confirmation)
#   stat_score >= 5.4 -> STAT-critical  (immediate-break-eligible statistics)
_STAT_WARNING = 1.8
_STAT_BREAK_CANDIDATE = 3.6
_STAT_CRITICAL = 5.4


def stat_status_from_score(points: float) -> str:
    """Internal statistical condition from the raw break score (Step 6).

    HEALTHY (< 1.8) / WARNING (< 3.6) / BREAK_CANDIDATE (>= 3.6). ``BREAK_CANDIDATE``
    is INTERNAL only -- statistics alone never emit a public BREAK; confirmation on a
    scheduled validation day (or an immediate news break) is required.
    """
    if points < _STAT_WARNING:
        return HEALTHY
    if points < _STAT_BREAK_CANDIDATE:
        return WARNING
    return BREAK_CANDIDATE


def statistical_break_score(
    *, recent_corr: float, signal_z: float, adf_pvalue: float, coint_pvalue: float,
    half_life: float, policy: "Policy", spread_vol_ratio: float = 1.0,
) -> Tuple[float, str, List[str]]:
    """Graded statistical-deterioration score (Steps 3/5/6).

    Returns ``(points, level, factors)`` where ``points`` is the RAW (unscaled)
    deterioration score and ``level`` is the internal LOW/MEDIUM/HIGH/CRITICAL band.

    Step 5/6: detection is made ~10% stricter at the BOUNDARY level (not by scaling
    the score): the band boundaries are 1.8 / 3.6 / 5.4 (down ~10% from 2 / 4 / 6),
    so the SAME observed statistics map to a higher band sooner. Per-metric
    thresholds and metric definitions (z-score, ADF, cointegration, half-life,
    correlation, spread-vol) are unchanged. ``stat_status_from_score`` maps the same
    raw score onto HEALTHY / WARNING / BREAK_CANDIDATE.
    """
    import math as _m
    points = 0.0
    factors: List[str] = []

    if recent_corr < 0.35:
        points += 4; factors.append(f"rolling-correlation collapse ({recent_corr:.2f})")
    elif recent_corr < 0.45:
        points += 3; factors.append(f"sharp correlation drop ({recent_corr:.2f})")
    elif recent_corr < 0.60:
        points += 2; factors.append(f"moderate correlation decline ({recent_corr:.2f})")
    elif recent_corr < 0.75:
        points += 1; factors.append(f"mild correlation softening ({recent_corr:.2f})")

    hlmax = policy.half_life_max
    if (isinstance(half_life, float) and (_m.isnan(half_life) or _m.isinf(half_life))) \
            or half_life > 4 * hlmax:
        points += 3; factors.append(f"extreme half-life ({half_life:.0f}d)")
    elif half_life > 2 * hlmax:
        points += 2; factors.append(f"long half-life ({half_life:.0f}d)")
    elif half_life > hlmax:
        points += 1; factors.append(f"half-life above ideal ({half_life:.0f}d)")

    if adf_pvalue > 0.20:
        points += 3; factors.append(f"severe ADF failure (p={adf_pvalue:.3f})")
    elif adf_pvalue > 2 * policy.adf_pmax:
        points += 2; factors.append(f"ADF failure (p={adf_pvalue:.3f})")
    elif adf_pvalue > policy.adf_pmax:
        points += 1; factors.append(f"ADF slightly weak (p={adf_pvalue:.3f})")

    if coint_pvalue > 0.20:
        points += 3; factors.append(f"severe cointegration failure (p={coint_pvalue:.3f})")
    elif coint_pvalue > 2 * policy.coint_pmax:
        points += 2; factors.append(f"cointegration failure (p={coint_pvalue:.3f})")
    elif coint_pvalue > policy.coint_pmax:
        points += 1; factors.append(f"cointegration slightly weak (p={coint_pvalue:.3f})")

    if abs(signal_z) > 3.5:
        points += 2; factors.append(f"extreme spread z ({signal_z:.1f})")
    elif abs(signal_z) > 2.5:
        points += 1; factors.append(f"stretched spread z ({signal_z:.1f})")

    if spread_vol_ratio > 2.0:
        points += 2; factors.append(f"spread-volatility regime shift (x{spread_vol_ratio:.1f})")
    elif spread_vol_ratio > 1.5:
        points += 1; factors.append(f"elevated spread volatility (x{spread_vol_ratio:.1f})")

    # Step 5/6: 10%-stricter BOUNDARIES on the raw score (1.8 / 3.6 / 5.4).
    level = ("CRITICAL" if points >= _STAT_CRITICAL else "HIGH" if points >= _STAT_BREAK_CANDIDATE
             else "MEDIUM" if points >= _STAT_WARNING else "LOW")
    return points, level, factors


# Leg-specific events that are UNFAVOURABLE to a pair relationship. A market-neutral
# pair trades on co-movement, so an adverse event hitting ONE leg (asymmetric
# guidance/earnings shock, adverse regulatory action, analyst downgrade, a
# leg-specific M&A overhang, ...) weakens the economic logic linking the pair.
_PAIR_UNFAVORABLE_EVENTS = {"regulatory", "analyst", "m&a", "earnings"}
_FIN_SENTIMENTS = ("strong_positive", "positive", "neutral", "negative", "strong_negative")


def _leg_financial_sentiment(leg: Dict[str, object]) -> str:
    """Map a leg's news summary to a FINANCIAL-market sentiment band (one of
    strong_positive/positive/neutral/negative/strong_negative). Prefers an explicit
    ``financial_sentiment`` (e.g. from the LLM), else derives it from the code summary
    (``sentiment`` + ``risk`` -> 'strong' when leg risk is high)."""
    fs = str(leg.get("financial_sentiment") or "").strip().lower()
    if fs in _FIN_SENTIMENTS:
        return fs
    s = str(leg.get("sentiment") or "neutral").strip().lower()
    strong = (leg.get("risk") == "high")
    if s == "positive":
        return "strong_positive" if strong else "positive"
    if s == "negative":
        return "strong_negative" if strong else "negative"
    return "neutral"


def news_break_level(news_summary: Dict[str, object]) -> Tuple[float, str, List[str], bool]:
    """3-level news relationship-break assessment (Steps 8/9). Semantic relationship
    FIRST, financial-market sentiment SECOND.

    Returns ``(score, level, factors, is_break)`` where ``level`` is one of the SAME
    3-level public structure as the relationship lifecycle -- HEALTHY / WARNING /
    BREAK (never LOW/MEDIUM/HIGH/CRITICAL) -- and ``is_break`` is ``level == BREAK``.

    Rules:
      * a direct structural relationship-breaking event, or an explicit
        ``semantic_relationship_signal == 'broken_candidate'`` -> BREAK;
      * same strong directional sentiment on BOTH legs (both positive or both
        negative) -> at least WARNING (same direction reduces market-neutral spread
        usefulness);
      * any strong directional sentiment, any negative/pair-unfavourable leg, or a
        semantic 'weakening' signal -> at least WARNING;
      * a stable semantic relationship with neutral/mixed sentiment -> HEALTHY.
    A WARNING from news alone never switches the pair; only a BREAK (the immediate
    news-break exception) or a confirmed statistical break does.
    """
    legs = [v for k, v in news_summary.items() if isinstance(v, dict)]
    factors: List[str] = []

    # (A) SEMANTIC relationship first -- a direct structural break -> BREAK.
    crit_leg = next((l for l in legs if l.get("critical")), None)
    if crit_leg is not None:
        ev = crit_leg.get("critical_event") or "structural relationship-breaking event"
        factors.append(f"structural relationship-breaking news: {ev}")
        return 4.0, BREAK, factors, True
    sem = str(news_summary.get("semantic_relationship_signal") or "").strip().lower()
    if sem == "broken_candidate":
        factors.append("semantic relationship broken (structural divergence)")
        return 4.0, BREAK, factors, True

    # (B) FINANCIAL sentiment second.
    fss = [_leg_financial_sentiment(l) for l in legs]
    dirs = ["pos" if "positive" in f else "neg" if "negative" in f else "neu" for f in fss]
    strongs = [f for f in fss if f.startswith("strong_")]
    nonneu = [d for d in dirs if d != "neu"]
    neg = [d for d in dirs if d == "neg"]
    # A leg-specific adverse event only counts when the leg is NOT clearly positive
    # (a positive earnings beat is not pair-unfavourable).
    adverse = any((_PAIR_UNFAVORABLE_EVENTS & set(l.get("events") or []))
                  and d != "pos" for l, d in zip(legs, dirs))
    same_direction = (len(nonneu) >= 2 and len(set(nonneu)) == 1)

    if same_direction:
        factors.append("both legs share the same directional sentiment "
                       "(reduced market-neutral usefulness)")
        return 2.0, WARNING, factors, False
    if strongs or neg or adverse or sem == "weakening":
        factors.append("pair-unfavourable / asymmetric financial sentiment on a leg")
        return 2.0, WARNING, factors, False
    return 0.0, HEALTHY, factors, False


def _candidate_critical_event(news_summary: Dict[str, object]) -> str:
    """The keyword-detected critical-event phrase from a leg's news (or 'unknown')."""
    for v in news_summary.values():
        if isinstance(v, dict) and v.get("critical"):
            return str(v.get("critical_event") or "critical event")
    return "unknown"


@dataclass
class BreakAssessment:
    relationship_break_detected: bool
    reasoning: str
    break_type: str
    confidence: float
    should_switch_pair: bool
    model: str
    raw_response: str = ""
    # --- pre-break risk state (Step 10) ---
    relationship_status: str = HEALTHY       # HEALTHY / WARNING / BREAK (3-level)
    pre_break_risk: str = "LOW"              # LOW / MEDIUM / HIGH / CRITICAL (internal)
    should_force_flat: bool = False
    risk_factors: List[str] = field(default_factory=list)
    # --- conservative scoring + cadence (Steps 3/4/5/8) ---
    statistical_risk: str = "LOW"            # internal LOW/MEDIUM/HIGH/CRITICAL (legacy granularity)
    news_risk: str = HEALTHY                 # 3-level HEALTHY/WARNING/BREAK (news_break_level)
    llm_risk: str = HEALTHY                  # 3-level HEALTHY/WARNING/BREAK (LLM news evidence)
    statistical_break_score: float = 0.0
    news_break_score: float = 0.0
    llm_break_score: float = 0.0
    monitoring_event_type: str = "scheduled_break_validation"
    days_since_last_break_validation: int = 0
    persistence_count: int = 0
    validation_window_count: int = 0
    scheduled_break_validation: bool = True
    critical_news_exception: bool = False
    break_confirmed: bool = False
    previous_relationship_status: str = HEALTHY
    critical_news_verification: Dict[str, object] = field(default_factory=dict)


STATARB_SELECTION_PROMPT = (
    "You are selecting ONE pair for CLASSICAL STATISTICAL ARBITRAGE.\n"
    "\n"
    "You MUST select exactly one pair. Do NOT return NO_TRADE. Pair selection always\n"
    "tracks a pair; whether to actually trade it today is a separate decision made by\n"
    "the trading-action layer (which may go FLAT).\n"
    "\n"
    "Use the following horizon priority for eligibility:\n"
    "  1Y qualified pair > 3Y > 5Y > 10Y > 20Y > best available unqualified pair.\n"
    "However, reason over the FULL Multi-Horizon Relationship Profile (1Y/3Y/5Y/10Y/20Y\n"
    "metrics + short/medium/long-term strength, horizon consistency, stability and the\n"
    "relationship trend), not just the selected horizon.\n"
    "\n"
    "Prefer:\n"
    "- significant cointegration (low p-value on the LOG spread)\n"
    "- a stationary log spread (low ADF p-value)\n"
    "- a reasonable log-spread half-life\n"
    "- a stable relationship across horizons\n"
    "- an improving or stable recent relationship\n"
    "- sufficient historical spread opportunities\n"
    "- no relationship-breaking news\n"
    "\n"
    "Avoid:\n"
    "- pairs previously failed for similar reasons (see FAILURE MEMORY)\n"
    "- pairs whose recent relationship is weakening\n"
    "- pairs with only superficial high correlation\n"
    "- pairs with weak cointegration\n"
    "- pairs with a long half-life\n"
    "- pairs affected by idiosyncratic relationship-breaking news\n"
    "Prefer patterns resembling SUCCESS MEMORY; avoid patterns resembling FAILURE MEMORY.\n"
    "\n"
    "You are given pre-computed TEXT summaries only (no raw prices/returns/correlation\n"
    "matrices). When no candidate qualifies you still pick the closest-to-qualified pair\n"
    "and set qualified=false, forced_pair_selection=true, listing the failed thresholds.\n"
    "\n"
    'Return ONLY JSON: {"selected_pair": ["A", "B"], '
    '"selection_horizon": "1Y/3Y/5Y/10Y/20Y/FALLBACK", "qualified": true/false, '
    '"forced_pair_selection": true/false, "reasoning": "...", '
    '"failed_thresholds": ["..."], "confidence": 0.0-1.0, "risks": ["..."], '
    '"multi_horizon_profile_used": true}'
)


RELATIONSHIP_BREAK_PROMPT = (
    "You assess the PAST-7-DAY NEWS for an ACTIVE statistical-arbitrage PAIR (leg A,\n"
    "leg B) and judge whether the news threatens the mean-reverting RELATIONSHIP.\n"
    "Evaluate the SEMANTIC RELATIONSHIP FIRST, then the FINANCIAL-MARKET SENTIMENT.\n"
    "Provide CONFIRMING evidence only -- the final BREAK is enforced by code rules.\n"
    "\n"
    "A. SEMANTIC RELATIONSHIP (first):\n"
    "   - Does the news affect only ONE leg (asymmetric business risk)?\n"
    "   - Does it weaken the economic linkage between the two stocks?\n"
    "   - Does it increase structural divergence or reduce mean-reversion plausibility?\n"
    "\n"
    "B. FINANCIAL SENTIMENT (second; financial-market sentiment, NOT emotional tone):\n"
    "   Positive: earnings beat, guidance raise, analyst upgrade, strong demand,\n"
    "     market-share gain, strategic advantage, short sellers turning bullish.\n"
    "   Negative: earnings miss, guidance cut, analyst downgrade, demand weakness,\n"
    "     regulatory pressure, accounting concern, product failure, lawsuit,\n"
    "     management instability, short sellers turning more bearish.\n"
    "\n"
    "Rules (3-level HEALTHY / WARNING / BREAK -- no LOW/MEDIUM/HIGH/CRITICAL):\n"
    "   - stable semantic relationship + neutral or mixed sentiment -> HEALTHY;\n"
    "   - mild pair-unfavourable semantic news -> WARNING;\n"
    "   - BOTH legs same strong direction (both strong_positive OR both strong_negative)\n"
    "     -> at least WARNING (same strong direction reduces market-neutral usefulness);\n"
    "   - one strong_positive and the other neutral/negative -> at least WARNING;\n"
    "   - one strong_negative and the other neutral/positive -> at least WARNING;\n"
    "   - a direct relationship-breaking event, or structural divergence that\n"
    "     invalidates the pair logic -> BREAK.\n"
    "\n"
    'Return ONLY JSON: {'
    '"semantic_relationship_signal": "stable|weakening|broken_candidate", '
    '"leg_a_financial_sentiment": "strong_positive|positive|neutral|negative|strong_negative", '
    '"leg_b_financial_sentiment": "strong_positive|positive|neutral|negative|strong_negative", '
    '"sentiment_symmetry": "same_strong_direction|asymmetric|mixed_or_neutral", '
    '"news_break_level": "HEALTHY|WARNING|BREAK", '
    '"is_relationship_break_signal": true|false, '
    '"reason": "brief explanation"}'
)


CRITICAL_NEWS_VERIFY_PROMPT = (
    "You verify whether a CANDIDATE news event (found by keyword match) is a genuine\n"
    "RELATIONSHIP-BREAKING event for a statistical-arbitrage PAIR. A relationship-\n"
    "breaking event materially and durably changes one leg's business so the pair's\n"
    "historical co-movement no longer holds. Examples: a leg being acquired / merged /\n"
    "taken private, bankruptcy or going-concern doubt, delisting, a trading halt on a\n"
    "leg, fraud / accounting scandal / financial restatement, a severe leg-specific\n"
    "regulatory action, or a catastrophic earnings/guidance shock to ONE leg.\n"
    "\n"
    "Routine mentions (the company merely commenting on industry M&A, an article about a\n"
    "DIFFERENT company's bankruptcy, ordinary earnings/analyst chatter) are NOT critical.\n"
    "\n"
    'Return ONLY JSON: {"event_type": "...", "affects_pair_relationship": true/false, '
    '"severity": "LOW/MEDIUM/HIGH/CRITICAL", "reasoning": "..."}'
)


def classify_pair_outcome(
    *, cum_pnl: float, relationship_break: bool = False, spread_reverted: bool = True,
    half_life: Optional[float] = None, half_life_max: float = 60.0,
    news_invalidated: bool = False,
) -> str:
    """Classify a closed/evaluated pair as SUCCESS / FAILURE / NEUTRAL (Step 7).

    FAILURE if any of: cumulative PnL < 0, a relationship break occurred, the spread
    failed to revert, the half-life became too long (or non-finite), or news
    invalidated the historical relationship. SUCCESS if cumulative PnL > 0 and none
    of the failure conditions hold. NEUTRAL otherwise (e.g. flat PnL).
    """
    failure = bool(cum_pnl < 0 or relationship_break or (spread_reverted is False)
                   or news_invalidated)
    if half_life is not None:
        if half_life != half_life or half_life > half_life_max:  # NaN or too long
            failure = True
    if failure:
        return "FAILURE"
    if cum_pnl > 0:
        return "SUCCESS"
    return "NEUTRAL"


@dataclass
class PairReflectionMemory:
    """Persistent memory that learns from previous pair outcomes (Step 8).

    Holds BOTH a Failure Memory (failed pairs, why they failed, their statistical
    and news patterns) and a Success Memory (successful pairs, why they worked),
    plus general AVOID (failure) and PREFER (success) lessons. The full rendered
    memory is injected into every future pair-selection prompt so selection evolves
    over time, and the structured entries drive reflection-adjusted ranking (Step 9).
    """
    reflections: List[str] = field(default_factory=list)
    failed_pairs: List[Dict[str, object]] = field(default_factory=list)
    success_pairs: List[Dict[str, object]] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)          # AVOID rules (failures)
    success_lessons: List[str] = field(default_factory=list)  # PREFER rules (successes)

    @staticmethod
    def _entry(pair: Tuple[str, str], reason: str,
               stats: Optional[Dict[str, object]],
               news: Optional[Dict[str, object]]) -> Dict[str, object]:
        stats = stats or {}
        news = news or {}
        return {
            "pair": "/".join(pair),
            "stocks": [pair[0], pair[1]],
            "theme": pair_theme(pair[0], pair[1]),
            "reason": reason,
            "half_life": stats.get("half_life_days"),
            "coint_pvalue": stats.get("cointegration_pvalue"),
            "news_risk": news.get("pair_level_risk", "low"),
        }

    def add_failure(self, *, pair: Tuple[str, str], reflection_text: str,
                    reason: str, lessons: List[str],
                    stats: Optional[Dict[str, object]] = None,
                    news: Optional[Dict[str, object]] = None) -> None:
        self.reflections.append(reflection_text)
        self.failed_pairs.append(self._entry(pair, reason, stats, news))
        for l in lessons:
            if l and l not in self.lessons:
                self.lessons.append(l)

    def add_success(self, *, pair: Tuple[str, str], reflection_text: str,
                    reason: str, lessons: List[str],
                    stats: Optional[Dict[str, object]] = None,
                    news: Optional[Dict[str, object]] = None) -> None:
        self.reflections.append(reflection_text)
        self.success_pairs.append(self._entry(pair, reason, stats, news))
        for l in lessons:
            if l and l not in self.success_lessons:
                self.success_lessons.append(l)

    # Backward-compatible alias (legacy callers/tests treat ``add`` as a failure).
    def add(self, *, pair: Tuple[str, str], reflection_text: str,
            reason: str, lessons: List[str], **kw) -> None:
        self.add_failure(pair=pair, reflection_text=reflection_text,
                         reason=reason, lessons=lessons, **kw)

    def render(self) -> str:
        cur = self.reflections[-1] if self.reflections else "(none yet)"
        failed = "\n".join(
            f"- {fp['pair']} (theme {fp.get('theme')}): {fp['reason']}"
            for fp in self.failed_pairs
        ) or "- (none yet)"
        succeeded = "\n".join(
            f"- {sp['pair']} (theme {sp.get('theme')}): {sp['reason']}"
            for sp in self.success_pairs
        ) or "- (none yet)"
        avoid = "\n".join(f"- {l}" for l in self.lessons) or "- (none yet)"
        prefer = "\n".join(f"- {l}" for l in self.success_lessons) or "- (none yet)"
        return (
            "PAIR REFLECTION MEMORY (current):\n" + cur + "\n\n"
            "FAILED PAIR HISTORY (FAILURE MEMORY):\n" + failed + "\n\n"
            "SUCCESSFUL PAIR HISTORY (SUCCESS MEMORY):\n" + succeeded + "\n\n"
            "GENERAL LESSONS LEARNED -- AVOID (from failures):\n" + avoid + "\n\n"
            "GENERAL LESSONS LEARNED -- PREFER (from successes):\n" + prefer + "\n"
        )

    def has_content(self) -> bool:
        return bool(self.reflections or self.failed_pairs or self.success_pairs
                    or self.lessons or self.success_lessons)


@dataclass
class ActionPromptMemory:
    """Learned action-selection prompt: base prompt + accepted action lessons (Step 2).

    The active action prompt evolves as ``base_action_prompt + accepted lessons``.
    A candidate prompt (base + accepted + new lessons) is evaluated each training
    cycle; if it improves training PnL it is accepted (lessons appended, version++),
    otherwise it is rolled back (lessons recorded as rejected). This is tracked
    SEPARATELY from numeric policy-parameter updates (Step 7).
    """
    base_action_prompt: str
    accepted_action_lessons: List[Dict[str, object]] = field(default_factory=list)
    rejected_action_lessons: List[Dict[str, object]] = field(default_factory=list)
    current_action_prompt_version: int = 1
    max_action_lessons: int = 8  # MAX_ACTION_LESSONS

    @staticmethod
    def _text(lesson: Dict[str, object]) -> str:
        return str(lesson.get("lesson_text") or lesson.get("text") or "")

    def _block(self, lessons: Sequence[Dict[str, object]]) -> str:
        if not lessons:
            return ""
        bullets = "\n".join(f"- {self._text(l)}" for l in lessons)
        return ("\n\nAccepted Action Lessons (learned from training reflection):\n"
                + bullets)

    def _recent_accepted(self) -> List[Dict[str, object]]:
        return self.accepted_action_lessons[-self.max_action_lessons:]

    def active_prompt(self) -> str:
        return self.base_action_prompt + self._block(self._recent_accepted())

    def candidate_prompt(self, new_lessons: Sequence[Dict[str, object]]) -> str:
        combined = (self.accepted_action_lessons + list(new_lessons))[-self.max_action_lessons:]
        return self.base_action_prompt + self._block(combined)

    @staticmethod
    def _bias(lessons: Sequence[Dict[str, object]]) -> float:
        return float(sum({"DIRECTIONAL": 1.0, "CAUTION": -1.0}.get(str(l.get("tag")), 0.0)
                         for l in lessons))

    def active_bias(self) -> float:
        return self._bias(self._recent_accepted())

    def candidate_bias(self, new_lessons: Sequence[Dict[str, object]]) -> float:
        combined = (self.accepted_action_lessons + list(new_lessons))[-self.max_action_lessons:]
        return self._bias(combined)

    def accept(self, new_lessons: Sequence[Dict[str, object]], *, source_cycle: int,
               reward_before: float, reward_after: float) -> None:
        for l in new_lessons:
            self.accepted_action_lessons.append({
                "source_cycle": source_cycle, "lesson_text": self._text(l),
                "tag": l.get("tag"), "reason": l.get("reason", ""),
                "reward_before": reward_before, "reward_after": reward_after,
                "accepted": True})
        self.current_action_prompt_version += 1

    def reject(self, new_lessons: Sequence[Dict[str, object]], *, source_cycle: int,
               reward_before: float, reward_after: float, rollback_reason: str) -> None:
        for l in new_lessons:
            self.rejected_action_lessons.append({
                "source_cycle": source_cycle, "lesson_text": self._text(l),
                "tag": l.get("tag"), "reason": l.get("reason", ""),
                "reward_before": reward_before, "reward_after": reward_after,
                "accepted": False, "rollback_reason": rollback_reason})

    def to_dict(self) -> Dict[str, object]:
        import hashlib
        return {
            "current_action_prompt_version": self.current_action_prompt_version,
            "base_action_prompt": self.base_action_prompt,
            "active_action_prompt": self.active_prompt(),
            "active_action_prompt_hash": hashlib.sha256(
                self.active_prompt().encode()).hexdigest()[:16],
            "accepted_action_lessons": self.accepted_action_lessons,
            "rejected_action_lessons": self.rejected_action_lessons,
        }


@dataclass
class DirectionPromptMemory:
    """Learned Direction-Discovery prompt: base prompt + accepted TRANSFERABLE lessons.

    The active prompt is ``base_prompt + accepted lessons``. A candidate prompt is
    evaluated each training round; if it improves training PnL it is accepted, else
    rolled back (existing accept/reject rule). Every lesson must pass
    ``sanitize_reflection`` -- forbidden specifics (tickers, pairs, dates, specific
    LONG_SHORT/SHORT_LONG / pair preferences) are rejected and NEVER enter the prompt.
    """
    base_prompt: str
    symbols: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_UNIVERSE))
    accepted_lessons: List[Dict[str, object]] = field(default_factory=list)
    rejected_lessons: List[Dict[str, object]] = field(default_factory=list)
    version: int = 1
    max_lessons: int = 8

    @staticmethod
    def _text(l: Dict[str, object]) -> str:
        return str(l.get("lesson_text") or l.get("text") or "")

    def _safe(self, lessons: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        return [l for l in lessons if sanitize_reflection(self._text(l), self.symbols)]

    def _block(self, lessons: Sequence[Dict[str, object]]) -> str:
        if not lessons:
            return ""
        return ("\n\nLearned Decision Principles (transferable; from training reflection):\n"
                + "\n".join(f"- {self._text(l)}" for l in lessons))

    def _recent(self) -> List[Dict[str, object]]:
        return self.accepted_lessons[-self.max_lessons:]

    def active_prompt(self) -> str:
        return self.base_prompt + self._block(self._recent())

    def candidate_prompt(self, new_lessons: Sequence[Dict[str, object]]) -> str:
        combined = (self.accepted_lessons + self._safe(new_lessons))[-self.max_lessons:]
        return self.base_prompt + self._block(combined)

    def accept(self, new_lessons: Sequence[Dict[str, object]], *, source_round: int,
               reward_before: float, reward_after: float) -> int:
        kept = self._safe(new_lessons)
        for l in kept:
            self.accepted_lessons.append({
                "source_round": source_round, "lesson_text": self._text(l),
                "reward_before": reward_before, "reward_after": reward_after, "accepted": True})
        self.version += 1
        return len(kept)

    def reject(self, new_lessons: Sequence[Dict[str, object]], *, source_round: int,
               reward_before: float, reward_after: float, reason: str) -> None:
        for l in new_lessons:
            self.rejected_lessons.append({
                "source_round": source_round, "lesson_text": self._text(l),
                "reward_before": reward_before, "reward_after": reward_after,
                "accepted": False, "reject_reason": reason})

    def to_dict(self) -> Dict[str, object]:
        return {"version": self.version, "base_prompt": self.base_prompt,
                "active_prompt": self.active_prompt(),
                "accepted_lessons": self.accepted_lessons,
                "rejected_lessons": self.rejected_lessons}


def _passes_hard_minimums(summary: Dict[str, object], policy: "Policy") -> bool:
    """Hard qualification gate from a pair summary + the policy's thresholds.

    Note: this is a QUALIFICATION test, no longer a NO_TRADE gate. Pair selection
    always returns a pair; failing this test only marks the pair unqualified/forced.
    """
    try:
        return (
            float(summary.get("cointegration_pvalue", 1.0)) <= policy.coint_pmax
            and float(summary.get("adf_pvalue", 1.0)) <= policy.adf_pmax
            and policy.half_life_min <= float(summary.get("half_life_days", 1e9))
            <= policy.half_life_max
            and not bool(summary.get("relationship_breakdown", False))
            and float(summary.get("composite_score", 0.0)) >= policy.no_trade_threshold
        )
    except (TypeError, ValueError):
        return False


def _failed_thresholds(summary: Dict[str, object], policy: "Policy") -> List[str]:
    """Explain which hard minimums a (legacy) pair summary fails (Step 1/4)."""
    failed: List[str] = []
    try:
        cp = float(summary.get("cointegration_pvalue", 1.0))
        if cp > policy.coint_pmax:
            failed.append(f"cointegration p={cp:.3f} > {policy.coint_pmax:.3f}")
        ap = float(summary.get("adf_pvalue", 1.0))
        if ap > policy.adf_pmax:
            failed.append(f"ADF p={ap:.3f} > {policy.adf_pmax:.3f}")
        hl = float(summary.get("half_life_days", 1e9))
        if not (policy.half_life_min <= hl <= policy.half_life_max):
            failed.append(f"half-life {hl:.1f}d outside "
                          f"[{policy.half_life_min:.0f},{policy.half_life_max:.0f}]")
        comp = float(summary.get("composite_score", 0.0))
        if comp < policy.no_trade_threshold:
            failed.append(f"composite {comp:.3f} < {policy.no_trade_threshold:.3f}")
        if bool(summary.get("relationship_breakdown", False)):
            failed.append("active structural breakdown")
    except (TypeError, ValueError):
        failed.append("unparseable metrics")
    return failed


def _is_no_trade_value(v) -> bool:
    """True if an LLM 'selected_pair' value is NO_TRADE / malformed (forces override)."""
    if isinstance(v, str):
        return v.strip().upper().replace(" ", "_") == NO_TRADE
    if isinstance(v, (list, tuple)):
        return (len(v) < 2) or (len(v) == 1) or (
            isinstance(v[0], str) and v[0].strip().upper().replace(" ", "_") == NO_TRADE)
    return True


class PolicyAgent:
    def __init__(
        self,
        *,
        backend: str = "offline",
        llm=None,
        symbols: Optional[Sequence[str]] = None,
        action_prompt: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.llm = llm
        self.symbols = list(symbols or [])
        if backend == "open_source" and llm is None:
            raise ValueError("backend='open_source' requires an llm instance")
        # Active action-selection prompt (Step 8): base from action_selection.txt,
        # optionally extended with accepted training-reflection action lessons.
        self.base_action_prompt = load_base_action_prompt()
        self.action_prompt = action_prompt if action_prompt is not None else self.base_action_prompt
        self.action_directional_bias = 0.0
        self.action_prompt_version = 1

    def set_action_prompt(self, text: str, directional_bias: float = 0.0,
                          version: Optional[int] = None) -> None:
        """Set the active action prompt + offline directional bias for subsequent calls."""
        self.action_prompt = text
        self.action_directional_bias = float(directional_bias)
        if version is not None:
            self.action_prompt_version = int(version)

    def extract_action_lessons(self, reflection: str, run) -> List[Dict[str, object]]:
        """Derive concise action-selection lessons from a training run (Step 3).

        Deterministic (works offline and openai). Lessons reinforce the base
        prompt's news-semantic, directional-preference policy -- they never
        contradict it. One lesson per cycle keeps the prompt concise.
        """
        recs = [r for r in run.records if not getattr(r, "no_trade", False)]
        n = len(recs) or 1
        flat_ratio = sum(1 for r in recs if r.action == "FLAT") / n
        cum = float(getattr(run, "cumulative_pnl", 0.0))
        # A candidate lesson always proposes a real behavioural change (the acceptance
        # gate decides if it helps): trade MORE directionally when FLAT-heavy/profitable,
        # or be MORE selective when losing.
        if cum < 0:
            return [{"tag": "CAUTION",
                     "text": ("Open a directional position only when the news semantics and "
                              "the spread z-score agree; under a high-risk (pre-break) WARNING "
                              "require a convincing news signal before trading."),
                     "reason": f"training PnL {cum:+.4f} negative"}]
        return [{"tag": "DIRECTIONAL",
                 "text": ("Avoid excessive FLAT: when one leg has clearly stronger "
                          "company-specific news than the other, express a directional "
                          "LONG_SHORT/SHORT_LONG view rather than staying FLAT."),
                 "reason": f"FLAT ratio {flat_ratio:.0%}; reinforce directional preference"}]

    # ----------------------------- pair selection -------------------------- #
    def select_pair(self, pair_summaries: List[Dict[str, object]], policy: Policy,
                    pair_memory: Optional[PairReflectionMemory] = None) -> PairChoice:
        """Choose exactly ONE pair from CODE-computed TEXT summaries (Step 1).

        Pair selection ALWAYS returns a pair -- never NO_TRADE. A pair that passes
        the hard minimums (cointegration/ADF p-values, half-life range, no active
        breakdown, composite >= threshold) is ``qualified``. If NONE qualify, the
        closest-to-qualified pair (highest composite) is force-selected with
        ``qualified=False`` and ``forced_pair_selection=True``, and the failed
        thresholds are recorded. The Pair Reflection Memory (success + failure) is
        injected so selection evolves over time.
        """
        if not pair_summaries:
            raise ValueError("No candidate pairs provided")
        allowed = {tuple(s["pair"]) for s in pair_summaries}
        passing = [s for s in pair_summaries if _passes_hard_minimums(s, policy)]
        mem_used = bool(pair_memory and pair_memory.has_content())
        qualified = bool(passing)
        forced = not qualified
        # summaries are ranked by composite desc -> [0] is the best candidate
        top = passing[0] if passing else pair_summaries[0]
        failed = [] if qualified else _failed_thresholds(top, policy)
        succ_used = bool(pair_memory and getattr(pair_memory, "success_pairs", None))
        fail_used = bool(pair_memory and getattr(pair_memory, "failed_pairs", None))

        def _finish(pair, reasoning, conf, risks, model, prompt, raw):
            return PairChoice(
                pair=pair, reasoning=reasoning, confidence=conf, risks=risks,
                model=model, prompt=prompt, raw_response=raw,
                pair_reflection_used=mem_used, qualified=qualified,
                forced_pair_selection=forced, failed_thresholds=failed,
                base_score=float(top.get("composite_score", 0.0) or 0.0),
                adjusted_score=float(top.get("composite_score", 0.0) or 0.0),
                success_memory_used=succ_used, failure_memory_used=fail_used)

        if self.backend == "offline":
            pair = (str(top["pair"][0]), str(top["pair"][1]))
            risks = _statarb_risks(top)
            conf = float(min(1.0, max(0.0, float(top.get("composite_score", 0.0)))))
            conf = conf if qualified else min(conf, 0.4)  # lower confidence when forced
            reasoning = (
                f"{'Highest composite among qualifying pairs' if qualified else 'FORCED: no pair qualified; best available'}"
                f" ({top.get('composite_score')}): cointegration p={top.get('cointegration_pvalue')}, "
                f"ADF p={top.get('adf_pvalue')}, half-life {top.get('half_life_days')}d."
            )
            return _finish(pair, reasoning, conf, risks, "offline-deterministic",
                           "(offline rule: argmax composite; never NO_TRADE)",
                           json.dumps({"selected_pair": list(pair),
                                       "qualified": qualified}))

        # open-weight backend -- send TEXT summaries + memory only
        memory_text = pair_memory.render() if pair_memory else "(no pair reflection memory yet)"
        system = STATARB_SELECTION_PROMPT + "\n\n" + memory_text
        text_blocks = [s.get("text", "") for s in pair_summaries]
        payload = {"candidate_pairs": text_blocks,
                   "candidates_passing_hard_minimums": ["/".join(s["pair"]) for s in passing],
                   "any_pair_qualified": qualified}
        raw = self.llm.call_text(system_prompt=system, payload=payload)
        data = _extract_json(raw)
        pv = data.get("selected_pair") or data.get("pair")
        prompt_full = system + "\n\nPAYLOAD:\n" + json.dumps(payload, ensure_ascii=True)
        risks = data.get("risks") or []
        if isinstance(risks, str):
            risks = [risks]
        if isinstance(pv, str) and "/" in pv:
            pv = [p.strip() for p in pv.split("/")]
        # NEVER NO_TRADE: a NO_TRADE / malformed answer is overridden to the top pair.
        if _is_no_trade_value(pv):
            pair = (str(top["pair"][0]).upper(), str(top["pair"][1]).upper())
        else:
            pair = (str(pv[0]).upper(), str(pv[1]).upper())
            if pair not in allowed and (pair[1], pair[0]) in allowed:
                pair = (pair[1], pair[0])
        choice = _finish(
            pair, str(data.get("reasoning", "")).strip(),
            float(data.get("confidence", 0.5) or 0.5), [str(r) for r in risks],
            getattr(self.llm, "model", "open_source"), prompt_full, raw)
        choice.pair_reflection_used = bool(data.get("pair_reflection_used", mem_used))
        if "qualified" in data:
            choice.qualified = bool(data.get("qualified"))
            choice.forced_pair_selection = bool(data.get("forced_pair_selection", not choice.qualified))
        return choice

    # ------------------------- Stage 2: Direction Discovery ---------------- #
    @staticmethod
    def _build_direction_payload(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Per-candidate payload for Direction Discovery (Steps 5/9).

        Each entry carries, for the chosen pair (legs A/B):
          * spread_state (cointegration rank/p, beta, half-life, current spread);
          * long_horizon_z_score  -- deviation from the 10Y historical equilibrium
            (from the long-horizon StatArbPair spread mean/std);
          * trading_window_z_score -- near-term spread pressure from the short trading
            window (supplied by the engine; falls back to the long-horizon z);
          * current prices price_A_T / price_B_T and daily deltas
            price_*_delta = price(T) - price(T-1) (+ percentage delta);
          * pair-centric news: A summary, B summary, and an A-vs-B RELATIVE comparison.
        """
        from utils.statarb import summarize_stock_news
        out: List[Dict[str, object]] = []
        for i, c in enumerate(candidates, 1):
            a, b = c["pair"]
            na = summarize_stock_news(c.get("news_a") or [])
            nb = summarize_stock_news(c.get("news_b") or [])
            ss = build_spread_state(i, c["pc"])
            long_z = ss["current_zscore"]
            tw_z = c.get("trading_window_z")
            trading_z = round(float(tw_z), 3) if tw_z is not None else long_z

            def _delta(cur, prev):
                if cur is None or prev is None:
                    return None, None
                d = round(float(cur) - float(prev), 6)
                pct = round((d / float(prev) * 100.0), 4) if prev else None
                return d, pct
            pa, pb = c.get("price_a"), c.get("price_b")
            da, da_pct = _delta(pa, c.get("prev_price_a"))
            db, db_pct = _delta(pb, c.get("prev_price_b"))
            out.append({
                "pair": [a, b], "rank": i,
                "spread_state": ss,
                "long_horizon_z_score": long_z,
                "trading_window_z_score": trading_z,
                "price_A_T": (round(float(pa), 4) if pa is not None else None),
                "price_B_T": (round(float(pb), 4) if pb is not None else None),
                "price_A_delta": da, "price_A_delta_pct": da_pct,
                "price_B_delta": db, "price_B_delta_pct": db_pct,
                "news_A_summary": {"asset": a, "sentiment": na.get("sentiment"),
                                   "major_event": na.get("major_event"), "risk": na.get("risk")},
                "news_B_summary": {"asset": b, "sentiment": nb.get("sentiment"),
                                   "major_event": nb.get("major_event"), "risk": nb.get("risk")},
                "relative_A_vs_B": _relative_news(a, na, b, nb)})
        return out

    def discover_direction(
        self, *, candidates: List[Dict[str, object]], action_prompt: Optional[str] = None,
        pair_memory: Optional[PairSelectionMemory] = None, entry_z: float = 1.0,
    ) -> DirectionDecision:
        """Mode A -- PAIR SELECTION: pick ONE pair from the causal Top-5 (no action here).

        ``candidates`` is the ranked Top-5 (rank 1 = strongest 10Y cointegration). The LLM
        (openai) sees per-candidate spread-state + relative news + COMPANY CONTEXT (business
        only) + Pair Selection memory, and returns ``{selected_pair,
        pair_selection_reason}``. The chosen pair's daily ACTION is decided separately by
        ``discover_daily_direction`` (Direction-Discovery prompt). The offline backend uses
        a deterministic rule. The returned ``action`` is a placeholder (FLAT) -- the engine
        ignores it and runs Direction Discovery for the action.
        """
        from utils.company_profiles import profiles_block
        cand_payload = self._build_direction_payload(candidates)

        if self.backend != "open_source":
            return self._discover_offline(candidates, cand_payload, entry_z)

        tickers = sorted({t for c in candidates for t in c["pair"]})
        system = load_pair_selection_discovery_prompt() + "\n\n" + profiles_block(tickers)
        if pair_memory is not None and pair_memory.has_content():
            system = system + "\n\n" + pair_memory.render()
        model = getattr(self.llm, "model", "open_source")
        raw = ""
        try:
            raw = self.llm.call_text(system_prompt=system,
                                     payload={"top5_candidates": cand_payload},
                                     task_type="pair_selection")
        except InferenceError:
            raise
        except Exception:
            return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat,
                                     "pair-selection LLM exception -> top pair", model, raw,
                                     failed=True, failure_type="llm_exception")
        try:
            data = _extract_json(raw)
        except Exception:
            return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat,
                                     "unparseable JSON -> top pair", model, raw,
                                     failed=True, failure_type="parse_error")
        allowed = {tuple(c["pair"]) for c in candidates}
        pv = data.get("selected_pair") or data.get("pair")
        if isinstance(pv, str) and "/" in pv:
            pv = [p.strip() for p in pv.split("/")]
        sel = None
        if isinstance(pv, (list, tuple)) and len(pv) == 2:
            cand = (str(pv[0]).upper(), str(pv[1]).upper())
            sel = (cand if cand in allowed else
                   cand[::-1] if cand[::-1] in allowed else None)
        if sel is None:                       # invalid/unknown pair -> top-ranked pair
            return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat,
                                     "invalid selected_pair -> top pair", model, raw,
                                     failed=True, failure_type="invalid_pair")
        reason = str(data.get("pair_selection_reason") or data.get("reason", "")).strip()
        return DirectionDecision(sel, Action.flat, reason, model, raw)

    def discover_daily_direction(
        self, *, candidate: Dict[str, object], action_prompt: Optional[str] = None,
        pair_memory: Optional[PairSelectionMemory] = None, entry_z: float = 1.0,
        recent_memory: Optional[List[Dict[str, object]]] = None,
        lifecycle_state: str = "HEALTHY",
    ) -> DirectionDecision:
        """Mode B -- DAILY Direction Discovery on the ALREADY-ACTIVE pair (no re-selection).

        ``candidate`` is a single dict ``{"pair":(A,B),"pc":StatArbPair,"news_a","news_b",
        "price_a","price_b","prev_price_a","prev_price_b","trading_window_z"}`` for the
        currently held pair. Returns ONLY an action (the pair is fixed). The offline
        backend uses the deterministic mean-reversion rule on that pair's z; the openai
        backend asks the LLM for an action only, with structured failure flags. The pair
        in the returned decision is ALWAYS the active pair -- it is never changed here.
        """
        active_pair = tuple(candidate["pair"])
        payload = self._build_direction_payload([candidate])[0]
        if self.backend != "open_source":
            # deterministic: trade the active pair if its z is stretched, else FLAT
            z = float(payload.get("trading_window_z_score")
                      if payload.get("trading_window_z_score") not in (None, 0.0)
                      else payload.get("long_horizon_z_score") or 0.0)
            if abs(z) < entry_z:
                return DirectionDecision(active_pair, Action.flat,
                                         f"daily: |z|={abs(z):.2f}<{entry_z} -> FLAT")
            act = Action.short_long if z > 0 else Action.long_short
            return DirectionDecision(active_pair, act, f"daily mean-reversion z={z:.2f}")

        from utils.company_profiles import profiles_block
        system = (action_prompt or load_direction_discovery_prompt()) + "\n\n" + \
            profiles_block(list(active_pair))
        if pair_memory is not None and pair_memory.has_content():
            system = system + "\n\n" + pair_memory.render()
        model = getattr(self.llm, "model", "open_source")
        raw = ""
        try:
            raw = self.llm.call_text(system_prompt=system,
                                     payload={"active_pair": list(active_pair),
                                              "active_candidate": payload,
                                              "lifecycle_state": lifecycle_state,
                                              "past_3_days_direction_memory": list(recent_memory or [])},
                                     task_type="daily_direction")
        except InferenceError:
            raise
        except Exception:
            return DirectionDecision(active_pair, Action.flat,
                                     "daily LLM exception -> FLAT", model, raw,
                                     failed=True, failure_type="llm_exception")
        try:
            data = _extract_json(raw)
        except Exception:
            return DirectionDecision(active_pair, Action.flat, "daily unparseable -> FLAT",
                                     model, raw, failed=True, failure_type="parse_error")
        try:
            action = action_from_label(data.get("action"))
        except Exception:
            return DirectionDecision(active_pair, Action.flat, "daily invalid action -> FLAT",
                                     model, raw, failed=True, failure_type="invalid_action")
        reason = str(data.get("direction_reason") or data.get("reason", "")).strip()
        return DirectionDecision(active_pair, action, reason, model, raw)

    def _discover_offline(self, candidates, cand_payload, entry_z: float) -> DirectionDecision:
        """Deterministic Direction Discovery: trade the most-stretched spread; combine
        the relative-news edge with the mean-reversion (z) signal; else FLAT.

        Mean reversion: z>0 -> leg A rich -> SHORT_LONG; z<0 -> leg A cheap -> LONG_SHORT.
        The candidate with the largest |z| beyond ``entry_z`` is chosen; if none is
        stretched, the top-ranked pair is held FLAT.
        """
        best = max(cand_payload, key=lambda c: abs(c["spread_state"]["current_zscore"]))
        z = float(best["spread_state"]["current_zscore"])
        pair = tuple(best["pair"])
        if abs(z) < entry_z:
            return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat,
                                     f"no stretched spread (max |z|={abs(z):.2f}<{entry_z}) -> FLAT")
        action = Action.short_long if z > 0 else Action.long_short
        rel = best["relative_A_vs_B"]["summary"]
        return DirectionDecision(pair, action,
                                 f"z={z:.2f} (mean-reversion); {rel}")

    # ------------------------- multi-horizon pair selection ---------------- #
    def select_pair_multihorizon(
        self, mh_summaries: List[Dict[str, object]], policy: Policy, *,
        selection_horizon: str, qualified: bool,
        pair_memory: Optional[PairReflectionMemory] = None,
    ) -> PairChoice:
        """Choose ONE pair from MULTI-HORIZON profiles (Steps 4-6).

        ``mh_summaries`` are ``summarize_multihorizon_text`` dicts for the eligible
        candidates (already reflection-adjusted and ranked by adjusted_score). The
        FULL multi-horizon profile text is sent to GPT-4.1-mini. Always returns a
        pair; ``qualified``/``forced_pair_selection`` reflect the horizon-priority
        outcome. NO_TRADE is never returned.
        """
        if not mh_summaries:
            raise ValueError("No candidate multi-horizon profiles provided")
        allowed = {tuple(s["pair"]) for s in mh_summaries}
        top = mh_summaries[0]
        mem_used = bool(pair_memory and pair_memory.has_content())
        succ_used = bool(pair_memory and getattr(pair_memory, "success_pairs", None))
        fail_used = bool(pair_memory and getattr(pair_memory, "failed_pairs", None))
        forced = not qualified

        def _finish(pair, summary, reasoning, conf, risks, model, prompt, raw):
            return PairChoice(
                pair=pair, reasoning=reasoning, confidence=conf, risks=risks,
                model=model, prompt=prompt, raw_response=raw,
                pair_reflection_used=mem_used, qualified=qualified,
                forced_pair_selection=forced,
                failed_thresholds=list(summary.get("failed_thresholds", [])),
                multi_horizon_profile_used=True,
                selection_horizon=selection_horizon,
                base_score=float(summary.get("base_score", 0.0) or 0.0),
                success_bonus=float(summary.get("success_bonus", 0.0) or 0.0),
                failure_penalty=float(summary.get("failure_penalty", 0.0) or 0.0),
                adjusted_score=float(summary.get("adjusted_score", 0.0) or 0.0),
                reflection_adjustment_reason=str(summary.get("reflection_adjustment_reason", "")),
                success_memory_used=succ_used, failure_memory_used=fail_used)

        if self.backend == "offline":
            pair = (str(top["pair"][0]), str(top["pair"][1]))
            conf = float(min(1.0, max(0.0, float(top.get("adjusted_score", 0.0)))))
            conf = conf if qualified else min(conf, 0.4)
            reasoning = (
                f"{'Top adjusted-score qualified pair' if qualified else 'FORCED: best available unqualified pair'} "
                f"at horizon {selection_horizon}; adjusted_score={top.get('adjusted_score')} "
                f"(base {top.get('base_score')} + bonus {top.get('success_bonus')} "
                f"- penalty {top.get('failure_penalty')}); trend {top.get('relationship_trend')}."
            )
            return _finish(pair, top, reasoning, conf,
                           _statarb_risks_mh(top), "offline-deterministic",
                           "(offline rule: argmax adjusted_score; never NO_TRADE)",
                           json.dumps({"selected_pair": list(pair),
                                       "selection_horizon": selection_horizon,
                                       "qualified": qualified}))

        memory_text = pair_memory.render() if pair_memory else "(no pair reflection memory yet)"
        system = STATARB_SELECTION_PROMPT + "\n\n" + memory_text
        payload = {
            "candidate_pairs": [s.get("text", "") for s in mh_summaries],
            "selection_horizon": selection_horizon,
            "any_pair_qualified": qualified,
        }
        raw = self.llm.call_text(system_prompt=system, payload=payload)
        data = _extract_json(raw)
        pv = data.get("selected_pair") or data.get("pair")
        if isinstance(pv, str) and "/" in pv:
            pv = [p.strip() for p in pv.split("/")]
        if _is_no_trade_value(pv):
            pair = (str(top["pair"][0]).upper(), str(top["pair"][1]).upper())
            chosen = top
        else:
            pair = (str(pv[0]).upper(), str(pv[1]).upper())
            if pair not in allowed and (pair[1], pair[0]) in allowed:
                pair = (pair[1], pair[0])
            chosen = next((s for s in mh_summaries if tuple(s["pair"]) == pair
                           or tuple(s["pair"]) == pair[::-1]), top)
        risks = data.get("risks") or []
        if isinstance(risks, str):
            risks = [risks]
        prompt_full = system + "\n\nPAYLOAD:\n" + json.dumps(payload, ensure_ascii=True)
        choice = _finish(pair, chosen, str(data.get("reasoning", "")).strip(),
                         float(data.get("confidence", 0.5) or 0.5), [str(r) for r in risks],
                         getattr(self.llm, "model", "open_source"), prompt_full, raw)
        if "qualified" in data:
            choice.qualified = bool(data.get("qualified"))
            choice.forced_pair_selection = bool(data.get("forced_pair_selection", not choice.qualified))
        if data.get("selection_horizon"):
            choice.selection_horizon = str(data.get("selection_horizon"))
        choice.multi_horizon_profile_used = bool(data.get("multi_horizon_profile_used", True))
        return choice

    # ------------------------- critical-news verification ------------------ #
    def _verify_critical_news(
        self, pair: Tuple[str, str], news_summary: Dict[str, object],
        candidate_event: str,
    ) -> Tuple[bool, Dict[str, object]]:
        """Verify a keyword critical-news CANDIDATE before allowing an immediate BREAK.

        Returns ``(is_critical, info)``. On the openai backend the LLM decides:
        critical iff ``severity == CRITICAL`` AND ``affects_pair_relationship``. The
        offline backend has no verifier, so the deterministic keyword match stands.
        """
        if self.backend != "open_source":
            return True, {"event_type": candidate_event, "affects_pair_relationship": True,
                          "severity": "CRITICAL", "reasoning": "offline keyword match (no LLM verifier)",
                          "verifier": "keyword"}
        payload = {"pair": list(pair), "candidate_event": candidate_event,
                   "past_week_news_summary": news_summary}
        try:
            raw = self.llm.call_text(system_prompt=CRITICAL_NEWS_VERIFY_PROMPT, payload=payload,
                                     task_type="critical_news_verification")
            data = _extract_json(raw)
            severity = str(data.get("severity", "LOW")).strip().upper()
            affects = bool(data.get("affects_pair_relationship", False))
            is_crit = (severity == "CRITICAL" and affects)
            return is_crit, {"event_type": str(data.get("event_type", candidate_event)),
                             "affects_pair_relationship": affects, "severity": severity,
                             "reasoning": str(data.get("reasoning", "")).strip(),
                             "verifier": "llm"}
        except InferenceError:
            raise
        except Exception:
            # On verifier failure, do NOT escalate to BREAK from a keyword alone.
            return False, {"event_type": candidate_event, "affects_pair_relationship": False,
                           "severity": "MEDIUM", "reasoning": "verifier error", "verifier": "llm-error"}

    # ------------------------- relationship-break detector ----------------- #
    def detect_relationship_break(
        self, *, pair: Tuple[str, str], recent_corr: float, signal_z: float,
        adf_pvalue: float, coint_pvalue: float, half_life: float,
        news_summary: Dict[str, object], pair_memory: Optional[PairReflectionMemory],
        policy: Policy, spread_vol_ratio: float = 1.0,
        event_type: str = "scheduled_break_validation",
        days_since_last_validation: int = 0,
        prev_status: str = HEALTHY,
        risk_validation_streak: int = 0,
    ) -> BreakAssessment:
        """Pre-break risk classification -> public 3-level HEALTHY/WARNING/BREAK
        (Steps 5/6/7). Statistics and news are scored separately:

          * statistical raw score -> stat_status HEALTHY / WARNING / BREAK_CANDIDATE
            via boundaries 1.8 / 3.6 (``stat_status_from_score``); BREAK_CANDIDATE is
            INTERNAL only;
          * news -> news_break_level HEALTHY / WARNING / BREAK (3-level).

        BREAK confirmation (Step 7), keeping the 23-day validation cadence:
          * news_break_level == BREAK -> IMMEDIATE break (any day; the supported
            immediate-news-break exception; verified on the openai backend);
          * on a SCHEDULED validation day, BREAK iff
              (A) stat_score >= 5.4, OR
              (B) stat_score >= 3.6 AND (news in {WARNING,BREAK} OR llm_confirmed
                  OR persistence >= 2).

        Off validation days only HEALTHY/WARNING are emitted (a BREAK_CANDIDATE
        becomes a HIGH-pre_break_risk WARNING that forces FLAT but never switches the
        pair). ``risk_validation_streak`` counts consecutive pre-break validation
        windows for the persistence path.
        """
        is_validation = (event_type == "scheduled_break_validation")

        # --- code scores (always computed) ---
        stat_pts, stat_lvl, stat_factors = statistical_break_score(
            recent_corr=recent_corr, signal_z=signal_z, adf_pvalue=adf_pvalue,
            coint_pvalue=coint_pvalue, half_life=half_life, policy=policy,
            spread_vol_ratio=spread_vol_ratio)
        stat_status = stat_status_from_score(stat_pts)          # HEALTHY/WARNING/BREAK_CANDIDATE
        news_pts, news_status, news_factors, news_break = news_break_level(news_summary)

        # --- news-BREAK VERIFICATION (structural candidate -> LLM -> immediate break) ---
        # A structural/keyword news BREAK is only a CANDIDATE; on the openai backend the
        # LLM verifier must confirm it truly breaks the pair before it forces an
        # immediate break. Unverified -> downgrade to a WARNING (still pair-unfavourable).
        critical_verification: Dict[str, object] = {}
        if news_break:
            candidate_event = _candidate_critical_event(news_summary)
            verified, critical_verification = self._verify_critical_news(
                pair, news_summary, candidate_event)
            if not verified:
                news_break, news_status = False, WARNING
                news_pts = min(news_pts, 2.0)
                news_factors = news_factors + [
                    f"news-break candidate '{candidate_event}' NOT verified "
                    f"({critical_verification.get('severity')}/"
                    f"affects={critical_verification.get('affects_pair_relationship')})"]

        # --- optional LLM confirming evidence (openai + scheduled validation only) ---
        # Uses the 3-level semantic-first news prompt; provides confirming evidence only.
        llm_news_level, llm_confirmed, llm_weakening = HEALTHY, False, False
        raw = ""
        model = "offline-deterministic"
        if self.backend == "open_source" and is_validation and not news_break:
            model = getattr(self.llm, "model", "open_source")
            memory_text = pair_memory.render() if pair_memory else "(no memory)"
            payload = {
                "pair": list(pair),
                "statistical_deterioration": stat_factors,
                "statistical_status": stat_status,
                "recent_rolling_correlation": round(recent_corr, 3),
                "half_life_days": round(half_life, 1),
                "adf_pvalue": round(adf_pvalue, 4),
                "cointegration_pvalue": round(coint_pvalue, 4),
                "current_spread_z": round(signal_z, 3),
                "spread_volatility_ratio": round(spread_vol_ratio, 2),
                "code_news_break_level": news_status,
                "past_week_news_summary": news_summary,
                "monitoring_event_type": event_type,
                "days_since_last_break_validation": days_since_last_validation,
            }
            try:
                raw = self.llm.call_text(
                    system_prompt=RELATIONSHIP_BREAK_PROMPT + "\n\n" + memory_text,
                    payload=payload)
                data = _extract_json(raw)
                lvl = str(data.get("news_break_level", "HEALTHY")).strip().upper()
                if lvl == "DANGER":          # backward-compat: never a public status
                    lvl = WARNING
                llm_news_level = lvl if lvl in (HEALTHY, WARNING, BREAK) else HEALTHY
                llm_confirmed = bool(data.get("is_relationship_break_signal",
                                              llm_news_level == BREAK))
                llm_weakening = llm_news_level in (WARNING, BREAK)
            except InferenceError:
                raise
            except Exception:
                llm_news_level, llm_confirmed, llm_weakening = HEALTHY, False, False

        # --- classification: public 3-level HEALTHY/WARNING/BREAK (Steps 6/7) ---
        news_confirms = news_status in (WARNING, BREAK)
        # persistence: a pre-break condition this validation window
        provisional_risk = (stat_status == BREAK_CANDIDATE) or news_confirms or llm_weakening
        if is_validation:
            effective_streak = (risk_validation_streak + 1) if provisional_risk else 0
        else:
            effective_streak = risk_validation_streak

        # Step 7 -- scheduled-day BREAK confirmation: (A) stat>=5.4 OR
        # (B) stat>=3.6 AND (news in {WARNING,BREAK} OR llm_confirmed OR persistence>=2).
        break_eligible = is_validation and (
            stat_pts >= _STAT_CRITICAL
            or (stat_pts >= _STAT_BREAK_CANDIDATE
                and (news_confirms or llm_confirmed or effective_streak >= 2)))
        immediate_news_break = bool(news_break)   # verified structural news, any day

        if immediate_news_break or break_eligible:
            status, pre_break_risk = BREAK, "CRITICAL"
        elif (stat_status == BREAK_CANDIDATE) or (news_status == BREAK):
            # serious pre-break stress short of confirmation -> HIGH-risk WARNING (forces FLAT)
            status, pre_break_risk = WARNING, "HIGH"
        elif (stat_status == WARNING) or (news_status == WARNING) or llm_weakening:
            status, pre_break_risk = WARNING, "MEDIUM"
        else:
            status, pre_break_risk = HEALTHY, "LOW"

        break_confirmed = (status == BREAK)
        critical_exception = bool(immediate_news_break and break_confirmed)
        # switching only on a scheduled validation day OR an immediate news-break
        should_switch = break_confirmed and (is_validation or immediate_news_break)
        # Lifecycle is INFORMATIONAL, never an action constraint: NO state (HEALTHY,
        # WARNING, or BREAK) forces FLAT. A confirmed BREAK only triggers pair
        # REPLACEMENT (``should_switch_pair`` -> retire + cooldown + re-selection); until
        # a replacement pair is chosen, Direction Discovery still freely picks
        # LONG_SHORT / SHORT_LONG / FLAT. ``pre_break_risk`` is a DIAGNOSTIC label only.
        should_force_flat = False
        risk_factors = stat_factors + news_factors
        monitoring_event_type = ("critical_news_exception" if critical_exception
                                 else event_type)
        reasoning = (
            f"status={status} (stat_score={stat_pts:.1f}/{stat_status}, news={news_status}, "
            f"llm={llm_news_level}; event={monitoring_event_type}, "
            f"persistence={effective_streak}). "
            + ("; ".join(risk_factors) if risk_factors else "relationship stable"))
        conf = min(1.0, 0.5 + 0.1 * (stat_pts + news_pts)) if status != HEALTHY else 0.2

        return BreakAssessment(
            relationship_break_detected=break_confirmed,
            reasoning=reasoning,
            break_type=(risk_factors[0].split("(")[0].strip() if risk_factors else "none"),
            confidence=conf, should_switch_pair=should_switch, model=model, raw_response=raw,
            relationship_status=status, pre_break_risk=pre_break_risk,
            should_force_flat=should_force_flat, risk_factors=risk_factors,
            statistical_risk=stat_lvl, news_risk=news_status, llm_risk=llm_news_level,
            statistical_break_score=stat_pts, news_break_score=news_pts,
            llm_break_score=float(news_status in (WARNING, BREAK)),
            monitoring_event_type=monitoring_event_type,
            days_since_last_break_validation=days_since_last_validation,
            persistence_count=effective_streak, scheduled_break_validation=is_validation,
            critical_news_exception=critical_exception, break_confirmed=break_confirmed,
            previous_relationship_status=prev_status,
            critical_news_verification=critical_verification)

    # ------------------------- pair reflection (on close/eval) ------------- #
    def reflect_on_pair(
        self, *, pair: Tuple[str, str], outcome_type: str,
        stats: Dict[str, object], news_summary: Dict[str, object],
        break_assessment: Optional[BreakAssessment] = None,
        performance: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Generate a SUCCESS / FAILURE / NEUTRAL Pair Reflection (Steps 7/8).

        Returns a structured record: reflection text, lessons, outcome_type,
        why_worked / why_failed bullets, and the statistical / news patterns. The
        record feeds the Success Memory or Failure Memory (Step 8) and reflection-
        adjusted ranking (Step 9). Offline uses a deterministic template; openai
        asks the model for the prose while the structured fields stay code-computed.
        """
        a, b = pair
        stats = stats or {}
        perf = performance or {}
        hl = float(stats.get("half_life_days") or 0.0)
        coint = float(stats.get("cointegration_pvalue") or 1.0)
        cum = float(perf.get("cumulative_pnl", 0.0) or 0.0)
        reverts = perf.get("reversions")
        news_risk = str(news_summary.get("pair_level_risk", "low"))
        reasoning = break_assessment.reasoning if break_assessment else "window evaluation"
        theme = pair_theme(a, b)

        lessons: List[str] = []
        why_worked: List[str] = []
        why_failed: List[str] = []
        statistical_patterns = [f"cointegration_p={coint:.3f}", f"half_life~{hl:.0f}d"]
        news_patterns = [f"pair_news_risk={news_risk}"]

        if outcome_type == "SUCCESS":
            why_worked.append(f"cumulative PnL positive ({cum:+.4f})")
            if coint <= 0.05:
                why_worked.append(f"cointegration strong (p={coint:.3f})")
                lessons.append("prefer cointegration p<=0.05")
            if 2 <= hl <= 20:
                why_worked.append(f"half-life {hl:.0f} days")
                lessons.append("prefer half-life < 20")
            if reverts:
                why_worked.append(f"spread reverted {reverts} times")
            if news_risk == "low":
                why_worked.append("no relationship-breaking news")
            lessons.append(f"prefer {theme} pairs")
            lessons.append("prefer stable rolling correlation")
        elif outcome_type == "FAILURE":
            why_failed.append(reasoning)
            if cum < 0:
                why_failed.append(f"cumulative PnL negative ({cum:+.4f})")
            if hl > 60:
                why_failed.append("half-life became too long")
                lessons.append("avoid pairs with half-life above 60 days")
            if coint > 0.05:
                lessons.append("do not rely on high correlation without significant cointegration")
            if news_risk == "high":
                lessons.append("avoid pairs where one leg has a recent idiosyncratic shock")
            if "rolling-correlation" in reasoning:
                lessons.append("exit when the rolling correlation collapses")
        else:  # NEUTRAL
            why_worked.append("flat/neutral outcome -- no decisive statistical signal")

        if self.backend == "offline":
            head = {"SUCCESS": "Why it worked", "FAILURE": "Why it failed",
                    "NEUTRAL": "Outcome"}.get(outcome_type, "Outcome")
            bullets = why_worked if outcome_type == "SUCCESS" else (
                why_failed if outcome_type == "FAILURE" else why_worked)
            text = (
                f"Previous Pair:\n{a} / {b}  (theme: {theme})\n"
                f"Outcome: {outcome_type}\n"
                f"{head}:\n" + "\n".join(f"- {x}" for x in bullets) + "\n"
                "Lessons:\n" + "\n".join(f"- {l}" for l in lessons or ['monitor relationship stability'])
            )
        else:
            payload = {"pair": [a, b], "theme": theme, "outcome_type": outcome_type,
                       "why_worked": why_worked, "why_failed": why_failed,
                       "stats": stats, "performance": perf, "news_summary": news_summary,
                       "reasoning": reasoning}
            system = (
                f"Write a concise PAIR REFLECTION (<=6 lines) classified as {outcome_type}. "
                "State why it worked or failed and 1-3 lessons to improve FUTURE pair "
                "selection. End with a line 'Lessons:' followed by the lessons.")
            text = self.llm.call_text(system_prompt=system, payload=payload).strip()

        return {
            "reflection_text": text, "lessons": lessons, "outcome_type": outcome_type,
            "why_worked": why_worked, "why_failed": why_failed,
            "statistical_patterns": statistical_patterns, "news_patterns": news_patterns,
        }

    # ----------------------------- action decision ------------------------- #
    def decide_action(
        self,
        *,
        pair: Tuple[str, str],
        position_state: PositionState,
        signal_z: float,
        recent_pnl_3d: float = 0.0,   # deprecated: no longer used (no 3-day-PnL stance)
        news_sentiment: str,
        policy: Policy,
        relationship_status: str = HEALTHY,
        pre_break_risk: str = "LOW",
        news_summary: Optional[Dict[str, object]] = None,
    ) -> Decision:
        """Return exactly one explicit action: LONG_SHORT / SHORT_LONG / FLAT.

        ``signal_z`` is the CODE-computed spread z-score (positive => leg A rich
        vs leg B). The decision follows the ACTIVE action prompt (base prompt +
        accepted training action lessons): prefer a directional view from news
        semantics; FLAT is a last resort. The relationship status gates caution --
        HEALTHY normal, WARNING more cautious (a HIGH ``pre_break_risk`` WARNING --
        the old DANGER level -- needs a convincing news signal), BREAK forces FLAT.
        """
        if self.backend == "offline":
            return self._decide_offline(
                signal_z=signal_z,
                news_sentiment=news_sentiment,
                policy=policy,
                relationship_status=relationship_status,
                pre_break_risk=pre_break_risk,
                directional_bias=self.action_directional_bias,
            )
        return self._decide_chat(
            pair=pair,
            position_state=position_state,
            signal_z=signal_z,
            news_sentiment=news_sentiment,
            news_summary=news_summary,
            policy=policy,
            relationship_status=relationship_status,
            pre_break_risk=pre_break_risk,
        )

    def _decide_offline(
        self, *, signal_z, news_sentiment, policy: Policy,
        relationship_status: str = HEALTHY, pre_break_risk: str = "LOW",
        directional_bias: float = 0.0,
    ) -> Decision:
        # BREAK forces FLAT regardless of signal.
        if relationship_status == BREAK:
            return Decision(Action.flat, "relationship BREAK -> force FLAT", relationship_status)

        az = abs(signal_z)
        strong_news = news_sentiment in ("Positive", "Negative")
        # A HIGH ``pre_break_risk`` WARNING is the old DANGER level (serious pre-break
        # stress); it gets the strongest caution. A normal (MEDIUM) WARNING is milder.
        high_risk_warning = (relationship_status == WARNING and pre_break_risk == "HIGH")
        # Base policy PREFERS a directional view. Accepted DIRECTIONAL action lessons
        # (bias > 0) LOWER the trade threshold (express a position on a weaker case --
        # avoid excessive FLAT); CAUTION lessons (bias < 0) raise it. Relationship-status
        # caution adds to the threshold on top. There is no 3-day-PnL stance.
        if directional_bias >= 0:
            bias_factor = 1.0 / (1.0 + 0.45 * directional_bias)  # <= 1 -> trade more
        else:
            bias_factor = 1.0 + 0.35 * (-directional_bias)       # > 1 -> trade less
        floor = policy.exit_band * bias_factor
        if az < floor:
            return Decision(Action.flat, f"z={signal_z:.2f} below floor {floor:.2f} -> FLAT",
                            relationship_status)
        caution = (0.7 if high_risk_warning
                   else {HEALTHY: 0.0, WARNING: 0.35}.get(relationship_status, 0.0))
        thr = floor + max(0.0, policy.entry_z - policy.exit_band) * caution

        # High-risk (pre-break) WARNING -- the old DANGER safety: allow a directional
        # trade only when news provides a convincing signal.
        if high_risk_warning and not strong_news and az < policy.entry_z * 1.2:
            return Decision(Action.flat,
                            f"high-risk WARNING without convincing news -> FLAT (z={signal_z:.2f})",
                            relationship_status)
        if az < thr:
            return Decision(Action.flat,
                            f"z={signal_z:.2f} below threshold {thr:.2f} "
                            f"({relationship_status}) -> FLAT", relationship_status)
        if signal_z > 0:  # leg A rich -> short A, long B
            return Decision(Action.short_long,
                            f"z={signal_z:.2f}>0 (A rich) -> SHORT_LONG ({relationship_status})",
                            relationship_status)
        return Decision(Action.long_short,
                        f"z={signal_z:.2f}<0 (A cheap) -> LONG_SHORT ({relationship_status})",
                        relationship_status)

    def _decide_chat(
        self, *, pair, position_state, signal_z, news_sentiment, news_summary, policy,
        relationship_status: str = HEALTHY, pre_break_risk: str = "LOW",
    ) -> Decision:
        # BREAK is decided in code (force FLAT) -- never spend an LLM call.
        if relationship_status == BREAK:
            return Decision(Action.flat, "relationship BREAK -> force FLAT", relationship_status)
        # The ACTIVE action prompt (base prompt + accepted action lessons) is the system
        # prompt; it already specifies the action set, news-semantic priority, the
        # HEALTHY/WARNING/BREAK rules (with pre_break_risk) and the JSON schema.
        system = self.action_prompt
        payload = {
            "pair": [pair[0], pair[1]],
            "current_position_state": position_state.label,
            "spread_zscore": round(float(signal_z), 3),
            "news_sentiment": news_sentiment,
            "news_summary": news_summary or {},
            "relationship_status": relationship_status,
            "pre_break_risk": pre_break_risk,
        }
        text = self.llm.call_text(system_prompt=system, payload=payload,
                                  task_type="trading_action")
        try:
            data = _extract_json(text)
            action = action_from_label(data.get("action"))
        except Exception:
            # A malformed/truncated LLM response must never crash the run -> FLAT.
            return Decision(Action.flat, "unparseable LLM action response -> FLAT",
                            relationship_status)
        return Decision(action, str(data.get("rationale", "")).strip(), relationship_status)

    # ------------------- Direction-Discovery reflection -------------------- #
    def reflect_direction(self, direction_memory) -> Dict[str, object]:
        """Round reflection from Direction Memory -> TRANSFERABLE decision principles only.

        Returns ``{"reflection_text": str, "lessons": [{"lesson_text": ...}, ...]}``.
        Lessons must improve the DECISION PROCESS (risk management, confidence,
        news/mean-reversion interpretation, when FLAT is appropriate, balancing news vs
        spread) and must NOT reference any pair/ticker/date/specific direction -- callers
        filter via ``sanitize_reflection`` (and ``DirectionPromptMemory`` enforces it).
        """
        recs = list(getattr(direction_memory, "records", []))
        graded = [r for r in recs if r.get("realized_pnl") is not None]
        n = len(graded) or 1
        wins = sum(1 for r in graded if (r.get("realized_pnl") or 0.0) > 0)
        flats = sum(1 for r in recs if str(r.get("action")) == "FLAT")
        avg = sum((r.get("realized_pnl") or 0.0) for r in graded) / n
        win_rate = wins / n
        # ----- counterfactual learning signals (Step 4) -----
        cf = [r for r in recs if r.get("right_or_wrong") is not None]
        n_cf = len(cf) or 1
        right = sum(1 for r in cf if r.get("right_or_wrong") == "RIGHT")
        accuracy = right / n_cf
        best = [str(r.get("best_action")) for r in cf]
        best_ls = best.count("LONG_SHORT")
        best_sl = best.count("SHORT_LONG")
        best_flat = best.count("FLAT")
        # how often each action would have been the right call vs how often it was chosen
        chose_flat = sum(1 for r in cf if str(r.get("action")) == "FLAT")
        flat_overused = chose_flat > best_flat            # chose FLAT more than it was optimal
        flat_underused = best_flat > chose_flat
        dir_bias = ("LONG_SHORT systematically better" if best_ls > best_sl * 1.5
                    else "SHORT_LONG systematically better" if best_sl > best_ls * 1.5
                    else "no systematic directional edge")
        from utils.counterfactual import counterfactual_diagnostics
        diag = counterfactual_diagnostics(cf)
        cf_stats = {"counterfactual_accuracy": round(accuracy, 3),
                    "best_action_counts": {"LONG_SHORT": best_ls, "SHORT_LONG": best_sl,
                                           "FLAT": best_flat},
                    "chose_flat": chose_flat, "flat_overused": flat_overused,
                    "flat_underused": flat_underused, "directional_bias": dir_bias,
                    "flat_best_before_deadband": diag["flat_best_before_deadband"],
                    "flat_best_after_deadband": diag["flat_best_after_deadband"],
                    "deadband_right_via_profit": diag["deadband_right_via_profit"],
                    "deadband_right_via_profit_pct": diag["deadband_right_via_profit_pct"]}
        if self.backend == "offline":
            if flat_overused and best_flat < n_cf * 0.5:
                lesson = ("Counterfactual review shows FLAT was over-used relative to when it "
                          "was actually optimal; when a clear relative edge exists, commit to a "
                          "directional position instead of defaulting to FLAT.")
            elif accuracy < 0.5:
                lesson = ("Counterfactual review shows the chosen side was often wrong; when the "
                          "daily price delta contradicts the spread-reversion assumption, weight "
                          "the near-term move more and reconsider the direction before trading.")
            else:
                lesson = ("Counterfactual review confirms the directional reads were mostly "
                          "correct; keep aligning the relative-news edge with the spread signal "
                          "and avoid unnecessary FLAT.")
            text = (f"Round: {len(graded)} graded, win rate {win_rate:.0%}, avg PnL {avg:+.4f}, "
                    f"FLAT {flats}; counterfactual accuracy {accuracy:.0%}, "
                    f"best-action LS/SL/FLAT={best_ls}/{best_sl}/{best_flat} ({dir_bias}). "
                    f"Principle: {lesson}")
            return {"reflection_text": text, "lessons": [{"lesson_text": lesson}]}
        # openai: ask for transferable principles, then the caller filters specifics out.
        system = (
            "You are a trading coach. From the round's DAILY decisions with COUNTERFACTUAL PnL "
            "(what LONG_SHORT/SHORT_LONG/FLAT would each have earned) write a SHORT reflection "
            "(<=6 lines) and 1-3 TRANSFERABLE decision PRINCIPLES that improve future direction "
            "decisions. NEVER mention a specific ticker, pair, date, step, or a specific "
            "LONG_SHORT/SHORT_LONG recommendation. Learn from: how often the chosen action matched "
            "the counterfactual best action, repeated wrong-side decisions, whether one direction "
            "was systematically better, whether FLAT was over- or under-used, whether the daily "
            "price delta contradicted spread reversion, and whether news supported continuation or "
            "reversion. End with a line 'Principles:' followed by the principles, one per line.")
        payload = {"decisions": [{"action": r.get("action"),
                                  "realized_pnl": r.get("realized_pnl"),
                                  "cf_long_short": r.get("cf_long_short"),
                                  "cf_short_long": r.get("cf_short_long"),
                                  "cf_flat": r.get("cf_flat"),
                                  "best_action": r.get("best_action"),
                                  "right_or_wrong": r.get("right_or_wrong")} for r in cf],
                   "win_rate": round(win_rate, 3), "avg_pnl": round(avg, 4),
                   "counterfactual_summary": cf_stats}
        try:
            text = self.llm.call_text(system_prompt=system, payload=payload).strip()
        except InferenceError:
            raise
        except Exception:
            text = "Balance the relative-news signal with the spread mean-reversion signal."
        lessons = []
        if "Principles:" in text:
            for line in text.split("Principles:", 1)[1].splitlines():
                s = line.strip(" -\t")
                if s:
                    lessons.append({"lesson_text": s})
        if not lessons:
            lessons = [{"lesson_text": "Balance the relative-news signal with the spread "
                        "mean-reversion signal; prefer FLAT when they conflict."}]
        return {"reflection_text": text, "lessons": lessons}

    # ----------------------------- reflection ------------------------------ #
    def reflect(
        self,
        *,
        policy: Policy,
        pnl_history: Sequence[float],
        trades: Sequence[Dict[str, object]],
        pair_selections: Sequence[str],
        previous_reflections: Sequence[str],
    ) -> str:
        """Generate an OVERALL reflection from the training cycle's experience."""
        wins = [t for t in trades if float(t.get("pnl", 0.0)) > 0]
        losses = [t for t in trades if float(t.get("pnl", 0.0)) < 0]
        cum = sum(pnl_history)
        if self.backend == "offline":
            win_rate = len(wins) / len(trades) if trades else 0.0
            return (
                f"Cycle cumulative PnL={cum:+.4f} over {len(trades)} active trades "
                f"(win rate {win_rate:.0%}; {len(wins)} wins / {len(losses)} losses). "
                f"Pairs used: {sorted(set(pair_selections))}. "
                + (
                    "Losses cluster on aggressive entries -> widen entry_z."
                    if cum < 0
                    else "Profitable mean-reversion -> entries are well-calibrated; "
                    "keep entry_z, consider tightening exit_band."
                )
            )
        system = (
            "You are a trading coach. Write an OVERALL reflection (<=8 lines) over "
            "this training cycle, using the PnL history, wins, losses, pair "
            "selections, and previous reflections. End with one concrete lesson.\n\n"
            + policy.render_context()
        )
        payload = {
            "cumulative_pnl": cum,
            "n_trades": len(trades),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "pairs_used": sorted(set(pair_selections)),
            "previous_reflections": list(previous_reflections)[-3:],
        }
        return self.llm.call_text(system_prompt=system, payload=payload).strip()

    # Deterministic neighbourhood explored by the offline coordinate search over the
    # behaviorally-active parameters only (entry_z, exit_band). On losses, become more
    # selective (widen entry_z) first; on gains, tighten exits first, then probe
    # trading a touch more. (risk_aversion removed -- it was inert.)
    _LOSING_ORDER = [
        ("entry_z", +0.25), ("entry_z", -0.25),
        ("exit_band", +0.05), ("exit_band", -0.05),
    ]
    _WINNING_ORDER = [
        ("exit_band", -0.05), ("exit_band", +0.05),
        ("entry_z", -0.25), ("entry_z", +0.25),
    ]
    _BOUNDS = {"entry_z": (0.5, 3.0), "exit_band": (0.1, 1.0)}

    def n_candidate_moves(self) -> int:
        return len(self._LOSING_ORDER)

    def propose_policy_update(
        self, *, policy: Policy, reflection: str, recent_cum_pnl: float, attempt: int = 0
    ) -> Policy:
        """Propose a CANDIDATE policy (acceptance/rollback handled by the trainer).

        Offline: deterministic coordinate search over the policy parameters. The
        ``attempt`` index walks a fixed neighbourhood; once it is exhausted the
        method returns an identical (no-op) candidate so the trainer's
        convergence test (overlap >= 80% AND |PnL change| < 3%) fires cleanly.

        OpenAI: ask the model to adjust parameters within bounds.
        """
        candidate = policy.copy()
        candidate.version = policy.version + 1
        candidate.reflection_text = reflection

        if self.backend == "offline":
            order = self._LOSING_ORDER if recent_cum_pnl < 0 else self._WINNING_ORDER
            if attempt >= len(order):
                # neighbourhood exhausted with no improvement -> propose no change
                candidate.lessons = policy.lessons + [
                    f"v{policy.version}: no improving neighbour -> policy stable."
                ]
                return candidate
            param, step = order[attempt]
            lo, hi = self._BOUNDS[param]
            newval = round(min(hi, max(lo, getattr(policy, param) + step)), 3)
            setattr(candidate, param, newval)
            candidate.lessons = policy.lessons + [
                f"v{policy.version}: try {param} {getattr(policy, param)}->{newval} "
                f"(recent PnL {recent_cum_pnl:+.4f})."
            ]
            return candidate

        system = (
            "You tune a pair-trading policy. Given the reflection and recent PnL, "
            "propose adjusted numeric parameters. Stay in bounds: entry_z in "
            "[0.5,3.0], exit_band in [0.1,1.0].\n"
            "Return ONLY JSON: {\"entry_z\": x, \"exit_band\": y, \"lesson\": \"one line\"}"
        )
        payload = {
            "current": {
                "entry_z": policy.entry_z,
                "exit_band": policy.exit_band,
            },
            "reflection": reflection,
            "recent_cumulative_pnl": recent_cum_pnl,
        }
        try:
            data = _extract_json(self.llm.call_text(system_prompt=system, payload=payload))
            candidate.entry_z = float(min(3.0, max(0.5, data.get("entry_z", policy.entry_z))))
            candidate.exit_band = float(min(1.0, max(0.1, data.get("exit_band", policy.exit_band))))
            lesson = str(data.get("lesson", "")).strip()
            if lesson:
                candidate.lessons = policy.lessons + [f"v{policy.version}: {lesson}"]
        except InferenceError:
            raise
        except Exception:
            return policy.copy()  # fall back to no-op candidate
        return candidate


def _statarb_risks(summary: Dict[str, object]) -> List[str]:
    """Derive risk flags from a stat-arb pair summary (offline reasoning)."""
    risks: List[str] = []
    if summary.get("relationship_breakdown"):
        risks.append("structural relationship breakdown detected")
    if not summary.get("half_life_ok", True):
        risks.append(f"half-life {summary.get('half_life_days')}d outside 2-60 day range")
    try:
        if float(summary.get("cointegration_pvalue", 1.0)) > 0.05:
            risks.append(f"weak cointegration (p={summary.get('cointegration_pvalue')})")
    except (TypeError, ValueError):
        pass
    news = summary.get("news_summary", {}) or {}
    if news.get("pair_level_risk") in ("medium", "high"):
        risks.append(f"news pair-level risk: {news.get('pair_level_risk')}")
    return risks or ["none material"]


def _statarb_risks_mh(summary: Dict[str, object]) -> List[str]:
    """Risk flags from a multi-horizon summary (offline reasoning)."""
    risks: List[str] = []
    if summary.get("failed_thresholds"):
        risks.append("forced/unqualified: " + "; ".join(summary["failed_thresholds"][:2]))
    trend = str(summary.get("relationship_trend", ""))
    if trend in ("weakening recently", "inconsistent", "long-term only"):
        risks.append(f"relationship trend {trend}")
    news = summary.get("news_summary", {}) or {}
    if news.get("pair_level_risk") in ("medium", "high"):
        risks.append(f"news pair-level risk: {news.get('pair_level_risk')}")
    return risks or ["none material"]


def _extract_json(text: str) -> Dict[str, object]:
    import re

    cleaned = (text or "").strip().replace("```json", "").replace("```", "").strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in response: {text!r}")
    return json.loads(m.group(0))
