"""Deterministic pair-trading simulation engine.

A single ``simulate_window`` function drives the portfolio over a date range
using a given ``Policy`` and ``PolicyAgent``. It is reused for:
  * training-cycle baseline/candidate evaluation,
  * convergence action-overlap comparisons,
  * the final TEST phase.

Key invariants enforced here (audit fixes):
  * NO HIDDEN POSITIONS / NO RESIDUAL EXPOSURE: every day we build the target
    from a zeroed vector and only set the *active pair's* two legs. Any other
    symbol is therefore forced flat every single step.
  * NO TWO-PAIR EXPOSURE: switching pairs goes Current -> FLAT -> verify flat ->
    New pair. We never open a new pair while the old one still has exposure.
  * NO HOLD: a fresh explicit target (LONG_SHORT / SHORT_LONG / FLAT) is emitted
    every day; positions are never carried implicitly.
  * REJECTED TARGETS DON'T REPLAY: if the portfolio rejects a target we clear to
    FLAT for that pair and record the rejection -- we never resubmit it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from strategies.types import Action, PositionState, action_to_state
from utils.portfolio import (
    PortfolioManager,
    compute_hedge_ratio,
    effective_hedge_ratio,
    size_pair_position_fractional,
)
from utils.pair_stats import _sentiment_label
from utils.counterfactual import (counterfactual_pnls, best_action as _cf_best,
                                  is_right as _cf_right, memory_sentence as _cf_sentence)
from utils.statarb import (
    rank_statarb_pairs,
    summarize_pair_text,
    summarize_stock_news,
    pair_news_risk,
    current_zscore,
)


def compute_spread_z(
    price_history: Dict[date, Dict[str, Dict[str, float]]],
    pair: Tuple[str, str],
    lookback: int = 20,
) -> Tuple[float, Optional[float]]:
    """Code-computed spread z-score for the pair. Returns (z, hedge_ratio).

    spread = log(A) - beta*log(B); z = (spread - mean)/std over the lookback.
    Positive z => leg A is rich relative to leg B.
    """
    a, b = pair
    beta = compute_hedge_ratio(price_history, pair, lookback)
    dates = sorted(price_history.keys())[-lookback:]
    spreads: List[float] = []
    if beta is None:
        beta = 1.0
    for d in dates:
        prices = price_history[d].get("price", {})
        pa, pb = prices.get(a), prices.get(b)
        if pa is None or pb is None or pa <= 0 or pb <= 0:
            continue
        spreads.append(math.log(pa) - beta * math.log(pb))
    if len(spreads) < 3:
        return 0.0, beta
    s = pd.Series(spreads)
    mu, sd = s.mean(), s.std()
    if sd is None or sd <= 1e-12 or math.isnan(sd):
        return 0.0, beta
    z = (spreads[-1] - mu) / sd
    if math.isnan(z) or math.isinf(z):
        return 0.0, beta
    return float(z), beta


@dataclass
class StepRecord:
    date: str
    selected_pair: str
    action: str
    position_state: str
    stance: str
    signal_z: float
    daily_pnl: float
    cumulative_pnl: float
    normalized_equity: float
    news_sentiment: str
    rationale: str
    rejected_target: bool
    pair_switched: bool
    no_trade: bool = False
    relationship_break_detected: bool = False
    break_reasoning: str = ""
    should_switch_pair: bool = False
    reason: Optional[str] = None
    # --- pre-break risk state (Steps 10/11) ---
    relationship_status: str = "HEALTHY"
    pre_break_risk: str = "LOW"
    should_force_flat: bool = False
    risk_factors: List[str] = field(default_factory=list)
    # --- conservative break cadence/scoring (15-day validation rule) ---
    monitoring_event_type: str = "daily_risk_check"
    days_since_last_break_validation: int = 0
    scheduled_break_validation: bool = False
    critical_news_exception: bool = False
    statistical_risk: str = "LOW"
    news_risk: str = "LOW"
    statistical_break_score: float = 0.0
    news_break_score: float = 0.0
    persistence_count: int = 0
    # --- pair lifecycle (ACTIVE/WARNING/BREAK/RETIRED/REPLACED) ---
    lifecycle_state: str = "ACTIVE"
    # --- selection provenance (Steps 1/4) ---
    qualified: bool = True
    forced_pair_selection: bool = False
    selection_horizon: str = ""


@dataclass
class WindowResult:
    records: List[StepRecord] = field(default_factory=list)
    action_by_date: Dict[str, str] = field(default_factory=dict)
    pair_selections: List[str] = field(default_factory=list)
    switch_events: List[Dict[str, object]] = field(default_factory=list)
    rejection_events: List[Dict[str, object]] = field(default_factory=list)
    trades: List[Dict[str, object]] = field(default_factory=list)
    normalized_equity: List[float] = field(default_factory=list)
    monitoring: List[Dict[str, object]] = field(default_factory=list)
    pair_reflections: List[Dict[str, object]] = field(default_factory=list)
    break_events: List[Dict[str, object]] = field(default_factory=list)
    pair_lifecycle: List[Dict[str, object]] = field(default_factory=list)
    beta_fallbacks: List[Dict[str, object]] = field(default_factory=list)
    decisions: List[Dict[str, object]] = field(default_factory=list)
    pair_pool: List[Dict[str, object]] = field(default_factory=list)
    pair_discovery_diagnostics: List[Dict[str, object]] = field(default_factory=list)
    cumulative_pnl: float = 0.0
    settlement: Optional[Dict[str, object]] = None

    def daily_pnls(self) -> List[float]:
        return ([r.daily_pnl for r in self.records]
                + ([self.settlement["daily_pnl"]] if self.settlement else []))

    def n_no_trade_days(self) -> int:
        return sum(1 for r in self.records if r.no_trade)


def simulate_window(
    *,
    dates: Sequence[date],
    env_data: Dict[date, Dict],
    symbols: Sequence[str],
    agent,
    policy,
    history_seed: Optional[Dict[date, Dict]] = None,
    lookback: int = 20,
    monitor_frequency: int = 10,
    break_check_frequency: int = 23,
    pair_reselection_cooldown: int = 2,
    gross_allocation: float = 1.0,
    top_k_pairs: int = 5,
    fixed_pair: Optional[Tuple[str, str]] = None,
    pair_candidates: Optional[List] = None,
    candidate_provider=None,
    pair_memory=None,
    news_week_days: int = 7,
) -> WindowResult:
    """Run the policy over ``dates`` and return a full ``WindowResult``.

    Pair persistence (Step 3): the current pair is KEPT across days. Re-selection
    happens only when the relationship-break detector returns
    ``should_switch_pair`` (or no pair is held yet). ``monitor_frequency`` is how
    often the break detector runs (not a fixed switch cycle).

    NO_TRADE (Step 6): if no candidate passes the hard minimums the portfolio
    stays FLAT. A Pair Reflection (Step 4) is generated whenever a held pair is
    closed/replaced, and the reflection memory is injected into future selections.
    """
    from llm.policy_agent import PairReflectionMemory
    symbols = list(symbols)
    prompt_llm = getattr(agent, "llm", None)
    if callable(getattr(prompt_llm, "set_prompt_calendar", None)):
        prompt_llm.set_prompt_calendar(dates)
    portfolio = PortfolioManager(symbols, initial_cash=1.0)
    price_history: Dict[date, Dict[str, Dict[str, float]]] = {}
    if history_seed:
        for d in sorted(history_seed.keys()):
            price_history[d] = {"price": history_seed[d]["price"]}
    if pair_memory is None:
        pair_memory = PairReflectionMemory()

    from llm.policy_agent import classify_pair_outcome

    result = WindowResult()
    selected_pair: Optional[Tuple[str, str]] = fixed_pair
    current_pc = None
    prev_norm = 1.0
    recent_rewards: List[float] = []
    cum_pnl = 0.0
    # pre-break risk state carried day-to-day (Steps 10/11)
    cur_status, cur_risk, cur_force_flat = "HEALTHY", "LOW", False
    cur_risk_factors: List[str] = []
    # pair-break validation cadence + persistence (Steps: 15-day rule + persistence)
    last_val_i = -break_check_frequency      # so the first held day is a validation
    days_since_validation = 0
    risk_streak = 0                          # consecutive pre-break (high-risk) validation windows
    validation_window_count = 0
    cur_event_type = "daily_risk_check"
    cur_stat_risk, cur_news_risk = "LOW", "LOW"
    cur_stat_score, cur_news_score = 0.0, 0.0
    cur_persistence = 0
    cur_scheduled, cur_critical = False, False
    cur_break_type = "none"
    cur_lifecycle_state = "ACTIVE"
    # per-pair holding bookkeeping for outcome classification (Step 7)
    pair_open_cum = 0.0
    pair_reversions = 0
    prev_pos_nonflat = False
    # selection provenance carried to the daily record (Steps 1/4)
    cur_qualified, cur_forced, cur_sel_horizon = True, False, ""
    # ----- pair lifecycle + cooldown (ACTIVE/WARNING/BREAK/RETIRED/REPLACED) -----
    retired_cooldown: Dict[str, int] = {}     # pair_str -> validation-cycle index it is free again
    cur_lifecycle: Optional[Dict[str, object]] = None
    _STATE_RANK = {"ACTIVE": 0, "WARNING": 1, "BREAK": 2,
                   "RETIRED": 3, "REPLACED": 4}

    def _lifecycle_state(status: str) -> str:
        return "ACTIVE" if status == "HEALTHY" else status

    def _active_cooldown() -> set:
        return {p for p, until in retired_cooldown.items() if until > validation_window_count}

    def _new_lifecycle(pair, dstr: str) -> Dict[str, object]:
        return {"pair": "/".join(pair), "selected_date": dstr, "state": "ACTIVE",
                "max_state": "ACTIVE", "states_seen": ["ACTIVE"],
                "warning_days": 0, "active_days": 0,
                "retired_date": None, "replaced_by": None, "cooldown_until": None}

    def _lifecycle_touch(lc, state: str) -> None:
        if lc is None:
            return
        lc["state"] = state
        if state not in lc["states_seen"]:
            lc["states_seen"].append(state)
        if _STATE_RANK.get(state, 0) > _STATE_RANK.get(lc["max_state"], 0):
            lc["max_state"] = state
        if state == "ACTIVE":
            lc["active_days"] = int(lc.get("active_days", 0)) + 1
        elif state == "WARNING":
            lc["warning_days"] = int(lc.get("warning_days", 0)) + 1

    def _emit_pair_reflection(closed_pair, pc, dstr, status, reasoning,
                              open_cum, reversions, news_summary, new_pair_label):
        """Classify the closed pair's outcome and store a SUCCESS/FAILURE/NEUTRAL reflection."""
        pnl_delta = cum_pnl - open_cum
        hl = float(pc.half_life) if pc is not None else None
        outcome = classify_pair_outcome(
            cum_pnl=pnl_delta, relationship_break=(status == "BREAK"),
            half_life=hl, half_life_max=policy.half_life_max)
        stats = {"cointegration_pvalue": pc.coint_pvalue if pc else None,
                 "adf_pvalue": pc.adf_pvalue if pc else None,
                 "half_life_days": pc.half_life if pc else None}
        rec = agent.reflect_on_pair(
            pair=closed_pair, outcome_type=outcome, stats=stats,
            news_summary=news_summary, break_assessment=_mk_break(reasoning),
            performance={"cumulative_pnl": pnl_delta, "reversions": reversions})
        if outcome == "SUCCESS":
            pair_memory.add_success(pair=closed_pair, reflection_text=rec["reflection_text"],
                                    reason=("profitable, stable relationship" if not reasoning
                                            else reasoning), lessons=rec["lessons"],
                                    stats=stats, news=news_summary)
        elif outcome == "FAILURE":
            pair_memory.add_failure(pair=closed_pair, reflection_text=rec["reflection_text"],
                                    reason=reasoning or "re-evaluation", lessons=rec["lessons"],
                                    stats=stats, news=news_summary)
        entry = {
            "date": dstr, "closed_pair": "/".join(closed_pair),
            "new_pair": new_pair_label, "reason": reasoning or "window evaluation",
            "reflection": rec["reflection_text"], "lessons": rec["lessons"],
            "outcome_type": outcome, "why_worked": rec["why_worked"],
            "why_failed": rec["why_failed"], "statistical_patterns": rec["statistical_patterns"],
            "news_patterns": rec["news_patterns"], "pnl_while_held": round(pnl_delta, 6)}
        result.pair_reflections.append(entry)
        return outcome

    def _candidates_for(d, exclude=()):
        cands = (pair_candidates if pair_candidates is not None
                 else rank_statarb_pairs(_history_to_df(price_history, symbols),
                                         symbols, top_k=top_k_pairs))
        # Cooldown: a retired pair cannot be re-selected until its cooldown expires.
        if exclude:
            cands = [pc for pc in cands
                     if "/".join(pc.pair) not in exclude and "/".join(pc.pair[::-1]) not in exclude]
        wn = _week_news_items(env_data, d, news_week_days, symbols)
        sums = []
        for rk, pc in enumerate(cands, 1):
            a, b = pc.pair
            sums.append(summarize_pair_text(
                pc, rk, news_a=summarize_stock_news(wn.get(a, [])),
                news_b=summarize_stock_news(wn.get(b, [])),
                current_z=current_zscore(pc, prices[a], prices[b])
                if a in prices and b in prices else None))
        return cands, sums

    for i, d in enumerate(dates):
        if callable(getattr(prompt_llm, "set_prompt_step", None)):
            prompt_llm.set_prompt_step(d)
        prices = env_data[d]["price"]
        news = env_data[d].get("news", {})
        price_history[d] = {"price": prices}
        week_news = _week_news_items(env_data, d, news_week_days, symbols)
        dstr = d.isoformat()

        pair_switched = False
        is_monitor_day = (i % monitor_frequency == 0)
        break_detected = False
        break_reasoning = ""
        should_switch = False

        # ----- relationship monitoring on the CURRENT pair (Steps 3/5/8) ------
        # Daily risk check every held day (code-only gating); a FORMAL pair-break
        # validation runs only every ``break_check_frequency`` (=23) trading days,
        # and is the only day (besides a critical-news exception) a switch is allowed.
        if fixed_pair is None and selected_pair is not None and current_pc is not None:
            prev_status = cur_status
            days_since_validation = i - last_val_i
            is_validation_day = days_since_validation >= break_check_frequency
            event_type = "scheduled_break_validation" if is_validation_day else "daily_risk_check"

            news_sum = _pair_news_summary(week_news, selected_pair)
            sig_z, beta_mon = compute_spread_z(price_history, selected_pair, lookback)
            recent_corr = _recent_log_corr(price_history, selected_pair)
            vol_ratio = _recent_spread_vol_ratio(
                price_history, selected_pair, beta_mon, lookback, current_pc.spread_vol)

            ba = agent.detect_relationship_break(
                pair=selected_pair, recent_corr=recent_corr, signal_z=sig_z,
                adf_pvalue=current_pc.adf_pvalue, coint_pvalue=current_pc.coint_pvalue,
                half_life=current_pc.half_life, news_summary=news_sum,
                pair_memory=pair_memory, policy=policy, spread_vol_ratio=vol_ratio,
                event_type=event_type, days_since_last_validation=days_since_validation,
                prev_status=prev_status, risk_validation_streak=risk_streak)

            break_detected = ba.relationship_break_detected
            break_reasoning = ba.reasoning
            should_switch = ba.should_switch_pair      # only true on validation / critical news
            cur_status = ba.relationship_status
            cur_risk = ba.pre_break_risk
            cur_force_flat = ba.should_force_flat
            cur_risk_factors = list(ba.risk_factors)
            cur_event_type = ba.monitoring_event_type
            cur_stat_risk, cur_news_risk = ba.statistical_risk, ba.news_risk
            cur_stat_score, cur_news_score = ba.statistical_break_score, ba.news_break_score
            cur_persistence = ba.persistence_count
            cur_scheduled = ba.scheduled_break_validation
            cur_critical = ba.critical_news_exception
            cur_break_type = ba.break_type
            cur_lifecycle_state = _lifecycle_state(cur_status)
            _lifecycle_touch(cur_lifecycle, cur_lifecycle_state)

            # update cadence + persistence bookkeeping
            if is_validation_day:
                last_val_i = i
                validation_window_count += 1
                risk_streak = ba.persistence_count   # carries the consecutive pre-break streak

            result.monitoring.append({
                "date": d.isoformat(), "current_pair": "/".join(selected_pair),
                "monitoring_event_type": cur_event_type,
                "days_since_last_break_validation": days_since_validation,
                "scheduled_break_validation": is_validation_day,
                "critical_news_exception": cur_critical,
                "validation_window_count": validation_window_count,
                "statistical_status": {"recent_rolling_correlation": round(recent_corr, 3),
                                       "signal_z": round(sig_z, 3),
                                       "spread_vol_ratio": round(vol_ratio, 3),
                                       "coint_pvalue": current_pc.coint_pvalue,
                                       "adf_pvalue": current_pc.adf_pvalue,
                                       "half_life_days": current_pc.half_life},
                "news_summary": news_sum,
                "previous_relationship_status": prev_status,
                "relationship_status": cur_status,
                "pre_break_risk": cur_risk,
                "statistical_risk": cur_stat_risk,
                "news_risk": cur_news_risk,
                "llm_risk": ba.llm_risk,
                "statistical_break_score": cur_stat_score,
                "news_break_score": cur_news_score,
                "llm_break_score": ba.llm_break_score,
                "persistence_count": cur_persistence,
                "should_force_flat": cur_force_flat,
                "should_switch_pair": should_switch,
                "break_confirmed": ba.break_confirmed,
                "relationship_break_detected": break_detected,
                "lifecycle_state": cur_lifecycle_state,
                "critical_news_verification": ba.critical_news_verification,
                "risk_factors": cur_risk_factors,
                "reasoning": break_reasoning})
            if should_switch:
                result.break_events.append({"date": d.isoformat(),
                                            "pair": "/".join(selected_pair),
                                            "break_type": ba.break_type,
                                            "relationship_status": cur_status,
                                            "monitoring_event_type": cur_event_type,
                                            "critical_news_exception": cur_critical,
                                            "reasoning": break_reasoning})

        # ----- (re)selection: only on relationship break, or when no pair held -
        need_select = False
        if fixed_pair is None:
            if selected_pair is None and (is_monitor_day or i == 0):
                need_select = True          # NO_TRADE / warm-up: retry selection
            elif selected_pair is not None and should_switch:
                need_select = True          # break detected -> switch
        if need_select:
            is_replacement = (selected_pair is not None and should_switch)
            old_pair = selected_pair

            # 1) On BREAK: RETIRE the current pair first -- flatten, FAILURE reflection +
            #    Failure-Memory update, and put it on cooldown -- BEFORE re-ranking, so the
            #    replacement selection sees the updated memory and cannot re-pick it.
            retire_outcome = None
            cooldown_until = None
            if is_replacement:
                if not portfolio.is_flat(list(old_pair)):
                    portfolio.step(prices, _zeroed_targets(symbols))
                assert portfolio.is_flat(list(old_pair)), "pair not flat before switch"
                cooldown_until = validation_window_count + pair_reselection_cooldown
                retired_cooldown["/".join(old_pair)] = cooldown_until
                retire_outcome = _emit_pair_reflection(
                    old_pair, current_pc, dstr, "BREAK", break_reasoning,
                    pair_open_cum, pair_reversions,
                    _pair_news_summary(week_news, old_pair), "(replacement pending)")

            # 2) Re-rank + select (cooldown pairs excluded). A retired pair is excluded,
            #    guaranteeing a DIFFERENT replacement pair -> the switch actually happens.
            #    With a candidate_provider, the pair statistics are recomputed CAUSALLY
            #    as-of day d (prices <= d only); otherwise the precomputed pair_candidates
            #    are used (legacy / tests).
            exclude = _active_cooldown()
            sel = None
            if candidate_provider is not None:
                choice, sel_pc = candidate_provider(d, prices, exclude, pair_memory)
                sel = (choice, sel_pc)
            else:
                candidates, summaries = _candidates_for(d, exclude)
                if not summaries:  # safety: everything on cooldown -> allow full set
                    candidates, summaries = _candidates_for(d)
                if summaries:
                    _ch = agent.select_pair(summaries, policy, pair_memory=pair_memory)
                    _pc = next((pc for pc in candidates
                                if tuple(pc.pair) == tuple(_ch.pair)
                                or tuple(pc.pair) == tuple(_ch.pair)[::-1]), current_pc)
                    sel = (_ch, _pc)
            if sel is not None:
                choice, _sel_pc = sel
                new_pair = tuple(choice.pair)
                current_pc = _sel_pc if _sel_pc is not None else current_pc

                if is_replacement:
                    # 3) finalize lifecycle (RETIRED -> REPLACED) + switch log
                    if result.pair_reflections:
                        result.pair_reflections[-1]["new_pair"] = "/".join(new_pair)
                    refl = result.pair_reflections[-1] if result.pair_reflections else {}
                    lc = cur_lifecycle or _new_lifecycle(old_pair, dstr)
                    lc.update(state="REPLACED", max_state="REPLACED", retired_date=dstr,
                              replaced_by="/".join(new_pair), cooldown_until=cooldown_until,
                              break_reason=break_reasoning, break_type=cur_break_type,
                              outcome=retire_outcome)
                    if "RETIRED" not in lc["states_seen"]:
                        lc["states_seen"].append("RETIRED")
                    lc["states_seen"].append("REPLACED")
                    result.pair_lifecycle.append(lc)
                    result.switch_events.append({
                        "old_pair": "/".join(old_pair), "new_pair": "/".join(new_pair),
                        "switch_date": dstr, "date": dstr, "verified_flat": True,
                        "switch_reason": break_reasoning or "relationship break",
                        "break_type": cur_break_type, "critical_news": cur_critical,
                        "statistical_risk": cur_stat_risk, "news_risk": cur_news_risk,
                        "pair_reflection": refl.get("reflection", ""),
                        "cooldown_until": cooldown_until, "failure_memory_update": True,
                        "outcome_type": retire_outcome,
                        "trigger": break_reasoning or "relationship break"})
                    pair_switched = True

                # 4) new pair becomes ACTIVE -> reset holding/cadence bookkeeping
                selected_pair = new_pair
                if is_replacement or old_pair is None:
                    pair_open_cum = cum_pnl
                    pair_reversions = 0
                    prev_pos_nonflat = False
                    cur_status, cur_risk, cur_force_flat, cur_risk_factors = \
                        "HEALTHY", "LOW", False, []
                    last_val_i = i              # restart the 23-day validation countdown
                    days_since_validation = 0
                    risk_streak = 0
                    cur_event_type = "daily_risk_check"
                    cur_stat_risk, cur_news_risk = "LOW", "LOW"
                    cur_stat_score, cur_news_score = 0.0, 0.0
                    cur_persistence = 0
                    cur_scheduled, cur_critical = False, False
                    cur_break_type = "none"
                    cur_lifecycle_state = "ACTIVE"
                    cur_lifecycle = _new_lifecycle(new_pair, dstr)
                # carry selection provenance for the daily record
                cur_qualified = choice.qualified
                cur_forced = choice.forced_pair_selection
                cur_sel_horizon = choice.selection_horizon

        # ----- NO_TRADE / warmup: stay FLAT, record a NO_TRADE day ------------
        if selected_pair is None:
            portfolio.step(prices, _zeroed_targets(symbols))
            norm = float(portfolio.normalized_equity_history[-1])
            reward = (norm / prev_norm - 1.0) if prev_norm > 0 else 0.0
            prev_norm = norm
            cum_pnl += reward
            recent_rewards.append(reward)
            result.records.append(StepRecord(
                date=d.isoformat(), selected_pair="NO_TRADE", action="FLAT",
                position_state="FLAT", stance="neutral", signal_z=0.0, daily_pnl=reward,
                cumulative_pnl=cum_pnl, normalized_equity=norm,
                news_sentiment="", rationale="warm-up: awaiting first pair selection",
                rejected_target=False, pair_switched=pair_switched, no_trade=True,
                relationship_break_detected=break_detected, break_reasoning=break_reasoning,
                should_switch_pair=should_switch,
                relationship_status=cur_status, pre_break_risk=cur_risk,
                should_force_flat=cur_force_flat, risk_factors=list(cur_risk_factors),
                monitoring_event_type=cur_event_type,
                days_since_last_break_validation=days_since_validation,
                scheduled_break_validation=cur_scheduled, critical_news_exception=cur_critical,
                statistical_risk=cur_stat_risk, news_risk=cur_news_risk,
                statistical_break_score=cur_stat_score, news_break_score=cur_news_score,
                persistence_count=cur_persistence, lifecycle_state="RETIRED"))
            result.action_by_date[d.isoformat()] = "FLAT"
            result.normalized_equity.append(norm)
            continue

        result.pair_selections.append("/".join(selected_pair))

        # ----- code-computed signal + risk context ----------------------------
        signal_z, beta = compute_spread_z(price_history, selected_pair, lookback)
        recent_pnl_3d = sum(recent_rewards[-3:])
        sentiment = _pair_sentiment(news, selected_pair)
        position_state = _infer_state(portfolio.positions, selected_pair)
        # per-leg news summary so the action prompt can reason on news SEMANTICS
        trade_news_sum = _pair_news_summary(week_news, selected_pair)

        decision = agent.decide_action(
            pair=selected_pair,
            position_state=position_state,
            signal_z=signal_z,
            recent_pnl_3d=recent_pnl_3d,
            news_sentiment=sentiment,
            policy=policy,
            relationship_status=cur_status,
            pre_break_risk=cur_risk,
            news_summary=trade_news_sum,
        )

        # ----- build FRESH target (no hidden / residual positions) ------------
        target = _zeroed_targets(symbols)  # everything flat unless set below
        if decision.action in (Action.long_short, Action.short_long):
            sizes = size_pair_position_fractional(
                prices=prices,
                pair=selected_pair,
                portfolio_value=max(portfolio._compute_equity(prices), 1e-9),
                action=decision.action,
                gross_fraction=gross_allocation,
                hedge_ratio=beta,
            )
            target.update(sizes)
        # FLAT leaves the zeroed target as-is (both legs explicitly closed).

        state = portfolio.step(prices, target)
        rejected = bool(state.get("rejected_target"))
        if rejected:
            # Do NOT replay: clear the target to FLAT for this pair and move on.
            portfolio.step(prices, _zeroed_targets(symbols))
            state = {
                "equity": portfolio.equity_history[-1],
                "normalized_equity": portfolio.normalized_equity_history[-1],
            }
            result.rejection_events.append(
                {"date": d.isoformat(), "attempted_action": decision.action.label,
                 "resolved": "cleared_to_flat"}
            )
            effective_action = Action.flat
        else:
            effective_action = decision.action

        norm = float(state["normalized_equity"])
        reward = (norm / prev_norm - 1.0) if prev_norm > 0 else 0.0
        prev_norm = norm
        cum_pnl += reward
        recent_rewards.append(reward)

        eff_state = action_to_state(effective_action)
        # reversion bookkeeping: count non-flat -> flat transitions (Step 7)
        is_nonflat = (effective_action != Action.flat)
        if prev_pos_nonflat and not is_nonflat:
            pair_reversions += 1
        prev_pos_nonflat = is_nonflat
        rec = StepRecord(
            date=d.isoformat(),
            selected_pair="/".join(selected_pair),
            action=effective_action.label,
            position_state=eff_state.label,
            stance=decision.stance,
            signal_z=round(signal_z, 4),
            daily_pnl=reward,
            cumulative_pnl=cum_pnl,
            normalized_equity=norm,
            news_sentiment=sentiment,
            rationale=decision.rationale,
            rejected_target=rejected,
            pair_switched=pair_switched,
            no_trade=False,
            relationship_break_detected=break_detected,
            break_reasoning=break_reasoning,
            should_switch_pair=should_switch,
            relationship_status=cur_status,
            pre_break_risk=cur_risk,
            should_force_flat=cur_force_flat,
            risk_factors=list(cur_risk_factors),
            monitoring_event_type=cur_event_type,
            days_since_last_break_validation=days_since_validation,
            scheduled_break_validation=cur_scheduled,
            critical_news_exception=cur_critical,
            statistical_risk=cur_stat_risk,
            news_risk=cur_news_risk,
            statistical_break_score=cur_stat_score,
            news_break_score=cur_news_score,
            persistence_count=cur_persistence,
            lifecycle_state=cur_lifecycle_state,
            qualified=cur_qualified,
            forced_pair_selection=cur_forced,
            selection_horizon=cur_sel_horizon,
        )
        result.records.append(rec)
        result.action_by_date[d.isoformat()] = effective_action.label
        result.normalized_equity.append(norm)
        if effective_action != Action.flat:
            result.trades.append(
                {"date": d.isoformat(), "pair": "/".join(selected_pair),
                 "action": effective_action.label, "pnl": reward}
            )

    # ----- end of window: evaluate the final held pair (Step 7) ---------------
    if fixed_pair is None and selected_pair is not None and current_pc is not None:
        last_d = dates[-1].isoformat() if len(dates) else ""
        last_news = _pair_news_summary(
            _week_news_items(env_data, dates[-1], news_week_days, symbols), selected_pair) \
            if len(dates) else {"pair_level_risk": "low"}
        _emit_pair_reflection(selected_pair, current_pc, last_d, cur_status,
                              "completed-window evaluation", pair_open_cum,
                              pair_reversions, last_news, "(window end)")
        # record the still-active pair's lifecycle (ACTIVE at window end)
        if cur_lifecycle is not None:
            cur_lifecycle.update(final_state=cur_lifecycle.get("state", "ACTIVE"),
                                 retired_date=None, replaced_by=None)
            result.pair_lifecycle.append(cur_lifecycle)

    result.cumulative_pnl = cum_pnl
    return result


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _zeroed_targets(symbols: Sequence[str]) -> Dict[str, float]:
    return {s: 0.0 for s in symbols}


def _history_to_df(price_history: Dict[date, Dict], symbols: Sequence[str]) -> pd.DataFrame:
    rows = {}
    for d in sorted(price_history.keys()):
        rows[d] = price_history[d]["price"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=list(symbols))
    df.index = pd.to_datetime(list(rows.keys()))
    return df.sort_index()


def _infer_state(positions: Dict[str, float], pair: Tuple[str, str]) -> PositionState:
    a, b = pair
    pa, pb = positions.get(a, 0.0), positions.get(b, 0.0)
    if pa > 1e-9 and pb < -1e-9:
        return PositionState.long_short
    if pa < -1e-9 and pb > 1e-9:
        return PositionState.short_long
    return PositionState.flat


def _pair_sentiment(news: Dict[str, List[str]], pair: Tuple[str, str]) -> str:
    a, b = pair
    items = list(news.get(a, []) or []) + list(news.get(b, []) or [])
    return _sentiment_label(items)


def _recent_spread_vol_ratio(price_history: Dict[date, Dict], pair: Tuple[str, str],
                             beta: Optional[float], lookback: int,
                             hist_vol: float) -> float:
    """Recent log-spread volatility / historical spread volatility (Step 3).

    >1 means the spread is more volatile lately than its historical regime; a
    sustained spike contributes to the statistical-break score.
    """
    a, b = pair
    if beta is None:
        beta = 1.0
    dates = sorted(price_history.keys())[-(lookback + 1):]
    spreads: List[float] = []
    for d in dates:
        p = price_history[d].get("price", {})
        pa, pb = p.get(a), p.get(b)
        if pa and pb and pa > 0 and pb > 0:
            spreads.append(math.log(pa) - beta * math.log(pb))
    if len(spreads) < 5 or hist_vol is None or hist_vol <= 1e-9:
        return 1.0
    recent_vol = float(pd.Series(spreads).diff().dropna().std())
    if recent_vol != recent_vol:  # NaN
        return 1.0
    return float(recent_vol / hist_vol)


def _recent_log_corr(price_history: Dict[date, Dict], pair: Tuple[str, str],
                     window: int = 60) -> float:
    """Recent rolling correlation of LOG prices (used by the break detector)."""
    a, b = pair
    dates = sorted(price_history.keys())[-window:]
    la, lb = [], []
    for d in dates:
        p = price_history[d].get("price", {})
        pa, pb = p.get(a), p.get(b)
        if pa and pb and pa > 0 and pb > 0:
            la.append(math.log(pa)); lb.append(math.log(pb))
    if len(la) < 5:
        return 1.0
    c = pd.Series(la).corr(pd.Series(lb))
    return float(c) if c == c else 1.0  # NaN -> 1.0


def _pair_news_summary(week_news: Dict[str, List[str]], pair: Tuple[str, str]) -> Dict[str, object]:
    """Per-leg news summary + pair-level relationship risk for the current pair."""
    a, b = pair
    na = summarize_stock_news(week_news.get(a, []))
    nb = summarize_stock_news(week_news.get(b, []))
    return {a: na, b: nb, "pair_level_risk": pair_news_risk(na, nb)}


def _mk_break(reasoning: str):
    """Build a lightweight BreakAssessment for pair reflection on re-selection."""
    from llm.policy_agent import BreakAssessment
    return BreakAssessment(
        relationship_break_detected=bool(reasoning),
        reasoning=reasoning or "periodic re-evaluation",
        break_type=(reasoning.split(";")[0].split("(")[0].strip() if reasoning else "none"),
        confidence=0.6 if reasoning else 0.2, should_switch_pair=bool(reasoning),
        model="engine")


def _week_news_items(
    env_data: Dict[date, Dict], current: date, window_days: int, symbols: Sequence[str]
) -> Dict[str, List[str]]:
    """Raw news items per ticker from the PAST WEEK (sessions in (current-7d, current]).

    Only news available *before* the decision date is used -- no future leakage.
    """
    start = current - timedelta(days=window_days)
    items: Dict[str, List[str]] = {s: [] for s in symbols}
    for d, rec in env_data.items():
        if start < d <= current:
            day_news = rec.get("news", {}) or {}
            for s in symbols:
                vals = day_news.get(s) or day_news.get(s.lower()) or []
                if isinstance(vals, list):
                    items[s].extend(vals)
    return items


def _prev_trading_day(env_data: Dict[date, Dict], current: date) -> Optional[date]:
    """The most recent trading session strictly before ``current`` (Step 4).

    Used as the lower bound of the Direction-Discovery news window
    ``(previous trading day, T-1]`` so only news that arrived since the last session
    (e.g. for a Monday decision: the prior weekend) is visible -- never future news,
    never news from day T. Returns ``None`` if there is no earlier session.
    """
    earlier = [d for d in env_data.keys() if d < current]
    return max(earlier) if earlier else None


def _cadence_news_items(
    env_data: Dict[date, Dict], prev_date: Optional[date], current: date,
    symbols: Sequence[str]
) -> Dict[str, List[str]]:
    """News window aligned to the decision cadence: items published in
    ``(prev_date, current - 1]`` -- i.e. strictly BEFORE the current decision day
    ``current`` and after the previous decision day ``prev_date``. No information from
    ``current`` or later may be used (causal). ``prev_date=None`` includes everything
    before ``current``.
    """
    items: Dict[str, List[str]] = {s: [] for s in symbols}
    for d, rec in env_data.items():
        if d < current and (prev_date is None or d > prev_date):
            day_news = rec.get("news", {}) or {}
            for s in symbols:
                vals = day_news.get(s) or day_news.get(s.lower()) or []
                if isinstance(vals, list):
                    items[s].extend(vals)
    return items


def anti_flat_override_blocked(lifecycle_state: str = "HEALTHY", *,
                               break_triggered: bool = False,
                               in_cooldown: bool = False) -> bool:
    """Whether the consecutive-FLAT (anti-inactivity) override is blocked.

    Lifecycle state is INFORMATIONAL and NEVER blocks the override -- neither WARNING nor
    BREAK suppresses trading. The override (force a directional trade after
    ``max_flat_streak`` consecutive FLAT days) is blocked ONLY by an active cooldown on
    the held pair; a margin rejection is handled separately by the portfolio step.
    ``lifecycle_state`` / ``break_triggered`` are accepted for call-site clarity but do
    not affect the result.
    """
    return bool(in_cooldown)


# --------------------------------------------------------------------------- #
# Combined lifecycle recording (threshold + recording change). break_score is the
# PURE statistical score (0-6); news is tracked separately as a relation signal.
# --------------------------------------------------------------------------- #
def statistical_lifecycle_state(break_score: float) -> str:
    """Statistical lifecycle state from the pure statistical break_score (0-6):
    ``<= 3`` HEALTHY, ``3 < score < 5`` WARNING, ``>= 5`` BREAK."""
    if break_score >= 5:
        return "BREAK"
    if break_score > 3:
        return "WARNING"
    return "HEALTHY"


def news_relation_signal(news_status: str) -> str:
    """Map the code's 3-level news status (HEALTHY/WARNING/BREAK) to the lifecycle news
    relation signal NONE / WARNING / BREAK."""
    return {"HEALTHY": "NONE", "WARNING": "WARNING", "BREAK": "BREAK"}.get(
        str(news_status), "NONE")


def news_lifecycle_state(signal: str) -> str:
    """News lifecycle state from the news relation signal: NONE->HEALTHY, WARNING->
    WARNING, BREAK->BREAK."""
    return {"NONE": "HEALTHY", "WARNING": "WARNING", "BREAK": "BREAK"}.get(
        str(signal), "HEALTHY")


def combined_lifecycle_state(statistical_state: str, news_state: str) -> str:
    """Final recorded lifecycle state = the more severe of the statistical and news
    states: BREAK if either is BREAK; else WARNING if either is WARNING; else HEALTHY."""
    if statistical_state == "BREAK" or news_state == "BREAK":
        return "BREAK"
    if statistical_state == "WARNING" or news_state == "WARNING":
        return "WARNING"
    return "HEALTHY"


def break_score_trend(prev_score: Optional[float], cur_score: float) -> str:
    """Direction of the break_score vs the previous monitored day: ``constant`` (equal or
    no prior value), ``rising`` (increased), or ``falling`` (decreased). Never 'flat'."""
    if prev_score is None or cur_score == prev_score:
        return "constant"
    return "rising" if cur_score > prev_score else "falling"


def simulate_discovery_window(
    *,
    dates: Sequence[date],
    env_data: Dict[date, Dict],
    symbols: Sequence[str],
    agent,
    discovery_provider,
    history_seed: Optional[Dict[date, Dict]] = None,
    lookback: int = 20,
    selection_frequency: int = 23,
    gross_allocation: float = 1.0,
    top_k: int = 5,
    beta_hedged: bool = False,
    pair_memory=None,
    action_prompt: Optional[str] = None,
    direction_memory=None,
    entry_z: float = 1.0,
    lifecycle: bool = False,
    break_check_frequency: int = 23,
    pair_reselection_cooldown: int = 2,
    anti_flat_override: bool = True,
    max_flat_streak: int = 3,
    daily_direction: bool = True,
    settlement_date: Optional[date] = None,
    initial_pair: Optional[Tuple[str, str]] = None,
    allow_reselection: bool = True,
) -> WindowResult:
    """Pair Discovery + Direction Discovery decision flow (Steps 1/2/4/5/8/9).

    At each pair-selection event permitted by the cadence below:
      * Stage 1 (rule-based): ``discovery_provider(as_of, prices)`` returns the Top-5
        pairs ranked by 10Y cointegration (each ``{"pair":(A,B), "pc": StatArbPair}``).
      * News window ``(previous TRADING DAY, T-1]`` is built per candidate pair.
      * Stage 2 (merged LLM/offline call): ``agent.discover_direction`` picks ONE pair +
        an action (LONG_SHORT/SHORT_LONG/FLAT) + a reason.
    Daily direction decisions update the action while the selected pair is held. Sizing:
    ``beta_hedged=False`` -> +/-1; ``beta_hedged=True`` -> (+1,-beta)/(-1,+beta).

    ``lifecycle=False`` (default, used by foundation tests): no monitoring -- the pair is
    held until the next scheduled decision.

    ``lifecycle=True`` (production redesign, Steps 1/8): after a pair is selected the
    system enters daily relationship monitoring. Re-selection is considered ONLY on the
    global ``break_check_frequency`` (=23) grid, and ONLY if the preceding validation
    window (including today) contains BREAK. HEALTHY/WARNING retains the pair; a daily
    BREAK does not switch immediately or force FLAT. A triggered re-selection retires
    the pair to quality memory and applies the existing cooldown. There is no independent
    periodic re-selection while a pair is held. Initial selection is attempted at index 0;
    if no candidate is available, initialization is retried on validation boundaries.
    ``selection_frequency`` applies only to the non-lifecycle compatibility path.
    ``allow_reselection=False`` disables only replacement (A4), retaining monitoring,
    daily actions and counterfactual feedback. ``initial_pair`` fixes the matched
    control's first pair, which must belong to the same causal candidate set.
    Pair Selection Memory stores only pair-quality outcomes, never actions or direction.
    """
    from llm.policy_agent import PairSelectionMemory, Policy as _Policy
    symbols = list(symbols)
    if settlement_date is not None:
        from utils.data_utils import next_trading_day
        if not dates or settlement_date != next_trading_day(dates[-1]):
            raise ValueError("Settlement must follow the last decision on the next trading day")
        final_prices = env_data.get(settlement_date, {}).get("price", {})
        if any(not math.isfinite(float(final_prices.get(s, float("nan"))))
               or float(final_prices.get(s, 0)) <= 0 for s in symbols):
            raise ValueError("Missing valid terminal settlement prices")
    prompt_llm = getattr(agent, "llm", None)
    if callable(getattr(prompt_llm, "set_prompt_calendar", None)):
        prompt_llm.set_prompt_calendar(dates)
    portfolio = PortfolioManager(symbols, initial_cash=1.0)
    price_history: Dict[date, Dict[str, Dict[str, float]]] = {}
    if history_seed:
        for d in sorted(history_seed.keys()):
            price_history[d] = {"price": history_seed[d]["price"]}
    if pair_memory is None:
        pair_memory = PairSelectionMemory()
    mon_policy = _Policy()                      # default thresholds for break monitoring

    result = WindowResult()
    prev_norm = 1.0
    cum_pnl = 0.0
    cur_pair: Optional[Tuple[str, str]] = None
    cur_action = Action.flat
    cur_reason = ""
    cur_beta = 1.0
    current_pc = None                           # StatArbPair of the held pair (monitoring)
    # ---- lifecycle state ----
    last_val_i = -break_check_frequency
    risk_streak = 0
    validation_window_count = 0
    retired_cooldown: Dict[str, int] = {}
    cur_status, cur_risk = "HEALTHY", "LOW"
    prev_break_score: Optional[float] = None    # previous monitored day's break_score (trend)
    held_reversions = 0
    prev_nonflat = False
    flat_streak = 0                             # consecutive FLAT decision points (anti-degeneration)
    cf_pending = None                           # previous day's decision awaiting counterfactual
    cf_memory: List[Dict[str, object]] = []     # finalized daily CF summaries (for recent-3)

    def _finalize_cf(next_prices):
        """Finalize the pending (previous) daily decision once next prices are known."""
        nonlocal cf_pending
        if cf_pending is None:
            return
        p = cf_pending
        a, b = p["pair"]
        cf = counterfactual_pnls(p["pa"], p["pb"], next_prices.get(a),
                                 next_prices.get(b), p["beta"])
        actual = cf.get(p["action"], 0.0)
        best = _cf_best(cf)                       # dead-band best action
        right = _cf_right(p["action"], cf, realized_pnl=actual)   # dead-band RIGHT/WRONG
        sentence = _cf_sentence(
            date=p["date"], action=p["action"], pair="/".join(p["pair"]),
            pa=p["pa"], pb=p["pb"], da=p["da"], db=p["db"], reason=p["reason"],
            actual_pnl=actual, cf=cf, right=right, za_long=p["za_long"], tw_z=p["tw_z"])
        dec = result.decisions[p["dec_idx"]]
        dec.update(actual_pnl=actual, cf_long_short=cf["LONG_SHORT"],
                   cf_short_long=cf["SHORT_LONG"], cf_flat=cf["FLAT"],
                   best_action=best, right_or_wrong=("RIGHT" if right else "WRONG"),
                   memory_sentence=sentence)
        if direction_memory is not None and p["dm_idx"] is not None:
            direction_memory.set_counterfactual(
                p["dm_idx"], actual_pnl=actual, cf=cf, best=best, right=right,
                sentence=sentence, prices=(p["pa"], p["pb"]), deltas=(p["da"], p["db"]))
        cf_memory.append({"date": p["date"], "action": p["action"], "reason": p["reason"],
                          "actual_pnl": round(actual, 6),
                          "cf_long_short": round(cf["LONG_SHORT"], 6),
                          "cf_short_long": round(cf["SHORT_LONG"], 6), "cf_flat": 0.0,
                          "best_action": best,
                          "right_or_wrong": ("RIGHT" if right else "WRONG"),
                          "memory_sentence": sentence})
        cf_pending = None

    def _active_cooldown() -> set:
        return {p for p, u in retired_cooldown.items() if u > validation_window_count}

    def _build_cands(dd_date, dd_prices, exclude):
        top5 = discovery_provider(dd_date, dd_prices) or []
        if exclude:
            top5 = [c for c in top5
                    if "/".join(tuple(c["pair"])) not in exclude
                    and "/".join(tuple(c["pair"])[::-1]) not in exclude]
        prev_td = _prev_trading_day(env_data, dd_date)
        win_news = _cadence_news_items(env_data, prev_td, dd_date, symbols)
        hist_dates = sorted(price_history.keys())
        prev_prices = (price_history[hist_dates[-2]]["price"] if len(hist_dates) >= 2 else {})
        out = []
        for c in top5[:top_k]:
            a, b = tuple(c["pair"])
            tw_z, _bw = compute_spread_z(price_history, (a, b), lookback)
            out.append({
                "pair": (a, b), "pc": c["pc"],
                "news_a": win_news.get(a, []), "news_b": win_news.get(b, []),
                "price_a": dd_prices.get(a), "price_b": dd_prices.get(b),
                "prev_price_a": prev_prices.get(a), "prev_price_b": prev_prices.get(b),
                "trading_window_z": tw_z})
        return out

    def _active_candidate(dd_date, dd_prices, pair, pc):
        """Build the single-candidate dict for the ACTIVE pair (daily Direction mode)."""
        a, b = pair
        prev_td = _prev_trading_day(env_data, dd_date)
        win_news = _cadence_news_items(env_data, prev_td, dd_date, symbols)
        hist_dates = sorted(price_history.keys())
        prev_prices = (price_history[hist_dates[-2]]["price"] if len(hist_dates) >= 2 else {})
        tw_z, _bw = compute_spread_z(price_history, pair, lookback)
        return ({"pair": (a, b), "pc": pc,
                 "news_a": win_news.get(a, []), "news_b": win_news.get(b, []),
                 "price_a": dd_prices.get(a), "price_b": dd_prices.get(b),
                 "prev_price_a": prev_prices.get(a), "prev_price_b": prev_prices.get(b),
                 "trading_window_z": tw_z}, prev_td)

    def _pair_quality(pc, broke: bool, reversions: int) -> Tuple[str, str]:
        """Pair-level stability outcome (Step 2) -- NEVER direction/action/PnL based."""
        if broke:
            if pc is not None and pc.coint_pvalue > 2 * mon_policy.coint_pmax:
                return "cointegration_deterioration", "cointegration deteriorated"
            return "spread_breakdown", "relationship break confirmed"
        if pc is not None and reversions >= 1 and 2 <= pc.half_life <= mon_policy.half_life_max:
            return "mean_reversion_success", f"spread reverted {reversions}x, stable"
        return "mean_reversion_failure", "spread failed to mean-revert in range"

    def _retire_to_memory(pair, pc, broke: bool, reversions: int) -> None:
        quality, note = _pair_quality(pc, broke, reversions)
        pair_memory.add(pair=tuple(pair), quality=quality, note=note,
                        coint_pvalue=(pc.coint_pvalue if pc is not None else None),
                        half_life=(pc.half_life if pc is not None else None))

    for i, d in enumerate(dates):
        if callable(getattr(prompt_llm, "set_prompt_step", None)):
            prompt_llm.set_prompt_step(d)
        prices = env_data[d]["price"]
        price_history[d] = {"price": prices}
        dstr = d.isoformat()
        _finalize_cf(prices)                    # finalize yesterday's decision w/ today's prices
        pair_switched = False
        break_detected = False
        should_switch = False
        force_flat = False
        cur_break_score = 0.0           # diagnostic: raw statistical break score (this day)
        cur_news_break_flag = False     # diagnostic: news-break flag (this day)
        pair_retired_today = False      # whether the held pair was retired this day (BREAK)
        is_validation = (i > 0) and (i % break_check_frequency == 0)

        # ----- lifecycle monitoring on the held pair (Steps 1/8) -----
        if lifecycle and cur_pair is not None and current_pc is not None:
            days_since_validation = i - last_val_i
            # GLOBAL validation grid: validations land on a FIXED cadence (every
            # break_check_frequency trading days) INDEPENDENT of pair switches. This makes
            # the 23-day lookback windows contiguous and pair-agnostic, so a BREAK is
            # evaluated at the next validation day regardless of whether the pair that
            # broke is still active or survived to "its own" validation day.
            event_type = "scheduled_break_validation" if is_validation else "daily_risk_check"
            prev_td = _prev_trading_day(env_data, d)
            mon_news = _pair_news_summary(
                _cadence_news_items(env_data, prev_td, d, symbols), cur_pair)
            sig_z, beta_mon = compute_spread_z(price_history, cur_pair, lookback)
            recent_corr = _recent_log_corr(price_history, cur_pair)
            vol_ratio = _recent_spread_vol_ratio(
                price_history, cur_pair, beta_mon, lookback, current_pc.spread_vol)
            ba = agent.detect_relationship_break(
                pair=cur_pair, recent_corr=recent_corr, signal_z=sig_z,
                adf_pvalue=current_pc.adf_pvalue, coint_pvalue=current_pc.coint_pvalue,
                half_life=current_pc.half_life, news_summary=mon_news, pair_memory=None,
                policy=mon_policy, spread_vol_ratio=vol_ratio, event_type=event_type,
                days_since_last_validation=days_since_validation, prev_status=cur_status,
                risk_validation_streak=risk_streak)
            # ----- combined lifecycle recording (threshold + recording change) -----
            # break_score is the PURE statistical score (0-6); news is tracked separately.
            cur_break_score = float(ba.statistical_break_score)
            stat_state = statistical_lifecycle_state(cur_break_score)
            news_signal = news_relation_signal(ba.news_risk)        # NONE/WARNING/BREAK
            news_state = news_lifecycle_state(news_signal)
            final_state = combined_lifecycle_state(stat_state, news_state)
            trend = break_score_trend(prev_break_score, cur_break_score)
            prev_break_score = cur_break_score
            cur_status = final_state                                # recorded lifecycle_state
            cur_risk = ba.pre_break_risk                            # diagnostic only
            force_flat = False                                      # lifecycle NEVER force-flats
            cur_news_break_flag = (news_signal == "BREAK")
            break_detected = (final_state == "BREAK")              # may occur any monitoring day
            if is_validation:
                last_val_i = i
                validation_window_count += 1
                risk_streak = ba.persistence_count

            # ----- validation-day re-selection (GLOBAL, validation-centric) -----
            # On a validation day, look back over the previous break_check_frequency (=23)
            # trading days of ALL lifecycle records (ANY pair) and trigger re-selection if
            # ANY day in that window was BREAK. This does NOT depend on whether the pair
            # that broke is still active or survived to its own validation day.
            pair_reselection_triggered = False
            validation_window_has_break = False
            validation_window_break_source = "none"
            validation_window_break_count = 0
            validation_window_start = validation_window_end = None
            pair_reselection_reason = ""
            if is_validation:
                lo = i - break_check_frequency
                window = [(r.get("statistical_lifecycle_state"), r.get("news_lifecycle_state"))
                          for r in result.monitoring
                          if r.get("decision_index", -1) > lo
                          and r.get("lifecycle_state") == "BREAK"]
                if final_state == "BREAK":                           # include today
                    window.append((stat_state, news_state))
                validation_window_break_count = len(window)
                validation_window_has_break = bool(window)
                _lo_idx = max(0, lo + 1)
                validation_window_start = (dates[_lo_idx].isoformat()
                                           if _lo_idx < len(dates) else None)
                validation_window_end = dstr
                if validation_window_has_break:
                    has_stat = any(s == "BREAK" for s, _n in window)
                    has_news = any(n == "BREAK" for _s, n in window)
                    validation_window_break_source = ("both" if (has_stat and has_news)
                                                      else "statistical" if has_stat else "news")
                    pair_reselection_triggered = True
                    pair_reselection_reason = (
                        f"validation-window BREAK (source={validation_window_break_source})")
            # BREAK never forces an IMMEDIATE switch; re-selection is decided ONLY here,
            # at the validation day, from the global 23-day window.
            pair_reselection_triggered = pair_reselection_triggered and allow_reselection
            should_switch = pair_reselection_triggered

            result.monitoring.append({
                "date": dstr, "pair": "/".join(cur_pair),
                "current_pair": "/".join(cur_pair), "decision_index": i,
                "monitoring_event_type": event_type,
                "scheduled_break_validation": is_validation,
                "is_validation_day": is_validation,
                "break_score": cur_break_score,
                "statistical_lifecycle_state": stat_state,
                "news_relation_signal": news_signal,
                "news_lifecycle_state": news_state,
                "lifecycle_state": final_state, "relationship_status": final_state,
                "break_score_trend": trend,
                "validation_window_start": validation_window_start,
                "validation_window_end": validation_window_end,
                "validation_window_break_count": validation_window_break_count,
                "validation_window_has_break": validation_window_has_break,
                "validation_window_break_source": validation_window_break_source,
                "pair_reselection_triggered": pair_reselection_triggered,
                "pair_reselection_reason": pair_reselection_reason,
                # extra diagnostics (not behavior-driving)
                "warning_flag": (final_state == "WARNING"),
                "break_confirmed": (final_state == "BREAK"),
                "statistical_break_score": cur_break_score,
                "news_break_flag": cur_news_break_flag,
                "force_flat_triggered": False, "force_flat_reason": "",
                "cooldown_active": False,        # patched True below if retired this day
                "pair_retired": False,           # patched True below if retired this day
                "pre_break_risk": cur_risk,      # diagnostic only
                "should_switch_pair": should_switch, "reasoning": ba.reasoning})
            if should_switch:
                result.break_events.append({
                    "date": dstr, "pair": "/".join(cur_pair),
                    "break_type": validation_window_break_source,
                    "reasoning": pair_reselection_reason})

        # An active pair can be replaced only after validation finds a window BREAK.
        # Without an active pair, a boundary retry completes initial discovery instead.
        break_triggered = bool(lifecycle and is_validation and should_switch)
        if lifecycle:
            do_select = (i == 0) or (cur_pair is None and is_validation) or break_triggered
        else:
            do_select = i == 0 or (allow_reselection and i % selection_frequency == 0)

        is_pair_selection_day = False
        pair_selection_trigger = "none"

        if do_select:
            is_pair_selection_day = True
            pair_selection_trigger = ("break" if break_triggered else
                                      "initial" if lifecycle else "scheduled")
            exclude = _active_cooldown() if lifecycle else set()
            # Closing the current pair -> Pair Selection Memory (pair-quality only) + cooldown.
            if lifecycle and cur_pair is not None:
                # Held pairs reach this block only after a validation-window BREAK.
                broke = break_triggered
                _retire_to_memory(cur_pair, current_pc, broke, held_reversions)
                if broke:
                    retired_cooldown["/".join(cur_pair)] = (
                        validation_window_count + pair_reselection_cooldown)
                    exclude = _active_cooldown()
                    pair_retired_today = True
                    # reflect retirement + cooldown in today's lifecycle record (if any)
                    if result.monitoring and result.monitoring[-1].get("date") == dstr:
                        result.monitoring[-1]["pair_retired"] = True
                        result.monitoring[-1]["cooldown_active"] = True
            cands = _build_cands(d, prices, exclude)
            if not cands and exclude:                     # all on cooldown -> allow full set
                cands = _build_cands(d, prices, set())
            _diag = getattr(discovery_provider, "last_diagnostics", None)
            if _diag is not None:
                _dd = dict(_diag); _dd["date"] = dstr
                result.pair_discovery_diagnostics.append(_dd)
            if cands:
                # Mode A: select ONE pair from the Top-5 (the daily ACTION is decided
                # separately by Mode B / Direction Discovery below).
                if initial_pair is not None and cur_pair is None:
                    new_pair = tuple(initial_pair)
                    if not any(tuple(c["pair"]) in (new_pair, new_pair[::-1]) for c in cands):
                        raise ValueError("Matched initial pair is absent from the causal candidate set")
                else:
                    ddA = agent.discover_direction(candidates=cands, action_prompt=action_prompt,
                                                   pair_memory=pair_memory, entry_z=entry_z)
                    new_pair = tuple(ddA.selected_pair)
                # Reset holding state for a new pair, preserving the global validation grid.
                is_new_pair = (cur_pair is None) or (new_pair != cur_pair)
                pair_switched = cur_pair is not None and new_pair != cur_pair
                if pair_switched:
                    # Count actual pair changes, not initial selection or a BREAK
                    # validation that selects the same pair again. Both summary
                    # metrics and baseline comparisons consume this event list.
                    result.switch_events.append({
                        "date": dstr, "switch_date": dstr, "decision_index": i,
                        "old_pair": "/".join(cur_pair), "new_pair": "/".join(new_pair),
                        "trigger": pair_selection_trigger,
                        "switch_reason": (pair_reselection_reason if break_triggered
                                          else "scheduled pair selection"),
                    })
                cur_pair = new_pair
                current_pc = next((c["pc"] for c in cands
                                   if tuple(c["pair"]) == new_pair
                                   or tuple(c["pair"]) == new_pair[::-1]), current_pc)
                result.pair_selections.append("/".join(cur_pair))
                for rk, c in enumerate(cands, 1):         # pair-pool log: selection days only
                    cp = tuple(c["pair"]); pcc = c["pc"]
                    result.pair_pool.append({
                        "date": dstr, "rank": rk, "pair": "/".join(cp),
                        "coint_pvalue": getattr(pcc, "coint_pvalue", None),
                        "beta": getattr(pcc, "hedge_ratio", None),
                        "half_life": getattr(pcc, "half_life", None),
                        "trading_window_z": c.get("trading_window_z"),
                        "selected": (cp == cur_pair or cp == cur_pair[::-1])})
                if lifecycle and is_new_pair:              # NEW pair -> reset holding state
                    held_reversions = 0
                    prev_nonflat = False
                    cur_status, cur_risk = "HEALTHY", "LOW"
                    force_flat = False
                    risk_streak = 0
                # The break-validation cadence is GLOBAL (a fixed grid, see is_validation),
                # so it is NOT reset on a pair change -- the 23-day lookback windows stay
                # contiguous and pair-agnostic.

        # ===== FAST timescale: DAILY Direction Discovery on the ACTIVE pair =====
        # Mode A's action on selection days, Mode B (direction-only) every other day. The
        # pair is NEVER changed here; only the daily action is (re)chosen.
        run_direction = (cur_pair is not None) and (daily_direction or is_pair_selection_day)
        original_action = Action.flat
        day_reason = cur_reason
        day_failed, day_failure_type = False, None
        override_triggered = override_blocked = False
        override_reason, override_z, override_z_type = "", None, ""
        flat_streak_before = flat_streak
        sel_c, prev_td = None, _prev_trading_day(env_data, d)

        if run_direction:
            sel_c, prev_td = _active_candidate(d, prices, cur_pair, current_pc)
            # Direction Discovery runs EVERY day on the active pair (Mode A only picked the
            # pair on selection days). Pass past-3-days counterfactual memory (Step 3).
            ddB = agent.discover_daily_direction(
                candidate=sel_c, action_prompt=action_prompt,
                pair_memory=pair_memory, entry_z=entry_z,
                recent_memory=cf_memory[-3:],
                lifecycle_state=(cur_status if lifecycle else "HEALTHY"))
            original_action, day_reason = ddB.action, ddB.reason
            day_failed = getattr(ddB, "failed", False)
            day_failure_type = getattr(ddB, "failure_type", None)
            final_action = original_action

            # consecutive-FLAT override (DAILY). Lifecycle state (WARNING/BREAK) is
            # informational and NEVER blocks it; only an active cooldown on the held pair
            # does (margin rejection is handled by the portfolio step).
            cur_in_cooldown = bool(lifecycle and cur_pair is not None
                                   and "/".join(cur_pair) in _active_cooldown())
            lifecycle_block = bool(lifecycle and anti_flat_override_blocked(
                cur_status, break_triggered=break_triggered, in_cooldown=cur_in_cooldown))
            if (anti_flat_override and final_action == Action.flat
                    and flat_streak >= max_flat_streak):
                if lifecycle_block:
                    override_blocked = True
                    override_reason = "blocked by cooldown on the active pair"
                else:
                    twz = sel_c.get("trading_window_z")
                    lhz = getattr(current_pc, "current_z", None)
                    if twz is not None and abs(float(twz)) > 1e-9:
                        override_z, override_z_type = float(twz), "trading_window_z"
                    elif lhz is not None and abs(float(lhz)) > 1e-9:
                        override_z, override_z_type = float(lhz), "long_horizon_z"
                    if override_z is not None:
                        final_action = (Action.short_long if override_z > 0
                                        else Action.long_short)
                        override_triggered = True
                        override_reason = (f"flat_streak={flat_streak}>= {max_flat_streak}; "
                                           f"forced by {override_z_type}={override_z:.3f}")
            cur_action = final_action
            cur_reason = (day_reason if not override_triggered
                          else day_reason + f" | OVERRIDE: {override_reason}")
            cur_beta = (effective_hedge_ratio(
                compute_spread_z(price_history, cur_pair, lookback)[1],
                pair=cur_pair, as_of=dstr, log=result.beta_fallbacks) if beta_hedged else 1.0)

        # ----- build the day's target (FLAT/force-flat -> cash; else size held pair) -----
        day_action = Action.flat if (lifecycle and force_flat) else cur_action
        if cur_pair is None or day_action == Action.flat:
            target = _zeroed_targets(symbols)
            signal_z = 0.0
        else:
            signal_z, _beta = compute_spread_z(price_history, cur_pair, lookback)
            sizes = size_pair_position_fractional(
                prices=prices, pair=cur_pair,
                portfolio_value=max(portfolio._compute_equity(prices), 1e-9),
                action=day_action, gross_fraction=gross_allocation,
                hedge_ratio=(cur_beta if beta_hedged else 1.0))
            target = _zeroed_targets(symbols)
            target.update(sizes)

        state = portfolio.step(prices, target)
        rejected = bool(state.get("rejected_target"))
        if rejected:
            portfolio.step(prices, _zeroed_targets(symbols))  # clear to flat, no replay
        norm = float(portfolio.normalized_equity_history[-1])
        reward = (norm / prev_norm - 1.0) if prev_norm > 0 else 0.0
        prev_norm = norm
        cum_pnl += reward

        flat = (cur_pair is None or day_action == Action.flat or rejected)
        # DAILY flat-streak update (count executed FLAT trading days) when a direction ran.
        if run_direction:
            flat_streak = (flat_streak + 1) if (day_action == Action.flat) else 0
        # reversion bookkeeping (non-flat -> flat transition) for pair-quality outcomes
        is_nonflat = (not flat)
        if lifecycle and prev_nonflat and not is_nonflat:
            held_reversions += 1
        prev_nonflat = is_nonflat
        sel = "/".join(cur_pair) if cur_pair else "FLAT"
        result.records.append(StepRecord(
            date=dstr, selected_pair=(sel if cur_pair else "FLAT"),
            action=("FLAT" if flat else day_action.label),
            position_state=_infer_state(portfolio.positions, cur_pair).label if cur_pair else "FLAT",
            stance="neutral", signal_z=round(float(signal_z), 4), daily_pnl=reward,
            cumulative_pnl=cum_pnl, normalized_equity=norm, news_sentiment="",
            rationale=cur_reason, rejected_target=rejected, pair_switched=pair_switched,
            no_trade=flat,
            relationship_break_detected=break_detected, should_switch_pair=should_switch,
            relationship_status=(cur_status if lifecycle else "HEALTHY"),
            pre_break_risk=(cur_risk if lifecycle else "LOW"),
            should_force_flat=force_flat,
            lifecycle_state=(cur_status if lifecycle else "ACTIVE")))
        result.action_by_date[dstr] = ("FLAT" if flat else day_action.label)
        result.normalized_equity.append(norm)

        # ----- daily decision row + daily Direction Memory (one per trading day) -----
        if run_direction:
            _da = (None if sel_c.get("prev_price_a") is None or sel_c.get("price_a") is None
                   else round(sel_c["price_a"] - sel_c["prev_price_a"], 6))
            _db = (None if sel_c.get("prev_price_b") is None or sel_c.get("price_b") is None
                   else round(sel_c["price_b"] - sel_c["prev_price_b"], 6))
            result.decisions.append({
                "date": dstr, "decision_index": i,
                "active_pair": "/".join(cur_pair), "selected_pair": "/".join(cur_pair),
                "is_pair_selection_day": is_pair_selection_day,
                "pair_selection_trigger": pair_selection_trigger,
                "selected_action": day_action.label,
                "original_llm_action": original_action.label,
                "final_executed_action": day_action.label,
                "action_reason": cur_reason, "reason": cur_reason,
                "flat_streak_before_decision": flat_streak_before,
                "flat_override_triggered": override_triggered,
                "flat_override_blocked_by_lifecycle": override_blocked,
                "override_reason": override_reason,
                "override_z_score": override_z, "override_z_score_type": override_z_type,
                "beta": cur_beta,
                "long_horizon_z": getattr(current_pc, "current_z", None),
                "trading_window_z": sel_c.get("trading_window_z"),
                "price_A": sel_c.get("price_a"), "price_B": sel_c.get("price_b"),
                "price_A_delta": _da, "price_B_delta": _db,
                "news_window_start": (prev_td.isoformat() if prev_td else None),
                "news_window_end": dstr,
                "lifecycle_state": (cur_status if lifecycle else "ACTIVE"),
                "lifecycle_status": (cur_status if lifecycle else "ACTIVE"),
                "break_flag": break_detected, "break_triggered": break_triggered,
                "break_confirmed": break_detected,
                "warning_flag": (cur_status == "WARNING") if lifecycle else False,
                "break_score": cur_break_score,
                "statistical_break_score": cur_break_score,
                "news_break_flag": cur_news_break_flag,
                "force_flat_flag": bool(lifecycle and force_flat),
                "force_flat_triggered": bool(lifecycle and force_flat),
                "force_flat_reason": ("confirmed_break" if (lifecycle and force_flat) else ""),
                "cooldown_active": bool(lifecycle and cur_pair is not None
                                        and "/".join(cur_pair) in _active_cooldown()),
                "pair_retired": pair_retired_today,
                "direction_decision_failed": day_failed, "failure_type": day_failure_type})
            dm_idx = None
            if direction_memory is not None:
                dm_idx = direction_memory.add_decision(
                    selected_pair="/".join(cur_pair), action=day_action.label,
                    reason=cur_reason, decision_index=i, decision_date=dstr,
                    lifecycle_state=(cur_status if lifecycle else "ACTIVE"),
                    original_action=original_action.label)
            # queue this decision for counterfactual finalization on the NEXT trading day
            cf_pending = {
                "dec_idx": len(result.decisions) - 1, "dm_idx": dm_idx,
                "pair": cur_pair, "date": dstr, "action": day_action.label,
                "reason": cur_reason, "beta": (cur_beta if beta_hedged else 1.0),
                "pa": sel_c.get("price_a"), "pb": sel_c.get("price_b"),
                "da": _da, "db": _db,
                "za_long": getattr(current_pc, "current_z", None),
                "tw_z": sel_c.get("trading_window_z")}

    # Value/close the last position without another decision, news lookup or monitoring.
    if settlement_date is not None:
        final_prices = env_data[settlement_date]["price"]
        _finalize_cf(final_prices)
        portfolio.step(final_prices, _zeroed_targets(symbols))
        norm = float(portfolio.normalized_equity_history[-1])
        reward = (norm / prev_norm - 1.0) if prev_norm > 0 else 0.0
        cum_pnl += reward
        result.normalized_equity.append(norm)
        result.settlement = {"date": settlement_date.isoformat(), "daily_pnl": reward,
                             "normalized_equity": norm}

    # ----- end of window: record the final held pair's quality (Step 2) -----
    if lifecycle and cur_pair is not None and current_pc is not None:
        _retire_to_memory(cur_pair, current_pc, cur_status == "BREAK", held_reversions)

    result.cumulative_pnl = cum_pnl
    return result


def write_discovery_logs(result: "WindowResult", outdir, backend: str = "offline",
                         tag: str = "") -> Dict[str, str]:
    """Write the three audit CSVs (Step 10): decision_log, pair_pool_log, lifecycle_log.

    Returns a map of name -> path. Realized holding-period PnL is attributed to each
    decision via its recorded ``decision_index`` (same rule as ``_assign_realized_pnl``).

    ``tag`` disambiguates runs that share an ``outdir`` (e.g. a ``LLM_SIZING=both`` run
    that executes the No-Beta and Beta passes back-to-back). When non-empty it is appended
    to every filename (``lifecycle_log_<tag>.csv`` etc.) so the second pass no longer
    OVERWRITES the first pass's logs. Empty ``tag`` keeps the original unsuffixed names
    for single-mode runs (backward compatible).
    """
    import csv
    import os
    os.makedirs(outdir, exist_ok=True)
    suf = f"_{tag}" if tag else ""
    pnls = result.daily_pnls()
    n = len(pnls)
    starts = [int(dd["decision_index"]) for dd in result.decisions]

    def _pnl(k):
        # forward attribution: decision at start_k earns the return over (start_k, start_{k+1}]
        if k >= len(starts):
            return None
        lo = starts[k] + 1
        hi = (starts[k + 1] + 1) if k + 1 < len(starts) else n
        return round(sum(pnls[lo:hi]), 8)

    cooldowns = {be["pair"] for be in result.break_events}
    paths = {}
    dec_path = os.path.join(outdir, f"decision_log{suf}.csv")
    with open(dec_path, "w", newline="") as f:
        cols = ["date", "active_pair", "is_pair_selection_day", "pair_selection_trigger",
                "selected_pair", "selected_action", "original_llm_action",
                "final_executed_action", "action_reason", "flat_streak_before_decision",
                "flat_override_triggered", "flat_override_blocked_by_lifecycle",
                "override_reason", "override_z_score", "override_z_score_type",
                "backend", "beta", "long_horizon_z", "trading_window_z",
                "price_A", "price_B", "price_A_delta", "price_B_delta",
                "news_window_start", "news_window_end",
                "lifecycle_state", "break_flag", "break_confirmed", "warning_flag",
                "break_score", "statistical_break_score", "news_break_flag",
                "force_flat_flag", "force_flat_triggered", "force_flat_reason",
                "break_triggered", "cooldown_flag", "cooldown_active", "pair_retired",
                "direction_decision_failed", "failure_type", "realized_pnl"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k, dd in enumerate(result.decisions):
            row = dict(dd)
            row["backend"] = backend
            row["cooldown_flag"] = dd["selected_pair"] in cooldowns
            row["realized_pnl"] = _pnl(k)
            w.writerow(row)
    paths["decision_log"] = dec_path

    pool_path = os.path.join(outdir, f"pair_pool_log{suf}.csv")
    with open(pool_path, "w", newline="") as f:
        cols = ["date", "rank", "pair", "coint_pvalue", "beta", "half_life",
                "trading_window_z", "selected"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in result.pair_pool:
            w.writerow(row)
    paths["pair_pool_log"] = pool_path

    life_path = os.path.join(outdir, f"lifecycle_log{suf}.csv")
    with open(life_path, "w", newline="") as f:
        # Combined lifecycle recording: pure statistical break_score (0-6) + separate news
        # relation signal -> final 3-state lifecycle_state. pre_break_risk is a diagnostic.
        cols = ["date", "pair", "break_score", "statistical_lifecycle_state",
                "news_relation_signal", "news_lifecycle_state", "lifecycle_state",
                "is_validation_day", "validation_window_start", "validation_window_end",
                "validation_window_break_count", "validation_window_has_break",
                "validation_window_break_source", "pair_reselection_triggered",
                "pair_reselection_reason", "break_score_trend",
                "force_flat_triggered", "cooldown_active", "pair_retired",
                "pre_break_risk", "reasoning"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in result.monitoring:
            w.writerow(row)
    paths["lifecycle_log"] = life_path

    # Stage-1 Pair Discovery diagnostics (Top-20 + half-life filter outcome)
    if result.pair_discovery_diagnostics:
        diag_path = os.path.join(outdir, f"pair_discovery_diagnostics{suf}.csv")
        with open(diag_path, "w", newline="") as f:
            cols = ["date", "total_candidates", "n_pass_halflife", "fallback_used",
                    "final_pool", "top20"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for dg in result.pair_discovery_diagnostics:
                row = dict(dg)
                row["final_pool"] = "; ".join(dg.get("final_pool", []))
                row["top20"] = "; ".join(
                    f"{t['pair']}(p={t['coint_pvalue']},hl={t['half_life']})"
                    for t in dg.get("top20", []))
                w.writerow(row)
        paths["pair_discovery_diagnostics"] = diag_path
    return paths


def write_counterfactual_logs(result: "WindowResult", outdir, tag: str) -> Dict[str, str]:
    """Write direction_memory_<tag>.csv and counterfactual_pnl_<tag>.csv (Step 9).

    ``tag`` is e.g. ``train`` or ``test``. Both are derived from the finalized daily
    decisions (those with a counterfactual). Returns name -> path.
    """
    import csv
    import os
    os.makedirs(outdir, exist_ok=True)
    decs = [d for d in result.decisions if d.get("right_or_wrong") is not None]
    paths = {}

    mem_path = os.path.join(outdir, f"direction_memory_{tag}.csv")
    with open(mem_path, "w", newline="") as f:
        cols = ["date", "pair", "action", "reason", "price_stock1", "price_stock2",
                "delta_stock1", "delta_stock2", "actual_pnl", "cf_long_short",
                "cf_short_long", "cf_flat", "best_action", "right_or_wrong",
                "memory_sentence"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in decs:
            w.writerow({"date": d.get("date"), "pair": d.get("active_pair"),
                        "action": d.get("final_executed_action"), "reason": d.get("action_reason"),
                        "price_stock1": d.get("price_A"), "price_stock2": d.get("price_B"),
                        "delta_stock1": d.get("price_A_delta"), "delta_stock2": d.get("price_B_delta"),
                        "actual_pnl": d.get("actual_pnl"), "cf_long_short": d.get("cf_long_short"),
                        "cf_short_long": d.get("cf_short_long"), "cf_flat": d.get("cf_flat"),
                        "best_action": d.get("best_action"), "right_or_wrong": d.get("right_or_wrong"),
                        "memory_sentence": d.get("memory_sentence")})
    paths["direction_memory"] = mem_path

    cf_path = os.path.join(outdir, f"counterfactual_pnl_{tag}.csv")
    with open(cf_path, "w", newline="") as f:
        cols = ["date", "pair", "actual_action", "actual_pnl", "cf_long_short",
                "cf_short_long", "cf_flat", "best_action", "right_or_wrong", "reason",
                "error_analysis"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in decs:
            sent = d.get("memory_sentence") or ""
            err = (sent.split("Error analysis:", 1)[1].strip()
                   if d.get("right_or_wrong") == "WRONG" and "Error analysis:" in sent else "")
            w.writerow({"date": d.get("date"), "pair": d.get("active_pair"),
                        "actual_action": d.get("final_executed_action"),
                        "actual_pnl": d.get("actual_pnl"), "cf_long_short": d.get("cf_long_short"),
                        "cf_short_long": d.get("cf_short_long"), "cf_flat": d.get("cf_flat"),
                        "best_action": d.get("best_action"), "right_or_wrong": d.get("right_or_wrong"),
                        "reason": d.get("action_reason"), "error_analysis": err})
    paths["counterfactual_pnl"] = cf_path
    return paths


def decide_acceptance(
    reward_before: float, reward_after: float, rollback_threshold: float = -0.03
) -> Tuple[bool, bool, float]:
    """Reward-conditioned update decision (Section 5 + rollback rule).

    Returns ``(accepted, rollback_triggered, pnl_change_rate)`` where:
      * accepted iff ``reward_after > reward_before`` (else the previous policy
        is restored),
      * rollback_triggered iff a rejected candidate degraded performance below
        the ``-3%`` rollback threshold.
    """
    pnl_change_rate = (reward_after - reward_before) / max(abs(reward_before), 1e-9)
    accepted = reward_after > reward_before
    rollback = (not accepted) and (pnl_change_rate < rollback_threshold)
    return accepted, rollback, pnl_change_rate


# Behaviorally-active policy parameters whose change is measured as a relative delta
# and which gate numeric convergence. (risk_aversion removed entirely -- it was inert.)
_RELATIVE_PARAMS = ("entry_z", "exit_band")
_CONVERGENCE_PARAMS = _RELATIVE_PARAMS
# parameters that must be exactly unchanged
_FIXED_PARAMS = ("coint_pmax", "adf_pmax", "half_life_min", "half_life_max",
                 "no_trade_threshold", "relationship_break_threshold")


def parameter_convergence(
    prev_params: Dict[str, float], cur_params: Dict[str, float], pnl_change_rate: float,
    rel_tol: float = 0.05, pnl_tol: float = 0.03,
) -> Tuple[bool, Dict[str, object]]:
    """Policy-parameter convergence (Step 1, replaces action-overlap).

    Converged (this evaluation) iff ALL hold:
      * |entry_z change|, |exit_band change| < 5% (relative). (risk_aversion has been
        removed; only the behaviorally-active parameters gate convergence.)
      * pair-selection thresholds, no_trade threshold, relationship-break threshold
        are exactly unchanged
      * |PnL change| < 3%

    Returns ``(converged, deltas)`` where deltas records each parameter's relative
    change and per-condition pass flags (for logging/visualisation).
    """
    deltas: Dict[str, object] = {}
    rel_ok = True
    for p in _RELATIVE_PARAMS:
        old, new = float(prev_params.get(p, 0.0)), float(cur_params.get(p, 0.0))
        rel = abs(new - old) / max(abs(old), 1e-9)
        deltas[f"{p}_delta"] = rel
        deltas[f"{p}_ok"] = rel < rel_tol
        rel_ok = rel_ok and (rel < rel_tol)

    fixed_ok = True
    for p in _FIXED_PARAMS:
        unchanged = float(prev_params.get(p, 0.0)) == float(cur_params.get(p, 0.0))
        deltas[f"{p}_unchanged"] = unchanged
        fixed_ok = fixed_ok and unchanged

    pnl_ok = abs(pnl_change_rate) < pnl_tol
    deltas["pnl_change_rate"] = pnl_change_rate
    deltas["pnl_ok"] = pnl_ok

    converged = bool(rel_ok and fixed_ok and pnl_ok)
    deltas["all_ok"] = converged
    return converged, deltas
