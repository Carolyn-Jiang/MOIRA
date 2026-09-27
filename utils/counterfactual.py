"""Daily counterfactual PnL for the three pair-trading actions + memory narration.

For a decision made on day ``T`` on the active pair ``(A, B)`` (sized as a fraction of
portfolio value), the realized forward 1-day return of each action is computed once the
NEXT trading session's prices are known. This uses the SAME market-neutral pair-return
convention as the executed engine (a long/short leg pair sized to gross notional):

    spread_unit_return = [(pA' - pA) - beta*(pB' - pB)] / (pA + beta*pB)

where ``pA, pB`` are day-``T`` prices and ``pA', pB'`` the next session's prices.

    LONG_SHORT PnL = +spread_unit_return   (long A / short B)
    SHORT_LONG PnL = -spread_unit_return   (short A / long B)
    FLAT       PnL = 0.0

``beta`` is the effective hedge ratio actually used (1.0 in no-beta mode, the clamped
beta in beta-hedged mode), so the counterfactual matches the executed accounting. The
counterfactual is a POST-HOC analysis used only for memory/reflection -- it is never
visible to the causal day-``T`` decision (it needs day ``T+1`` prices).
"""
from __future__ import annotations

from typing import Dict, Optional

ACTIONS = ("LONG_SHORT", "SHORT_LONG", "FLAT")

# Single source of truth for the counterfactual dead-band (Step: CF correction).
# Inside |s| < COUNTERFACTUAL_DEADBAND the spread move is too small to imply a
# directional edge, so FLAT is the counterfactual-best action. Used identically by
# counterfactual labeling, RIGHT/WRONG evaluation, reflection, and statistics.
COUNTERFACTUAL_DEADBAND = 0.002


def counterfactual_pnls(pa: float, pb: float, pa_next: float, pb_next: float,
                        beta: float = 1.0) -> Dict[str, float]:
    """Return {LONG_SHORT, SHORT_LONG, FLAT} forward 1-day PnL fractions."""
    denom = pa + beta * pb
    if denom <= 0 or pa is None or pb is None or pa_next is None or pb_next is None:
        return {"LONG_SHORT": 0.0, "SHORT_LONG": 0.0, "FLAT": 0.0}
    ls = ((pa_next - pa) - beta * (pb_next - pb)) / denom
    return {"LONG_SHORT": float(ls), "SHORT_LONG": float(-ls), "FLAT": 0.0}


def best_action(cf: Dict[str, float], tau: float = COUNTERFACTUAL_DEADBAND) -> str:
    """Counterfactual-best action with a dead-band (Step: CF correction).

    With ``s = cf['LONG_SHORT']`` (the forward spread return; SHORT_LONG = -s):
        |s| < tau  -> FLAT  (no directional edge)
        s  > 0     -> LONG_SHORT
        else       -> SHORT_LONG
    Unlike the old ``argmax(s, -s, 0)`` (which could never return FLAT), FLAT is now
    the best action inside the dead-band, including the exact tie ``s == 0``.
    """
    s = float(cf.get("LONG_SHORT", 0.0))
    if abs(s) < tau:
        return "FLAT"
    return "LONG_SHORT" if s > 0 else "SHORT_LONG"


def is_right(actual_action: str, cf: Dict[str, float], *,
             realized_pnl: Optional[float] = None,
             tau: float = COUNTERFACTUAL_DEADBAND) -> bool:
    """RIGHT/WRONG using the dead-band best action + the dead-band tolerance rule.

      * RIGHT if the chosen action equals the (dead-band) best action.
      * Dead-band TOLERANCE: if |s| < tau (best == FLAT) and a DIRECTIONAL action was
        chosen and its realized PnL > 0, it is RIGHT -- a profitable low-conviction
        directional trade is not penalized merely for being inside the dead-band.
      * Otherwise WRONG.
    ``realized_pnl`` is the chosen action's realized return (``cf[actual_action]`` in the
    engine's accounting); when omitted it is treated as non-positive.
    """
    best = best_action(cf, tau)
    if actual_action == best:
        return True
    s = float(cf.get("LONG_SHORT", 0.0))
    if (abs(s) < tau and actual_action in ("LONG_SHORT", "SHORT_LONG")
            and (realized_pnl or 0.0) > 0):
        return True
    return False


def _old_best_action(cf: Dict[str, float]) -> str:
    """The PRE-dead-band best action (argmax over s/-s/0; ties -> LONG_SHORT)."""
    return max(ACTIONS, key=lambda a: cf.get(a, 0.0))


def counterfactual_diagnostics(records, tau: float = COUNTERFACTUAL_DEADBAND) -> Dict[str, object]:
    """Best-action distribution + dead-band diagnostics over finalized CF records.

    ``records`` are dicts carrying ``cf_long_short`` (s), ``best_action``,
    ``right_or_wrong`` and a chosen-action field (``final_executed_action`` or
    ``action``). Reports the new best-action counts, the FLAT-best count before vs
    after the dead-band, and ``deadband_right_via_profit`` (count + pct of records made
    RIGHT only by the profitable-directional tolerance rule).
    """
    recs = [r for r in records if r.get("right_or_wrong") is not None]
    n = len(recs)
    best_after = {"LONG_SHORT": 0, "SHORT_LONG": 0, "FLAT": 0}
    flat_best_before = 0
    dband_right_via_profit = 0
    for r in recs:
        s = float(r.get("cf_long_short") or 0.0)
        cf = {"LONG_SHORT": s, "SHORT_LONG": -s, "FLAT": 0.0}
        ba = r.get("best_action") or best_action(cf, tau)
        best_after[ba] = best_after.get(ba, 0) + 1
        if _old_best_action(cf) == "FLAT":
            flat_best_before += 1
        chosen = r.get("final_executed_action") or r.get("action")
        if (abs(s) < tau and chosen in ("LONG_SHORT", "SHORT_LONG")
                and r.get("right_or_wrong") == "RIGHT"):
            dband_right_via_profit += 1
    return {"n": n, "best_action_counts": best_after,
            "flat_best_before_deadband": flat_best_before,
            "flat_best_after_deadband": best_after["FLAT"],
            "deadband_right_via_profit": dband_right_via_profit,
            "deadband_right_via_profit_pct": (round(dband_right_via_profit / n, 4) if n else 0.0)}


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.5f}"


def error_analysis(actual_action: str, cf: Dict[str, float], *,
                   za_long: Optional[float], tw_z: Optional[float],
                   da: Optional[float], db: Optional[float]) -> str:
    """Deterministic, code-computed explanation of why a WRONG choice underperformed.

    Reasons over the realized counterfactual gap, the spread state (long-horizon and
    trading-window z), and the relative daily price move (delta_A - delta_B).
    """
    best = best_action(cf)
    gap = cf.get(best, 0.0) - cf.get(actual_action, 0.0)
    bits = [f"best action was {best} (+{gap:.5f} better than {actual_action})"]
    rel = None if (da is None or db is None) else (da - db)
    if rel is not None:
        if (best == "LONG_SHORT" and rel > 0) or (best == "SHORT_LONG" and rel < 0):
            bits.append(f"today's relative price move (ΔA-ΔB={rel:+.4f}) favoured {best} "
                        f"(momentum continued, not reverted)")
        elif rel != 0:
            bits.append(f"relative price move (ΔA-ΔB={rel:+.4f}) went against {actual_action}")
    if tw_z is not None and abs(tw_z) > 1e-9:
        exp = "SHORT_LONG" if tw_z > 0 else "LONG_SHORT"
        if exp != actual_action and best == exp:
            bits.append(f"trading-window z={tw_z:+.2f} implied {exp} (mean reversion) "
                        f"but {actual_action} was taken")
    if actual_action == "FLAT":
        bits.append("FLAT forfeited an available directional edge")
    return "Error analysis: " + "; ".join(bits) + "."


def memory_sentence(*, date: str, action: str, pair: str,
                    pa: Optional[float], pb: Optional[float],
                    da: Optional[float], db: Optional[float], reason: str,
                    actual_pnl: Optional[float], cf: Dict[str, float], right: bool,
                    za_long: Optional[float] = None, tw_z: Optional[float] = None) -> str:
    """Build the required counterfactual memory sentence (+ error analysis if WRONG)."""
    a, b = (pair.split("/") + ["?", "?"])[:2]
    s = (f"On day {date}, based on information available at T-1, I chose {action} on {pair}. "
         f"Prices were {_fmt(pa)}/{_fmt(pb)}, and price deltas were {_fmt(da)}/{_fmt(db)}. "
         f"I chose this based on: {reason}. "
         f"The realized PnL was {_fmt(actual_pnl)}. "
         f"If I had chosen the other actions, LONG_SHORT would have produced "
         f"{_fmt(cf.get('LONG_SHORT'))}, SHORT_LONG would have produced "
         f"{_fmt(cf.get('SHORT_LONG'))}, and FLAT would have produced "
         f"{_fmt(cf.get('FLAT'))}. "
         f"This indicates I made a {'RIGHT' if right else 'WRONG'} choice.")
    if not right:
        s += " " + error_analysis(action, cf, za_long=za_long, tw_z=tw_z, da=da, db=db)
    return s
