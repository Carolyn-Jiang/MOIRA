from typing import Any, Dict, List, Tuple

from strategies.types import Action
import math

# --------------------------------------------------------------------------- #
# Margin defaults suitable for market-neutral PAIR trading.
#
# A pair trade is dollar/beta-neutral: a long leg and a short leg of roughly
# equal market value. With Reg-T style *initial* margin of 50% on each leg, a
# fully-invested pair (gross notional == equity) requires only ~50% of equity in
# margin, leaving headroom instead of sitting exactly at the limit (the legacy
# 100%/100% defaults left zero buffer, so ordinary price drift rejected nearly
# every target -- "effectively disabling market-neutral leverage"). These are
# configurable via PortfolioManager(long_initial_margin=..., short_initial_margin=...).
# --------------------------------------------------------------------------- #
DEFAULT_LONG_INITIAL_MARGIN = 0.5
DEFAULT_SHORT_INITIAL_MARGIN = 0.5

# --------------------------------------------------------------------------- #
# Single source of truth for the hedge-ratio (beta) fallback rule (Step 7).
#
#   * If 0.5 <= |beta| <= 2.0  -> use |beta| (the estimated hedge ratio).
#   * Otherwise (|beta| < 0.5, |beta| > 2.0, missing, NaN, or infinite)
#     -> use beta = 1.0 (EXPLICIT dollar-neutral fallback, not a silent error).
#
# This rule is applied identically everywhere a beta-hedged position is built
# (engine, discovery engine, baselines, beta-hedged LLM variant, fixed-coint
# baselines). When ``log`` (a list) is provided, an applied fallback appends a
# record {pair, date, original_beta, fallback_beta, reason} for the experiment logs.
# --------------------------------------------------------------------------- #
BETA_MIN = 0.5
BETA_MAX = 2.0


def effective_hedge_ratio(beta_raw, *, pair=None, as_of=None, log=None) -> float:
    """Resolve a raw/None hedge ratio to the effective beta per the Step-7 rule.

    Returns a POSITIVE magnitude in ``[0.5, 2.0]`` or ``1.0``. ``beta_raw`` may be a
    signed float, ``None``, NaN or inf. A fallback to 1.0 is logged into ``log`` (if
    given) with the original value and the reason.
    """
    reason = None
    if beta_raw is None:
        reason = "invalid beta (missing)"
    else:
        try:
            b = float(beta_raw)
        except (TypeError, ValueError):
            b = None
            reason = "invalid beta (unparseable)"
        if reason is None:
            if not math.isfinite(b):
                reason = "invalid beta (NaN/inf)"
            elif not (BETA_MIN <= abs(b) <= BETA_MAX):
                reason = f"beta outside [{BETA_MIN}, {BETA_MAX}]"
    if reason is not None:
        if log is not None:
            log.append({"pair": ("/".join(pair) if isinstance(pair, (tuple, list)) else pair),
                        "date": as_of, "original_beta": beta_raw,
                        "fallback_beta": 1.0, "reason": reason})
        return 1.0
    return abs(float(beta_raw))

def compute_hedge_ratio(
    price_history: dict,
    pair: tuple[str, str],
    lookback: int = 20,
) -> float | None:
    """
    OLS hedge ratio on log prices:
        log(A) = c + beta * log(B)
    """
    a, b = pair
    dates = sorted(price_history.keys())[-lookback:]
    if len(dates) < max(5, lookback // 2):
        return None

    a_prices, b_prices = [], []
    for d in dates:
        prices = price_history[d].get("price", {})
        pa, pb = prices.get(a), prices.get(b)
        if pa is None or pb is None or pa <= 0 or pb <= 0:
            continue
        a_prices.append(pa)
        b_prices.append(pb)

    if len(a_prices) < max(5, lookback // 2):
        return None

    x = [math.log(float(p)) for p in b_prices]
    y = [math.log(float(p)) for p in a_prices]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denom = sum((xi - x_mean) ** 2 for xi in x)
    if denom <= 1e-12:
        return None

    beta = float(sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denom)
    if not math.isfinite(beta) or abs(beta) > 2 or abs(beta) < 0.5:
        return None
    return beta


class PortfolioManager:
    """
    Applies strategy target positions to a portfolio, updating cash and positions.

    Rules:
    - Short sale proceeds are sequestered in ``restricted_cash`` (counts toward equity, but
      cannot fund new long buys).
    - Equity = cash + restricted_cash + current market value of all positions.
    - Net buying power comes from equity minus restricted cash and margin requirements; targets
      that breach the margin buffer are rejected.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_cash: float = 1.0,
        long_initial_margin: float = DEFAULT_LONG_INITIAL_MARGIN,
        short_initial_margin: float = DEFAULT_SHORT_INITIAL_MARGIN,
    ):
        if long_initial_margin <= 0:
            raise ValueError("long_initial_margin must be positive")
        if short_initial_margin < 0:
            raise ValueError("short_initial_margin cannot be negative")

        self.symbols = symbols
        self.positions: Dict[str, float] = {s: 0.0 for s in symbols}
        self.entry_prices: Dict[str, float] = {}
        self.entry_shares: Dict[str, float] = {}
        self.cash: float = initial_cash  # available cash (unrestricted)
        self.restricted_cash: float = 0.0  # short proceeds held as collateral
        self.short_proceeds: Dict[str, float] = {s: 0.0 for s in symbols}

        self.long_initial_margin = long_initial_margin
        self.short_initial_margin = short_initial_margin

        self.initial_equity: float = initial_cash
        self.equity_history: List[float] = []
        self.normalized_equity_history: List[float] = []

    # --- Helpers -----------------------------------------------------------------
    def _update_entry_metadata(
        self,
        *,
        prev_positions: Dict[str, float],
        new_positions: Dict[str, float],
        prices: Dict[str, float],
        eps: float = 1e-12,
    ) -> None:
        for symbol in self.symbols:
            prev = float(prev_positions.get(symbol, 0.0))
            new = float(new_positions.get(symbol, 0.0))

            if abs(new) < eps:
                self.entry_prices.pop(symbol, None)
                self.entry_shares.pop(symbol, None)
                continue

            opened = abs(prev) < eps
            flipped = (prev * new) < 0
            if opened or flipped:
                self.entry_prices[symbol] = float(prices[symbol])
                self.entry_shares[symbol] = float(new)
                continue

            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = float(prices[symbol])
            if symbol not in self.entry_shares:
                self.entry_shares[symbol] = float(prev)

    def _market_values(self, prices: Dict[str, float], positions: Dict[str, float]) -> Tuple[float, float]:
        long_mv = sum(max(pos, 0.0) * prices[s] for s, pos in positions.items())
        short_mv = sum(max(-pos, 0.0) * prices[s] for s, pos in positions.items())
        return long_mv, short_mv

    def _compute_equity(
        self,
        prices: Dict[str, float],
        positions: Dict[str, float] | None = None,
        cash: float | None = None,
        restricted_cash: float | None = None,
    ) -> float:
        positions = positions or self.positions
        cash = self.cash if cash is None else cash
        restricted_cash = self.restricted_cash if restricted_cash is None else restricted_cash
        return cash + restricted_cash + sum(positions[s] * prices[s] for s in positions)

    def _compute_buying_power(
        self,
        prices: Dict[str, float],
        positions: Dict[str, float] | None = None,
        cash: float | None = None,
        restricted_cash: float | None = None,
    ) -> Tuple[float, float]:
        """
        Returns (net_buying_power, margin_buffer).
        - margin_buffer <= 0 means the portfolio is at/over its allowable exposure.
        - net_buying_power is the incremental long market value capacity implied by the
          margin buffer (ignores cash gating so equity-driven margin can be used).
        """
        positions = positions or self.positions
        cash = self.cash if cash is None else cash
        restricted_cash = self.restricted_cash if restricted_cash is None else restricted_cash

        equity = self._compute_equity(prices, positions, cash, restricted_cash)
        long_mv, short_mv = self._market_values(prices, positions)

        margin_need = long_mv * self.long_initial_margin + short_mv * self.short_initial_margin
        margin_buffer = equity - margin_need

        additional_capacity = margin_buffer / self.long_initial_margin if self.long_initial_margin else 0.0
        net_buying_power = max(0.0, additional_capacity)
        return net_buying_power, margin_buffer

    def _normalize_equity_value(self, equity: float) -> float:
        if self.initial_equity and self.initial_equity != 0:
            return equity / self.initial_equity
        return float("nan")

    def _record_equity(self, equity: float) -> float:
        normalized = self._normalize_equity_value(equity)
        self.equity_history.append(equity)
        self.normalized_equity_history.append(normalized)
        return normalized

    def _simulate_trades(
        self, prices: Dict[str, float], target_positions: Dict[str, float]
    ) -> Dict[str, Any]:
        cash = self.cash
        restricted_cash = self.restricted_cash
        positions = self.positions.copy()
        short_proceeds = self.short_proceeds.copy()
        trades: List[Dict[str, Any]] = []

        for symbol in self.symbols:
            target = target_positions.get(symbol, 0.0)
            current = positions[symbol]
            if abs(target - current) <= 1e-9:
                continue

            price = prices[symbol]
            before_cash = cash
            before_restricted = restricted_cash

            current_long = max(current, 0.0)
            target_long = max(target, 0.0)
            current_short = max(-current, 0.0)
            target_short = max(-target, 0.0)

            # Cover existing shorts first (releases restricted proceeds tied to those shares).
            cover_shares = max(current_short - target_short, 0.0)
            if cover_shares > 0:
                per_share_restricted = short_proceeds[symbol] / current_short if current_short > 0 else 0.0
                release_amount = per_share_restricted * cover_shares
                short_proceeds[symbol] -= release_amount
                restricted_cash -= release_amount
                cover_cost = cover_shares * price
                cash += release_amount - cover_cost

            # Trim long exposure, sending proceeds to available cash.
            reduce_long = max(current_long - target_long, 0.0)
            if reduce_long > 0:
                cash += reduce_long * price

            # Add to long exposure (spends available cash).
            increase_long = max(target_long - current_long, 0.0)
            if increase_long > 0:
                cash -= increase_long * price

            # Add to short exposure; short sale proceeds are sequestered.
            increase_short = max(target_short - current_short, 0.0)
            if increase_short > 0:
                proceeds = increase_short * price
                restricted_cash += proceeds
                short_proceeds[symbol] += proceeds

            positions[symbol] = target

            trades.append(
                {
                    "symbol": symbol,
                    "delta_shares": target - current,
                    "price": price,
                    "cash_change": cash - before_cash,
                    "restricted_change": restricted_cash - before_restricted,
                }
            )

        # Normalize tiny float residue.
        for symbol in self.symbols:
            if abs(positions[symbol]) < 1e-12:
                positions[symbol] = 0.0
            if abs(short_proceeds[symbol]) < 1e-12:
                short_proceeds[symbol] = 0.0

        restricted_cash = sum(short_proceeds.values())
        equity = self._compute_equity(prices, positions, cash, restricted_cash)
        net_buying_power, margin_buffer = self._compute_buying_power(
            prices, positions, cash, restricted_cash
        )

        valid = margin_buffer >= -1e-9 and restricted_cash >= -1e-9
        reason = None if valid else "Target positions exceed net buying power or cash constraints"

        return {
            "positions": positions,
            "cash": cash,
            "restricted_cash": restricted_cash,
            "short_proceeds": short_proceeds,
            "trades": trades,
            "equity": equity,
            "net_buying_power": net_buying_power,
            "margin_buffer": margin_buffer,
            "valid": valid,
            "reason": reason,
        }

    # --- Public API --------------------------------------------------------------
    def step(self, prices: Dict[str, float], desired_positions: Dict[str, float]) -> Dict[str, Any]:
        target_positions = {s: desired_positions.get(s, 0.0) for s in self.symbols}
        current_buying_power, current_margin_buffer = self._compute_buying_power(prices)

        prev_positions = self.positions.copy()
        projection = self._simulate_trades(prices, target_positions)
        if not projection["valid"]:
            equity = self._compute_equity(prices)
            normalized_equity = self._record_equity(equity)
            return {
                "equity": equity,
                "normalized_equity": normalized_equity,
                "cash": self.cash,
                "restricted_cash": self.restricted_cash,
                "positions": self.positions.copy(),
                "trades": [],
                "net_buying_power": current_buying_power,
                "margin_buffer": current_margin_buffer,
                "rejected_target": True,
                "reason": projection["reason"],
            }

        # Apply projected state
        self.positions = projection["positions"]
        self.cash = projection["cash"]
        self.restricted_cash = projection["restricted_cash"]
        self.short_proceeds = projection["short_proceeds"]
        self._update_entry_metadata(
            prev_positions=prev_positions,
            new_positions=self.positions,
            prices=prices,
        )

        normalized_equity = self._record_equity(projection["equity"])

        return {
            "equity": projection["equity"],
            "normalized_equity": normalized_equity,
            "cash": self.cash,
            "restricted_cash": self.restricted_cash,
            "positions": self.positions.copy(),
            "trades": projection["trades"],
            "net_buying_power": projection["net_buying_power"],
            "margin_buffer": projection["margin_buffer"],
            "rejected_target": False,
            "reason": None,
        }

    def is_flat(self, symbols: List[str] | None = None, eps: float = 1e-9) -> bool:
        """True iff the given symbols (default: all) hold no shares.

        Used to VERIFY the portfolio is genuinely flat before allowing a new
        pair's positions to be opened (pair-switch safety invariant).
        """
        check = symbols if symbols is not None else self.symbols
        return all(abs(self.positions.get(s, 0.0)) < eps for s in check)

    def nonzero_positions(self, eps: float = 1e-9) -> Dict[str, float]:
        """All symbols with non-trivial exposure -- used to detect stale holdings."""
        return {s: p for s, p in self.positions.items() if abs(p) >= eps}

    def open_positions_input(
        self, prices: Dict[str, float] | None = None
    ) -> Dict[str, Dict[str, float | None]]:
        """
        Snapshot of open positions for model inputs.

        Returns {SYMBOL: {"entry_price": float, "shares": float, ...}}
        """
        out: Dict[str, Dict[str, float | None]] = {}
        for symbol, shares in self.positions.items():
            if abs(shares) < 1e-12:
                continue
            item: Dict[str, float | None] = {
                "entry_price": (
                    None if symbol not in self.entry_prices else float(self.entry_prices[symbol])
                ),
                "shares": float(shares),
            }
            if prices is not None and symbol in prices and prices[symbol] is not None:
                item["current_price"] = float(prices[symbol])
            out[symbol] = item
        return out


def flat_targets(pair: Tuple[str, str]) -> Dict[str, float]:
    """Explicit FLAT target: both legs set to zero (close both legs)."""
    a, b = pair
    return {a: 0.0, b: 0.0}


def size_pair_position(
    prices: Dict[str, float],
    pair: Tuple[str, str],
    notional: float,
    action: Action,
    hedge_ratio: float | None = None,
) -> Dict[str, float]:
    """
    Turn a pair-trade action into signed share counts sized by gross notional.

    Action space (explicit, no HOLD):
      LONG_SHORT -> long leg A, short leg B (scaled by hedge_ratio)
      SHORT_LONG -> short leg A, long leg B
      FLAT       -> both legs zero

    FLAT always returns explicit zeros for both legs; there is no "empty mapping
    keep yesterday" behaviour, because positions are never held implicitly.
    """
    a, b = pair
    if action == Action.flat:
        return flat_targets(pair)
    if action not in (Action.long_short, Action.short_long):
        return flat_targets(pair)

    price_a = prices[a]
    price_b = prices[b]
    beta = abs(hedge_ratio) if hedge_ratio is not None else 1.0
    gross_unit_value = price_a + beta * price_b

    if notional <= 0 or gross_unit_value <= 0:
        return flat_targets(pair)

    scale = notional / gross_unit_value
    if action == Action.long_short:
        return {a: scale, b: -scale * beta}
    return {a: -scale, b: scale * beta}


def size_pair_position_fractional(
    prices: Dict[str, float],
    pair: Tuple[str, str],
    portfolio_value: float,
    action: Action,
    gross_fraction: float = 1.0,
    hedge_ratio: float | None = None,
) -> Dict[str, float]:
    """
    Size a pair trade using a fraction of the current portfolio value.
    gross_fraction=1.0 means 100% gross notional of the current portfolio value.
    """
    if portfolio_value <= 0 or gross_fraction <= 0:
        return {pair[0]: 0.0, pair[1]: 0.0}

    notional = portfolio_value * gross_fraction
    return size_pair_position(prices, pair, notional, action, hedge_ratio)
