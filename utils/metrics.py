import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def compute_final_value_and_ar(
    equity_series: pd.Series, starting_value: float, trading_days_per_year: int = 252
) -> Tuple[float, float]:
    """
    Return (final_value, annualized_return).
    final_value reflects the portfolio value at the end of the series (can be normalized).
    If inputs are invalid or the series is non-positive, annualized return is NaN.
    """
    if equity_series.empty or starting_value is None:
        return float("nan"), float("nan")

    final_value = float(equity_series.iloc[-1])
    if final_value > 0 and starting_value > 0:
        ann_return = (final_value / starting_value) ** (
            trading_days_per_year / len(equity_series)
        ) - 1
    else:
        ann_return = float("nan")

    return final_value, ann_return


def compute_performance_metrics(
    equity_series: pd.Series,
    starting_value: float,
    risk_free_daily: float = 0.000085,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Compute common performance metrics:
    - final_value: ending normalized portfolio value (or absolute if not normalized)
    - annualized_return: compounded from start to finish
    - sharpe_ratio: excess return over risk-free per unit risk, annualized
    - max_drawdown: max peak-to-trough decline on total equity
    - annualized_volatility: std of daily returns, annualized
    """
    final_value, ann_return = compute_final_value_and_ar(
        equity_series, starting_value, trading_days_per_year
    )

    if equity_series.empty or starting_value is None:
        return {
            "final_value": float("nan"),
            "annualized_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "annualized_volatility": float("nan"),
        }

    equity_total = equity_series
    daily_returns = equity_total.pct_change().dropna()

    # Flat (e.g. all-NO_TRADE) portfolio: returns are numerically ~0, so the
    # Sharpe ratio is undefined (tiny std blows it up). Treat it as flat.
    flat = (len(equity_total) == 0) or (
        float(equity_total.max() - equity_total.min()) < 1e-9)

    # Annualized volatility
    vol_daily = daily_returns.std()
    ann_vol = 0.0 if flat else (
        vol_daily * np.sqrt(trading_days_per_year) if not np.isnan(vol_daily) else np.nan)

    # Sharpe ratio (annualized)
    if flat:
        sharpe = np.nan  # no meaningful risk-adjusted return for a flat portfolio
    elif vol_daily is not None and vol_daily > 1e-12 and not np.isnan(vol_daily):
        sharpe = ((daily_returns.mean() - risk_free_daily) / vol_daily) * np.sqrt(
            trading_days_per_year
        )
    else:
        sharpe = np.nan

    # Max drawdown
    if equity_total.empty:
        max_dd = np.nan
    else:
        running_max = equity_total.cummax()
        drawdown = equity_total / running_max - 1.0
        max_dd = drawdown.min()

    return {
        "final_value": final_value,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "annualized_volatility": ann_vol,
    }

# Backwards compatibility alias
compute_final_equity_and_ar = compute_final_value_and_ar
