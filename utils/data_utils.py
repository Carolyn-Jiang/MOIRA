"""Data acquisition and news alignment.

Two important fixes / additions relative to the legacy version:

1. WEEKEND / HOLIDAY NEWS BUG FIX
   News for a trading day ``D`` is taken from the *most recent prior trading
   session*, NOT from ``D - 1 calendar day``. So Monday sees Friday's news and
   the first session after a holiday sees the last session before it. We never
   use calendar subtraction to locate news. See ``align_news_to_trading_days``.

2. OFFLINE / CACHED DATASET
   ``load_offline_dataset`` builds a fully reproducible env dataset from the
   bundled ``data/news/*.json`` files (which embed both prices and news), so the
   whole pipeline runs deterministically without hitting Yahoo Finance. The live
   ``download_price_data`` path supports up to 10 years of history with on-disk
   caching to avoid rate limits.
"""

import json
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PAIR_HISTORY_YEARS = 10


def pair_history_bounds(decision_dates):
    """Union of the rolling ten-calendar-year windows used by this experiment."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).normalize()
    if dates.empty or dates.isna().any():
        raise ValueError("Pair selection requires valid decision dates")
    return dates.min() - pd.DateOffset(years=PAIR_HISTORY_YEARS), dates.max()


def validate_pair_selection_history(frame, symbols, decision_dates):
    """Check actual coverage, rather than trusting a cache's '10y' filename.

    Up to seven calendar days at the first boundary accommodate market holidays.
    Every decision date must have a valid closing price for each requested stock.
    """
    import numpy as np
    decision_dates = list(decision_dates)
    start, end = pair_history_bounds(decision_dates)
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).normalize()
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate dates in pair-selection price history")
    frame = frame.sort_index().loc[start:end]
    for symbol in symbols:
        if symbol not in frame:
            raise ValueError(f"Missing ten-year price history for {symbol}")
        prices = pd.to_numeric(frame[symbol], errors="coerce")
        prices = prices.where(np.isfinite(prices) & (prices > 0))
        valid = prices.dropna()
        if (valid.empty or valid.index.min() > start + pd.Timedelta(days=7)
                or prices.reindex(dates).isna().any()):
            raise ValueError(f"{symbol} lacks ten-year history for the experiment's decision dates; "
                             f"required pool range {start.date()} to {end.date()}")
        frame[symbol] = prices
    return frame[list(symbols)]


def load_pair_selection_price_history(symbols, *, decision_dates, cache_dir="data/cache"):
    """Cache/download prices for causal ten-year pair selection, with no short fallback.

    Request [first decision - 10 years, last decision], not ten years back from
    today's date. Each actual selection later takes its own trailing 10Y slice.
    """
    from pathlib import Path
    symbols, decision_dates = list(symbols), list(decision_dates)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Supply distinct price-pool symbols")
    start, end = pair_history_bounds(decision_dates)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames, downloaded = [], []
    for symbol in symbols:
        path = cache / f"pool_{symbol}_10y.csv"
        frame = None
        if path.is_file():
            try:
                frame = validate_pair_selection_history(
                    pd.read_csv(path, parse_dates=["date"]).set_index("date"),
                    [symbol], decision_dates)
            except (ValueError, KeyError):
                # An existing filename cannot make an insufficient window valid.
                frame = None
        if frame is None:
            try:
                import yfinance as yf
                raw = yf.download(symbol, start=start.date().isoformat(),
                                  end=(end + pd.Timedelta(days=1)).date().isoformat(),
                                  auto_adjust=True, progress=False, threads=False)
                if raw is None or raw.empty:
                    raise ValueError("Price download returned no observations")
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                frame = validate_pair_selection_history(
                    raw[["Close"]].rename(columns={"Close": symbol}), [symbol], decision_dates)
            except Exception as exc:
                raise RuntimeError(f"Cannot obtain ten-year pair-selection prices for {symbol}. "
                                   f"Provide {path.name} covering {start.date()} to {end.date()}, "
                                   "or retry with market-data network access. No shorter horizon is used.") from exc
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index_label="date")
            temporary.replace(path)
            downloaded.append(symbol)
        frames.append(frame)
    pool = pd.concat(frames, axis=1).sort_index()
    source = "yfinance_10y_refreshed" if downloaded else "cache_10y"
    return validate_pair_selection_history(pool, symbols, decision_dates), source


# --------------------------------------------------------------------------- #
# News loading + trading-day alignment
# --------------------------------------------------------------------------- #
def load_news_data(news_dir: str, tickers: List[str]) -> Dict[date, Dict[str, List[str]]]:
    """Load summarized news per ticker per date from ``data/news/*.json``.

    Returns mapping ``date -> {ticker: [news strings]}`` keyed by the *publication*
    date (no alignment yet -- alignment happens in ``align_news_to_trading_days``).
    """
    news_map: Dict[date, Dict[str, List[str]]] = {}

    for ticker in tickers:
        path = os.path.join(news_dir, f"{ticker}_market_data.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                continue

        for date_str, content in payload.items():
            try:
                news_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            news_list = content.get("news", []) if isinstance(content, dict) else []
            if not isinstance(news_list, list):
                news_list = []
            if news_list:
                news_map.setdefault(news_date, {})[ticker] = news_list

    return news_map


def align_news_to_trading_days(
    trading_dates: Sequence[date],
    news_by_publish_date: Dict[date, Dict[str, List[str]]],
) -> Dict[date, Dict[str, List[str]]]:
    """Map each trading day to the news from the most recent *prior* session.

    This is the core weekend/holiday fix. For trading day ``D`` we collect every
    news item published from the previous trading session's date (inclusive) up
    to -- but not including -- ``D``. News published on a session's own date is
    only actionable on the *next* session (no same-day lookahead), so:

      * Monday  -> Friday (+ Saturday/Sunday) news
      * Tuesday after a Monday holiday -> the prior Friday's news, etc.

    No exposure to *future* news: a trading day only ever sees news published
    strictly before that day.
    """
    trading_dates = sorted(set(trading_dates))
    publish_dates = sorted(news_by_publish_date.keys())
    aligned: Dict[date, Dict[str, List[str]]] = {}

    prev_session: Optional[date] = None
    for d in trading_dates:
        bucket: Dict[str, List[str]] = {}
        # window of publication dates available to session D but not yet shown to
        # an earlier session: [prev_session, D)
        for pd_date in publish_dates:
            if pd_date >= d:
                break
            if prev_session is not None and pd_date < prev_session:
                continue
            for ticker, items in news_by_publish_date[pd_date].items():
                bucket.setdefault(ticker, []).extend(items)
        aligned[d] = bucket
        prev_session = d
    return aligned


# --------------------------------------------------------------------------- #
# NYSE trading calendar (weekends + US market holidays)
# --------------------------------------------------------------------------- #
def _nyse_calendar():
    """A dependency-free NYSE holiday calendar (pandas built-in rules + Good Friday)."""
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar, Holiday, nearest_workday, GoodFriday,
        USMartinLutherKingJr, USPresidentsDay, USMemorialDay, USLaborDay,
        USThanksgivingDay,
    )

    class NYSECalendar(AbstractHolidayCalendar):
        rules = [
            Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
            # NYSE full-market closure for President Carter's national day of mourning.
            Holiday("CarterNationalDayOfMourning", year=2025, month=1, day=9),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday("Juneteenth", month=6, day=19, start_date="2021-01-01",
                    observance=nearest_workday),
            Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        ]
    return NYSECalendar()


def filter_to_trading_days(dates: Sequence[date]) -> List[date]:
    """Keep only real NYSE trading days: drop weekends AND US market holidays.

    This is what lets news from non-session days (Fri/Sat/Sun before a Monday, or a
    mid-week holiday) roll forward to the NEXT trading session in
    ``align_news_to_trading_days``. Without it (e.g. a daily calendar dataset),
    every calendar day looks like a session and weekend/holiday news is never rolled.
    """
    dates = sorted(set(dates))
    if not dates:
        return []
    weekdays = [d for d in dates if d.weekday() < 5]               # drop Sat/Sun
    hol = set(_nyse_calendar().holidays(start=dates[0], end=dates[-1]).date)
    return [d for d in weekdays if d not in hol]                   # drop holidays


def next_trading_day(last_date: date) -> date:
    """First NYSE session after a decision, including weekend/holiday gaps."""
    candidates = pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=14).date
    return filter_to_trading_days(candidates)[0]


def apply_cached_prices(env_data, pool_df, symbols, settlement_date):
    """Use one adjusted-price series for decisions and the price-only final day.

    News defines decision dates; extra historical/future prices never add decisions.
    Fail rather than forward-fill a missing quote or mix different price sources.
    """
    import numpy as np
    dates = sorted(env_data)
    if not dates or settlement_date != next_trading_day(dates[-1]):
        raise ValueError("Settlement must be the next trading day after the final decision")
    required = [*dates, settlement_date]
    prices = pool_df.reindex(pd.to_datetime(required))[list(symbols)]
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("Missing/invalid cached prices for a decision or terminal settlement")
    result = {d: {**env_data[d], "price": prices.loc[pd.Timestamp(d)].to_dict()} for d in dates}
    result[settlement_date] = {"price": prices.loc[pd.Timestamp(settlement_date)].to_dict()}
    return result


# --------------------------------------------------------------------------- #
# Offline dataset (reproducible, no network)
# --------------------------------------------------------------------------- #
def load_offline_dataset(
    tickers: List[str],
    news_dir: str = "data/news",
    trading_days_only: bool = True,
) -> Dict[date, Dict]:
    """Build an env dataset from the bundled per-ticker JSON files.

    Each file maps ``"YYYY-MM-DD" -> {"prices": <float>, "news": [...]}``. We
    assemble a price matrix across tickers and keep only dates where *every* ticker
    has a price. When ``trading_days_only`` (default), the session calendar is
    further restricted to **real NYSE trading days** (weekends + US market holidays
    removed) -- the bundled files contain a row for every calendar day, so without
    this filter weekends/holidays would be treated as trading sessions and their
    news would never roll forward. News is then aligned to the prior trading
    session: a Monday sees Friday + Saturday + Sunday news; the first session after
    a holiday sees the pre-holiday session + the holiday-gap news.
    """
    prices_by_ticker: Dict[str, Dict[date, float]] = {}
    raw_news: Dict[date, Dict[str, List[str]]] = {}

    for ticker in tickers:
        path = os.path.join(news_dir, f"{ticker}_market_data.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing offline data file: {path}")
        with open(path, "r") as _fh:
            payload = json.load(_fh)
        pmap: Dict[date, float] = {}
        for date_str, content in payload.items():
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if not isinstance(content, dict):
                continue
            price = content.get("prices")
            if price is not None and not (isinstance(price, float) and pd.isna(price)):
                pmap[d] = float(price)
            news_list = content.get("news", [])
            if isinstance(news_list, list) and news_list:
                raw_news.setdefault(d, {})[ticker] = news_list
        prices_by_ticker[ticker] = pmap

    # dates where every ticker has a price
    common_dates = None
    for ticker in tickers:
        ds = set(prices_by_ticker[ticker].keys())
        common_dates = ds if common_dates is None else (common_dates & ds)
    common_dates = sorted(common_dates or [])
    if trading_days_only:
        # Restrict sessions to real NYSE trading days (drop weekends + holidays). The
        # full raw_news is kept, so news published on those non-session days rolls
        # forward to the next trading session via align_news_to_trading_days.
        common_dates = filter_to_trading_days(common_dates)
    if len(common_dates) < 2:
        raise ValueError("Not enough common-price dates in offline dataset")

    aligned_news = align_news_to_trading_days(common_dates, raw_news)

    env_data: Dict[date, Dict] = {}
    for d in common_dates:
        env_data[d] = {
            "price": {t: prices_by_ticker[t][d] for t in tickers},
            "news": aligned_news.get(d, {}),
        }
    return env_data


# --------------------------------------------------------------------------- #
# Live download (cached, supports up to 10y)
# --------------------------------------------------------------------------- #
def download_price_data(
    start_day: str,
    end_day: str,
    tickers: List[str],
    news_dir: str = "data/news",
    cache_dir: Optional[str] = "data/cache",
) -> Dict[date, Dict]:
    """Download close prices from Yahoo Finance and attach prior-session news.

    Prices are cached per (ticker, start, end) under ``cache_dir`` so repeated
    runs avoid Yahoo rate limits. News alignment uses the prior trading session
    (NOT calendar ``-1``).
    """
    import yfinance as yf  # imported lazily so offline runs never need it

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    df_list = []
    for ticker in tickers:
        cache_path = (
            os.path.join(cache_dir, f"{ticker}_{start_day}_{end_day}.csv")
            if cache_dir
            else None
        )
        if cache_path and os.path.isfile(cache_path):
            df = pd.read_csv(cache_path, parse_dates=["date"])
            df["date"] = df["date"].dt.date
        else:
            print(f"Downloading {ticker}")
            raw = yf.download(
                ticker, start=start_day, end=end_day, auto_adjust=True, progress=False
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            raw.columns.name = None
            raw["Date"] = raw["Date"].dt.date
            df = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": ticker})
            if cache_path:
                df.to_csv(cache_path, index=False)
        df_list.append(df)

    df_merged = df_list[0]
    for df in df_list[1:]:
        df_merged = df_merged.merge(df, on="date", how="outer")
    df_merged = df_merged.sort_values("date").reset_index(drop=True)

    # build trading calendar = dates where every ticker has a price
    trading_dates: List[date] = []
    for _, row in df_merged.iterrows():
        prices = {t: row[t] for t in tickers}
        if all(not pd.isna(v) for v in prices.values()):
            trading_dates.append(row["date"])

    raw_news = load_news_data(news_dir=news_dir, tickers=tickers)
    aligned_news = align_news_to_trading_days(trading_dates, raw_news)

    env_data: Dict[date, Dict] = {}
    for _, row in df_merged.iterrows():
        prices = {t: row[t] for t in tickers}
        if all(pd.isna(v) for v in prices.values()):
            continue
        d = row["date"]
        env_data[d] = {
            "price": prices,
            "news": aligned_news.get(d, {}),
        }
    return env_data


# --------------------------------------------------------------------------- #
# 10-year pool price history (for correlation-based pair selection)
# --------------------------------------------------------------------------- #
def load_pool_price_history(
    symbols: Sequence[str],
    years: int = 10,
    end_day: Optional[str] = None,
    cache_dir: Optional[str] = "data/cache",
    news_dir: str = "data/news",
    allow_short_fallback: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Return (price_df, source) with ~``years`` years of daily close prices.

    Used to construct classical stat-arb pairs from the *full* 10y pool history.
    Prices are cached per ticker so repeated runs are reproducible and avoid
    Yahoo rate limits.

    Failure policy (per spec):
      * try the per-ticker cache first, then the live Yahoo download;
      * if BOTH are unavailable, FAIL CLEARLY (raise) rather than silently
        substituting the 5-month experiment window.
      * the short bundled window is only used when ``allow_short_fallback=True``
        (explicit fallback failure mode), labelled ``offline_fallback_SHORT``.

    ``price_df`` is indexed by date with one column per symbol.
    """
    symbols = list(symbols)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def _from_cache_or_download() -> Optional[pd.DataFrame]:
        frames = []
        try:
            import yfinance as yf
        except Exception:
            yf = None
        for ticker in symbols:
            cache_path = (
                os.path.join(cache_dir, f"pool_{ticker}_{years}y.csv") if cache_dir else None
            )
            if cache_path and os.path.isfile(cache_path):
                df = pd.read_csv(cache_path, parse_dates=["date"])
            elif yf is not None:
                raw = yf.download(
                    ticker, period=f"{years}y", auto_adjust=True, progress=False,
                    end=end_day,
                )
                if raw is None or len(raw) == 0:
                    return None
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.reset_index()
                raw.columns.name = None
                df = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": ticker})
                if cache_path:
                    df.to_csv(cache_path, index=False)
            else:
                return None
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df.set_index("date")[[ticker]] if ticker in df.columns
                          else df.set_index("date"))
        if not frames:
            return None
        out = pd.concat(frames, axis=1).sort_index()
        out.columns = symbols
        return out

    df = _from_cache_or_download()
    if df is not None and len(df) > 60:
        return df, f"yfinance_{years}y"

    if not allow_short_fallback:
        raise RuntimeError(
            f"Could not obtain {years}y of price history for the pool "
            f"(cache miss and live download failed). Refusing to substitute the "
            f"5-month experiment window for 10y pair construction. Re-run with "
            f"network access to populate '{cache_dir}', or pass "
            f"allow_short_fallback=True / set ALLOW_SHORT_HISTORY_FALLBACK=1 to "
            f"explicitly accept the degraded short-history fallback."
        )

    # Explicit fallback failure mode: bundled offline prices (short window).
    env = load_offline_dataset(symbols, news_dir=news_dir)
    rows = {d: env[d]["price"] for d in sorted(env)}
    fb = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=symbols)
    fb.index = pd.to_datetime(list(rows.keys()))
    return fb.sort_index(), "offline_fallback_SHORT"


# --------------------------------------------------------------------------- #
# Multi-horizon pool price history (Step 2: 1Y / 3Y / 5Y / 10Y / 20Y)
# --------------------------------------------------------------------------- #
DEFAULT_HORIZON_YEARS: Tuple[int, ...] = (1, 3, 5, 10, 20)


def _horizon_label(years: int) -> str:
    return f"{years}Y"


def load_multi_horizon_price_history(
    symbols: Sequence[str],
    horizons: Sequence[int] = DEFAULT_HORIZON_YEARS,
    end_day: Optional[str] = None,
    cache_dir: Optional[str] = "data/cache",
    news_dir: str = "data/news",
    allow_short_fallback: bool = False,
) -> Tuple[Dict[str, Dict[str, object]], str]:
    """Return ``({horizon_label: {df, metadata, source}}, base_source)`` for Step 2.

    For every stock we load the LONGEST available adjusted-close history (the
    cached/live pool frame), then slice it to each requested horizon. When the
    full horizon is NOT available (e.g. a requested 20Y but only ~10Y on disk),
    we use the MAXIMUM available data and record the *effective* horizon in the
    metadata block:

        requested_horizon, effective_horizon, actual_start_date,
        actual_end_date, number_of_observations

    All horizons share one cached download so offline runs stay deterministic and
    never hit Yahoo more than once. Raw prices are sliced here; every statistical
    metric downstream uses LOG prices (Step 3).
    """
    horizons = list(horizons)
    longest = max(horizons) if horizons else 10
    # Load the longest available base frame ONCE (cache-first, then live yfinance).
    # Request the longest horizon; the loader falls back to the longest cache it
    # finds, so a 20Y request still resolves to the ~10Y cache offline.
    base_df, source = load_pool_price_history(
        symbols, years=longest, end_day=end_day, cache_dir=cache_dir,
        news_dir=news_dir, allow_short_fallback=allow_short_fallback,
    )
    base_df = base_df.sort_index()
    last = base_df.index.max()

    out: Dict[str, Dict[str, object]] = {}
    for y in horizons:
        label = _horizon_label(y)
        start = last - pd.DateOffset(years=y)
        sl = base_df[base_df.index >= start].dropna(how="all")
        if len(sl) == 0:
            sl = base_df.copy()
        eff_start = sl.index.min()
        eff_end = sl.index.max()
        eff_years = round((eff_end - eff_start).days / 365.25, 2) if len(sl) else 0.0
        out[label] = {
            "df": sl,
            "source": source,
            "metadata": {
                "requested_horizon": label,
                "effective_horizon": f"{eff_years}Y",
                "actual_start_date": eff_start.date().isoformat() if len(sl) else None,
                "actual_end_date": eff_end.date().isoformat() if len(sl) else None,
                "number_of_observations": int(len(sl)),
                "full_horizon_available": bool(eff_years >= y - 0.1),
            },
        }
    return out, source
