from __future__ import annotations

import time, random
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from typing import Iterable
import numpy as np

# ---------- Rate-limit + backoff helpers ----------
def sleep_with_jitter(base: float, jitter: float = 0.4) -> None:
    time.sleep(base + random.random() * jitter)

def get_aggs_with_backoff(client, **kwargs):
    """Polygon get_aggs with exponential backoff for 429s."""
    max_retries = 6
    base = 1.5
    for i in range(max_retries):
        try:
            return client.get_aggs(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if (("429" in msg) or ("rate" in msg) or ("retry" in msg)) and i < max_retries - 1:
                delay = min(base * (2 ** i), 30.0)
                print(f"Rate-limited; retrying in {delay:.1f}s (attempt {i+1}/{max_retries})")
                time.sleep(delay)
                continue
            raise

# ---- Small internal I/O helpers ----
def read_wide_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError(f"{p} must contain a 'date' column.")
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    df.index = pd.to_datetime(df.index)
    return df

def write_wide_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_index()
    out.index.name = "date"
    out.reset_index().to_csv(p, index=False)

def fetch_polygon_adj_close_series(
    client,
    ticker: str,
    start_date: str,
    end_date: str,
    polygon_limit: int = 50000,
) -> pd.Series:
    """
    Fetch adjusted daily close for one ticker from Polygon, returning Series indexed by date.
    """
    aggs = get_aggs_with_backoff(
        client,
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start_date,
        to=end_date,
        adjusted=True,
        sort="asc",
        limit=polygon_limit,
    )
    if not aggs:
        return pd.Series(name=ticker, dtype=float)

    tmp = pd.DataFrame(
        [{"date": datetime.fromtimestamp(a.timestamp/1000, tz=timezone.utc).date(),
          ticker: a.close} for a in aggs]
    )
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.set_index("date").sort_index()
    return tmp[ticker]


# ----------------------------
# Price integrity
# ----------------------------
def missing_price_dates(
    daily_prices_csv: str | Path,
    tickers: Iterable[str],
    start_date: str,
    end_date: str | None = None,
    freq: str = "B",  # <-- business days
) -> dict[str, list[str]]:
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = read_wide_csv(daily_prices_csv)

    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq=freq)
    out: dict[str, list[str]] = {}

    for t in tickers:
        if prices.empty or t not in prices.columns:
            out[t] = [d.date().isoformat() for d in idx]
            continue
        s = prices.reindex(idx)[t]
        miss = s[s.isna()].index
        if len(miss) > 0:
            out[t] = [d.date().isoformat() for d in miss]

    return out

def ensure_prices(
    *,
    client,
    daily_prices_csv: str | Path,
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
    request_limit_per_min: int = 4,
    pause_between: float = 1.6,
    freq: str = "B",  # <-- business days
) -> pd.DataFrame:
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = read_wide_csv(daily_prices_csv)

    idx = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq=freq)

    if prices.empty:
        prices = pd.DataFrame(index=idx)
    else:
        prices = prices.reindex(prices.index.union(idx)).sort_index()

    req_count = 0
    window_start = time.time()

    def minute_guard():
        nonlocal req_count, window_start
        if req_count >= request_limit_per_min:
            elapsed = time.time() - window_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            window_start = time.time()
            req_count = 0

    for t in tickers:
        minute_guard()

        if t not in prices.columns:
            prices[t] = pd.NA

        # Only fetch if missing any BUSINESS DAYS in required window
        if not prices.loc[idx, t].isna().any():
            continue

        s = fetch_polygon_adj_close_series(client, t, start_date, end_date)
        if s.empty:
            print(f"{t}: no data returned.")
            continue

        aligned = s.reindex(prices.index)
        mask = prices[t].isna() & aligned.notna()
        prices.loc[mask, t] = aligned.loc[mask]
        
        req_count += 1
        sleep_with_jitter(pause_between, jitter=0.5)
        write_wide_csv(prices, daily_prices_csv)

    write_wide_csv(prices, daily_prices_csv)
    return prices


# ----------------------------
# Portfolio update
# ----------------------------
def update_portfolio(
    *,
    rebalance_date: str,
    holdings_info: dict[str, dict],
    target_shares: dict[str, float],
    portfolio_snapshot_csv: str | Path,
    daily_prices_csv: str | Path,
    client,
    benchmark_ticker: str = "IWM",
    benchmark_start_date: str = "2025-01-02",
    backfill_days_new_holdings: int = 365,
) -> None:
    """
    Update portfolio_snapshot.csv (wide shares) from rebalance_date onward to match target_shares
    for tickers in holdings_info; update daily_prices.csv for active tickers + benchmark.

    - Active set after rebalance = holdings_info keys
    - target_shares MUST contain every active ticker
    - New holdings (not already in daily_prices.csv) are backfilled up to 1 year before rebalance_date
    - Holdings removed from holdings_info are NOT fetched; their price columns remain as-is / may be missing
    """
    rb = pd.to_datetime(rebalance_date)

    # --- Snapshot update ---
    snap = read_wide_csv(portfolio_snapshot_csv)
    if snap.empty:
        raise ValueError("portfolio_snapshot_csv is empty or missing. Create it first.")

    active = sorted(holdings_info.keys())
    missing_qty = [t for t in active if t not in target_shares]
    if missing_qty:
        raise ValueError(f"target_shares missing tickers: {missing_qty}")

    for t in active:
        if t not in snap.columns:
            snap[t] = pd.NA

    snap.loc[snap.index >= rb, active] = pd.DataFrame(
        {t: float(target_shares[t]) for t in active},
        index=snap.index[snap.index >= rb],
    )

    write_wide_csv(snap.round(6), portfolio_snapshot_csv)

    # --- Price update ---
    prices = read_wide_csv(daily_prices_csv)
    existing_cols = set([] if prices.empty else prices.columns)

    needed = sorted(set(active) | {benchmark_ticker})

    # Benchmark window
    ensure_prices(
        client=client,
        daily_prices_csv=daily_prices_csv,
        tickers=[benchmark_ticker],
        start_date=benchmark_start_date,
    )

    # Active holdings window:
    # - for tickers newly appearing in daily_prices, backfill 1y before rebalance
    # - for tickers already in daily_prices, just ensure from rebalance_date - 1y as well (keeps risk calcs consistent)
    start_hold = (rb - pd.Timedelta(days=backfill_days_new_holdings)).strftime("%Y-%m-%d")
    ensure_prices(
        client=client,
        daily_prices_csv=daily_prices_csv,
        tickers=[t for t in needed if t != benchmark_ticker],
        start_date=start_hold,
    )


# ----------------------------
# Dashboard portfolio table
# ----------------------------
def build_live_portfolio_dashboard(
    *,
    portfolio_snapshot_csv: str | Path,
    daily_prices_csv: str | Path,
    cost_basis_csv: str | Path | None = None,
    asof_date: str | None = None,
    cash_ticker: str = "CASH",
) -> pd.DataFrame:
    snap = read_wide_csv(portfolio_snapshot_csv)
    prices = read_wide_csv(daily_prices_csv)

    if snap.empty:
        raise ValueError("portfolio_snapshot.csv is empty.")
    if prices.empty:
        raise ValueError("daily_prices.csv is empty.")

    asof = pd.to_datetime(asof_date) if asof_date else snap.index.max()
    if asof not in snap.index:
        asof = snap.index[snap.index <= asof].max()
        if pd.isna(asof):
            raise ValueError("No snapshot date <= asof_date.")

    shares = snap.loc[asof].dropna()
    shares = shares[shares != 0]

    # last known prices as of 'asof'
    px = prices.reindex(prices.index.union([asof])).sort_index().ffill()
    latest_px = px.loc[asof] if asof in px.index else px.iloc[-1]

    # optional cost basis
    cb = {}
    if cost_basis_csv is not None and Path(cost_basis_csv).exists():
        cb_df = pd.read_csv(cost_basis_csv)
        # tolerate minor format issues
        cb_df = cb_df.dropna(subset=["ticker", "total_cost_basis"])
        cb = dict(zip(cb_df["ticker"].astype(str), cb_df["total_cost_basis"].astype(float)))

    rows = []
    for t, sh in shares.items():
        if t == cash_ticker:
            p = 1.0  # cash price convention
        else:
            p = float(latest_px[t]) if (t in latest_px.index and pd.notna(latest_px[t])) else float("nan")

        curr_val = float(sh) * p if pd.notna(p) else float("nan")
        cost_basis = float(cb.get(t, np.nan)) if cb else float("nan")
        ret = (curr_val - cost_basis) / cost_basis if (pd.notna(curr_val) and pd.notna(cost_basis) and cost_basis != 0) else float("nan")

        rows.append({
            "Ticker": str(t),
            "shares": float(sh),
            "cost basis": float(cost_basis) if pd.notna(cost_basis) else np.nan,
            "current value": float(curr_val) if pd.notna(curr_val) else np.nan,
            "return": float(ret) if pd.notna(ret) else np.nan,
        })

    live = pd.DataFrame(rows)

    # weights computed over priced positions only
    total = live["current value"].sum(skipna=True)
    live["weights"] = live["current value"] / total if total and total > 0 else 0.0

    # cash first, then descending weights
    if cash_ticker in live["Ticker"].values:
        cash = live[live["Ticker"] == cash_ticker]
        rest = live[live["Ticker"] != cash_ticker].sort_values("weights", ascending=False)
        live = pd.concat([cash, rest], ignore_index=True)
    else:
        live = live.sort_values("weights", ascending=False).reset_index(drop=True)

    return live