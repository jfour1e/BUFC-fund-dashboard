# src/daily_update_job.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from polygon import RESTClient

from companies import HOLDINGS_INFO
from portfolio_data_manager import read_wide_csv, ensure_prices


def _project_root() -> Path:
    # src/ -> project root
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    # Default: <root>/data, but allow override via env var (useful on servers)
    root = _project_root()
    return Path(os.environ.get("DATA_DIR", str(root / "data")))


def _required_tickers(snapshot_cols: Iterable[str]) -> list[str]:
    # Union of snapshot columns + current holdings + benchmark, but never include CASH
    tickers = set(snapshot_cols) | set(HOLDINGS_INFO.keys()) | {"IWM"}
    tickers.discard("CASH")
    return sorted(tickers)


def run_daily_update(
    *,
    polygon_api_key: str,
    backfill_days: int = 365,
    benchmark_ticker: str = "IWM",
    benchmark_start_date: str = "2025-01-02",
) -> None:
    """
    Daily updater for small, repo-local CSVs.

    Reads:
      - data/portfolio_snapshot.csv (wide)
    Updates/creates:
      - data/daily_prices.csv (wide)

    Behavior:
      - Ensures benchmark (IWM) from benchmark_start_date.
      - Ensures all relevant tickers from (today - backfill_days) to today.
      - Excludes CASH by design (cash handled as price=1.0 in dashboard logic).
    """
    data_dir = _data_dir()
    snapshot_path = data_dir / "portfolio_snapshot.csv"
    prices_path = data_dir / "daily_prices.csv"

    snap = read_wide_csv(snapshot_path)
    if snap.empty:
        raise ValueError(f"Snapshot missing/empty: {snapshot_path}")

    tickers = _required_tickers(snap.columns)
    if not tickers:
        raise ValueError("No tickers found to update (snapshot + HOLDINGS_INFO empty?)")

    client = RESTClient(polygon_api_key)

    # Backfill window used for risk metrics
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=backfill_days)).strftime("%Y-%m-%d")

    # 1) Ensure benchmark from fixed start
    ensure_prices(
        client=client,
        daily_prices_csv=prices_path,
        tickers=[benchmark_ticker],
        start_date=benchmark_start_date,
        freq="B",  # business days to avoid weekend "missing"
    )

    # 2) Ensure holdings from rolling backfill window
    holdings_tickers = [t for t in tickers if t != benchmark_ticker]
    ensure_prices(
        client=client,
        daily_prices_csv=prices_path,
        tickers=holdings_tickers,
        start_date=start_date,
        freq="B",
    )

    print(
        f"[daily_update_job] Updated daily prices for {len(holdings_tickers)} tickers "
        f"+ {benchmark_ticker}. Wrote: {prices_path}"
    )


def main() -> None:
    """
    Entrypoint:
      POLYGON_API_KEY must be set in your environment.

    Optional:
      DATA_DIR can override the default <root>/data
      BACKFILL_DAYS can override default 365
    """
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env var: POLYGON_API_KEY")

    backfill_days = int(os.environ.get("BACKFILL_DAYS", "365"))
    run_daily_update(polygon_api_key=api_key, backfill_days=backfill_days)


if __name__ == "__main__":
    main()
