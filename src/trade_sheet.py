# src/trade_sheet_builder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class TradePlanConfig:
    lot_size: int = 1                # 1 for normal stocks, 100 if you ever want round lots, etc.
    min_trade_value: float = 0.0     # filter tiny trades (absolute $)
    cash_buffer: float = 0.00        # keep this fraction of AUM uninvested (0.02 => keep 2% cash)
    cash_ticker: str = "CASH"        # cash identifier in snapshots


# -----------------------------
# IO helpers (match your wide csv convention)
# -----------------------------
def read_wide_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


# -----------------------------
# Core logic
# -----------------------------
def _latest_prices_from_daily_prices(daily_prices_csv: str | Path) -> pd.Series:
    prices = read_wide_csv(daily_prices_csv).ffill()
    if prices.empty:
        raise ValueError("daily_prices_csv is empty.")
    latest = prices.iloc[-1].copy()
    latest.index = latest.index.astype(str)
    return latest


def _round_to_lot(x: float, lot: int) -> int:
    lot = max(1, int(lot))
    if not np.isfinite(x):
        return 0
    # For target shares: we usually want nearest, not always ceil.
    # But for trades: buys ceil, sells floor is safer.
    # We'll use "round to nearest lot" for targets and directional rounding for trades.
    return int(np.round(x / lot) * lot)


def _round_trade_to_lot(x: float, lot: int) -> int:
    lot = max(1, int(lot))
    if not np.isfinite(x):
        return 0
    if x < 0:
        return int(np.floor(x / lot) * lot)
    return int(np.ceil(x / lot) * lot)


def build_trade_sheet_from_optimizer_result(
    *,
    res: dict[str, Any],
    portfolio_snapshot_csv: str | Path,
    daily_prices_csv: str | Path,
    portfolio_value: float | None = None,
    cfg: TradePlanConfig = TradePlanConfig(),
) -> pd.DataFrame:
    """
    Build trades that move from CURRENT shares -> TARGET shares implied by optimized weights.

    Inputs:
      - res: output dict from optimize_quadratic_portfolio(...)
             must contain: res["weights"] (dict[ticker] -> weight)
      - portfolio_snapshot_csv: your wide snapshot of shares by date (like portfolio_snapshot.csv)
      - daily_prices_csv: your wide prices (like daily_prices.csv)
      - portfolio_value: optional override; if None, computed from current shares * latest prices (+ cash)
      - cfg: lot size / min trade filter / cash buffer / cash ticker

    Output columns (email/CSV friendly):
      Ticker, Action, Price, CurShares, TgtShares, TradeShares, Cur$, Tgt$, TradeValue, New$
      plus optional Sector if res has holdings_info / sectors (not required).
    """
    if "weights" not in res or not isinstance(res["weights"], dict):
        raise ValueError("res must contain res['weights'] as dict[ticker]->weight")

    # 1) Current shares as-of latest snapshot date
    snap = read_wide_csv(portfolio_snapshot_csv)
    if snap.empty:
        raise ValueError("portfolio_snapshot_csv is empty.")
    asof = snap.index.max()
    cur_shares = snap.loc[asof].dropna().copy()
    cur_shares.index = cur_shares.index.astype(str)

    # 2) Latest prices
    latest_px = _latest_prices_from_daily_prices(daily_prices_csv)

    # 3) Build target weights Series (normalize)
    w = pd.Series(res["weights"], dtype=float)
    w.index = w.index.astype(str)
    w = w.groupby(level=0).sum()

    # Optional: allow optimizer universe to be smaller than current holdings
    # If something exists in current holdings but not in w, target weight = 0 (sell down).
    universe = sorted(set(cur_shares.index) | set(w.index))
    w = w.reindex(universe).fillna(0.0)

    # Ensure weights sum to 1 (optimizer should, but guard)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Target weights sum to 0. Check res['weights'].")

    w = w / s

    # 4) Current dollars (including cash if present)
    # If a ticker has no price, assume $0 value EXCEPT cash.
    px = latest_px.reindex(universe)

    cur_dollars = pd.Series(0.0, index=universe, dtype=float)
    for t in universe:
        sh = float(cur_shares.get(t, 0.0))
        if t == cfg.cash_ticker:
            cur_dollars[t] = sh  # cash shares = dollars
        else:
            p = float(px.get(t, np.nan))
            cur_dollars[t] = sh * p if np.isfinite(p) and p > 0 else 0.0

    # Determine portfolio value if not provided
    if portfolio_value is None:
        portfolio_value = float(cur_dollars.sum())

    if portfolio_value <= 0:
        raise ValueError(f"portfolio_value must be > 0, got {portfolio_value}")

    spendable_value = float(portfolio_value) * (1.0 - float(cfg.cash_buffer))

    # 5) Target dollars (apply cash buffer by scaling spendable dollars)
    tgt_dollars = w * spendable_value

    # If you want cash explicitly to hold buffer:
    # - If CASH exists in universe, set it to portfolio_value - spendable_value + (any target cash weight * spendable_value)
    # - Otherwise, you’ll just end up “underinvested” vs total AUM (fine operationally).
    if cfg.cash_ticker in tgt_dollars.index:
        tgt_dollars[cfg.cash_ticker] += (float(portfolio_value) - spendable_value)

    # 6) Convert to target shares
    tgt_shares = pd.Series(0.0, index=universe, dtype=float)
    for t in universe:
        dollars = float(tgt_dollars.get(t, 0.0))
        if t == cfg.cash_ticker:
            tgt_shares[t] = dollars
            continue

        p = float(px.get(t, np.nan))
        if not np.isfinite(p) or p <= 0:
            # Freeze: keep current shares if price missing
            tgt_shares[t] = float(cur_shares.get(t, 0.0))
            continue

        raw = dollars / p
        tgt_shares[t] = _round_to_lot(raw, cfg.lot_size)

    # 7) Trades = target - current, then directional lot rounding for safety
    cur_shares_aligned = cur_shares.reindex(universe).fillna(0.0).astype(float)
    raw_trade = tgt_shares - cur_shares_aligned

    trade_shares = raw_trade.apply(lambda x: _round_trade_to_lot(float(x), cfg.lot_size)).astype(int)
    new_shares = (cur_shares_aligned + trade_shares).astype(float)

    # 8) Trade values
    trade_value = pd.Series(0.0, index=universe, dtype=float)
    new_dollars = pd.Series(0.0, index=universe, dtype=float)

    for t in universe:
        if t == cfg.cash_ticker:
            # CASH: trade value is change in dollars
            trade_value[t] = float(trade_shares[t])
            new_dollars[t] = float(new_shares[t])
        else:
            p = float(px.get(t, np.nan))
            if np.isfinite(p) and p > 0:
                trade_value[t] = float(trade_shares[t]) * p
                new_dollars[t] = float(new_shares[t]) * p
            else:
                trade_value[t] = 0.0
                new_dollars[t] = 0.0

    action = np.where(trade_shares.values > 0, "BUY", np.where(trade_shares.values < 0, "SELL", "HOLD"))

    sheet = pd.DataFrame({
        "Ticker": universe,
        "Action": action,
        "Price": [float(px.get(t, np.nan)) if t != cfg.cash_ticker else 1.0 for t in universe],

        "CurShares": cur_shares_aligned.values,
        "TgtShares": tgt_shares.values,
        "TradeShares": trade_shares.values,
        "NewShares": new_shares.values,

        "Cur$": cur_dollars.reindex(universe).values,
        "Tgt$": tgt_dollars.reindex(universe).values,
        "TradeValue": trade_value.values,
        "New$": new_dollars.values,
    })

    # Clean formatting
    # (keep shares as ints for non-cash; cash is dollars)
    def _fmt_shares(row, col):
        if row["Ticker"] == cfg.cash_ticker:
            return float(row[col])
        return int(np.round(float(row[col])))

    for col in ["CurShares", "TgtShares", "TradeShares", "NewShares"]:
        sheet[col] = sheet.apply(lambda r: _fmt_shares(r, col), axis=1)

    # Filter holds + tiny trades (but keep CASH if it meaningfully changes)
    sheet = sheet.loc[sheet["TradeShares"] != 0].copy()
    if float(cfg.min_trade_value) > 0:
        sheet = sheet.loc[sheet["TradeValue"].abs() >= float(cfg.min_trade_value)].copy()

    # Optional sector column if available in res (you can pass it in later if you want)
    # If you want: put holdings_info into res (or just pass HOLDINGS_INFO separately)
    if "alpha" in res:  # no-op marker; keep minimal here
        pass

    # Sort by absolute trade value
    sheet = sheet.sort_values("TradeValue", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    # Residual check: dollars after rounding vs target
    residual = float((sheet["TradeValue"].sum()))
    print("\n=== TRADE SHEET (ordered by $ volume) ===")
    with pd.option_context("display.max_rows", None, "display.float_format", "{:,.2f}".format):
        print(sheet[["Ticker", "Action", "TradeShares", "Price", "TradeValue", "CurShares", "NewShares"]].to_string(index=False))
    print(f"\nNet cash impact from listed trades (positive = spend): ${residual:,.2f}")

    return sheet


def save_trade_sheet_csv(sheet: pd.DataFrame, output_csv: str | Path) -> Path:
    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(p, index=False)
    print(f"\n✓ Trade sheet saved to: {p}")
    return p
