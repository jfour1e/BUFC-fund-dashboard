from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# If this file is in the project root, this makes `src.*` imports work
sys.path.append(str(Path(__file__).parent))

from src.portfolio_optimizer_temp import PortfolioOptimizer, OptimizationConfig  # v2 optimizer you created
from API_KEY import POLYGON_API_KEY
from polygon import RESTClient


# -----------------------------
# Trade planner
# -----------------------------
@dataclass
class TradePlanConfig:
    lot_size: int = 1
    min_trade_value: float = 0.0
    cash_buffer: float = 0.00


class TradePlanner:
    """
    Builds a trade list to move from current holdings to target weights.
    Assumes PortfolioOptimizer has: live_portfolio (with 'Ticker','shares','current value'),
    fund_aum, prices_data, sector_mapping (optional).
    """
    def __init__(self, optimizer: "PortfolioOptimizer", cfg: TradePlanConfig = TradePlanConfig()):
        self.opt = optimizer
        self.cfg = cfg

    def _latest_prices(self) -> pd.Series:
        if self.opt.prices_data is None or self.opt.prices_data.empty:
            raise ValueError("No price data available. Load data first.")
        px = self.opt.prices_data.ffill().iloc[-1]
        px.index = px.index.astype(str)
        return px

    @staticmethod
    def _round_to_lot(raw: pd.Series, lot: int) -> pd.Series:
        """
        Round shares to lot-size:
        - buys round up (ceil) so you don't under-buy
        - sells round down (floor, more negative) so you don't under-sell
        """
        lot = max(1, int(lot))

        def _r(x: float) -> int:
            if not np.isfinite(x):
                return 0
            if x < 0:
                return int(np.floor(x / lot) * lot)
            return int(np.ceil(x / lot) * lot)

        return raw.apply(_r).astype(int)

    def build_trade_sheet(self, target_weights: pd.Series) -> pd.DataFrame:
        if self.opt.live_portfolio is None or self.opt.fund_aum is None:
            raise ValueError("Live portfolio/AUM missing. Call load_portfolio_data() first.")

        # Normalize tickers
        target_weights = target_weights.copy()
        target_weights.index = target_weights.index.astype(str)

        # If duplicates somehow exist, combine them
        target_weights = target_weights.groupby(level=0).sum()

        live = self.opt.live_portfolio.copy()
        live["Ticker"] = live["Ticker"].astype(str)

        latest_prices = self._latest_prices()

        # Current dollars (fallback if 'current value' missing)
        if "current value" not in live.columns or live["current value"].isna().any():
            live["price"] = live["Ticker"].map(latest_prices)
            live["current value"] = live["shares"] * live["price"]

        # Build current series aligned to target universe
        cur_dollars = live.set_index("Ticker")["current value"].reindex(target_weights.index).fillna(0.0)
        cur_shares  = live.set_index("Ticker")["shares"].reindex(target_weights.index).fillna(0.0)

        # Target dollars (respect cash buffer)
        spendable_aum = float(self.opt.fund_aum) * (1.0 - float(self.cfg.cash_buffer))
        tgt_dollars = target_weights.fillna(0.0) * spendable_aum

        # Prices aligned
        px = latest_prices.reindex(target_weights.index)

        # If missing price: freeze that name (no trade)
        missing_price = px.isna() | (px <= 0)
        if missing_price.any():
            tgt_dollars.loc[missing_price] = cur_dollars.loc[missing_price]
            px = px.fillna(0.0)

        # Ideal target shares (continuous), then rounded
        raw_tgt_shares = tgt_dollars / px.replace(0.0, np.nan)
        tgt_shares = self._round_to_lot(raw_tgt_shares, lot=self.cfg.lot_size)

        # Compute trade shares to go from current -> target (then rounded to lot again)
        raw_trade_shares = (tgt_shares - cur_shares).astype(float)
        trade_shares = self._round_to_lot(raw_trade_shares, lot=self.cfg.lot_size)

        # Post-trade shares
        new_shares = (cur_shares + trade_shares).astype(int)

        # Trade values
        trade_value = trade_shares * px
        action = np.where(trade_value > 0, "BUY", np.where(trade_value < 0, "SELL", "HOLD"))

        # Recompute actual dollars after rounding (helpful to see drift)
        new_dollars = new_shares * px
        delta_dollars = tgt_dollars - cur_dollars

        sheet = pd.DataFrame({
            "Ticker": target_weights.index,
            "Action": action,
            "Price": px.round(4),

            "CurShares": cur_shares.astype(int),
            "TgtShares": tgt_shares.astype(int),
            "TradeShares": trade_shares.astype(int),
            "NewShares": new_shares.astype(int),

            "Cur$": cur_dollars.round(2),
            "Tgt$": tgt_dollars.round(2),
            "Delta$": delta_dollars.round(2),
            "TradeValue": trade_value.round(2),
            "New$": new_dollars.round(2),
        })

        # Remove zero trades
        sheet = sheet.loc[sheet["TradeShares"] != 0].copy()

        # Minimum trade value filter
        if float(self.cfg.min_trade_value) > 0:
            sheet = sheet.loc[sheet["TradeValue"].abs() >= float(self.cfg.min_trade_value)]

        # Sector (optional)
        if getattr(self.opt, "sector_mapping", None):
            sheet["Sector"] = sheet["Ticker"].map(self.opt.sector_mapping)

        # Sort by $ volume
        sheet = sheet.sort_values("TradeValue", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

        # Residual cash check (using rounded trades)
        rounded_spend = float(sheet["TradeValue"].sum())
        residual = float(tgt_dollars.sum() - cur_dollars.sum() - rounded_spend)

        self._print_trade_sheet(sheet, residual)
        return sheet

    def _print_trade_sheet(self, sheet: pd.DataFrame, residual_cash: float):
        base_cols = ["Ticker", "Action", "TradeShares", "Price", "TradeValue", "CurShares", "NewShares"]
        extra_cols = []
        if "Sector" in sheet.columns:
            extra_cols.append("Sector")

        view = sheet[base_cols + extra_cols]

        print("\n=== TRADE SHEET (ordered by $ volume) ===")
        with pd.option_context("display.max_rows", None, "display.float_format", "{:,.2f}".format):
            print(view.to_string(index=False))
        print(f"\nResidual cash after rounding: ${residual_cash:,.2f}")


# -----------------------------
# Main formatting entrypoint
# -----------------------------
def format_trades(
    optimized_csv: str = "optimized_portfolio_v2_20260122_230524.csv",
    lot_size: int = 1,
    min_trade_value: float = 0.0,
    cash_buffer: float = 0.00,
    output_csv: Optional[str] = None,
):
    """
    Loads the optimizer, reads optimized weights from CSV, and prints/saves a trade sheet.
    """
    # You only need the optimizer to get: live_portfolio, AUM, latest prices, sector mapping.
    config = OptimizationConfig(
        # these don't matter for formatting, but must be valid
        max_position_size=0.06,
        etf_min_total_weight=0.20,
        etf_max_total_weight=0.32,
        etf_max_single_weight=0.08,
        max_tracking_error=0.08,
        sector_penalty=5.0,
        risk_aversion=0.5,
        lookback_days=252,
        solver="MOSEK",
        verbose=False,
    )
    optimizer = PortfolioOptimizer(config)

    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY is not set.")
    client = RESTClient(POLYGON_API_KEY)

    print("\nLoading portfolio + prices for trade formatting...")
    optimizer.load_portfolio_data(client=client)

    # Read optimized csv
    p = Path(optimized_csv)
    if not p.exists():
        raise FileNotFoundError(f"Could not find: {optimized_csv}")

    df = pd.read_csv(p)
    if "Ticker" not in df.columns or "Weight" not in df.columns:
        raise ValueError("Optimized CSV must contain columns: Ticker, Weight")

    df["Ticker"] = df["Ticker"].astype(str)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    # Build target weights series
    target_weights = df.groupby("Ticker")["Weight"].sum()

    # Optional: enforce weights sum to 1 (sometimes rounding / filtering changes this)
    s = float(target_weights.sum())
    if s <= 0:
        raise ValueError("Target weights sum to 0. Check optimized CSV.")
    target_weights = target_weights / s

    planner = TradePlanner(
        optimizer,
        TradePlanConfig(lot_size=lot_size, min_trade_value=min_trade_value, cash_buffer=cash_buffer),
    )
    sheet = planner.build_trade_sheet(target_weights)

    # Save
    if output_csv is None:
        # reuse timestamp from file name if possible
        stem = p.stem.replace("optimized_portfolio_v2_", "")
        output_csv = f"trade_sheet_{stem}.csv"

    sheet.to_csv(output_csv, index=False)
    print(f"\n✓ Trade sheet saved to: {output_csv}")


if __name__ == "__main__":
    # Default run
    format_trades(
        optimized_csv="optimized_portfolio_v2_20260122_230524.csv",
        lot_size=1,
        min_trade_value=0.0,
        cash_buffer=0.00,
    )
