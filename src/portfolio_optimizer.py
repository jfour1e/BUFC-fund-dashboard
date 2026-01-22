from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import cvxpy as cp

# Import existing modules
from .data_get import (
    load_clean_holdings, load_clean_sector_allocations,
    fetch_price_data, build_live_portfolio, fetch_RUT_data
)

from .companies import (
    ETF_TICKERS, STOCK_TICKERS, NORMALIZED_SECTOR_MAP,
    RUSSELL_SECTOR_WEIGHTS, EXPECTED_RETURNS
)

# -----------------------------
# Config and Optimizer classes
# -----------------------------
@dataclass
class OptimizationConfig:
    """
    One-step optimizer config.

    Notes:
    - max_tracking_error is ANNUALIZED TE 
    - sector_penalty controls how hard you "hug" Russell sector weights (soft constraint).
    """
    # Position constraints
    max_position_size: float = 0.12     # 7% cap per name
    min_position_size: float = 0.00     # usually keep at 0 for feasibility in long-only selection

    # ETF sleeve controls
    etf_min_total_weight: float = 0.10  # e.g. 20%
    etf_max_total_weight: float = 1.0  # e.g. 35%
    etf_max_single_weight: float = 0.1

    # Risk constraints / penalties
    max_tracking_error: float = 0.12    # annualized TE vs benchmark
    risk_aversion: float = 0.01         # optional variance penalty

    # Sector drift penalty (soft constraint)
    sector_penalty: float = 5.0         # larger => closer to Russell sector weights (L1 penalty) 
    sector_te_penalty: float = 5.0
    # Data window
    lookback_days: int = 252

    # Benchmark
    benchmark_ticker: str = "IWM"       
    benchmark_source: str = "IWM"      
   
    # Solver
    solver: str = "MOSEK"
    solver_tolerance: float = 1e-6
    verbose: bool = False


@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    tracking_error: float
    information_ratio: float
    sector_deviations: pd.Series
    optimization_status: str
    solver_time: float
    benchmark_annual_return: float


# -----------------------------
# Optimizer
# -----------------------------
class PortfolioOptimizer:
    """
    One-step long-only optimizer over (stocks + ETFs).

    - Objective: maximize expected return - sector_penalty * L1(sector drift) - risk_aversion * variance
    - Constraint: Tracking Error vs benchmark <= max_tracking_error (computed from returns time series)
    - Constraints: long-only, sum weights = 1, per-name caps, ETF sleeve bounds, per-ETF cap
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

        # Static mappings / priors
        self.russell_sector_weights = pd.Series(RUSSELL_SECTOR_WEIGHTS, dtype=float)
        self.russell_sector_weights /= self.russell_sector_weights.sum()

        self._base_sector_map = NORMALIZED_SECTOR_MAP.copy()
        self.expected_returns = pd.Series(EXPECTED_RETURNS, dtype=float)

        # Live data containers
        self.sector_mapping: Optional[Dict[str, str]] = None
        self.current_weights: Optional[pd.Series] = None

        self.prices_data: Optional[pd.DataFrame] = None   # asset prices
        self.benchmark_prices: Optional[pd.Series] = None

        self.holdings_data: Optional[pd.DataFrame] = None
        self.sector_allocations: Optional[pd.DataFrame] = None
        self.live_portfolio: Optional[pd.DataFrame] = None
        self.fund_aum: Optional[float] = None

        # For reporting
        self._last_returns_matrix: Optional[np.ndarray] = None
        self._last_benchmark_returns: Optional[np.ndarray] = None
        self._last_asset_list: Optional[pd.Index] = None
        self._last_sector_list: Optional[pd.Index] = None

    # -----------------------------
    # Data loading
    # -----------------------------
    def load_portfolio_data(self, filepath: Optional[str] = None, client=None) -> None:
        """
        Loads:
        - holdings (from Excel)
        - prices for tickers present in holdings
        - benchmark prices (IWM by default, or RUT via fetch_RUT_data)
        - live_portfolio values, weights, and AUM
        """
        print("\nLoading portfolio data...")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if filepath is None:
            filepath = os.path.join(BASE_DIR, "BUFC_May_2025_Allocations.xlsx")

        self.holdings_data = load_clean_holdings(filepath)
        self.sector_allocations = load_clean_sector_allocations(filepath)
        print(f"✓ Loaded {len(self.holdings_data)} holdings")

        if client is None:
            raise ValueError("Polygon RESTClient required. Pass client=RESTClient(...)")

        # Ticketers from holdings
        tickers = (
            self.holdings_data.get("Ticker", pd.Series(dtype=str))
            .dropna().astype(str).unique().tolist()
        )

        # Fetch asset prices
        try:
            self.prices_data = fetch_price_data(tickers, client)
            print(f"✓ Loaded price data for {len(self.prices_data.columns)} tickers")
        except Exception as e:
            print(f"⚠️ Warning: could not fetch price data: {e}")
            self.prices_data = pd.DataFrame()

        # Fetch benchmark prices
        self.benchmark_prices = self._fetch_benchmark_prices(client)
        if self.benchmark_prices is None or len(self.benchmark_prices) == 0:
            raise RuntimeError("Could not load benchmark prices.")

        # Build live portfolio
        live_portfolio = build_live_portfolio(self.holdings_data, self.prices_data)

        latest_prices = self.prices_data.ffill().iloc[-1] if self.prices_data is not None and not self.prices_data.empty else pd.Series(dtype=float)
        mask = live_portfolio["Ticker"].isin(latest_prices.index)

        live_portfolio.loc[mask, "current value"] = (
            live_portfolio.loc[mask, "shares"] *
            live_portfolio.loc[mask, "Ticker"].map(latest_prices)
        )

        total_value = float(live_portfolio["current value"].sum())
        if total_value <= 0:
            raise RuntimeError("Computed AUM is non-positive; check holdings/prices.")

        live_portfolio["weights"] = live_portfolio["current value"] / total_value

        self.live_portfolio = live_portfolio
        self.current_weights = live_portfolio.set_index("Ticker")["weights"]
        self.fund_aum = total_value

        print(f"✓ Built live portfolio | AUM: ${total_value:,.2f}")
        print(f"  Current portfolio has {len(self.current_weights)} positions")

    def _fetch_benchmark_prices(self, client) -> pd.Series:
        try:
            ser = fetch_RUT_data(client)  # actually IWM proxy in your code
            ser = pd.Series(ser).dropna()
            ser.index = pd.to_datetime(ser.index).normalize()
            print(f"✓ Loaded benchmark (IWM proxy) ({len(ser)} records)")
            return ser
        except Exception as e:
            print(f"⚠️ Warning: could not fetch benchmark via fetch_RUT_data: {e}")
            return pd.Series(dtype=float)
        
    def get_current_portfolio_weights(self) -> pd.Series:
        if self.current_weights is None:
            raise ValueError("Current weights not initialized. Call load_portfolio_data() first.")
        return self.current_weights

    # -----------------------------
    # Core math helpers
    # -----------------------------
    def _build_universe(self) -> pd.Index:
        """
        Optimization universe = intersection of:
        - expected_returns keys
        - price columns available
        - (optional) you can restrict further if desired
        """
        if self.prices_data is None or self.prices_data.empty:
            raise ValueError("No asset prices loaded.")
        assets = pd.Index(self.expected_returns.index).intersection(self.prices_data.columns)
        if len(assets) == 0:
            raise ValueError("No common assets between expected_returns and available prices.")
        return assets

    def _sector_map_for_assets(self, assets: pd.Index) -> Dict[str, str]:
        sector_map = {t: self._base_sector_map.get(t) for t in assets}
        # Fail-safe default — but warn loudly so you fix mapping rather than silently bucket to IT
        missing = [t for t, s in sector_map.items() if s is None]
        if missing:
            warnings.warn(
                f"{len(missing)} tickers missing from NORMALIZED_SECTOR_MAP. "
                f"Defaulting to 'Information Technology': {missing[:10]}{'...' if len(missing)>10 else ''}"
            )
        sector_map = {t: (sector_map[t] or "Information Technology") for t in assets}
        return sector_map

    def _compute_returns_matrix(
        self,
        assets: pd.Index,
        lookback_days: int
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Returns:
        - R: (T x N) daily returns for assets
        - rb: (T,) daily returns for benchmark
        """
        if self.prices_data is None or self.prices_data.empty:
            raise ValueError("No prices_data available.")
        if self.benchmark_prices is None or len(self.benchmark_prices) == 0:
            raise ValueError("No benchmark prices available.")

        px_assets = self.prices_data.reindex(columns=assets).copy()
        px_bench = self.benchmark_prices.copy()

        # Align by date index intersection
        common_idx = px_assets.index.intersection(px_bench.index)
        px_assets = px_assets.loc[common_idx].ffill()
        px_bench = px_bench.loc[common_idx].ffill()

        # Compute returns
        rets_assets = px_assets.pct_change().dropna()
        rets_bench = px_bench.pct_change().dropna()

        # Re-align after pct_change drop
        common_idx2 = rets_assets.index.intersection(rets_bench.index)
        rets_assets = rets_assets.loc[common_idx2]
        rets_bench = rets_bench.loc[common_idx2]

        # Lookback
        if len(common_idx2) < max(30, lookback_days // 4):
            warnings.warn(f"Only {len(common_idx2)} return observations available; results may be unstable.")

        rets_assets = rets_assets.tail(lookback_days)
        rets_bench = rets_bench.tail(lookback_days)

        # Drop any asset columns that still have NaNs (insufficient history)
        bad = rets_assets.columns[rets_assets.isna().any()].tolist()
        if bad:
            rets_assets = rets_assets.drop(columns=bad)
            warnings.warn(
                f"Dropped {len(bad)} assets due to NaNs in lookback returns: "
                f"{bad[:10]}{'...' if len(bad)>10 else ''}"
            )

        assets_named = pd.Index(rets_assets.columns)

        R = rets_assets.to_numpy(dtype=float)   # T x N
        rb = rets_bench.to_numpy(dtype=float).reshape(-1)  # T

        return R, rb, rets_assets.index, assets_named

    def _annualize_mean(self, x: np.ndarray) -> float:
        return float(np.mean(x) * 252.0)
    
    def _sector_to_etf(self) -> Dict[str, str]:
        # invert ETF part of NORMALIZED_SECTOR_MAP
        etf_set = set([t.upper() for t in ETF_TICKERS])
        out = {}
        for tkr, sec in self._base_sector_map.items():
            if tkr.upper() in etf_set:
                # if multiple ETFs map to same sector, keep the first; you can override manually
                out.setdefault(sec, tkr)
        return out

    # -----------------------------
    # One-step optimization
    # -----------------------------
    def optimize_portfolio(self) -> OptimizationResult:
        """
        One-step solve:
        maximize    mu'w  -  gamma_sectorTE * Σ_s Var( active_sector_contrib_s )  -  lambda * w' Σ w
        subject to  sum(w)=1, w>=0
                    per-name caps (+ optional min pos)
                    ETF sleeve bounds + per-ETF cap
                    Tracking Error vs benchmark <= TE_cap (computed from return series)
        """
        cfg = self.config

        # 1) Universe + sector mapping
        assets = self._build_universe()
        sector_map = self._sector_map_for_assets(assets)
        self.sector_mapping = sector_map

        # 2) Returns matrix aligned with surviving tickers
        R, rb, idx, assets_named = self._compute_returns_matrix(assets, cfg.lookback_days)
        T, N = R.shape
        if N == 0:
            raise ValueError("No assets left after cleaning returns matrix.")
        if len(rb) != T:
            raise RuntimeError("Benchmark returns length mismatch after alignment.")

        # 3) Expected returns aligned to universe
        mu = self.expected_returns.reindex(assets_named).astype(float).to_numpy()

        # 4) Build annualized covariance for total-vol penalty (optional)
        Sigma = np.cov(R, rowvar=False) * 252.0
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma[np.diag_indices(N)] += 1e-12

        # 5) Decision variable
        w = cp.Variable(N, nonneg=True)

        constraints = [cp.sum(w) == 1.0]

        # Per-name caps
        constraints.append(w <= float(cfg.max_position_size))
        if float(cfg.min_position_size) > 0:
            constraints.append(w >= float(cfg.min_position_size))

        # 6) ETF sleeve constraints
        etf_set = set(t.upper() for t in ETF_TICKERS)
        is_etf = np.array([t.upper() in etf_set for t in assets_named], dtype=bool)

        if is_etf.any():
            constraints += [
                cp.sum(w[is_etf]) >= float(cfg.etf_min_total_weight),
                cp.sum(w[is_etf]) <= float(cfg.etf_max_total_weight),
                w[is_etf] <= float(cfg.etf_max_single_weight),
            ]
        else:
            if float(cfg.etf_min_total_weight) > 1e-12:
                raise ValueError("ETF min sleeve requested but no ETF tickers are in the optimization universe.")

        # 7) Global Tracking Error constraint vs benchmark
        tecap = float(cfg.max_tracking_error)
        if tecap <= 0:
            raise ValueError("max_tracking_error must be > 0.")

        ones = np.ones((T, 1), dtype=float)
        M = np.eye(T) - (ones @ ones.T) / float(T)  # de-meaning operator

        active = M @ (R @ w - rb)  # T-vector (demeaned)
        te2 = (252.0 / max(1, T - 1)) * cp.sum_squares(active)
        constraints.append(te2 <= tecap**2)

        # 8) Sector TE penalty (skip sectors without ETF proxies)
        # Build mapping sector -> ETF ticker using NORMALIZED_SECTOR_MAP restricted to ETF tickers.
        sector_to_etf = self._sector_to_etf()

        ticker_to_col = {t: j for j, t in enumerate(assets_named)}
        rsw = (self.russell_sector_weights / self.russell_sector_weights.sum()).copy()

        sector_te_terms = []
        used_sectors = []

        ann = 252.0
        for sector, w_bench_sector in rsw.items():
            bench_etf = sector_to_etf.get(sector, None)
            if bench_etf is None:
                # no sector ETF proxy (e.g., Communications, Staples, and Industrials in your current ETF list)
                continue
            if bench_etf not in ticker_to_col:
                # ETF proxy not in universe after return cleaning
                continue

            # which assets are in this sector?
            idxs = [j for j, t in enumerate(assets_named) if sector_map.get(t) == sector]
            if not idxs:
                continue

            # Portfolio sector contribution time series: c_s(t) = sum_{i in s} w_i r_i(t)
            R_s = R[:, idxs]           # T x k numpy
            c_s = R_s @ w[idxs]        # CVXPY expression (T,)

            # Benchmark sector contribution proxy: w_s^{Russell} * r_{ETF_s}(t)
            r_b_s = R[:, ticker_to_col[bench_etf]]  # (T,) numpy
            c_b = float(w_bench_sector) * r_b_s     # (T,) numpy

            # Active sector contribution
            a_s = c_s - c_b

            # Penalize annualized variance of a_s:
            # Var(a_s) ≈ (1/(T-1))*||M a_s||^2
            a_centered = M @ a_s
            sector_te_terms.append((ann / max(1, T - 1)) * cp.sum_squares(a_centered))
            used_sectors.append(sector)
        
        scale = np.sqrt(252.0 / max(1, T - 1))
        te_s = scale * cp.norm(M @ a_s, 2)   # this is sector TE (annualized stdev)
        sector_te_terms.append(te_s)
        sector_te_penalty = cp.sum(sector_te_terms)

        # 9) Objective
        gamma = float(getattr(cfg, "sector_te_penalty", 0.0))   # NEW coefficient
        lam = float(cfg.risk_aversion)

        objective = cp.Maximize(mu @ w - gamma * sector_te_penalty - lam * cp.quad_form(w, Sigma))
   
        # 10) Solve
        t0 = time.time()
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=getattr(cp, cfg.solver), verbose=bool(getattr(cfg, "verbose", False)))
        solve_time = time.time() - t0

        if prob.status in ("infeasible", "unbounded") or w.value is None:
            raise RuntimeError(f"Optimization failed: {prob.status}")

        w_val = np.clip(np.asarray(w.value).reshape(-1), 0.0, None)
        if w_val.sum() <= 0:
            raise RuntimeError("Optimization returned all-zero weights.")
        w_val = w_val / w_val.sum()

        weights = pd.Series(w_val, index=assets_named).sort_values(ascending=False)

        # 11) Reporting metrics
        port_daily = (R @ w_val).reshape(-1)
        active_daily = port_daily - rb
        te = float(np.std(active_daily, ddof=1) * np.sqrt(252.0))

        bench_ann = self._annualize_mean(rb)
        port_ret = float(mu @ w_val)  # expected return input
        port_vol = float(np.sqrt(w_val.T @ Sigma @ w_val))
        ir = float((port_ret - bench_ann) / te) if te > 1e-12 else 0.0

        # Sector deviations (weights, for display only)
        sectors_all = rsw.index.tolist()
        A = np.zeros((len(sectors_all), N), dtype=float)
        sec_to_i = {s: i for i, s in enumerate(sectors_all)}
        for j, t in enumerate(assets_named):
            s = sector_map.get(t, "Information Technology")
            if s in sec_to_i:
                A[sec_to_i[s], j] = 1.0
        sector_weights = pd.Series(A @ w_val, index=sectors_all)
        sector_devs = (sector_weights - rsw).reindex(rsw.index)

        # Save for debugging
        self._last_returns_matrix = R
        self._last_benchmark_returns = rb
        self._last_asset_list = assets_named
        self._last_sector_list = pd.Index(sectors_all)

        # Optional: print which sectors got TE penalty
        if used_sectors:
            print(f"\nSector TE penalty applied to: {used_sectors}")

        return OptimizationResult(
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            tracking_error=te,
            information_ratio=ir,
            sector_deviations=sector_devs,
            optimization_status=str(prob.status),
            solver_time=float(solve_time),
            benchmark_annual_return=float(bench_ann),
        )


    # -----------------------------
    # Output helpers
    # -----------------------------
    def sector_te_report(self, weights: pd.Series) -> pd.DataFrame:
        """
        Computes sector active-contribution TE for sectors with ETF proxies.
        Returns a DataFrame with:
        - sector
        - bench_weight (Russell)
        - portfolio_weight
        - TE_sector (annualized stdev of active sector contribution)
        - var_share (share of summed sector active variances)
        """
        if self._last_returns_matrix is None or self._last_benchmark_returns is None or self._last_asset_list is None:
            raise ValueError("No cached returns found. Run optimize_portfolio() first.")

        R = self._last_returns_matrix          # T x N
        rb = self._last_benchmark_returns      # T
        assets = list(self._last_asset_list)   # length N
        T, N = R.shape

        w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)

        rsw = (self.russell_sector_weights / self.russell_sector_weights.sum()).copy()
        sector_to_etf = self._sector_to_etf()
        sector_map = self.sector_mapping or {}

        ticker_to_col = {t: j for j, t in enumerate(assets)}

        rows = []
        variances = []

        for sector, bench_w in rsw.items():
            bench_etf = sector_to_etf.get(sector)
            if bench_etf is None:
                continue
            if bench_etf not in ticker_to_col:
                continue

            idxs = [j for j, t in enumerate(assets) if sector_map.get(t) == sector]
            if not idxs:
                continue

            # portfolio sector contribution
            c_p = R[:, idxs] @ w[idxs]  # (T,)

            # benchmark sector proxy contribution
            r_s = R[:, ticker_to_col[bench_etf]]  # (T,)
            c_b = float(bench_w) * r_s

            active = c_p - c_b
            te_s = float(np.std(active, ddof=1) * np.sqrt(252.0))
            var_s = float(np.var(active, ddof=1))
            variances.append(var_s)

            port_w = float(weights.loc[[t for t in weights.index if sector_map.get(t) == sector]].sum())

            rows.append({
                "Sector": sector,
                "BenchWeight": float(bench_w),
                "PortWeight": port_w,
                "ActiveContribTE": te_s,
                "BenchETFProxy": bench_etf
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        total_var = sum(variances) if sum(variances) > 0 else 1.0
        df["VarShare"] = [v / total_var for v in variances]

        # Sort by TE or share
        df = df.sort_values("ActiveContribTE", ascending=False).reset_index(drop=True)
        return df
    
    def active_return_report(self, weights: pd.Series) -> dict:
        if self._last_returns_matrix is None or self._last_benchmark_returns is None or self._last_asset_list is None:
            raise ValueError("No cached returns found. Run optimize_portfolio() first.")

        R = self._last_returns_matrix
        rb = self._last_benchmark_returns
        assets = list(self._last_asset_list)

        w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)

        rp = R @ w
        active = rp - rb

        ann_active_mean = float(np.mean(active) * 252.0)
        te = float(np.std(active, ddof=1) * np.sqrt(252.0))

        # Beta to benchmark
        var_b = np.var(rb, ddof=1)
        beta = float(np.cov(rp, rb, ddof=1)[0, 1] / var_b) if var_b > 1e-12 else np.nan

        # Residual TE from regression rp ~ alpha + beta * rb
        # residual = rp - (a + beta rb)
        a = float(np.mean(rp) - beta * np.mean(rb)) if np.isfinite(beta) else 0.0
        resid = rp - (a + beta * rb)
        resid_te = float(np.std(resid, ddof=1) * np.sqrt(252.0))

        return {
            "ActiveMeanAnn": ann_active_mean,
            "TE": te,
            "BetaToBenchmark": beta,
            "ResidualTE": resid_te,
        }


    def get_portfolio_summary(self, result: OptimizationResult) -> pd.DataFrame:
        rows = []
        aum = float(self.fund_aum) if self.fund_aum is not None else 1_300_000.0

        for t, w in result.weights.items():
            sector = self.sector_mapping.get(t, "Unknown") if self.sector_mapping else "Unknown"
            rows.append({
                "Ticker": t,
                "Weight": float(w),
                "Sector": sector,
                "NMV": float(w) * aum,
                "Expected_Return": float(self.expected_returns.get(t, 0.0))
            })
        df = pd.DataFrame(rows).sort_values("Weight", ascending=False).reset_index(drop=True)

        return df


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
        return self.opt.prices_data.ffill().iloc[-1]

    def build_trade_sheet(self, target_weights: pd.Series) -> pd.DataFrame:
        if self.opt.live_portfolio is None or self.opt.fund_aum is None:
            raise ValueError("Live portfolio/AUM missing. Call load_portfolio_data() first.")

        live = self.opt.live_portfolio.copy()
        live["Ticker"] = live["Ticker"].astype(str)
        latest_prices = self._latest_prices()

        # Current dollars (fallback if 'current value' missing)
        if "current value" not in live.columns or live["current value"].isna().any():
            live["price"] = live["Ticker"].map(latest_prices)
            live["current value"] = live["shares"] * live["price"]

        # Build current series
        cur_dollars = live.set_index("Ticker")["current value"].reindex(target_weights.index).fillna(0.0)
        cur_shares  = live.set_index("Ticker")["shares"].reindex(target_weights.index).fillna(0.0)

        # Target dollars (respect cash buffer)
        spendable_aum = float(self.opt.fund_aum) * (1.0 - float(self.cfg.cash_buffer))
        tgt_dollars = target_weights.fillna(0.0) * spendable_aum

        # Prices
        px = latest_prices.reindex(target_weights.index)
        missing_price = px.isna()
        if missing_price.any():
            tgt_dollars.loc[missing_price] = cur_dollars.loc[missing_price]  # no change
            px = px.fillna(0.0)

        # Dollar & share deltas
        dollar_delta = tgt_dollars - cur_dollars
        raw_shares_delta = dollar_delta / px.replace(0.0, np.nan)

        lot = max(1, int(self.cfg.lot_size))
        shares_delta = raw_shares_delta.apply(
            lambda x: 0 if np.isnan(x) else (np.floor(x/lot)*lot if x < 0 else np.ceil(x/lot)*lot)
        ).astype(int)

        trade_value = shares_delta * px
        action = np.where(trade_value > 0, "BUY", np.where(trade_value < 0, "SELL", "HOLD"))

        sheet = pd.DataFrame({
            "Ticker": target_weights.index,
            "Action": action,
            "Shares": shares_delta,
            "Price": px.round(4),
            "TradeValue": trade_value.round(2),
            "Cur$": cur_dollars.round(2),
            "Tgt$": tgt_dollars.round(2),
            "Delta$": dollar_delta.round(2),
        }).query("Shares != 0")

        if self.cfg.min_trade_value > 0:
            sheet = sheet.loc[sheet["TradeValue"].abs() >= self.cfg.min_trade_value]

        if getattr(self.opt, "sector_mapping", None):
            sheet["Sector"] = sheet["Ticker"].map(self.opt.sector_mapping)

        sheet = sheet.sort_values("TradeValue", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

        rounded_spend = sheet["TradeValue"].sum()
        residual = (tgt_dollars.sum() - cur_dollars.sum()) - rounded_spend

        self._print_trade_sheet(sheet, residual)
        return sheet

    def _print_trade_sheet(self, sheet: pd.DataFrame, residual_cash: float):
        cols = ["Ticker", "Action", "Shares", "Price", "TradeValue"]
        extra = [c for c in ["Sector"] if c in sheet.columns]
        view = sheet[cols + extra]
        print("\n=== TRADE SHEET (ordered by $ volume) ===")
        with pd.option_context("display.max_rows", None, "display.float_format", "{:,.2f}".format):
            print(view.to_string(index=False))
        print(f"\nResidual cash after rounding: ${residual_cash:,.2f}")
