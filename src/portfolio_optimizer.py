import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import warnings
import os
from dataclasses import dataclass


# Import existing modules
from .data_get import (
    load_clean_holdings, load_clean_sector_allocations,
    fetch_price_data, build_live_portfolio, fetch_RUT_data
)

from .companies import (
    companies, sector_designations, 
    ETF_TICKERS, STOCK_TICKERS, NORMALIZED_SECTOR_MAP, 
    RUSSELL_SECTOR_WEIGHTS, EXPECTED_RETURNS
)

# -----------------------------
# Helper functions
# -----------------------------

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    # Fund parameters
    min_position_size: float = 0.01  # 1%
    max_position_size: float = 0.07  # 6%
    
    # Tracking error constraints
    tracking_error_band: float = 0.20  
    max_tracking_error = 0.07
    
    # Risk model parameters - 1 year lookback
    covariance_lookback_years: int = 1
    covariance_shrinkage: bool = False 
    
    # Optimization parameters
    solver: str = "MOSEK"
    solver_tolerance: float = 1e-6
    max_iterations: int = 1000


@dataclass
class OptimizationResult:
    """Results from portfolio optimization"""
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    information_ratio: float
    tracking_error: float
    sector_deviations: pd.Series
    turnover: float
    optimization_status: str
    solver_time: float


class PortfolioOptimizer:
    """
    Main portfolio optimization class implementing information ratio maximization
    with sector constraints and tracking error management.
    
    This class integrates with existing fund_dashboard and data_get modules to:
    - Load current portfolio holdings
    - Get current portfolio weights
    - Fetch Russell 2000 (IWM) prices
    - Calculate Russell sector weights from Excel data
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config

        self.russell_sector_weights: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.expected_returns: Optional[pd.Series] = None  # set from companies.EXPECTED_RETURNS

        self.sector_mapping: Optional[Dict[str, str]] = None
        self.current_weights: Optional[pd.Series] = None
        self.prices_data: Optional[pd.DataFrame] = None
        self.holdings_data: Optional[pd.DataFrame] = None
        self.rut_prices: Optional[pd.Series] = None
        self.live_portfolio: Optional[pd.DataFrame] = None
        self.fund_aum: Optional[float] = None

        self.russell_sector_weights = pd.Series(RUSSELL_SECTOR_WEIGHTS, dtype=float)
        self.russell_sector_weights /= self.russell_sector_weights.sum()

        # Static sector map; filtered to actual universe on optimize()
        self._base_sector_map = NORMALIZED_SECTOR_MAP.copy()

        # Set expected returns from companies.py (can overwrite later if needed)
        self.expected_returns = pd.Series(EXPECTED_RETURNS, dtype=float)

        # Benchmark return for IR (can overwrite per run)
        self.benchmark_return: float = 0.08
        
    def load_portfolio_data(self, filepath: Optional[str] = None, client=None) -> None:
        """
        Load and initialize full portfolio dataset including:
        - Holdings and sector allocations
        - Live price data and Russell 2000 benchmark
        - Live portfolio values and weights (auto-calculated)
        """

        print("\n6. Loading portfolio data...")

        # === Filepaths ===
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if filepath is None:
            filepath = os.path.join(BASE_DIR, "BUFC_May_2025_Allocations.xlsx")

        # === Load core holdings & sector data ===
        self.holdings_data = load_clean_holdings(filepath)
        self.sector_allocations = load_clean_sector_allocations(filepath)
        print(f"✓ Loaded {len(self.holdings_data)} holdings")

        # === Fetch prices and benchmark (require Polygon client) ===
        if client is None:
            raise ValueError("Polygon RESTClient is required. Pass a valid client to load_portfolio_data().")
        tickers = (
            self.holdings_data.get('Ticker', pd.Series(dtype=str))
            .dropna().astype(str).unique().tolist()
        )
        try:
            self.prices_data = fetch_price_data(tickers, client)
            print(f"✓ Loaded price data for {len(self.prices_data.columns)} tickers")
        except Exception as e:
            print(f"⚠️ Warning: could not fetch price data: {e}")
            self.prices_data = pd.DataFrame()

        try:
            self.rut_prices = fetch_RUT_data(client)
            print(f"✓ Loaded Russell 2000 benchmark ({len(self.rut_prices)} records)")
        except Exception as e:
            print(f"⚠️ Warning: could not fetch Russell 2000 data: {e}")
            self.rut_prices = pd.Series(dtype=float)

        # === Build live portfolio ===
        try:
            live_portfolio = build_live_portfolio(self.holdings_data, self.prices_data)
            latest_prices = self.prices_data.ffill().iloc[-1]
            mask = live_portfolio['Ticker'].isin(latest_prices.index)

            # Calculate live values
            live_portfolio.loc[mask, 'current value'] = (
                live_portfolio.loc[mask, 'shares'] *
                live_portfolio.loc[mask, 'Ticker'].map(latest_prices)
            )

            # Compute fund AUM and weights
            total_value = live_portfolio['current value'].sum()
            live_portfolio['weights'] = live_portfolio['current value'] / total_value

            # Store results
            self.live_portfolio = live_portfolio
            self.current_weights = live_portfolio.set_index('Ticker')['weights']
            self.fund_aum = total_value

            print(f"✓ Built live portfolio | AUM: ${total_value:,.2f}")
            print(f"  Current portfolio has {len(self.current_weights)} positions")

        except Exception as e:
            raise RuntimeError(f"Error building live portfolio: {e}")
        
    def get_current_portfolio_weights(self) -> pd.Series:
        """Return current portfolio weights computed in load_portfolio_data."""
        if self.current_weights is None:
            raise ValueError("Current weights not initialized. Call load_portfolio_data() first.")
        return self.current_weights
        
    def build_covariance_matrix(self, returns_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        1-year sample covariance (252 trading days), annualized; no shrinkage.
        """
        if returns_data is None:
            if self.prices_data is None or self.prices_data.empty:
                raise ValueError("No price data available. Call load_portfolio_data() first or provide returns_data.")
            returns_data = self.prices_data

        lookback_days = 252
        recent = returns_data.tail(lookback_days)
        daily_rets = recent.pct_change().dropna()
        cov_df = daily_rets.cov() * 252  # annualize
        self.covariance_matrix = cov_df
        print(f"Built covariance matrix with {lookback_days} days of data.")
        return cov_df
    
    # ------------- Optimization (two-stage, MOSEK only) -------------

    def optimize_portfolio(self, current_weights: Optional[pd.Series] = None) -> OptimizationResult:
        """
        Stage A: Optimize stocks only (long-only, fully invested; position limits; SOFT sector bands).
        Stage B: Add ETFs only to (i) cover sector LOWER-bound deficits, (ii) minimize TE vs Russell,
                under per-ETF and total ETF caps; then scale stocks to keep sum=1.
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be set before optimization")

        # Universe alignment (only names present in both ER and Sigma)
        all_assets = pd.Index(self.expected_returns.index).intersection(self.covariance_matrix.index)
        if len(all_assets) == 0:
            raise ValueError("No common assets between expected returns and covariance matrix")

        # Sector mapping for in-universe names
        sector_map = {t: self._base_sector_map.get(t) for t in all_assets}
        # Default unmapped to IT to avoid None
        sector_map = {t: (sector_map[t] or 'Information Technology') for t in all_assets}
        self.sector_mapping = sector_map

        # Split stocks vs ETFs
        etf_set = set(ETF_TICKERS)
        stock_assets = [t for t in all_assets if t not in etf_set]
        etf_assets = [t for t in all_assets if t in etf_set]
        if len(stock_assets) == 0:
            raise ValueError("No stocks available after filtering; cannot run Stage A.")

        # Inputs
        mu_all = self.expected_returns.reindex(all_assets).astype(float)
        Sigma_all = self.covariance_matrix.reindex(index=all_assets, columns=all_assets).astype(float)
        Sigma_all = 0.5 * (Sigma_all + Sigma_all.T)
        Sigma_all.values[range(len(all_assets)), range(len(all_assets))] += 1e-12

        # Config
        br    = float(self.benchmark_return)
        tecap = float(self.config.max_tracking_error)
        band  = float(self.config.tracking_error_band)
        minp  = float(self.config.min_position_size)
        maxp  = float(self.config.max_position_size)

        # Sector weights (normalized)
        rsw = self.russell_sector_weights.copy()
        rsw /= rsw.sum()

        # Helpers
        def sector_blocks(asset_list: List[str]) -> Dict[str, List[int]]:
            blk = {}
            for i, t in enumerate(asset_list):
                s = sector_map.get(t)
                blk.setdefault(s, []).append(i)
            return blk

        def w_bench(asset_list: List[str]) -> np.ndarray:
            w_b = np.zeros(len(asset_list))
            blocks = sector_blocks(asset_list)
            for s, idxs in blocks.items():
                sw = float(rsw.get(s, 0.0))
                if sw > 0 and len(idxs) > 0:
                    w_b[idxs] = sw / len(idxs)
            return w_b

        # ===== Stage A: STOCKS ONLY, soft sector bands, no hard TE =====
        assets_s = pd.Index(stock_assets)
        n_s = len(assets_s)
        mu_s = mu_all.reindex(assets_s).values
        Sigma_s = Sigma_all.loc[assets_s, assets_s].values
        Sigma_s = 0.5 * (Sigma_s + Sigma_s.T)
        Sigma_s[np.diag_indices(n_s)] += 1e-12

        min_pos_s = 0.0 if n_s * minp > 1.0 else minp

        w_s = cp.Variable(n_s, nonneg=True)
        cons_s = [cp.sum(w_s) == 1.0, w_s <= maxp]
        if min_pos_s > 0:
            cons_s.append(w_s >= min_pos_s)

        slo_vars, shi_vars = [], []
        blocks_s = sector_blocks(list(assets_s))
        for s, idxs in blocks_s.items():
            lower = max(0.0, float(rsw.get(s, 0.0) * (1 - band)))
            upper = min(1.0, float(rsw.get(s, 0.0) * (1 + band)))
            slo, shi = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
            cons_s += [cp.sum(w_s[idxs]) + slo >= lower,
                    cp.sum(w_s[idxs]) - shi <= upper]
            slo_vars.append(slo); shi_vars.append(shi)

        lam_sec = 10.0  # keep sector violations small but not infeasible
        obj_s = cp.Maximize(mu_s @ w_s - br - lam_sec * (cp.sum(slo_vars) + cp.sum(shi_vars)))
        prob_s = cp.Problem(obj_s, cons_s)
        prob_s.solve(solver=cp.MOSEK, verbose=False)
        if prob_s.status in ("infeasible", "unbounded"):
            raise RuntimeError(f"Stage A (stocks) failed: {prob_s.status}")

        w_s_ser = pd.Series(np.maximum(w_s.value, 0.0), index=assets_s)
        stock_sector_w = (
            w_s_ser.groupby(w_s_ser.index.map(sector_map)).sum().reindex(rsw.index, fill_value=0.0)
        )

        # ===== Stage B: ETFs top-off & TE minimization (QP, MOSEK) =====
        if len(etf_assets) == 0:
            w_etf_ser = pd.Series(dtype=float)
            w_total_etf = 0.0
        else:
            # Lower bounds and deficits
            lower_bounds = (rsw * (1 - band)).clip(lower=0.0)
            deficits = (lower_bounds - stock_sector_w).clip(lower=0.0)

            assets_e = pd.Index(etf_assets)
            n_e = len(assets_e)

            # Build incidence blocks for ETFs
            blocks_e: Dict[str, List[int]] = {}
            for j, t in enumerate(assets_e):
                s = sector_map.get(t)
                blocks_e.setdefault(s, []).append(j)

            # Decision
            w_etf = cp.Variable(n_e, nonneg=True)

            # Caps
            per_cap = 0.03  # 3% per ETF
            tot_cap = 0.15  # 15% total ETFs
            base_caps = [w_etf <= per_cap, cp.sum(w_etf) <= tot_cap]

            # Deficit constraints (only sectors that actually have ETFs)
            deficit_cons = []
            for s, idxs in blocks_e.items():
                need = float(deficits.get(s, 0.0))
                if need > 1e-12 and len(idxs) > 0:
                    deficit_cons.append(cp.sum(w_etf[idxs]) >= need)

            # Build combined TE expression over all_assets
            idx_map = {t: i for i, t in enumerate(all_assets)}
            E = np.zeros((len(all_assets), n_e))
            for j, t in enumerate(assets_e):
                E[idx_map[t], j] = 1.0

            w_s_full = np.array([w_s_ser.get(t, 0.0) for t in all_assets])
            w_b_full = w_bench(list(all_assets))
            w_c = (1 - cp.sum(w_etf)) * w_s_full + E @ w_etf
            diff = w_c - w_b_full
            te2 = cp.quad_form(diff, Sigma_all.values)

            # Objective: minimize TE
            obj_e = cp.Minimize(te2)

            # --- Solve attempts (relaxation cascade) ---
            # 1) TE cap + deficits + caps
            prob_e = cp.Problem(obj_e, base_caps + deficit_cons + [te2 <= tecap**2])
            prob_e.solve(solver=cp.MOSEK, verbose=False)
            val = w_etf.value

            # 2) If infeasible, drop TE cap (keep deficits + caps)
            if (val is None) or (prob_e.status in ("infeasible", "unbounded")):
                prob_e = cp.Problem(obj_e, base_caps + deficit_cons)
                prob_e.solve(solver=cp.MOSEK, verbose=False)
                val = w_etf.value

            # 3) If still infeasible, drop deficits too (keep caps only)
            if (val is None) or (prob_e.status in ("infeasible", "unbounded")):
                prob_e = cp.Problem(obj_e, base_caps)
                prob_e.solve(solver=cp.MOSEK, verbose=False)
                val = w_etf.value

            # 4) Final guard: if solver still returned None, use zeros (no ETF fill)
            if val is None:
                w_etf_ser = pd.Series(0.0, index=assets_e)
            else:
                w_etf_ser = pd.Series(np.clip(np.asarray(val, dtype=float), 0.0, None), index=assets_e)

            w_total_etf = float(w_etf_ser.sum())

        # Scale stocks to make room for ETFs (true "top-off")
        if len(etf_assets) > 0 and w_total_etf > 0:
            w_s_ser = w_s_ser * max(0.0, 1.0 - w_total_etf)

        combined = pd.concat([w_s_ser, w_etf_ser]).astype(float)
        combined = combined[combined > 1e-10]
        combined /= combined.sum()  # safety renorm

        # Metrics
        mu_c = mu_all.reindex(combined.index).values
        Sigma_c = Sigma_all.loc[combined.index, combined.index].values
        w_c = combined.values

        w_b_c = w_bench(list(combined.index))
        te = float(np.sqrt((w_c - w_b_c).T @ Sigma_c @ (w_c - w_b_c)))
        ret = float(mu_c @ w_c)
        var = float(w_c.T @ Sigma_c @ w_c)
        vol = float(np.sqrt(var))
        ir = (ret - br) / te if te > 1e-12 else 0.0

        sector_devs = (
            combined.groupby(combined.index.map(sector_map)).sum().reindex(rsw.index, fill_value=0.0) - rsw
        )

        return OptimizationResult(
            weights=combined,
            expected_return=ret,
            expected_volatility=vol,
            information_ratio=ir,
            tracking_error=te,
            sector_deviations=sector_devs,
            turnover=0.0,
            optimization_status="optimal" if np.isfinite(ret) else "unknown",
            solver_time=float(getattr(getattr(prob_s, 'solver_stats', None), 'solve_time', 0.0))
        )

    
    def get_portfolio_summary(self, result: OptimizationResult) -> pd.DataFrame:
        rows = []
        for t, w in result.weights.items():
            sector = self.sector_mapping.get(t, 'Unknown') if self.sector_mapping else 'Unknown'
            rows.append({
                'Ticker': t,
                'Weight': w,
                'Sector': sector,
                'NMV': w * (self.fund_aum if self.fund_aum else 1_200_000.0),
                'Expected_Return': float(self.expected_returns.get(t, 0.0)) if self.expected_returns is not None else 0.0
            })
        df = pd.DataFrame(rows).sort_values('Weight', ascending=False).reset_index(drop=True)

        # Pretty sector print
        sector_totals = df.groupby('Sector')['Weight'].sum()
        print("\nSector Allocations:")
        for s, w in sector_totals.items():
            b = float(self.russell_sector_weights.get(s, 0.0)) if self.russell_sector_weights is not None else 0.0
            print(f"{s}: {w:.1%} (Benchmark: {b:.1%}, Deviation: {w-b:+.1%})")
        return df
    

@dataclass
class TradePlanConfig:
    lot_size: int = 1                 # round to whole shares (or odd lots)
    min_trade_value: float = 0.0      # e.g., 500 to suppress tiny trades
    cash_buffer: float = 0.00         # hold back cash as fraction of AUM (e.g., 0.01 = 1%)

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
        live['Ticker'] = live['Ticker'].astype(str)
        latest_prices = self._latest_prices()

        # Current dollars (fallback if 'current value' missing)
        if 'current value' not in live.columns or live['current value'].isna().any():
            live['price'] = live['Ticker'].map(latest_prices)
            live['current value'] = live['shares'] * live['price']

        # Build current series
        cur_dollars = live.set_index('Ticker')['current value'].reindex(target_weights.index).fillna(0.0)
        cur_shares  = live.set_index('Ticker')['shares'].reindex(target_weights.index).fillna(0.0)

        # Target dollars (respect cash buffer)
        spendable_aum = float(self.opt.fund_aum) * (1.0 - float(self.cfg.cash_buffer))
        tgt_dollars = target_weights.fillna(0.0) * spendable_aum

        # Prices
        px = latest_prices.reindex(target_weights.index)
        missing_price = px.isna()
        if missing_price.any():
            # Drop tickers with missing price from trade plan
            tgt_dollars.loc[missing_price] = cur_dollars.loc[missing_price]  # no change
            px = px.fillna(0.0)

        # Dollar & share deltas
        dollar_delta = tgt_dollars - cur_dollars
        raw_shares_delta = dollar_delta / px.replace(0.0, np.nan)
        # Round to lot size; handle buys vs sells with symmetric rounding
        lot = max(1, int(self.cfg.lot_size))
        shares_delta = raw_shares_delta.apply(
            lambda x: 0 if np.isnan(x) else (np.floor(x/lot)*lot if x < 0 else np.ceil(x/lot)*lot)
        ).astype(int)

        trade_value = shares_delta * px
        action = np.where(trade_value > 0, "BUY", np.where(trade_value < 0, "SELL", "HOLD"))

        # Build sheet
        sheet = pd.DataFrame({
            'Ticker': target_weights.index,
            'Action': action,
            'Shares': shares_delta,
            'Price': px.round(4),
            'TradeValue': trade_value.round(2),
            'Cur$': cur_dollars.round(2),
            'Tgt$': tgt_dollars.round(2),
            'Delta$': dollar_delta.round(2),
        }).query('Shares != 0')

        # Optional filter tiny trades
        if self.cfg.min_trade_value > 0:
            sheet = sheet.loc[sheet['TradeValue'].abs() >= self.cfg.min_trade_value]

        # Attach sector (if available)
        if getattr(self.opt, 'sector_mapping', None):
            sheet['Sector'] = sheet['Ticker'].map(self.opt.sector_mapping)

        # Sort by absolute dollar volume
        sheet = sheet.sort_values('TradeValue', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

        # Residual cash due to rounding
        rounded_spend = sheet['TradeValue'].sum()
        residual = (tgt_dollars.sum() - cur_dollars.sum()) - rounded_spend

        # Pretty print
        self._print_trade_sheet(sheet, residual)

        return sheet

    def _print_trade_sheet(self, sheet: pd.DataFrame, residual_cash: float):
        cols = ['Ticker','Action','Shares','Price','TradeValue']
        extra = [c for c in ['Sector'] if c in sheet.columns]
        view = sheet[cols + extra]
        print("\n=== TRADE SHEET (ordered by $ volume) ===")
        with pd.option_context('display.max_rows', None, 'display.float_format', '{:,.2f}'.format):
            print(view.to_string(index=False))
        print(f"\nResidual cash after rounding: ${residual_cash:,.2f}")
