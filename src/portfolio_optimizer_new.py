# src/portfolio_optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
import mosek.fusion as mf

# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class OptimizerBands:
    # Sleeve bands (fractions of portfolio)
    etf_min: float = 0.30
    etf_max: float = 0.70
    stock_min: float = 0.30
    stock_max: float = 0.70

    # Position caps
    max_weight_etf: float = 0.30
    max_weight_stock: float = 0.12

    # Soft benchmark penalty strength
    # Larger => closer to Russell sector weights (for representable sectors)
    sector_penalty_gamma: float = 5.0


@dataclass(frozen=True)
class OptimizerRunConfig:
    te_min: float = 0.00          # annual tracking error
    te_max: float = 0.12          # annual tracking error
    te_points: int = 50           # sweep points
    lookback_days: int = 252      # covariance window
    annualization: int = 252      # daily->annual
    benchmark_ticker: str = "IWM" # only used for plotting/perf later (not required here)

    # If you want to exclude illiquid/irrelevant columns from prices:
    drop_if_all_nan: bool = True

def _read_wide_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df

def compute_alpha_vector(
    *,
    tickers: list[str],
    holdings_info: dict[str, dict[str, Any]],
) -> np.ndarray:
    """
    Alpha convention:
      - ETFs (asset_type == 'index'): alpha = -expense_ratio (if present) else 0
      - Stocks: alpha = expected_return(stock) - expected_return(sector_etf_proxy)

    This makes stocks an 'excess return vs sector ETF' and ETFs roughly neutral
    (or slightly negative due to fees if you store expense_ratio).
    """
    # sector -> ETF expected return
    etf_er_by_sector: dict[str, float] = {}
    for t, meta in holdings_info.items():
        if meta.get("asset_type") == "index":
            sec = meta.get("sector")
            if sec is not None:
                etf_er_by_sector[sec] = float(meta.get("expected_return", 0.0))

    alpha = np.zeros(len(tickers), dtype=float)
    for i, t in enumerate(tickers):
        meta = holdings_info.get(t, {})
        atype = meta.get("asset_type")
        if atype == "index":
            alpha[i] = -float(meta.get("expense_ratio", 0.0))
        else:
            sec = meta.get("sector")
            er_stock = float(meta.get("expected_return", 0.0))
            er_etf = float(etf_er_by_sector.get(sec, 0.0))
            alpha[i] = er_stock - er_etf
    return alpha

def _build_benchmark_weights(
    *,
    tickers: list[str],
    holdings_info: dict[str, dict[str, Any]],
    russell_sector_weights: dict[str, float],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Returns:
      w_b (N,) benchmark weights in asset space (ETF proxies only)
      sectors_used: list[str]
      b_sec (m,) benchmark sector weights (renormalized over representable sectors)

    Sector -> ETF proxy is inferred from HOLDINGS_INFO:
      any ticker with asset_type == 'index' is considered the ETF proxy for its sector.
      If multiple exist for a sector, we pick the first in sorted ticker order (stable).
    """
    # infer sector -> ETF proxy from HOLDINGS_INFO
    sector_to_etf: dict[str, str] = {}
    for t in sorted(holdings_info.keys()):
        meta = holdings_info[t]
        if meta.get("asset_type") == "index":
            sec = meta.get("sector")
            if sec and sec not in sector_to_etf:
                sector_to_etf[sec] = t

    rsw = pd.Series(russell_sector_weights, dtype=float)

    # representable sectors = those with an ETF proxy in universe
    sectors_used = [s for s in rsw.index if s in sector_to_etf]
    if not sectors_used:
        raise ValueError("No representable sectors found (no ETF proxies in HOLDINGS_INFO).")

    # renormalize Russell weights over representable sectors only
    b_sec = rsw.loc[sectors_used].copy()
    b_sec = b_sec / b_sec.sum()

    # asset-space benchmark weights: assign each sector weight to its ETF proxy
    w_b = pd.Series(0.0, index=tickers, dtype=float)
    for sec, w in b_sec.items():
        etf = sector_to_etf[sec]
        if etf in w_b.index:
            w_b.loc[etf] += float(w)

    if w_b.sum() <= 1e-12:
        raise ValueError("Benchmark weights mapped to ETFs sum to ~0. Check HOLDINGS_INFO ETF sectors.")
    w_b = (w_b / w_b.sum()).to_numpy(dtype=float)

    return w_b, sectors_used, b_sec.to_numpy(dtype=float)


def _build_sector_exposure_matrix(
    *,
    tickers: list[str],
    holdings_info: dict[str, dict[str, Any]],
    sectors_used: list[str],
) -> np.ndarray:
    """
    S is (m x N), where (S w)_k = total portfolio weight in sector k
    (summing weights of stocks+ETF in that sector).
    """
    m = len(sectors_used)
    n = len(tickers)
    sec_index = {s: k for k, s in enumerate(sectors_used)}

    S = np.zeros((m, n), dtype=float)
    for j, t in enumerate(tickers):
        sec = holdings_info.get(t, {}).get("sector")
        if sec in sec_index:
            S[sec_index[sec], j] = 1.0
    return S


# -----------------------------
# Main optimizer
# -----------------------------
def optimize_quadratic_portfolio(
    *,
    daily_prices_csv: str | Path,
    holdings_info: dict[str, dict[str, Any]],
    russell_sector_weights: dict[str, float],
    te_cap: float,
    bands: OptimizerBands,
    benchmark_ticker: str = "IWM",
    lookback_days: int = 252,
) -> dict[str, Any] | None:
    """
    Solve:
      max alpha^T w  - gamma * || S w - b ||^2
      s.t.
        sum w = 1
        w >= 0
        per-asset caps
        ETF/Stock sleeve bands
        TE constraint: sqrt((w-wb)^T Sigma (w-wb)) <= te_cap

    Notes:
      - We do NOT force hard sector bands.
      - We ONLY penalize sector deviations for sectors representable by ETF proxies.
      - Russell weights are renormalized over representable sectors.
      - benchmark_ticker is not used inside the solve (TE is vs ETF-mapped benchmark),
        but you can keep it for compatibility / future extension.
    """
    if mf is None:
        raise RuntimeError("MOSEK Fusion not available. Install/enable MOSEK Python package.")

    # Universe = HOLDINGS_INFO only
    tickers = sorted(holdings_info.keys())
    n = len(tickers)
    if n == 0:
        raise ValueError("holdings_info is empty.")

    # Load prices and build returns (use last lookback_days of trading rows)
    prices = _read_wide_csv(daily_prices_csv)
    missing_cols = [t for t in tickers if t not in prices.columns]
    if missing_cols:
        raise ValueError(f"Missing price columns for universe tickers: {missing_cols}")

    px = prices[tickers].ffill()
    rets = px.pct_change().dropna()
    if len(rets) < max(60, min(lookback_days, 252)):
        raise ValueError(f"Not enough return history ({len(rets)} rows).")

    rets = rets.iloc[-lookback_days:] if len(rets) > lookback_days else rets
    if rets.isna().any().any():
        # Should not happen after your checks, but be strict.
        raise ValueError("NaNs present in returns. Fix price history.")

    # Covariance (annualized)
    Sigma = LedoitWolf().fit(rets.to_numpy(dtype=float)).covariance_ * 252.0

    # Alpha (objective linear term)
    alpha = compute_alpha_vector(tickers=tickers, holdings_info=holdings_info)

    # Benchmark weights (asset space), and sector penalty target b_sec
    w_b, sectors_used, b_sec = _build_benchmark_weights(
        tickers=tickers,
        holdings_info=holdings_info,
        russell_sector_weights=russell_sector_weights,
    )

    # Sector exposure matrix
    S = _build_sector_exposure_matrix(
        tickers=tickers,
        holdings_info=holdings_info,
        sectors_used=sectors_used,
    )
    m = len(sectors_used)

    # Identify groups
    is_etf = np.array([holdings_info[t].get("asset_type") == "index" for t in tickers], dtype=float)
    is_stock = 1.0 - is_etf

    # Upper bounds per asset
    ub = np.ones(n, dtype=float)
    for i, t in enumerate(tickers):
        if holdings_info[t].get("asset_type") == "index":
            ub[i] = float(bands.max_weight_etf)
        else:
            ub[i] = float(bands.max_weight_stock)

    # --- MOSEK model ---
    M = mf.Model("opt_quadratic_portfolio")
    w = M.variable("w", n, mf.Domain.inRange(0.0, ub.tolist()))

    # Budget
    M.constraint("budget", mf.Expr.sum(w), mf.Domain.equalsTo(1.0))

    # ETF / stock sleeve bands
    if is_etf.sum() > 0:
        M.constraint("etf_min", mf.Expr.dot(is_etf.tolist(), w), mf.Domain.greaterThan(bands.etf_min))
        M.constraint("etf_max", mf.Expr.dot(is_etf.tolist(), w), mf.Domain.lessThan(bands.etf_max))
    if is_stock.sum() > 0:
        M.constraint("stock_min", mf.Expr.dot(is_stock.tolist(), w), mf.Domain.greaterThan(bands.stock_min))
        M.constraint("stock_max", mf.Expr.dot(is_stock.tolist(), w), mf.Domain.lessThan(bands.stock_max))

    # --- Tracking error constraint ---
    # || L (w - w_b) ||_2 <= te_cap, where Sigma = L L^T
    ridge = 1e-10
    Sigma2 = Sigma + ridge * np.eye(n)

    try:
        L = np.linalg.cholesky(Sigma2)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma2)
        vals = np.clip(vals, 0.0, None)
        L = vecs @ np.diag(np.sqrt(vals))

    a = mf.Expr.sub(w, w_b.tolist())                     # active weights
    y = mf.Expr.mul(mf.Matrix.dense(L), a)               # y = L a
    M.constraint("te_cone", mf.Expr.vstack(float(te_cap), y), mf.Domain.inQCone())

    # --- Soft sector deviation penalty ---
    # z = S w - b_sec
    # t >= ||z||^2   enforced via rotated QCone: [t, 1, z] in RQCone
    # objective: max alpha^T w - gamma * t
    z = mf.Expr.sub(mf.Expr.mul(mf.Matrix.dense(S), w), b_sec.tolist())
    t = M.variable("sector_dev_sq", 1, mf.Domain.greaterThan(0.0))
    M.constraint("sector_rqcone", mf.Expr.vstack(t, 1.0, z), mf.Domain.inRotatedQCone())

    gamma = float(bands.sector_penalty_gamma)
    obj = mf.Expr.sub(mf.Expr.dot(alpha.tolist(), w), mf.Expr.mul(gamma, t))
    M.objective("obj", mf.ObjectiveSense.Maximize, obj)

    # Solve
    try:
        M.solve()
        ps = str(M.getPrimalSolutionStatus()).lower()
        if ps != "optimal":
            M.dispose()
            return None
    except Exception:
        M.dispose()
        return None

    w_opt = np.array(w.level(), dtype=float)

    # Post-metrics
    active = w_opt - w_b
    te = float(np.sqrt(active @ Sigma @ active))
    active_ret = float(alpha @ w_opt)
    ir = active_ret / te if te > 1e-12 else np.nan

    # Sector deviation report
    sector_w = S @ w_opt
    sec_dev = sector_w - b_sec

    M.dispose()

    return {
        "tickers": tickers,
        "weights": dict(zip(tickers, w_opt)),
        "te": te,
        "active_return": active_ret,
        "ir": ir,
        "sectors_used": sectors_used,
        "benchmark_sector_weights": dict(zip(sectors_used, b_sec)),
        "portfolio_sector_weights": dict(zip(sectors_used, sector_w)),
        "sector_deviation": dict(zip(sectors_used, sec_dev)),
        "alpha": dict(zip(tickers, alpha)),
    }



# -----------------------------
# Public API
# -----------------------------
# def run_te_sweep_and_select_best_ir(
#     *,
#     daily_prices_csv: str | Path,
#     holdings_info: Dict[str, dict],
#     russell_sector_weights: Dict[str, float],
#     investable_universe: Optional[List[str]] = None,
#     bands: OptimizerBands = OptimizerBands(),
#     run_cfg: OptimizerRunConfig = OptimizerRunConfig(),
#     cash_ticker: str = "CASH",
#     save_plot: bool = True,
# ) -> Dict:
#     """
#     Returns:
#       {
#         "best": {...},
#         "frontier": DataFrame(te_cap, te, active_return, ir),
#         "tickers": [...],
#         "plot_path": Path | None
#       }
#     """
#     root = project_root()

#     prices = read_prices_wide(daily_prices_csv)

#     # Determine tickers for optimization:
#     # - use investable_universe if provided, else HOLDINGS_INFO keys
#     base = set(investable_universe) if investable_universe else set(holdings_info.keys())
#     base.discard("CASH")  # cash handled explicitly

#     # --- Determine tickers for optimization: ONLY HOLDINGS_INFO keys (+ CASH) ---
#     universe = sorted(set(holdings_info.keys()))
#     if cash_ticker not in universe:
#         universe.append(cash_ticker)

#     # We need prices for all non-cash assets
#     prices = read_prices_wide(daily_prices_csv).sort_index()
#     prices = prices.ffill()

#     missing_cols = [t for t in universe if t != cash_ticker and t not in prices.columns]
#     if missing_cols:
#         raise ValueError(
#             f"daily_prices is missing columns for these HOLDINGS_INFO tickers: {missing_cols}\n"
#             "Fix: ensure_prices(...) for these tickers, or remove them from HOLDINGS_INFO."
#         )

#     tickers = universe

#     # Build returns matrix for covariance
#     # Use lookback window ending at latest available
#     prices_sub = prices[[t for t in tickers if t != cash_ticker]].copy()

#     if run_cfg.drop_if_all_nan:
#         prices_sub = prices_sub.dropna(axis=1, how="all")

#     end_dt = prices_sub.index.max()
#     start_dt = end_dt - pd.Timedelta(days=int(run_cfg.lookback_days * 1.5))
#     prices_sub = prices_sub.loc[prices_sub.index >= start_dt].copy()

#     returns = compute_daily_returns(prices_sub).dropna(how="all")

#     # Align columns to tickers (excluding CASH)
#     risk_tickers = list(prices_sub.columns)

#     # LW covariance on risk assets
#     Sigma_risk = ledoit_wolf_cov(returns[risk_tickers], annualization=run_cfg.annualization)

#     # Expand Sigma to include CASH with zero variance/cov
#     n = len(tickers)
#     Sigma = np.zeros((n, n), dtype=float)
#     idx_map = {t: i for i, t in enumerate(tickers)}
#     for i, ti in enumerate(risk_tickers):
#         for j, tj in enumerate(risk_tickers):
#             Sigma[idx_map[ti], idx_map[tj]] = Sigma_risk[i, j]
#     # CASH row/col stays 0

#     # alpha + benchmark vector
#     alpha, w_b, _, _ = compute_alpha_vector(
#         tickers=tickers,
#         holdings_info=holdings_info,
#         russell_sector_weights=russell_sector_weights,
#         cash_ticker=cash_ticker,
#     )

#     # TE grid
#     te_caps = np.linspace(run_cfg.te_min, run_cfg.te_max, run_cfg.te_points)

#     rows = []
#     solutions = []

#     for te_cap in te_caps:
#         sol = solve_max_alpha_under_te(
#             tickers=tickers,
#             alpha=alpha,
#             Sigma=Sigma,
#             w_b=w_b,
#             holdings_info=holdings_info,
#             russell_sector_weights=russell_sector_weights,
#             sector_band=bands.sector_band,
#             te_cap=float(te_cap),
#             bands=bands,
#             cash_ticker=cash_ticker,
#         )
#         if sol is None:
#             rows.append({"te_cap": te_cap, "te": np.nan, "active_return": np.nan, "ir": np.nan})
#             solutions.append(None)
#             continue

#         rows.append({"te_cap": te_cap, "te": sol["te"], "active_return": sol["active_return"], "ir": sol["ir"]})
#         solutions.append(sol)

#     frontier = pd.DataFrame(rows)

#     # choose best IR among feasible solutions
#     best_idx = frontier["ir"].replace([np.inf, -np.inf], np.nan).idxmax()
#     best = solutions[int(best_idx)] if pd.notna(best_idx) else None

#     plot_path = None
#     if save_plot:
#         plot_path = root / "optimizer_frontier.png"
#         _plot_frontier(frontier, plot_path)

#     return {
#         "best": best,
#         "frontier": frontier,
#         "tickers": tickers,
#         "plot_path": plot_path,
#     }


def _plot_frontier(frontier: pd.DataFrame, path: Path) -> None:
    df = frontier.dropna(subset=["te", "active_return", "ir"]).copy()
    if df.empty:
        return

    # 1) alpha vs TE
    plt.figure()
    plt.plot(df["te"], df["active_return"])
    plt.xlabel("Tracking Error (annual)")
    plt.ylabel("Expected Active Return (annual)")
    plt.title("Active Frontier: Max Alpha subject to TE cap")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # Optional second plot (IR vs TE) saved alongside
    path2 = path.with_name(path.stem + "_ir.png")
    plt.figure()
    plt.plot(df["te"], df["ir"])
    plt.xlabel("Tracking Error (annual)")
    plt.ylabel("Information Ratio (alpha / TE)")
    plt.title("Information Ratio vs Tracking Error")
    plt.tight_layout()
    plt.savefig(path2)
    plt.close()


def weights_to_target_shares(
    *,
    weights: dict[str, float],
    portfolio_value: float,
    latest_prices: pd.Series,
    cash_ticker: str = "CASH",
    price_floor: float = 1e-12,
) -> dict[str, float]:
    """
    Convert portfolio weights -> target shares.

    - For CASH: shares = dollars (since we treat cash as $1.00 per unit)
    - For others: shares = (weight * portfolio_value) / price

    Args:
        weights: dict[ticker] = weight (sums to ~1)
        portfolio_value: total portfolio dollars
        latest_prices: pd.Series indexed by ticker (latest close)
        cash_ticker: ticker used for cash in your snapshot
        price_floor: guard against zero/near-zero prices

    Returns:
        dict[ticker] = target shares (float)
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be > 0")

    tgt: dict[str, float] = {}
    for t, w in weights.items():
        w = float(w)
        dollars = w * float(portfolio_value)

        if t == cash_ticker:
            # Treat cash as $1.00 per "share"
            tgt[t] = dollars
            continue

        if t not in latest_prices.index or pd.isna(latest_prices[t]):
            raise KeyError(f"Missing latest price for {t}. Ensure daily_prices has this ticker and is up to date.")

        px = float(latest_prices[t])
        if px <= price_floor:
            raise ValueError(f"Non-positive price for {t}: {px}")

        tgt[t] = dollars / px

    return tgt
