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
    debug: bool = False,
    mosek_log: bool = False,
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

    Debug mode runs staged feasibility checks to pinpoint infeasibility.
    """

    def _dbg(msg: str) -> None:
        if debug:
            print(f"[OPT-DBG] {msg}")

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
        raise ValueError("NaNs present in returns. Fix price history.")

    # Covariance (annualized)
    Sigma = LedoitWolf().fit(rets.to_numpy(dtype=float)).covariance_ * 252.0
    Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize

    # Basic Sigma diagnostics
    if debug:
        evals = np.linalg.eigvalsh(Sigma)
        _dbg(f"Sigma shape={Sigma.shape}, eig_min={evals.min():.3e}, eig_max={evals.max():.3e}, finite={np.isfinite(Sigma).all()}")

    # Alpha (objective linear term)
    alpha = compute_alpha_vector(tickers=tickers, holdings_info=holdings_info)
    if debug:
        _dbg(f"Alpha min={alpha.min():.6f}, max={alpha.max():.6f}, finite={np.isfinite(alpha).all()}")

    # Benchmark weights (asset space), and sector penalty target b_sec
    w_b, sectors_used, b_sec = _build_benchmark_weights(
        tickers=tickers,
        holdings_info=holdings_info,
        russell_sector_weights=russell_sector_weights,
    )
    if debug:
        _dbg(f"w_b sum={w_b.sum():.6f}, min={w_b.min():.6f}, max={w_b.max():.6f}, len={len(w_b)}")
        _dbg(f"Representable sectors used={len(sectors_used)}; missing Russell sectors ignored automatically.")

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

    if debug:
        _dbg(f"Universe n={n}, ETFs={int(is_etf.sum())}, Stocks={int(is_stock.sum())}")
        _dbg(f"UB sum={ub.sum():.6f}, UB min={ub.min():.6f}, UB max={ub.max():.6f}")
        _dbg(f"ETF UB sum={ub[is_etf.astype(bool)].sum():.6f}, Stock UB sum={ub[is_stock.astype(bool)].sum():.6f}")
        _dbg(f"Sleeves: etf_min={bands.etf_min}, etf_max={bands.etf_max}, stock_min={bands.stock_min}, stock_max={bands.stock_max}")

        # quick feasibility sanity checks
        if ub.sum() < 1.0 - 1e-9:
            _dbg("❌ infeasible: ub.sum() < 1.0")
            return None
        if is_etf.sum() > 0:
            if ub[is_etf.astype(bool)].sum() + 1e-12 < bands.etf_min:
                _dbg("❌ infeasible: ETF ub-sum < etf_min")
                return None
        if is_stock.sum() > 0:
            if ub[is_stock.astype(bool)].sum() + 1e-12 < bands.stock_min:
                _dbg("❌ infeasible: Stock ub-sum < stock_min")
                return None

    # ---- helper to build a model with selective constraints (for debug isolation) ----
    def _solve_stage(
        stage_name: str,
        add_sleeves: bool,
        add_te: bool,
        add_sector_penalty: bool,
    ):
        M = mf.Model(stage_name)
        if mosek_log:
            import sys
            M.setLogHandler(sys.stdout)

        w = M.variable("w", n, mf.Domain.inRange(0.0, ub.tolist()))
        M.constraint("budget", mf.Expr.sum(w), mf.Domain.equalsTo(1.0))

        if add_sleeves:
            if is_etf.sum() > 0:
                M.constraint("etf_min", mf.Expr.dot(is_etf.tolist(), w), mf.Domain.greaterThan(float(bands.etf_min)))
                M.constraint("etf_max", mf.Expr.dot(is_etf.tolist(), w), mf.Domain.lessThan(float(bands.etf_max)))
            if is_stock.sum() > 0:
                M.constraint("stock_min", mf.Expr.dot(is_stock.tolist(), w), mf.Domain.greaterThan(float(bands.stock_min)))
                M.constraint("stock_max", mf.Expr.dot(is_stock.tolist(), w), mf.Domain.lessThan(float(bands.stock_max)))

        if add_te:
            # || L (w - w_b) || <= te_cap
            ridge = 1e-10
            Sigma2 = Sigma + ridge * np.eye(n)
            Sigma2 = 0.5 * (Sigma2 + Sigma2.T)

            try:
                L = np.linalg.cholesky(Sigma2)
                _dbg("Cholesky OK (Sigma2 PSD).")
            except np.linalg.LinAlgError:
                _dbg("Cholesky failed; using eig sqrt.")
                vals, vecs = np.linalg.eigh(Sigma2)
                vals = np.clip(vals, 0.0, None)
                L = vecs @ np.diag(np.sqrt(vals))

            a = mf.Expr.sub(w, w_b.tolist())
            # IMPORTANT: use .tolist() for Fusion
            y = mf.Expr.mul(mf.Matrix.dense(L.tolist()), a)
            M.constraint("te_cone", mf.Expr.vstack(float(te_cap), y), mf.Domain.inQCone())

        if add_sector_penalty:
            # Rotated cone: 2*t*1 >= ||z||^2  => t >= ||z||^2/2
            z = mf.Expr.sub(mf.Expr.mul(mf.Matrix.dense(S.tolist()), w), b_sec.tolist())
            tvar = M.variable("sector_dev_sq", 1, mf.Domain.greaterThan(0.0))
            M.constraint("sector_rqcone", mf.Expr.vstack(tvar, 1.0, z), mf.Domain.inRotatedQCone())
            gamma = float(bands.sector_penalty_gamma)
            obj = mf.Expr.sub(mf.Expr.dot(alpha.tolist(), w), mf.Expr.mul(gamma, tvar))
        else:
            obj = mf.Expr.dot(alpha.tolist(), w)

        M.objective("obj", mf.ObjectiveSense.Maximize, obj)

        try:
            M.solve()
            ps = str(M.getPrimalSolutionStatus()).lower()
            if "optimal" not in ps:
                _dbg(f"{stage_name}: primal status={ps} (not optimal)")
                M.dispose()
                return None
        except Exception as e:
            _dbg(f"{stage_name}: solve exception: {e}")
            M.dispose()
            return None

        w_opt = np.array(w.level(), dtype=float)
        M.dispose()
        return w_opt

    # ---- Debug staged feasibility isolation ----
    if debug:
        _dbg("Stage A: bounds+budget only (should ALWAYS be feasible if ub.sum>=1)")
        wA = _solve_stage("stage_A", add_sleeves=False, add_te=False, add_sector_penalty=False)
        if wA is None:
            _dbg("❌ Infeasible at Stage A => UB/budget is broken (or Fusion domain issue).")
            return None
        _dbg(f"Stage A ok. sum={wA.sum():.6f}, min={wA.min():.6f}, max={wA.max():.6f}")

        _dbg("Stage B: add sleeve constraints")
        wB = _solve_stage("stage_B", add_sleeves=True, add_te=False, add_sector_penalty=False)
        if wB is None:
            _dbg("❌ Infeasible at Stage B => sleeve constraints contradict UB/budget.")
            return None
        _dbg(f"Stage B ok. ETF={float(is_etf@wB):.4f}, STOCK={float(is_stock@wB):.4f}")

        _dbg("Stage C: add TE constraint (this is where your issue most likely is)")
        wC = _solve_stage("stage_C", add_sleeves=True, add_te=True, add_sector_penalty=False)
        if wC is None:
            _dbg("❌ Infeasible at Stage C => TE constraint modeling or te_cap too tight.")
            return None
        activeC = wC - w_b
        teC = float(np.sqrt(activeC @ Sigma @ activeC))
        _dbg(f"Stage C ok. realized TE={teC:.6f} cap={te_cap}")

        _dbg("Stage D: add sector penalty cone (should remain feasible)")
        wD = _solve_stage("stage_D", add_sleeves=True, add_te=True, add_sector_penalty=True)
        if wD is None:
            _dbg("❌ Infeasible at Stage D => sector penalty cone modeling issue (unexpected).")
            return None
        _dbg("Stage D ok. Proceeding to final solve.")

    # ---- Final solve (same as Stage D) ----
    w_opt = _solve_stage("final", add_sleeves=True, add_te=True, add_sector_penalty=True)
    if w_opt is None:
        return None

    active = w_opt - w_b
    te = float(np.sqrt(active @ Sigma @ active))
    active_ret = float(alpha @ w_opt)
    ir = active_ret / te if te > 1e-12 else np.nan

    sector_w = S @ w_opt
    sec_dev = sector_w - b_sec

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

def plot_frontier_in_notebook(frontier: pd.DataFrame) -> None:
    df = frontier.replace([np.inf, -np.inf], np.nan).dropna(subset=["te", "active_return", "ir"]).copy()
    if df.empty:
        print("No feasible points to plot.")
        return

    # Active return vs TE
    plt.figure()
    plt.plot(df["te"], df["active_return"], marker="o", linestyle="-")
    plt.xlabel("Tracking Error (annual)")
    plt.ylabel("Expected Active Return (annual)")
    plt.title("Active Frontier: objective vs TE cap")
    plt.tight_layout()
    plt.show()

    # IR vs TE
    plt.figure()
    plt.plot(df["te"], df["ir"], marker="o", linestyle="-")
    plt.xlabel("Tracking Error (annual)")
    plt.ylabel("Information Ratio (active_return / TE)")
    plt.title("Information Ratio vs Tracking Error")
    plt.tight_layout()
    plt.show()

def run_te_sweep_frontier(
    *,
    daily_prices_csv: str | Path,
    holdings_info: dict[str, dict[str, Any]],
    russell_sector_weights: dict[str, float],
    bands,
    te_min: float = 0.00,
    te_max: float = 0.12,
    te_points: int = 50,
    lookback_days: int = 252,
    debug: bool = False,
    mosek_log: bool = False,
    plot: bool = True,
) -> dict[str, Any]:
    """
    Notebook-friendly TE sweep that repeatedly calls optimize_quadratic_portfolio(...).

    Returns:
      {
        "best": dict | None,
        "frontier": DataFrame(te_cap, te, active_return, ir, status),
        "solutions": list[dict|None] aligned to te grid,
        "te_grid": np.ndarray,
        "best_index": int | None,
      }
    """
    te_grid = np.linspace(float(te_min), float(te_max), int(te_points))

    rows: list[dict[str, Any]] = []
    solutions: list[dict[str, Any] | None] = []

    for te_cap in te_grid:
        res = optimize_quadratic_portfolio(
            daily_prices_csv=daily_prices_csv,
            holdings_info=holdings_info,
            russell_sector_weights=russell_sector_weights,
            te_cap=float(te_cap),
            bands=bands,
            lookback_days=int(lookback_days),
            debug=bool(debug),
            mosek_log=bool(mosek_log),
        )

        if res is None:
            rows.append({
                "te_cap": float(te_cap),
                "te": np.nan,
                "active_return": np.nan,
                "ir": np.nan,
                "status": "infeasible_or_failed",
            })
            solutions.append(None)
        else:
            rows.append({
                "te_cap": float(te_cap),
                "te": float(res.get("te", np.nan)),
                "active_return": float(res.get("active_return", np.nan)),
                "ir": float(res.get("ir", np.nan)),
                "status": "ok",
            })
            solutions.append(res)

    frontier = pd.DataFrame(rows)

    # pick best by IR
    frontier_ok = frontier.replace([np.inf, -np.inf], np.nan).dropna(subset=["ir"]).copy()
    best = None
    best_index = None
    if not frontier_ok.empty:
        best_index = int(frontier_ok["ir"].idxmax())
        best = solutions[best_index]

    if plot:
        plot_frontier_in_notebook(frontier)

    return {
        "best": best,
        "frontier": frontier,
        "solutions": solutions,
        "te_grid": te_grid,
        "best_index": best_index,
    }