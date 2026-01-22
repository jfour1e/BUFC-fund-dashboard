"""
BUFC Fund Portfolio Optimization (One-step)
==========================================
One-step solve over stocks + ETFs:
- True Tracking Error constraint vs benchmark returns (IWM proxy via fetch_RUT_data or IWM ticker)
- Soft sector drift penalty vs Russell sector weights
- ETF sleeve bounds and per-name caps
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.portfolio_optimizer import PortfolioOptimizer, OptimizationConfig
from API_KEY import POLYGON_API_KEY
import pandas as pd
from polygon import RESTClient


def main():
    print("BUFC Fund Portfolio Optimization (One-step)")
    print("=" * 50)

    # Config & optimizer (tune these if needed)
    config = OptimizationConfig(
        max_position_size=0.10,
        min_position_size=0.00,
        etf_min_total_weight=0.00,
        etf_max_total_weight=0.40,
        etf_max_single_weight=0.12,
        max_tracking_error=0.09,
        sector_te_penalty=4.0,
        risk_aversion=0.20,
        lookback_days=252,
        benchmark_source="IWM",   # uses your fetch_RUT_data() (which returns IWM proxy series)
        benchmark_ticker="IWM",   # ignored if benchmark_source="RUT"
        solver="MOSEK",
        verbose=False,
    )
    optimizer = PortfolioOptimizer(config)

    if not POLYGON_API_KEY:
        print("✗ POLYGON_API_KEY is not set. Please export it in your environment.")
        return
    client = RESTClient(POLYGON_API_KEY)

    # 1) Load portfolio data (holdings, prices, benchmark)
    print("\n1. Loading portfolio data...")
    try:
        optimizer.load_portfolio_data(client=client)
    except Exception as e:
        print(f"✗ Error loading portfolio data: {e}")
        return

    # 2) (Optional) show current weights / AUM
    print("\n2. Getting current portfolio weights...")
    try:
        current_weights = optimizer.get_current_portfolio_weights()
        print(f"✓ Current portfolio has {len(current_weights)} positions")
        print(f"  Fund AUM: ${optimizer.fund_aum:,.2f}")
    except Exception as e:
        print(f"✗ Error retrieving current weights: {e}")
        return

    # 3) Optimize
    print("\n3. Running optimization...")
    print("-" * 50)
    try:
        result = optimizer.optimize_portfolio()

        print("\n✓ Optimization completed successfully!\n")
        print("Optimization Results:")
        print(f"  Expected Return (mu·w):       {result.expected_return:.1%}")
        print(f"  Expected Volatility (1y):     {result.expected_volatility:.1%}")
        print(f"  Benchmark Annual Return:      {result.benchmark_annual_return:.1%}")
        print(f"  Tracking Error vs benchmark:  {result.tracking_error:.1%}")
        print(f"  Information Ratio:            {result.information_ratio:.3f}")
        print(f"  Status:                       {result.optimization_status}")
        print(f"  Solve time:                   {result.solver_time:.3f}s")

        # 4) Summary table
        print("\n4. Generating portfolio summary...")
        summary = optimizer.get_portfolio_summary(
            result
        )

        # Top positions
        print("\n" + "=" * 50)
        print("Top Holdings:")
        print("=" * 50)
        top_n = summary.head(25)
        for _, row in top_n.iterrows():
            print(f"{row['Ticker']:6} {row['Weight']:6.1%} {row['Sector']:25} ${row['NMV']:>12,.0f}")

        # 5) Active risk report (new)
        print("\n" + "=" * 50)
        print("Active Risk Report:")
        print("=" * 50)
        ar = optimizer.active_return_report(result.weights)
        print(f"Active mean (ann):     {ar['ActiveMeanAnn']:.2%}")
        print(f"Tracking Error (ann):  {ar['TE']:.2%}")
        print(f"Beta to benchmark:     {ar['BetaToBenchmark']:.3f}")
        print(f"Residual TE (ann):     {ar['ResidualTE']:.2%}")

        # 6) Sector active-contribution TE (new)
        print("\n" + "=" * 50)
        print("Sector Active-Contribution TE (proxy ETFs):")
        print("=" * 50)
        sector_te = optimizer.sector_te_report(result.weights)

        if sector_te.empty:
            print("No sector TE report available (missing proxies or data).")
        else:
            tmp = sector_te.copy()
            tmp["BenchWeight"] = (tmp["BenchWeight"] * 100).round(2)
            tmp["PortWeight"] = (tmp["PortWeight"] * 100).round(2)
            tmp["ActiveContribTE"] = (tmp["ActiveContribTE"] * 100).round(2)
            tmp["VarShare"] = (tmp["VarShare"] * 100).round(1)
            print(
                tmp[["Sector", "BenchETFProxy", "BenchWeight", "PortWeight", "ActiveContribTE", "VarShare"]]
                .rename(columns={
                    "BenchETFProxy": "Proxy",
                    "BenchWeight": "BenchW(%)",
                    "PortWeight": "PortW(%)",
                    "ActiveContribTE": "TE_s(%)",
                    "VarShare": "VarShare(%)",
                })
                .to_string(index=False)
            )

        # 7) Save CSV
        out_name = f"optimized_portfolio_{pd.Timestamp.now(tz='America/New_York').strftime('%Y%m%d_%H%M%S')}.csv"
        summary.to_csv(out_name, index=False)
        print(f"\n✓ Results saved to: {out_name}")

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
