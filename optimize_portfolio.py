"""
BUFC Fund Portfolio Optimization Example
=======================================

This script demonstrates how to use the portfolio optimization system
with the simplified API where fund_aum is auto-calculated and 
expected returns are annualized (not idiosyncratic).

Usage:
    python optimize_portfolio.py
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationConfig
)
from src.companies import companies, sector_designations
import pandas as pd
import numpy as np
from polygon import RESTClient

# File-wide API key (set this to your Polygon API key)
POLYGON_API_KEY = "apRxKKpQoM2_K8sPhJ5a0IFvs7C0tGs1" 

def main():
    """
    Main function to run the portfolio optimization with the new architecture:
      - EXPECTED_RETURNS loaded from companies.py inside PortfolioOptimizer.__init__
      - Russell sector weights loaded from companies.py (can be replaced later if needed)
      - No manual sector mapping creation (comes from companies.py normalization map)
      - MOSEK-only, two-stage solve (stocks → ETF top-offs)
    """
    print("BUFC Fund Portfolio Optimization")
    print("=" * 50)

    # Config & optimizer
    config = OptimizationConfig()
    optimizer = PortfolioOptimizer(config)

    # Polygon client
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

    # 2) Current weights (fund AUM computed during load)
    print("\n2. Getting current portfolio weights...")
    try:
        current_weights = optimizer.get_current_portfolio_weights()
        print(f"✓ Current portfolio has {len(current_weights)} positions")
        print(f"  Fund AUM: ${optimizer.fund_aum:,.2f}")
    except Exception as e:
        print(f"✗ Error retrieving current weights: {e}")
        return

    # 3) Covariance (1y, no shrinkage)
    print("\n3. Building covariance matrix...")
    try:
        optimizer.build_covariance_matrix()
        print("✓ Covariance matrix built")
    except Exception as e:
        print(f"✗ Error building covariance matrix: {e}")
        return

    # 4) Expected returns are already set from companies.EXPECTED_RETURNS in __init__.
    #    If you want to override with another source, assign here:
    #    optimizer.expected_returns = pd.Series(your_dict, dtype=float)

    # 5) Optimize (MOSEK, two-stage stocks→ETF top-offs)
    print("\n4. Running optimization...")
    print("-" * 50)
    try:
        result = optimizer.optimize_portfolio(current_weights=current_weights)

        print("\n✓ Optimization completed successfully!")
        print("\nOptimization Results:")
        print(f"  Expected Return: {result.expected_return:.1%}")
        print(f"  Expected Volatility: {result.expected_volatility:.1%}")
        print(f"  Information Ratio: {result.information_ratio:.3f}")
        print(f"  Tracking Error: {result.tracking_error:.1%}")
        print(f"  Status: {result.optimization_status}")

        # 6) Summary
        print("\n5. Generating portfolio summary...")
        summary = optimizer.get_portfolio_summary(result)

        # Top positions (adjust N as you like)
        print("\n" + "=" * 50)
        print("Top Holdings:")
        print("=" * 50)
        top_n = summary.head(25)
        for _, row in top_n.iterrows():
            print(f"{row['Ticker']:6} {row['Weight']:6.1%} {row['Sector']:25} ${row['NMV']:>12,.0f}")

        # Sector allocations vs Russell
        print("\n" + "=" * 50)
        print("Sector Allocations:")
        print("=" * 50)
        sector_totals = summary.groupby('Sector')['Weight'].sum()
        rsw = optimizer.russell_sector_weights  # Series from companies.py (normalized)
        for sector in rsw.index:
            actual = float(sector_totals.get(sector, 0.0))
            bench = float(rsw.get(sector, 0.0))
            dev = actual - bench
            print(f"{sector:25} {actual:6.1%} (Benchmark: {bench:6.1%}, Deviation: {dev:+6.1%})")

        # Optional: save CSV
        out_name = f"optimized_portfolio_{pd.Timestamp.now(tz='America/New_York').strftime('%Y%m%d_%H%M%S')}.csv"
        summary.to_csv(out_name, index=False)
        print(f"\n✓ Results saved to: {out_name}")

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()