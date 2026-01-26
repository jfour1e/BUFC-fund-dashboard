import os
import pandas as pd
import numpy as np
from datetime import datetime

import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

from polygon import RESTClient
from API_KEY import POLYGON_API_KEY
from companies import HOLDINGS_INFO, RUSSELL_SECTOR_WEIGHTS

from portfolio_data_manager import (
    read_wide_csv,
    ensure_prices,
    build_live_portfolio_dashboard,
)
from dashboard_utils import (
    compute_daily_pct_change, assign_color,
    create_treemap, create_sparkline,
    create_holdings_table, compute_cumulative_returns,
    create_sector_donut,
    sector_df_from_live_portfolio, log_step
)

"""
Fetch Data
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PORTFOLIO_SNAPSHOT_CSV = os.path.join(DATA_DIR, "portfolio_snapshot.csv")
DAILY_PRICES_CSV      = os.path.join(DATA_DIR, "daily_prices.csv")
COST_BASIS_CSV        = os.path.join(DATA_DIR, "cost_basis_snapshot.csv")

client = RESTClient(POLYGON_API_KEY)

# Load snapshot (wide shares)
log_step(f"Reading snapshot: {PORTFOLIO_SNAPSHOT_CSV}")
snapshot = read_wide_csv(PORTFOLIO_SNAPSHOT_CSV)
log_step(f"Snapshot dates: {snapshot.index.min().date()} -> {snapshot.index.max().date()} | cols={len(snapshot.columns)}")

if snapshot.empty:
    raise ValueError("portfolio_snapshot.csv is empty/missing.")


"""
Build portfolio Snapshot
"""
tickers_needed = sorted((set(snapshot.columns) | set(HOLDINGS_INFO.keys()) | {"IWM"}) - {"CASH"})

log_step(f"Tickers needed (incl IWM): {len(tickers_needed)}")

# Ensure price history exists / up-to-date (business days)
ensure_prices(
    client=client,
    daily_prices_csv=DAILY_PRICES_CSV,
    tickers=tickers_needed,
    start_date="2025-01-02",
    freq="B",
)
log_step("Prices ensured")

prices_data = read_wide_csv(DAILY_PRICES_CSV)
log_step(f"Prices range: {prices_data.index.min().date()} -> {prices_data.index.max().date()} | cols={len(prices_data.columns)}")

# Build live_portfolio in exact schema required by dashboard
live_portfolio = build_live_portfolio_dashboard(
    portfolio_snapshot_csv=PORTFOLIO_SNAPSHOT_CSV,
    daily_prices_csv=DAILY_PRICES_CSV,
    cost_basis_csv=COST_BASIS_CSV,
    cash_ticker="CASH",
)
log_step(f"Live portfolio rows: {len(live_portfolio)} | total weight={live_portfolio['weights'].sum():.4f}")

# Percent changes + colors
pct_change_map = compute_daily_pct_change(prices_data)
live_portfolio["pct_change"] = live_portfolio["Ticker"].map(pct_change_map).fillna(0.0)
live_portfolio["color"] = live_portfolio["pct_change"].apply(assign_color)

# Benchmark series (IWM)
rut_series = prices_data["IWM"].ffill() if "IWM" in prices_data.columns else pd.Series(dtype=float)

# Cumulative returns (portfolio vs IWM)
portfolio_cum, benchmark_cum = compute_cumulative_returns(live_portfolio, prices_data, rut_series)

# Sector donut data: compute from live_portfolio + HOLDINGS_INFO
sector_df = sector_df_from_live_portfolio(live_portfolio, HOLDINGS_INFO)

print("___________ Fetched Data ___________")

"""
Create Dash app 
"""
app = dash.Dash(__name__)
app.title = "BUFC Fund Dashboard"

app.layout = html.Div([
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Holdings', children=[
            html.H1('Holdings', style={'text-align': 'center', 'margin-top': '20px'}),

            html.Div(
                dcc.Graph(figure=create_treemap(live_portfolio)),
                style={'width': '100%', 'display': 'inline-block'}
            ),

            html.Div(
                create_holdings_table(live_portfolio, prices_data),
                style={'padding': '20px'}
            )
        ]),

        dcc.Tab(label='Performance', children=[
            html.H1('Performance vs Benchmark', style={'text-align': 'center', 'margin-top': '20px'}),

            html.Div(
                dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Scatter(
                                x=portfolio_cum.index,
                                y=portfolio_cum.values,
                                mode='lines',
                                name='Portfolio'
                            ),
                            go.Scatter(
                                x=benchmark_cum.index,
                                y=benchmark_cum.values,
                                mode='lines',
                                name='IWM (Russell 2000)'
                            )
                        ],
                        layout=go.Layout(
                            title='Portfolio vs IWM Cumulative Return',
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Cumulative Return'),
                            legend=dict(x=0, y=1),
                            margin=dict(l=40, r=20, t=40, b=40),
                            hovermode='x unified'
                        )
                    )
                ),
                style={'padding': '20px'}
            ),

            html.Div(
                dcc.Graph(figure=create_sector_donut(sector_df)),
                style={'width': '50%', 'margin': 'auto', 'padding': '20px'}
            )
        ])
    ])
])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True, use_reloader=False, dev_tools_hot_reload=False)
