import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import time
import dash
import os
from dash import dcc, html
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

from polygon import RESTClient
from dateutil.relativedelta import relativedelta

from companies import companies, sector_designations
from data_get import (
    load_clean_holdings, load_clean_sector_allocations,
    fetch_price_data, build_live_portfolio, 
    fetch_RUT_data
)
from dashboard_utils import (
    compute_daily_pct_change, assign_color,
    create_treemap, create_sparkline,
    create_holdings_table, compute_cumulative_returns,
    create_sector_donut, compute_var_time_series, create_var_histogram,
    create_monte_carlo_var_hist_5d 
)


"""
Fetch Data 
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
filepath = os.path.join(BASE_DIR, "BUFC_May_2025_Allocations.xlsx")

holdings = load_clean_holdings(filepath)
sector_allocations = load_clean_sector_allocations(filepath)

API_KEY = "apRxKKpQoM2_K8sPhJ5a0IFvs7C0tGs1" 
client = RESTClient(API_KEY)

prices_data = fetch_price_data(companies, client)
rut_series = fetch_RUT_data(client)

print("___________ Fetched Data ___________")

"""
Build portfolio Snapshot
"""
live_portfolio = build_live_portfolio(holdings, prices_data)
ticker_shares = list(zip(live_portfolio['Ticker'], live_portfolio['shares']))

latest_prices = prices_data.ffill().iloc[-1]
mask = live_portfolio['Ticker'].isin(latest_prices.index)

live_portfolio.loc[mask, 'current value'] = (
    live_portfolio.loc[mask, 'shares'] * live_portfolio.loc[mask, 'Ticker'].map(latest_prices)
)
live_portfolio['weights'] = live_portfolio['current value'] / live_portfolio['current value'].sum()

# Percent change map
pct_change_map = compute_daily_pct_change(prices_data)
live_portfolio['pct_change'] = live_portfolio['Ticker'].map(pct_change_map)
live_portfolio['color'] = live_portfolio['pct_change'].apply(assign_color)

# Cumulative returns (portfolio vs IWM)
portfolio_cum, benchmark_cum = compute_cumulative_returns(live_portfolio, prices_data, rut_series)

# Create data for sector donut 
sector_df = sector_allocations.copy()
if 'Value' not in sector_df.columns and '% of Fund' in sector_df.columns:
    sector_df = sector_df.rename(columns={'% of Fund': 'Value'})

if 'Sector' in sector_df.columns:
    sector_df = sector_df[sector_df['Sector'].str.lower() != 'total']
var_5d, _ = compute_var_time_series(live_portfolio, prices_data, days=5, window=60)

var_timeseries_fig = go.Figure()
var_timeseries_fig.add_trace(go.Scatter(
    x=var_5d.index, y=var_5d.values, mode='lines', name='5-Day VaR', line=dict(color='red')
))
var_timeseries_fig.update_layout(
    title="Rolling 5-Day VaR (95% Confidence)",
    xaxis_title="Date", yaxis_title="VaR ($)",
    hovermode="x unified"
)
var_fig = create_var_histogram(var_5d, bins=25, confidence=0.95, days=5)
if not var_5d.empty:
    print(f"The portfolio's latest 5-day Value at Risk (95% confidence) is ${-var_5d.iloc[-1]:,.2f}")
else:
    print("Not enough data to compute rolling VaR.")


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
        ]),

        dcc.Tab(label='VaR', children=[
            html.H1('Value at Risk (VaR)', style={'text-align': 'center', 'margin-top': '20px'}),

            html.Div(
                dcc.Graph(figure=var_fig),
                style={'padding': '20px'}
            ),
            html.Div(
                dcc.Graph(figure=var_timeseries_fig),
                style={'padding': '20px'}
            ),
            html.Div(
                dcc.Graph(figure=create_monte_carlo_var_hist_5d(live_portfolio, prices_data)),
                style={'padding': '20px'}
            )
        ])
    ])
])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True, use_reloader=False, dev_tools_hot_reload=False)
