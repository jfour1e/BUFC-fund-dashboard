import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import time
import dash
from dash import dcc, html
import plotly.graph_objects as go

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
    create_sector_donut
)

#fetch data 
filepath = "../BUFC_May_2025_Allocations.xlsx"

holdings = load_clean_holdings(filepath)
sector_allocations = load_clean_sector_allocations(filepath)

API_KEY = "FnlGiHgIUqRipoOra1mzJQpYJTrMUqTS" 
client = RESTClient(API_KEY)

prices_data = fetch_price_data(companies, client)
rut_series = fetch_RUT_data(companies, client)

current_prices = prices_data.iloc[-1].to_dict()

live_portfolio = build_live_portfolio(holdings, prices_data)
live_portfolio['current value'] = live_portfolio['Ticker'].map(current_prices) * live_portfolio['shares']
live_portfolio['weights'] = live_portfolio['current value'] / live_portfolio['current value'].sum()

print("___________ Fetched Data ___________")

#create dash app 
app = dash.Dash(__name__)
app.title = "BUFC Fund Dashboard"

pct_change_map = compute_daily_pct_change(prices_data)

# Map pct change into a “color” column in live_portfolio
live_portfolio['pct_change'] = live_portfolio['Ticker'].map(pct_change_map)
live_portfolio['color'] = live_portfolio['pct_change'].apply(assign_color)

portfolio_cum, benchmark_cum = compute_cumulative_returns(live_portfolio, prices_data, rut_series)

# assemble the dash app

app.layout = html.Div([
    dcc.Tabs(id='tabs', children=[
        # --------------------------------------------------------------------------
        # Tab 1: “Holdings”
        # --------------------------------------------------------------------------
        dcc.Tab(label='Holdings', children=[
            html.H1('Holdings', style={'text-align': 'center', 'margin-top': '20px'}),

            # 7a) Treemap / Mosaic at top
            html.Div(
                dcc.Graph(figure=create_treemap(live_portfolio)),
                style={'width': '100%', 'display': 'inline-block'}
            ),

            # 7b) Portfolio holdings table (with embedded sparklines)
            html.Div(
                create_holdings_table(live_portfolio, prices_data),
                style={'padding': '20px'}
            )
        ]),

        # --------------------------------------------------------------------------
        # Tab 2: “Performance”
        # --------------------------------------------------------------------------
        dcc.Tab(label='Performance', children=[
            html.H1('Performance vs Benchmark', style={'text-align': 'center', 'margin-top': '20px'}),

            # 7c) Cumulative returns figure
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

            # 7d) Sector allocation “donut” chart
            html.Div(
                dcc.Graph(figure=create_sector_donut(sector_allocations)),
                style={'width': '50%', 'margin': 'auto', 'padding': '20px'}
            )
        ])
    ])
])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)