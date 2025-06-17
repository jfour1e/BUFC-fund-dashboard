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

# BUFC-inspired color scheme
BUFC_COLORS = {
    'primary': '#003366',  # DARK BLUE
    'secondary': '#FF6600',  # Orange (QUESTIONABLE)
    'background': '#F5F5F5',
    'text': '#333333'
}
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
        # Tab 1: Holdings (unchanged)
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Tab 2: Performance (updated version)
        # --------------------------------------------------------------------------
        dcc.Tab(label='Performance', children=[
            # Top Section: Metrics and Cumulative Returns
            html.Div([
                # Metrics Table (Left)
                html.Div([
                    html.H3('Performance Metrics', style={
                        'text-align': 'center',
                        'color': BUFC_COLORS['primary'],
                        'margin-bottom': '15px'
                    }),
                    create_performance_metrics_table(portfolio_cum, benchmark_cum)
                ], style={
                    'width': '30%',
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'padding': '20px',
                    'background-color': BUFC_COLORS['background'],
                    'border-radius': '10px'
                }),
                
                # Cumulative Returns Chart (Right)
                html.Div([
                    dcc.Graph(
                        figure=create_cumulative_returns_chart(portfolio_cum, benchmark_cum),
                        style={'height': '400px'}
                    )
                ], style={
                    'width': '68%',
                    'display': 'inline-block',
                    'padding': '20px'
                })
            ], style={
                'display': 'flex',
                'margin-bottom': '20px'
            }),
            
            # Bottom Section: Sector Allocation
            html.Div([
                html.H3('Sector Allocation', style={
                    'text-align': 'center',
                    'color': BUFC_COLORS['primary'],
                    'margin-bottom': '15px'
                }),
                dcc.Graph(
                    figure=create_sector_donut(sector_allocations),
                    style={'height': '400px'}
                )
            ], style={
                'width': '80%',
                'margin': '0 auto',
                'padding': '20px',
                'background-color': BUFC_COLORS['background'],
                'border-radius': '10px'
            })
        ])
    ])
])

def create_performance_metrics_table(portfolio_cum, benchmark_cum):
    # Calculate metrics
    port_returns = portfolio_cum.pct_change().dropna()
    bench_returns = benchmark_cum.pct_change().dropna()
    excess_return = (portfolio_cum.iloc[-1] - benchmark_cum.iloc[-1]) / benchmark_cum.iloc[-1]
    
    metrics = [
        ("Annualized Return", f"{portfolio_cum.iloc[-1]**(252/len(portfolio_cum))-1:.2%}"),
        ("Benchmark Return", f"{benchmark_cum.iloc[-1]**(252/len(benchmark_cum))-1:.2%}"),
        ("Excess Return", f"{excess_return:.2%}"),
        ("Volatility", f"{port_returns.std()*np.sqrt(252):.2%}"),
        ("Sharpe Ratio", f"{(port_returns.mean()/port_returns.std())*np.sqrt(252):.2f}")
    ]
    
    rows = [html.Tr([
        html.Td(html.Strong(name), style={
            'padding': '10px',
            'border-bottom': '1px solid #ddd',
            'color': BUFC_COLORS['primary']
        }),
        html.Td(value, style={
            'padding': '10px',
            'text-align': 'right',
            'border-bottom': '1px solid #ddd'
        })
    ]) for name, value in metrics]
    
    return html.Table(rows, style={
        'width': '100%',
        'border-collapse': 'collapse',
        'margin-top': '10px'
    })

def create_cumulative_returns_chart(portfolio_cum, benchmark_cum):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_cum.index,
        y=portfolio_cum.values,
        mode='lines',
        name='Portfolio',
        line=dict(color=BUFC_COLORS['primary'], width=3)
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_cum.index,
        y=benchmark_cum.values,
        mode='lines',
        name='IWM (Russell 2000)',
        line=dict(color=BUFC_COLORS['secondary'], width=3)
    ))
    fig.update_layout(
        title='Portfolio vs Benchmark Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=BUFC_COLORS['text']),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    return fig
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)