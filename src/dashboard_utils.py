import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

def compute_daily_pct_change(prices_df):
    """
    Compute the percent change between the last two trading days for each ticker.
    Return a dict: {ticker: percent_change}.
    """
    pct_change = {}
    for ticker in prices_df.columns:
        if len(prices_df[ticker]) >= 2:
            today_price = prices_df[ticker].iloc[-1]
            prev_price  = prices_df[ticker].iloc[-2]
            pct_change[ticker] = (today_price - prev_price) / prev_price * 100
        else:
            pct_change[ticker] = 0.0
    return pct_change

def assign_color(pct):
    """
    Assign a color string based on the percent change thresholds:
      - pct >=  1.5%  => “darkgreen”
      - 0    <  pct  < 1.5%  => “lightgreen”
      - pct <= -1.5% => “darkred”
      - -1.5% <  pct <  0    => “lightcoral”
      - exactly 0     => “grey”
    """
    if pct >= 1.5:
        return 'darkgreen'
    elif pct > 0:
        return 'lightgreen'
    elif pct <= -1.5:
        return 'darkred'
    elif pct < 0:
        return 'lightcoral'
    else:
        return 'grey'

def create_treemap(df):
    """
    Create a Plotly Treemap showing each position’s weight (size) and daily % change (color).
    """
    labels  = df['Ticker']
    values  = df['weights']
    colors  = df['color']
    parents = [''] * len(df)   # Single root parent for all rectangles

    fig = go.Figure(go.Treemap(
        labels       = labels,
        parents      = parents,
        values       = values,
        marker=dict(colors=colors),
        customdata   = df['pct_change'],
        hovertemplate=
            "<b>%{label}</b><br>" +
            "Weight: %{value:.2%}<br>" +
            "Daily Change: %{customdata:.2f}%<extra></extra>"
    ))
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    return fig

def create_sparkline(prices_series: pd.Series) -> go.Figure:
    """
    Create a small Plotly line chart (“sparkline”) for a given price series.
    Axes are hidden, margins are minimized, and height is small (40px).
    """
    fig = go.Figure(data=[
        go.Scatter(
            x         = prices_series.index,
            y         = prices_series.values,
            mode      = 'lines',
            line=dict(color='blue'),
            hoverinfo = 'none'
        )
    ])
    fig.update_layout(
        xaxis    = dict(visible=False),
        yaxis    = dict(visible=False),
        margin   = dict(l=0, r=0, t=2, b=2),
        height   = 40
    )
    return fig


def create_holdings_table(portfolio_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Table:
    """
    Construct an HTML <table> showing:
      - Ticker
      - Company
      - % of Portfolio
      - Shares Owned
      - Current Value
      - Cost Basis
      - Price History (YTD sparkline)
    Alternates row background for readability.
    """
    # 4a) Table Header Row
    header = html.Tr([
        html.Th('Ticker',                style={'padding': '8px', 'text-align': 'left'}),
        html.Th('Company',               style={'padding': '8px', 'text-align': 'left'}),
        html.Th('% of Portfolio',        style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Shares Owned',          style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Current Value',         style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Cost Basis',            style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Price History (YTD)',   style={'padding': '8px', 'text-align': 'center'})
    ], style={'backgroundColor': '#CCCCCC'})

    # 4b) Determine “start of current year” for YTD filtering
    start_of_year = pd.Timestamp(datetime.today().year, 1, 1)

    # 4c) Build each row
    rows = []
    for idx, row in portfolio_df.iterrows():
        ticker     = row['Ticker']
        company    = row['Company']
        weight_pct = row['weights']
        shares     = row['shares']
        curr_val   = row['current value']
        cost       = row['cost basis']

        # Extract YTD price series for sparkline
        if ticker in prices_df.columns:
            prices_series = prices_df[ticker][prices_df.index >= start_of_year]
        else:
            prices_series = pd.Series(dtype=float)

        sparkline_fig = create_sparkline(prices_series)

        # Alternate row color
        bg_color = '#F9F9F9' if idx % 2 == 0 else 'white'

        rows.append(
            html.Tr([
                html.Td(ticker, style={'padding': '8px'}),
                html.Td(company, style={'padding': '8px'}),
                html.Td(f"{weight_pct:.2%}", style={'padding': '8px', 'text-align': 'right'}),
                html.Td(f"{shares:,}", style={'padding': '8px', 'text-align': 'right'}),
                html.Td(f"${curr_val:,.2f}", style={'padding': '8px', 'text-align': 'right'}),
                html.Td(f"${cost:,.2f}", style={'padding': '8px', 'text-align': 'right'}),
                html.Td(
                    dcc.Graph(figure=sparkline_fig, config={'displayModeBar': False}),
                    style={'padding': '2px', 'width': '120px'}
                )
            ], style={'backgroundColor': bg_color})
        )

    table = html.Table(
        [header] + rows,
        style={'width': '100%', 'border-collapse': 'collapse'}
    )
    return table


def compute_cumulative_returns(
    portfolio_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    benchmark_series: pd.Series
) -> (pd.Series, pd.Series):
    """
    Compute two time series:
      (a) portfolio_cum: the portfolio’s cumulative return over time
      (b) benchmark_cum: the benchmark’s cumulative return over time (e.g. Russell 2000)
    Steps:
      1. Multiply each ticker’s daily price by number of shares → daily position values
      2. Sum across tickers → daily portfolio total value
      3. Compute daily % returns, then take (1 + returns).cumprod()
      4. Do the same for benchmark_series
    Returns:
      portfolio_cum, benchmark_cum  (both pandas.Series indexed by date)
    """
    # (1) Build a “shares” series aligned to prices_df’s columns
    shares_series = portfolio_df.set_index('Ticker')['shares']
    # Filter for tickers actually present in prices_df
    common_tickers = [t for t in shares_series.index if t in prices_df.columns]

    # (2) Compute daily portfolio values:
    #     prices_df[common_tickers] is (dates × tickers). Multiply columnwise by shares_series.
    daily_values = prices_df[common_tickers].multiply(
        shares_series[common_tickers], axis=1
    ).sum(axis=1)

    # (3) Daily returns and cumulative product
    port_returns = daily_values.pct_change().fillna(0)
    portfolio_cum = (1 + port_returns).cumprod()

    # (4) Benchmark returns & cumulative product
    bench_returns = benchmark_series.pct_change().fillna(0)
    benchmark_cum = (1 + bench_returns).cumprod()

    return portfolio_cum, benchmark_cum


def create_sector_donut(df: pd.DataFrame) -> go.Figure:
    """
    Create a Plotly “donut” chart (pie chart with a hole) from sector_allocations DataFrame,
    which has columns ['Sector', 'Value'].
    """
    fig = go.Figure(go.Pie(
        labels      = df['Sector'],
        values      = df['Value'],
        hole        = 0.4,
        hoverinfo   = 'label+percent'
    ))
    fig.update_layout(
        title_text = 'Sector Allocation',
        annotations=[
            dict(text='Sectors', x=0.5, y=0.5, font_size=20, showarrow=False)
        ]
    )
    return fig
