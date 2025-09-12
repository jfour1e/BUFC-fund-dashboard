import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import math

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

def create_treemap(portfolio_df) -> go.Figure:
    labels_leaf  = portfolio_df['Ticker'].astype(str).tolist()
    values_leaf  = portfolio_df['weights'].astype(float).tolist()
    changes_leaf = portfolio_df['pct_change'].astype(float).tolist()  # percent units, e.g. 1.23

    # ----- Build root + leaves -----
    labels  = ["Portfolio"] + labels_leaf
    parents = [""] + ["Portfolio"] * len(labels_leaf)
    values  = [sum(values_leaf)] + values_leaf

    # ----- Color mapping (explicit hex per point) -----
    # thresholds in percent units
    NEUTRAL_BAND = 0.10  # ±0.10% stays grey
    MAX_ABS      = 6.0   # clamp beyond ±6% to deepest colors
    SHADES       = 10

    def clamp(x, a, b): return max(a, min(b, x))

    def lerp_hex(a, b, t):
        # a,b are '#RRGGBB'; t in [0,1]
        ar, ag, ab = int(a[1:3],16), int(a[3:5],16), int(a[5:7],16)
        br, bg, bb = int(b[1:3],16), int(b[3:5],16), int(b[5:7],16)
        rr = round(ar + (br - ar)*t)
        rg = round(ag + (bg - ag)*t)
        rb = round(ab + (bb - ab)*t)
        return f"#{rr:02x}{rg:02x}{rb:02x}"

    # palette endpoints
    DEEP_GREEN = "#00441b"   # forest green at >= +6%
    PALE_GREEN = "#74e274"
    DEEP_RED   = "#7f0000"   # deep red at <= -6%
    PALE_RED   = "#e47272"
    NEUTRAL_G  = "#a6a6a6"   #

    def _hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (1,3,5))
    def _rgb_to_hex(r,g,b): return f"#{r:02x}{g:02x}{b:02x}"
    def _lerp(a,b,t): return a + (b-a)*t

    def _ramp(c1, c2, steps):
        r1,g1,b1 = _hex_to_rgb(c1); r2,g2,b2 = _hex_to_rgb(c2)
        return [
            _rgb_to_hex(int(_lerp(r1,r2,t)), int(_lerp(g1,g2,t)), int(_lerp(b1,b2,t)))
            for t in [i/(steps-1) for i in range(steps)]
        ]

    greens = _ramp(PALE_GREEN, DEEP_GREEN, SHADES)  # light → deep
    reds   = _ramp(PALE_RED,   DEEP_RED,   SHADES)  # light → deep

    def color_for_change(pct):
        # pct in percent units
        if abs(pct) < NEUTRAL_BAND or pct != pct:   # NaN-safe neutral
            return NEUTRAL_G
        # map to 10 bins, clamped to ±MAX_ABS
        if pct > 0:
            pos = min(MAX_ABS, max(NEUTRAL_BAND, pct))
            t   = (pos - NEUTRAL_BAND) / (MAX_ABS - NEUTRAL_BAND)
            idx = min(SHADES-1, int(round(t*(SHADES-1))))
            return greens[idx]
        else:
            neg = min(MAX_ABS, max(NEUTRAL_BAND, -pct))
            t   = (neg - NEUTRAL_BAND) / (MAX_ABS - NEUTRAL_BAND)
            idx = min(SHADES-1, int(round(t*(SHADES-1))))
            return reds[idx]
        
    colors = ["#ffffff"] + [color_for_change(p) for p in changes_leaf]

    # Per-point hover: none for root, show for leaves
    leaf_hover = "<b>%{label}</b><br>Weight: %{value:.2%}<br>Daily Change: %{customdata:.2f}%<extra></extra>"
    hovertemplates = [""] + [leaf_hover] * len(labels_leaf)

    fig = go.Figure(go.Treemap(
        labels       = labels,
        parents      = parents,
        values       = values,
        branchvalues = "total",
        marker=dict(
            colors=colors,                         # explicit colors (no colorscale)
            line=dict(width=0.2, color="#f0f0f0")
        ),
        texttemplate = "<b>%{label}</b>",
        textfont     = dict(color="white", size=14),
        textposition = "middle center",
        customdata   = [None] + changes_leaf,
        hovertemplate= hovertemplates,
        tiling       = dict(pad=0),               # no gutters
    ))

    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    # If any breadcrumb band still shows on your Plotly version, cover it:
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0.955, x1=1, y1=1,
        fillcolor="white", line=dict(width=0), layer="above"
    )

    return fig


def create_sparkline(prices_series: pd.Series) -> go.Figure:
    """
    Create a small Plotly line chart (“sparkline”) for a given price series.
    Axes are hidden, margins are minimized, and height is small (20px).
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
        autosize=False, 
        height=24,   
        width=120,    
        xaxis    = dict(visible=False),
        yaxis    = dict(visible=False),
        margin   = dict(l=0, r=0, t=2, b=2),
        showlegend=False
    )
    return fig


def create_holdings_table(portfolio_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Table:
    """
    Construct an HTML <table> showing:
      - Ticker
      - % of Portfolio
      - Shares Owned
      - Current Value
      - Cost Basis
      - Return (e.g. P/L or pct, if you want to show that)
      - Price History (YTD sparkline)

    Alternates row background for readability.
    """
    # 1) Header Row (no "Company" column, since portfolio_df doesn’t have it)
    header = html.Tr([
        html.Th('Ticker',                  style={'padding': '8px', 'text-align': 'left'}),
        html.Th('% of Portfolio',          style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Shares Owned',            style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Current Value',           style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Cost Basis',              style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Return',                  style={'padding': '8px', 'text-align': 'right'}),
        html.Th('Price History (YTD)',      style={'padding': '8px', 'text-align': 'center'}),
    ], style={'backgroundColor': '#CCCCCC'})
        
    # 2) Determine “start of current period” for 3M filtering
    start_of_year = pd.Timestamp(datetime.today().year, 1, 1)

    # 3) Build each row
    rows = []
    for idx, row in portfolio_df.iterrows():
        ticker     = row['Ticker']
        weight_pct = row['weights']
        shares     = row['shares']
        curr_val   = row['current value']
        cost_basis = row['cost basis']
        ret_val    = row.get('return', None)  # Use .get in case "return" is missing

        shares_display = math.ceil(shares - 1e-6) 

        # Extract price series for sparkline
        if ticker in prices_df.columns:
            prices_series = prices_df[ticker][prices_df.index >= start_of_year]
        else:
            prices_series = pd.Series(dtype=float)

        # Create the sparkline figure (you must have defined create_sparkline elsewhere)
        sparkline_fig = create_sparkline(prices_series)

        # Alternate row color shading
        bg_color = '#F9F9F9' if (idx % 2 == 0) else 'white'

        # Create the <tr> for this row
        rows.append(
            html.Tr([
                html.Td(ticker,                                    style={'padding': '8px'}),
                html.Td(f"{weight_pct:.2%}",                       style={'padding': '8px', 'text-align': 'right'}),
                html.Td(f"{shares_display:,.2f}",         
                    style={'padding': '8px', 'text-align': 'right'}), 
                html.Td(f"${curr_val:,.2f}",                       style={'padding': '8px', 'text-align': 'right'}),
                html.Td(f"${cost_basis:,.2f}",                     style={'padding': '8px', 'text-align': 'right'}),
                html.Td(f"{ret_val:.2%}" if ret_val is not None else "—",
                        style={'padding': '8px', 'text-align': 'right'}),
                html.Td(
                    dcc.Graph(figure=sparkline_fig, config={'displayModeBar': False, 'staticPlot': True, 'responsive': False}),
                    style={'padding': '2px', 'width': '120px', 'height': '24px'}
                )
            ], style={'backgroundColor': bg_color})
        )

    # Assemble the entire <table>
    table = html.Table(
        [header] + rows,
        style={'width': '100%', 'border-collapse': 'collapse'}
    )
    return table


def compute_cumulative_returns(
    portfolio_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    benchmark_series: pd.Series
):
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
