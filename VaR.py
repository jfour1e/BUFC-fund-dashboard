import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import re

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)

df = pd.read_csv("BUFC_May_2025_Allocations.csv")

df["Total Cost Basis"] = df["Total Cost Basis"].replace({',': '', r'\$': ''}, regex=True).astype(float)
df["Average Cost Basis"] = df["Average Cost Basis"].replace({',': '', r'\$': ''}, regex=True).astype(float)

tickers = []
shares_list = []

for i, val in enumerate(df["Unnamed: 0"][2:]):
    match = re.search(r":\s*([A-Za-z0-9]+)\)", str(val))
    if match:
        ticker = match.group(1).strip()
        tickers.append(ticker)
        share = df["Total Cost Basis"].iloc[i+2] / df["Average Cost Basis"].iloc[i+2]
        shares_list.append(share)

seen = set()
ticker_shares = []
for t, s in zip(tickers, shares_list):
    if t not in seen:
        ticker_shares.append([t, s])
        seen.add(t)

adj_close_df = yf.download(
    [ts[0] for ts in ticker_shares],
    start=start_date,
    end=end_date,
    auto_adjust=True
)["Close"]

adj_close_df = adj_close_df.dropna(axis=1, how='all')
valid_tickers = adj_close_df.columns.tolist()

ticker_shares = [[t, s] for t, s in ticker_shares if t in valid_tickers]

log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

position_values = np.array([s for t, s in ticker_shares]) * adj_close_df.iloc[-1].values
weights = position_values / np.sum(position_values)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights)

def std_portfolio(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

cov_matrix = log_returns.cov()
port_return = expected_return(weights, log_returns)
port_std = std_portfolio(weights, cov_matrix)
port_val = position_values.sum()

#print("\nTicker and Shares:", ticker_shares)
#print("Portfolio Expected Return:", port_return)
#print("Portfolio Standard Deviation:", port_std)

def random_z_score():
    return np.random.normal(0,1)

days = 5 # change based on days

def scenario_gain_loss(port_value, port_standarddev, randz_score, days):
    return port_value * port_return * days + port_value * port_standarddev * randz_score * np.sqrt(days)

simulations = 10000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(port_val, port_std, z_score, days))

confidence_interval = 0.95
VaR = -np.percentile(scenarioReturn, 100 * (1 - confidence_interval))

plt.hist(scenarioReturn, bins=50, density=True, edgecolor='black')
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')
plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2, label='VaR at 95% confidence')
plt.legend()
plt.show()
print(f"The portfolio's 5-day Value at Risk (95% confidence) is ${VaR:,.2f}")
