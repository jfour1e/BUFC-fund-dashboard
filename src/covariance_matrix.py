import pandas as pd

prices = pd.read_csv("/Users/henryidone/Documents/GitHub/BUFC-fund-dashboard/daily_prices.csv", index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()
cov_matrix = returns.cov()
cov_matrix_annualized = cov_matrix * 252
corr_matrix_annualized = returns.corr()
cov_matrix_annualized.to_csv("holdings_covariance_matrix.csv", float_format="%.6f")
corr_matrix_annualized.to_csv("holdings_corr_matrix.csv", float_format="%.6f")

