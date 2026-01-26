### Rework the following files: 
- Portfolio_Optimizer.py 

Must design a robust portfolio optimizer that outputs a csv into the /data folder called optimized_portfolio_{date}.csv in addition to a trade_sheet_{date}.csv that has a list of trades formatted for the new trades. This trade sheet will be sent via an email API to Prof Salemy. 

This optimization process will be done manually via a Jupyter Notebook called Portfolio Optimizer that will call the pipeline function from within src 

We must reconcile the optimizer API with the new data fetching functions and double check that the update_portfolio function actually works. I.e. after we rebalance, those changed weights are reflected both in the portfolio_snapshot.csv, cost_basis_snapshot.csv AND the fund_dashboard. 

### How to Design the optimizer? 

The inputs are the following: 
- all price / portfolio data csvs 
- Investable Universe (I.e. all the companies/ETFS we want to own through the rebalance)
- Expected returns for all assets, tickers, sectors, ect. 
- I do think that a covariance matrix is important to avoid overloading on specific factors 

Ideally, we would use a relatively simple optimizer here, Mean-Variance Is a simple choice, but I am struggling to identify the actual goal of this optimizer. 

Our fund is Benchmarked to the Russell 2000 -- we cannot buy any companies above the 10B market cap, and we can buy specific sector ETFS to manage our risk. 

It would be very nice to be able to suggest to the optimizer what split from ETFS-STOCKS we want, and how much cash in reserve to hold. 

Now I don't understand how to set up constraints / goals. Since we are benchmarked to the russell in terms of performance, one suggestion from a PM was to try to maximize information ratio which I like. However, I don't know how to maximize this and where to set up tracking error level constraints -- in each sector AND / OR globally? 

The other hard thing is that the returns for each holding is based on a stock pitch and model with a 3y time horizon. All expected returns are annualized, and they are unknown, also the expected return of the Russell is Unknown. 

Our fund is unique in that we have no leverage, and currently, no expenses (other than ETF expense ratios) that need to be managed, the fund is around 1.3M long only equities. 

Please evaluate the requests and help me design a system for the type of optimizr (for Mean variance, the MOSEK library may be a good idea since I already have it configured) but I would like criticism for other options and evaluate the most suitable optimizer to construct and how to set up constraints, assumptions, ect 