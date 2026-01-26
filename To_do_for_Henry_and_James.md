### My ideas for next steps to implement 

- Implement optional cash buffer into the optimizer (pre-specify amount of cash to keep to the side)
- then fix that cash buffer logic inside of trade_sheet.py 
- Implement e-mail API thing, should take trade_sheet_{date}.csv from the optimization outputs (lives in the /data folder) and email to Prof Salemy via email API. 
- Adjust the optimizer function to automatically save a csv of the new portfolio weights to the /data folder 

We must reconcile the optimizer API with the new data fetching functions and double check that the update_portfolio function actually works. I.e. after we rebalance, those changed weights are reflected both in the portfolio_snapshot.csv, cost_basis_snapshot.csv AND the fund_dashboard. 

This optimization process will be done manually via a Jupyter Notebook called Portfolio Optimizer that will call the pipeline function from within src 

### Overall 

Still a lot of stuff to test that the full pipeline works: 
- (Data ingestion --> csvs updated --> dashboard --> portfolio rebalance --> csvs and dashboard). Up until 3rd step is good, but I haven't designed a function to take the rebalanced portfolio and update the portfolio_snapshot.csv and then those changes are reflected on the website. 
- could be some formatting things to correct on the dashboard since I changed the data fetching mechanisms around 
