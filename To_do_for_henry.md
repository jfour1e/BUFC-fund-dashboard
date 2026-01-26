### My ideas for next steps to implement 

- Implement optional cash buffer into the optimizer (pre-specify amount of cash to keep to the side)
- then fix that cash buffer logic inside of trade_sheet.py 
- Implement e-mail API thing, should take trade_sheet_{date}.csv from the optimization outputs (lives in the /data folder) and email to Prof Salemy via email API. 
- Adjust the optimizer function to automatically save a csv of the new portfolio weights to the /data folder 
- Consider adding the portfolio frontier thing (see line 455 of portfolio_optimizer.py for my shit code) and add as a cool visual to simulate many optimizations of portfolios for post-hoc analysis stuff 