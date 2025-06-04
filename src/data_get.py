import pandas as pd
from datetime import datetime
import time
from polygon import RESTClient

def load_clean_holdings(filepath: str, sheet_name=0): 

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df[:-2]
    df = df.drop(df.iloc[1].name)
    df = df.drop(df.iloc[14].name)

    # Fix the first-row current price/value/cost‐basis
    first_idx = df.index[0]
    df.at[first_idx, 'Current Price'] = 1.0
    df.at[first_idx, 'Current Value'] = df.at[first_idx, 'Total Cost Basis']
    df.at[first_idx, 'Average Cost Basis'] = 1.0

    # Rename “Unnamed: 0” → “Company”
    df = df.rename(columns={"Unnamed: 0": "Company"})

    # Extract “Ticker” from “Company” via regex
    df["Ticker"] = df["Company"].str.extract(r":([A-Z]+)\)")

    # Reset index and drop the old index column
    df = df.reset_index(drop=True)

    return df

def load_clean_sector_allocations(filepath: str, sheet_name=1):

    df = pd.read_excel(filepath, sheet_name=sheet_name)

    last_idx = df.index[-1]
    df.at[last_idx, '% of Fund'] = 1.0

    return df


def fetch_price_data(companies, client):

    prices_data = pd.DataFrame()

    end_date = datetime.today()
    start_date = pd.to_datetime("2025-01-01")
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    for stock in companies:
        try:
            aggs = client.get_aggs(
                ticker=stock,
                multiplier=1,
                timespan="day",
                from_=start_str,
                to=end_str,
                adjusted=True,
                sort="asc",
                limit=120
            )
            temp_df = pd.DataFrame([{
                "date": datetime.utcfromtimestamp(a.timestamp / 1000).date(),
                stock: a.close
            } for a in aggs])

            temp_df.set_index("date", inplace=True)
            prices_data = prices_data.join(temp_df, how="outer") if not prices_data.empty else temp_df
        except Exception as e:
            print(f"Failed to fetch {stock}: {e}")
        time.sleep(15)

    return prices_data

def fetch_RUT_data(companies, client): 

    start_date = "2025-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    aggs = client.get_aggs(
        ticker="IWM",
        multiplier=1,
        timespan="day",
        from_=start_date,
        to=end_date,
        adjusted=True,
        sort="asc",
        limit=5000
    )

    rut_df = pd.DataFrame([{
        "date": pd.to_datetime(a.timestamp, unit="ms"),
        "IWM": a.close
    } for a in aggs])

    rut_df['date'] = pd.to_datetime(rut_df['date'])
    rut_df.set_index('date', inplace=True)
    rut_df.rename(columns={'IWM': 'price'}, inplace=True)
    rut_series = rut_df['price']

    return rut_series

def build_live_portfolio(holdings: pd.DataFrame, prices_data: pd.DataFrame, cash_ticker: str = "SPAXX"): 

    columns = ['Ticker', 'shares', 'cost basis', 'current value', 'return', 'weights']
    live_portfolio = pd.DataFrame(columns=columns)

    # 2. Get the most recent known price for each ticker
    latest_prices = prices_data.ffill().iloc[-1]

    # 3. Assume first row of holdings is cash
    cash_row = holdings.iloc[0]
    cash_amount = cash_row["Total Cost Basis"]

    # 4. Insert cash row with zero return & weight (weight will be computed later)
    live_portfolio.loc[len(live_portfolio)] = [
        cash_ticker,
        cash_amount,
        cash_amount,
        cash_amount,
        0.0,
        0.0
    ]

    # 5. Loop over the remaining holdings (skip row 0)
    for _, row in holdings.iloc[1:].iterrows():
        ticker = row["Ticker"]
        cost_basis = row["Total Cost Basis"]
        avg_cost_basis = row["Average Cost Basis"]

        # number of shares = total cost / average cost per share
        shares = cost_basis / avg_cost_basis

        # lookup current price and compute current value & return
        current_price = latest_prices[ticker]
        current_value = shares * current_price
        ret = (current_value - cost_basis) / cost_basis

        live_portfolio.loc[len(live_portfolio)] = [
            ticker,
            shares,
            cost_basis,
            current_value,
            ret,
            0.0
        ]

    # 6. Compute total portfolio value and assign weights
    portfolio_value = live_portfolio["current value"].sum()
    live_portfolio["weights"] = live_portfolio["current value"] / portfolio_value

    # 7. Sort by descending weight and reset index
    live_portfolio = (
        live_portfolio
        .sort_values("weights", ascending=False)
        .reset_index(drop=True)
    )

    return live_portfolio