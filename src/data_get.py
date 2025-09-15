import pandas as pd
import time, os
import random 
from pathlib import Path
from polygon import RESTClient
from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PRICE_CSV = ROOT_DIR / "daily_prices.csv"
DEFAULT_RUT_CSV   = ROOT_DIR / "rut_daily.csv"

def _resolve_to_root(path_like) -> Path:
    """Return an absolute Path. Relative paths are resolved against project root."""
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT_DIR / p)

def _read_cached_wide(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    return pd.DataFrame()

def _write_cached(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    out = df.sort_index()
    out.index.name = "date"
    df.sort_index().to_csv(path, date_format="%Y-%m-%d")

def _sleep_with_jitter(base: float, jitter: float = 0.4):
    time.sleep(base + random.random() * jitter)

def _get_aggs_with_backoff(client, **kwargs):
    """Polygon get_aggs with exponential backoff for 429s."""
    max_retries = 6
    base = 1.5
    for i in range(max_retries):
        try:
            return client.get_aggs(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if (("429" in msg) or ("rate" in msg) or ("retry" in msg)) and i < max_retries - 1:
                delay = min(base * (2 ** i), 30.0)
                print(f"Rate-limited; retrying in {delay:.1f}s (attempt {i+1}/{max_retries})")
                time.sleep(delay)
                continue
            raise

def load_clean_holdings(filepath: str, sheet_name: int = 0) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df[:-2]                          # drop last two rows
    df = df.drop(df.iloc[1].name)        # drop row 1
    df = df.drop(df.iloc[14].name)       # drop row 14

    first_idx = df.index[0]
    df.at[first_idx, 'Current Price'] = 1.0
    df.at[first_idx, 'Current Value'] = df.at[first_idx, 'Total Cost Basis']
    df.at[first_idx, 'Average Cost Basis'] = 1.0

    df = df.rename(columns={"Unnamed: 0": "Company"})
    df["Ticker"] = df["Company"].str.extract(r":([A-Z]+)\)")
    df = df.reset_index(drop=True)
    return df

def load_clean_sector_allocations(filepath: str, sheet_name=1):

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    last_idx = df.index[-1]
    df.at[last_idx, '% of Fund'] = 1.0

    return df
    
def fetch_price_data(
    companies, 
    client, 
    csv_path: str | Path = DEFAULT_PRICE_CSV,
    start_date: str | datetime = "2025-01-01",
    end_date: datetime | None = None,
    request_limit_per_min: int = 4,   
    pause_between: float = 1.6,       
    polygon_limit: int = 365,
) -> pd.DataFrame:
    
    """
    Fetch daily close prices for `companies` from Polygon, with CSV caching and incremental updates.

    Behavior:
      - Loads existing CSV if present.
      - For each ticker, only requests missing dates (latest_date+1 → today).
      - Respects a simple per-minute rate limit and a small delay between calls.
      - Returns a wide DataFrame indexed by date; columns are tickers.
    """
    
    csv_path = _resolve_to_root(csv_path)
    prices = _read_cached_wide(csv_path)

    start_dt = pd.to_datetime(start_date)
    end_dt = end_date or datetime.today()
    end_str = pd.to_datetime(end_dt).strftime("%Y-%m-%d")

    request_count = 0
    window_start = time.time()

    def _minute_guard():
        nonlocal request_count, window_start
        if request_count >= request_limit_per_min:
            elapsed = time.time() - window_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            window_start = time.time()
            request_count = 0

    for ticker in companies:
        # incremental start date for this ticker
        if ticker in prices.columns:
            existing = prices[ticker].dropna()
            if not existing.empty:
                latest = existing.index.max()
                # consider 1-day lag to avoid partial/intraday issues
                if pd.to_datetime(latest) >= pd.to_datetime(end_str) - pd.Timedelta(days=1):
                    print(f"{ticker}: up-to-date.")
                    continue
                fetch_from_dt = latest + pd.Timedelta(days=1)
            else:
                fetch_from_dt = start_dt
        else:
            fetch_from_dt = start_dt

        if fetch_from_dt >= end_dt:
            print(f"{ticker}: no new data needed.")
            continue

        fetch_from_str = fetch_from_dt.strftime("%Y-%m-%d")
        _minute_guard()

        try:
            aggs = _get_aggs_with_backoff(
                client,
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=fetch_from_str,
                to=end_str,
                adjusted=True,
                sort="asc",
                limit=polygon_limit,
            )

            temp = pd.DataFrame(
                [{"date": datetime.fromtimestamp(a.timestamp/1000, tz=timezone.utc).date(),
                  ticker: a.close} for a in aggs]
            )
            if temp.empty:
                print(f"{ticker}: no data returned.")
                continue

            temp.set_index("date", inplace=True)
            temp.index = pd.to_datetime(temp.index)

            # overlap-safe update
            if prices.empty:
                prices = temp
            elif ticker in prices.columns:
                # ensure the DF has rows for all new dates
                prices = prices.reindex(prices.index.union(temp.index))
                prices[ticker] = prices[ticker].combine_first(temp[ticker])
            else:
                prices = prices.join(temp, how="outer")

            print(f"{ticker}: fetched {len(temp)} new rows.")
            request_count += 1

            _sleep_with_jitter(pause_between, jitter=0.5)
            _write_cached(prices, csv_path)  # persist incrementally

        except Exception as e:
            print(f"{ticker}: failed with error {e}")

    _write_cached(prices, csv_path)
    return prices


def fetch_RUT_data(
    client,
    csv_path: str | Path = DEFAULT_RUT_CSV,
    start_date: str | datetime = "2025-01-01",
    end_date: datetime | None = None,
    polygon_limit: int = 5000,  # allow long spans
) -> pd.Series:
    """
    Returns a Series named 'price' for IWM (Russell 2000 ETF proxy) with a root-level CSV cache.
      - loads cache if present
      - fetches missing dates
      - writes back after updates
    """
    csv_path = _resolve_to_root(csv_path)

    # Load cache as df (index=date, col='price')
    if Path(csv_path).exists():
        cached = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
        cached.index = pd.to_datetime(cached.index)
        if "price" not in cached.columns:  # backward compatibility with 'IWM'
            # handle a previous cache with 'IWM' col
            if "IWM" in cached.columns:
                cached = cached.rename(columns={"IWM": "price"})
        rut_df = cached[["price"]]
    else:
        rut_df = pd.DataFrame(columns=["price"])

    start_dt = pd.to_datetime(start_date)
    end_dt = end_date or datetime.today()
    end_str = pd.to_datetime(end_dt).strftime("%Y-%m-%d")

    # Determine incremental start
    if not rut_df.empty and not rut_df["price"].dropna().empty:
        latest = rut_df.index.max()
        if pd.to_datetime(latest) >= pd.to_datetime(end_str) - pd.Timedelta(days=1):
            # up-to-date: return as Series
            return rut_df["price"].sort_index()
        fetch_from_dt = latest + pd.Timedelta(days=1)
    else:
        fetch_from_dt = start_dt

    if fetch_from_dt < end_dt:
        aggs = _get_aggs_with_backoff(
            client,
            ticker="IWM",
            multiplier=1,
            timespan="day",
            from_=fetch_from_dt.strftime("%Y-%m-%d"),
            to=end_str,
            adjusted=True,
            sort="asc",
            limit=polygon_limit,
        )
        temp = pd.DataFrame(
            [{"date": pd.to_datetime(a.timestamp, unit="ms"), "price": a.close} for a in aggs]
        )
        if not temp.empty:
            temp.set_index("date", inplace=True)
            temp.index = pd.to_datetime(temp.index)

            if rut_df.empty:
                rut_df = temp
            else:
                # combine_first on the 'price' column
                combined = rut_df.copy()
                for idx, val in temp["price"].items():
                    if idx not in combined.index or pd.isna(combined.at[idx, "price"]):
                        combined.loc[idx, "price"] = val
                rut_df = combined.sort_index()

            # persist cache (index->date)
            out = rut_df.sort_index().rename_axis("date").reset_index()
            out.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

    return rut_df["price"].sort_index()


def build_live_portfolio(
    holdings: pd.DataFrame,
    prices_data: pd.DataFrame,
    cash_ticker: str = "SPAXX",
    forward_fill: bool = True,
) -> pd.DataFrame:
    
    columns = ['Ticker', 'shares', 'cost basis', 'current value', 'return', 'weights']
    live_portfolio = pd.DataFrame(columns=columns)

    # Get the most recent known price for each ticker
    latest_prices = prices_data.ffill().iloc[-1]

    # Assume first row of holdings is cash
    cash_row = holdings.iloc[0]
    cash_amount = cash_row["Total Cost Basis"]

    # Insert cash row with zero return & weight (weight will be computed later)
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

    # Compute total portfolio value and assign weights
    portfolio_value = live_portfolio["current value"].sum()
    live_portfolio["weights"] = live_portfolio["current value"] / portfolio_value

    # Sort by descending weight and reset index
    live_portfolio = (
        live_portfolio
        .sort_values("weights", ascending=False)
        .reset_index(drop=True)
    )

    return live_portfolio