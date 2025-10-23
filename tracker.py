from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as  yf

BASE_DIR=BASE_DIR = Path.cwd()
DATA_DIR=BASE_DIR / "data"
OUTPUT_DIR=BASE_DIR / "output"

TICKERS_FILE=DATA_DIR / "tickers.csv"
PRICES_FILE=DATA_DIR / "prices_raw.csv"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

tickers_df=pd.read_csv(
    TICKERS_FILE,
    sep=";",
    dtype={"ticker": "string", "name": "string", "start_date": "string"}
)

if tickers_df.empty:
    raise ValueError("RU: файл trackers пуст\nENG: Ticker's file is empty")

if not {"ticker", "name", "start_date"}.issubset(tickers_df.columns):
    raise ValueError("RU: В tickers.csv должны быть колонки: ticker;name;start_date\nENG: Tickers.csv should have columns: ticker;name;start_date")

start_date = tickers_df["start_date"].min()
ticker_symbol = tickers_df["ticker"].tolist()

prices=yf.download(
    tickers=ticker_symbol,
    start=start_date,
    progress=True,
    auto_adjust=False,   
)["Adj Close"]

if isinstance(prices, pd.Series):
    prices = prices.to_frame(name=ticker_symbol[0])

prices = prices.dropna(how="all")
prices.index = pd.to_datetime(prices.index)

prices = prices.loc[:, [c for c in prices.columns if c in ticker_symbol]]

prices.to_csv(PRICES_FILE, index_label="date")
print(f"RU: Готово. Скачано {prices.shape[0]} строк по {prices.shape[1]} тикерам.\n ENG: Done. Downloaded {prices.shape[0]} rows of {prices.shape[1]} tickers")
