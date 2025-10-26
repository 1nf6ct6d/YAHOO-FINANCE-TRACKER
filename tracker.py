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

prices = pd.read_csv(PRICES_FILE, index_col="date" , parse_dates=["date"])

rets=prices.pct_change().dropna(how="all")
tickers_df = pd.read_csv(TICKERS_FILE, sep=";")

if "weight" not in tickers_df.columns:
    tickers_df["weight"]=1.0/len(tickers_df)
    
weights = tickers_df.set_index("ticker")["weight"]
weights = weights / weights.sum()

D = 252 #Количество торговых дней в году
rf = 0.00 #Безрисковая годовая ставка

#Вычисление метрик
rets = rets.loc[:, rets.columns.intersection(weights.index)]
weighted = rets.mul(weights, axis=1) 
r_portfel = weighted.sum(axis=1)  
equity = (1.0 + r_portfel).cumprod() 

n_size=r_portfel.shape[0]
total_return = equity.iloc[-1] - 1.0
cagr = equity.iloc[-1] ** (D / n_size) - 1.0
ann_vol = r_portfel.std(ddof=0) * np.sqrt(D)
sharpe = (cagr - rf) / ann_vol if ann_vol > 0 else np.nan
dd = equity / equity.cummax() - 1.0
max_dd = dd.min()

metrics = pd.DataFrame({
    "TotalReturn":[total_return],
    "CAGR":[cagr],
    "ANN_Vol":[ann_vol],
    "Sharpe":[sharpe],
    "MaxDrawdown":[max_dd]
})

rets.to_csv(DATA_DIR / "returns_by_ticker.csv", index_label="date")
pd.DataFrame({"r_portfel": r_portfel, "equity": equity}).to_csv(DATA_DIR / "portfolio_series.csv", index_label="date")
metrics.to_csv(DATA_DIR / "metrics.csv", index=False)

print("RU: Доходности и метрики посчитаны.\nENG: Returns and metrics computed.")
print(f"Файлы:\n - {DATA_DIR/'returns_by_ticker.csv'}\n - {DATA_DIR/'portfolio_series.csv'}\n - {DATA_DIR/'metrics.csv'}")

