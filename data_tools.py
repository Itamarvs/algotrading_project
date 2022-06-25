import pandas as pd
from pandas_datareader import data
import yfinance as yf

nasdaq_stocks = pd.read_csv("./nasdaq_stocks.csv")


def get_sp500_tickers():
    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    second_table = payload[1]
    df = first_table
    symbols = df['Symbol'].values.tolist()
    symbols = filter(lambda s: "." not in s, symbols)
    return symbols


# def micro_tickers_between_1_10(tickers):
#     micros = []
#     for ticker in tickers:
#         quote = data.get_quote_yahoo(ticker)
#         if 'marketCap' in quote.columns:
#             market_cap = quote['marketCap'][0]
#             print(f"{ticker} market_cap: {market_cap}")
#             if market_cap < 300000000:
#                 micros.append(ticker)
#     return micros


def between_1_10(tickers):
    between_1_10 = []
    for ticker in tickers:
        ticker_data = yf.download(ticker, period="1d")
        if len(ticker_data['Close']) > 0:
            print(f"{ticker}...{ticker_data['Close'][0]}")
            if 1 <= ticker_data['Close'][0] <= 10:
                between_1_10.append(ticker)
    print(between_1_10)
    return between_1_10


# NASDAQ stocks tickers complying with:
  # Micro Market Cap (less than 300M$)
  # Stock Price between 1$ - 10$
  # Daily Average Volume of at least 1M$
  # IPO date at least 1 year
  # Healthcare Sector
# as extracted from https://finviz.com/screener.ashx on June 2022
tickers = ["SNOA", "NURO", "YMTX", "BTTX", "ASRT", "ALVR", "GOVX", "DARE",
           "EVOK", "XERS", "HGEN", "OSUR", "XXII", "ATOS", "GRTS", "VSTM",
           "ALLK", "TNXP", "CBIO", "CRIS", "SELB", "PRAX", "CMRX", "CLVS",
           "CMRA", "RIGL", "BLUE"]