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


def save_micro_tickers_between_1_10():
    print(nasdaq_stocks.columns)
    micro_size = nasdaq_stocks['Market Cap'].between(50000000,300000000)
    micro_tickers_between_1_10 = list(nasdaq_stocks[micro_size]['Symbol'].values)
    micro_tickers_between_1_10 = between_1_10(micro_tickers_between_1_10)
    pd.DataFrame(micro_tickers_between_1_10).to_csv("./micro_tickers_between_1_10")