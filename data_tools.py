import pandas as pd


def get_sp500_tickers():
    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    second_table = payload[1]
    df = first_table
    symbols = df['Symbol'].values.tolist()
    symbols = filter(lambda s: "." not in s, symbols)
    return symbols


get_sp500_tickers()