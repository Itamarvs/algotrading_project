import random
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt

# data = yf.download(["AMZN", "AAPL", "GOOG"], period="20y", interval="1d")
from finance_tools import pfe

data = yf.download(["AMZN"], period="20y", interval="1d")
# data['Return'] = 100 * (data['Close'].pct_change())
# data['Log returns'] = np.log(data['Close']/data['Close'].shift())
# print(type(np.log(data['Close']/data['Close'].shift())))


# k_period = 14
# d_period = 3
# # Adds a "n_high" column with max value of previous k_period periods
# data['n_high'] = data['High'].rolling(k_period).max()
# # Adds an "n_low" column with min value of previous k_period periods
# data['n_low'] = data['Low'].rolling(k_period).min()
# # Uses the min/max values to calculate the %k (as a percentage)
# data['%K'] = (data['Close'] - data['n_low']) * 100 / (data['n_high'] - data['n_low'])
# # Uses the %k to calculates a SMA over the past d_period values of %k
# data['%D'] = data['%K'].rolling(d_period).mean()


def calc_position(data, price_fast_momentum_lbw = 14, price_slow_momentum_lbw = 3):
    # price momentum STOCHk_{}_{}_3 and STOCHd_{}_{}_3
    data.ta.stoch(high='high', low='low', k=price_fast_momentum_lbw, d=price_slow_momentum_lbw, append=True)


dates = []
for timestamp in list(data.index):
    date = timestamp.to_pydatetime()
    dates.append(date)


# print(dates)

# print(data.columns)
print(data)


def volume_trigger(date, diff_factor=1.5, lbw=20):
    vol = data['Volume'][date.isoformat()]
    lbw_vol_data = data['Volume'][(date - timedelta(lbw)).isoformat():date.isoformat()]
    is_trigger = vol > lbw_vol_data.mean() * diff_factor
    return is_trigger


def price_momentum(date, lbw):
    prices = list(data['Close'][(date - timedelta(lbw)).isoformat():date.isoformat()])
    return pfe(prices)


# prices = []
# momentum_k = []
# momentum_d = []
# volume_triggers = []
# rand_day = random.randint(11, len(dates))
# for date in dates[rand_day : rand_day + 100]:
#     prices.append(data['Close'][date])
#     momentum_k.append(data['STOCHk_14_3_3'][date])
#     momentum_d.append(data['STOCHd_14_3_3'][date])
#     # volume_triggers.append(volume_trigger(date))
#
# plt.plot(prices)
# plt.show()
# plt.plot(momentum_k, 'r+')
# plt.plot(momentum_d, 'b+')
# plt.show()