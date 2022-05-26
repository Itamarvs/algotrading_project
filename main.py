import random
from datetime import datetime, timedelta
from typing import Union, Any

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from finance_tools import pfe


def add_SSO_feature(data, lbw=14, smoothing_factor=3):
    # price momentum (Stochastic Oscillator) STOCHk_{}_{}_3 and STOCHd_{}_{}_3
    # data.ta.stoch(high='high', low='low', k=price_fast_momentum_lbw, d=price_slow_momentum_lbw, append=True)

    lbw_high = f'lbw_{lbw}_high'
    lbw_low = f'lbw_{lbw}_low'
    # Adds a "n_high" column with max value of previous lbw periods
    data[lbw_high] = data['High'].rolling(lbw).max()
    # Adds an "n_low" column with min value of previous lbw periods
    data[lbw_low] = data['Low'].rolling(lbw).min()
    # Uses the min/max values to calculate the SSO (as a percentage)
    data['SSO'] = (data['Close'] - data[lbw_low]) / (data[lbw_high] - data[lbw_low])
    # Uses the SSO to calculates a SMA over the past smoothing_factor values of SSO
    data['smooth_SSO'] = data['SSO'].rolling(smoothing_factor).mean()
    return 0


def add_volume_vs_lbw_volume_mean_feature(data, lbw):
    lbw_volume_mean = f'lbw_{lbw}_volume_mean'
    data[lbw_volume_mean] = data['Volume'].rolling(lbw).mean()
    data['volume_vs_lbw_volume_mean'] = data['Volume'] / data[lbw_volume_mean]
    return 0


def add_rolling_price_feature(data, lbw):
    data['lbw_rolling_price'] = data['Close'].rolling(lbw).mean()
    data['close_vs_rolling_price'] = data['Close'] / data['lbw_rolling_price']


def add_volume_trigger(data, volume_trigger_factor=1.5):
    for i in data.index:
        if abs(float(data.at[i, 'volume_vs_lbw_volume_mean'])) > volume_trigger_factor:
            data.at[i, 'volume_trigger'] = 1
        else:
            data.at[i, 'volume_trigger'] = 0


def add_volume_trigger_holds(data, volume_trigger_duration):
    data['volume_trigger_holds'] = data['volume_trigger'].rolling(volume_trigger_duration).max()



def add_model_features(data,
                       SSO_lbw=14,
                       SSO_smoothing_factor=3,
                       SSO_th=0.5,
                       volume_trigger_lbw=20,
                       volume_trigger_factor=1.5,
                       volume_trigger_duration=5,
                       rolling_price_lbw=14):
    add_volume_vs_lbw_volume_mean_feature(data, volume_trigger_lbw)
    add_volume_trigger(data, volume_trigger_factor)
    add_volume_trigger_holds(data, volume_trigger_duration)
    add_SSO_feature(data, SSO_lbw, SSO_smoothing_factor)
    add_rolling_price_feature(data, rolling_price_lbw)


def calc_open_positions(data, close_rolling_price_diff_factor=1, SSO_boundaries=0.2):
    positions = []
    for i in data.index:
        open_position = 0
        if np.isfinite(data.at[i, 'volume_trigger_holds']) and int(data.at[i, 'volume_trigger_holds']) == 1:
            if data.loc[i, 'close_vs_rolling_price'] >= close_rolling_price_diff_factor:  # close above rolling price
                j = i
                high_sso_since_vol_trigger = True
                while j >= 0 and high_sso_since_vol_trigger and int(data.loc[j, 'volume_trigger_holds']) == 1:
                    if data.loc[j, 'smooth_SSO'] < 1 - SSO_boundaries:
                        high_sso_since_vol_trigger = False
                    j = j - 1
                if high_sso_since_vol_trigger:
                    open_position = 1
            else:  # close under rolling price
                j = i
                low_sso_since_vol_trigger = True
                while j >= 0 and low_sso_since_vol_trigger and int(data.loc[j, 'volume_trigger_holds']) == 1:
                    if data.loc[j, 'smooth_SSO'] > SSO_boundaries:
                        low_sso_since_vol_trigger = False
                    j = j - 1
                if low_sso_since_vol_trigger:
                    open_position = -1
        data.loc[i, 'open_position'] = open_position


def calc_positions(data, SSO_boundaries=0.2):
    for i in data.index:
        position = 0
        if i > 0:
            if int(data.at[i-1, 'open_position']) == 1:
                if data.loc[i, 'smooth_SSO'] < 1 - SSO_boundaries:
                    position = 1
            if int(data.at[i - 1, 'open_position']) == -1:
                if data.loc[i, 'smooth_SSO'] > SSO_boundaries:
                    position = -1
        data.loc[i, 'position'] = position


# data = yf.download(["AMZN", "AAPL", "GOOG"], period="20y", interval="1d")
data = yf.download(["AMZN"], period="20y", interval="1d")
data = data.reset_index()

add_model_features(data)
calc_open_positions(data)
calc_positions(data)

# print(data.groupby(['position']).count())
# print(data.loc[data['position'] == -1])

data.to_csv("./results.csv")




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
