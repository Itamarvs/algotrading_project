import random
from datetime import datetime, timedelta
from typing import Union, Any

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from data_tools import get_sp500_tickers
from finance_tools import pfe, sharpe


def add_SSO(data, lbw=14, smoothing_factor=3):
    # price momentum (Stochastic Oscillator) STOCHk_{}_{}_3 and STOCHd_{}_{}_3
    # data.ta.stoch(high='high', low='low', k=price_fast_momentum_lbw, d=price_slow_momentum_lbw, append=True)

    lbw_high = f'lbw_{lbw}_high'
    lbw_low = f'lbw_{lbw}_low'
    # Adds a "n_high" column with max value of previous lbw periods
    data[lbw_high] = data['High'].rolling(lbw).max()
    # Adds an "n_low" column with min value of previous lbw periods
    data[lbw_low] = data['Low'].rolling(lbw).min()
    # Uses the min/max values to calculate the SSO (as a percentage)
    data[f'SSO'] = (data['Close'] - data[lbw_low]) / (data[lbw_high] - data[lbw_low])
    # Uses the SSO to calculates a SMA over the past smoothing_factor values of SSO
    data['smooth_SSO'] = data[f'SSO'].rolling(smoothing_factor).mean()


def add_pfe(data, pfe_lbw=10):
    pfe_period = pfe_lbw - 1
    pfe_values = []
    for k in range(0, pfe_period):
        pfe_values.append(0)
    for i in range(pfe_period, len(data)):
        diff = data['Close'][i] - data['Close'][i - pfe_period]
        top = (diff ** 2 + pfe_period ** 2) ** 0.5
        bottom = 0
        for j in range(0, pfe_period-1):
            bottom += (1 + (data['Close'][i - j] - data['Close'][i - j - 1]) ** 2) ** 0.5
        pfe_res = top / bottom
        if diff < 0:
            pfe_res = -pfe_res
        pfe_values.append(pfe_res)
    data[f'PFE'] = pfe_values
    data[f'PFE'] = data[f'PFE'].ewm(span=5, adjust=False).mean()


def add_volume_vs_lbw_volume_mean(data, lbw):
    lbw_volume_mean = f'lbw_{lbw}_volume_mean'
    data[lbw_volume_mean] = data['Volume'].rolling(lbw).mean()
    data['volume_vs_lbw_volume_mean'] = data['Volume'] / data[lbw_volume_mean]


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


def add_ATR(data, atr_lbw):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    # true_range = np.max(ranges, axis=1)
    true_range = ranges.max(axis=1)
    data[f'atr_lbw_{atr_lbw}'] = true_range.rolling(atr_lbw).sum() / atr_lbw


def add_model_features(data,
                       SSO_lbw=14,
                       SSO_smoothing_factor=3,
                       SSO_th=0.5,
                       volume_trigger_lbw=28,
                       volume_trigger_factor=1.75,
                       volume_trigger_duration=10,
                       rolling_price_lbw=14,
                       atr_lbw=14):
    add_volume_vs_lbw_volume_mean(data, volume_trigger_lbw)
    add_volume_trigger(data, volume_trigger_factor)
    add_volume_trigger_holds(data, volume_trigger_duration)
    add_SSO(data, SSO_lbw, SSO_smoothing_factor)
    add_pfe(data, SSO_lbw)
    add_rolling_price_feature(data, rolling_price_lbw)
    add_ATR(data, atr_lbw)


def calc_naive_momentum_SSO(data, SSO_boundaries=0.2):
    data.loc[0, 'naive_momentum_open_position'] = 0
    for i in data.index:
        open_position = 0
        if data.loc[i, 'smooth_SSO'] > 1 - SSO_boundaries:
            open_position = 1
        elif data.loc[i, 'smooth_SSO'] < SSO_boundaries:
            open_position = -1
        data.loc[i + 1, 'naive_momentum_open_position'] = open_position

    data.loc[0, 'naive_momentum_position'] = 0
    for i in data.index:
        if i > 0:
            position = np.sign(data.loc[i - 1, 'naive_momentum_position'] + data.loc[i, 'naive_momentum_open_position'])
            data.loc[i, 'naive_momentum_position'] = position

    calc_pl(data, prev_position_column='naive_momentum_prev_position', position_column='naive_momentum_position',
            pl='naive_momentum_pl', pl_acc='naive_momentum_pl_accumulate')


def calc_open_positions_SSO(data, close_rolling_price_diff_factor=1, SSO_boundaries=0.2):
    data.loc[0, 'open_position'] = 0
    for i in data.index:
        open_position = 0
        if np.isfinite(data.at[i, 'volume_trigger_holds']) and int(data.at[i, 'volume_trigger_holds']) == 1:
            if data.loc[i, 'close_vs_rolling_price'] >= close_rolling_price_diff_factor:  # close above rolling price
                j = i
                high_sso_since_vol_trigger = True
                while j >= 0 and high_sso_since_vol_trigger and int(data.loc[j, 'volume_trigger_holds']) == 1:
                    if data.loc[j, 'smooth_SSO'] < 1 - SSO_boundaries:
                        high_sso_since_vol_trigger = False
                    if int(data.loc[j, 'volume_trigger']) == 1:
                        break
                    j = j - 1
                if high_sso_since_vol_trigger:
                    open_position = 1
            else:  # close under rolling price
                j = i
                low_sso_since_vol_trigger = True
                while j >= 0 and low_sso_since_vol_trigger and int(data.loc[j, 'volume_trigger_holds']) == 1:
                    if data.loc[j, 'smooth_SSO'] > SSO_boundaries:
                        low_sso_since_vol_trigger = False
                    if int(data.loc[j, 'volume_trigger']) == 1:
                        break
                    j = j - 1
                if low_sso_since_vol_trigger:
                    open_position = -1
        data.loc[i + 1, 'open_position'] = open_position


def calc_open_positions_PFE(data, close_rolling_price_diff_factor=1, pfe_boundaries=0.4):
    data.loc[0, 'open_position'] = 0
    for i in data.index:
        open_position = 0
        if np.isfinite(data.at[i, 'volume_trigger_holds']) and int(data.at[i, 'volume_trigger_holds']) == 1:
            if data.loc[i, 'close_vs_rolling_price'] >= close_rolling_price_diff_factor:  # close above rolling price
                j = i
                high_sso_since_vol_trigger = True
                while j >= 0 and high_sso_since_vol_trigger and int(data.loc[j, 'volume_trigger_holds']) == 1:
                    if data.loc[j, 'PFE'] < 1 - pfe_boundaries:
                        high_sso_since_vol_trigger = False
                    if int(data.loc[j, 'volume_trigger']) == 1:
                        break
                    j = j - 1
                if high_sso_since_vol_trigger:
                    open_position = 1
            else:  # close under rolling price
                j = i
                low_sso_since_vol_trigger = True
                while j >= 0 and low_sso_since_vol_trigger and int(data.loc[j, 'volume_trigger_holds']) == 1:
                    if data.loc[j, 'smooth_SSO'] > -1 + pfe_boundaries:
                        low_sso_since_vol_trigger = False
                    if int(data.loc[j, 'volume_trigger']) == 1:
                        break
                    j = j - 1
                if low_sso_since_vol_trigger:
                    open_position = -1
        data.loc[i + 1, 'open_position'] = open_position



def calc_positions(data, SSO_boundaries=0.2):
    data.loc[0, 'position'] = 0
    for i in data.index:
        if i > 0:
            position = np.sign(data.loc[i - 1, 'position'] + data.loc[i, 'open_position'])
            if position == 1:
                if data.loc[i, 'smooth_SSO'] < 1 - SSO_boundaries:
                    position = 0
            if position == -1:
                if data.loc[i, 'smooth_SSO'] > SSO_boundaries:
                    position = 0
            data.loc[i, 'position'] = position


def calc_pl(data, transaction_cost=1, order_quantity=1,
            prev_position_column='prev_position', position_column='position', pl='pl', pl_acc='pl_accumulate'):
    data[prev_position_column] = data[position_column].shift()
    data['prev_close'] = data['Close'].shift()
    data[pl] = data.apply(lambda row: calc_row_pl(row, transaction_cost, order_quantity, prev_position_column, position_column),
                          axis=1)
    data[pl_acc] = data[pl].cumsum()


def calc_row_pl(row, transaction_cost, order_quantity, prev_position_column, position_column):
    if abs(row[position_column]) == 1 and row[prev_position_column] == 0:
        return (row.Close - row.Open) * order_quantity * row[position_column] - abs(order_quantity * transaction_cost * row.Open)
    elif row[position_column] == 0 and abs(row[prev_position_column]) == 1:
        return (row.Open - row.prev_close) * order_quantity * row[prev_position_column] - abs(order_quantity * transaction_cost * row.Open)
    elif abs(row[position_column]) == 1 and abs(row[prev_position_column]) == 1 and np.sign(row[position_column]) != np.sign(row[prev_position_column]):
        return (row.Close - row.Open) * order_quantity * row[position_column] - abs(order_quantity * transaction_cost * row.Open) \
            + (row.Open - row.prev_close) * order_quantity * row[prev_position_column] - abs(order_quantity * transaction_cost * row.Open)
    else:
        return (row.Close - row.prev_close) * order_quantity * row[position_column]


def calc_sharpe(data):
    sharpe_res = sharpe(data['pl_accumulate'])
    print("sharpe ratio: ", sharpe_res)
    return sharpe_res

# data = yf.download(["AMZN", "AAPL", "GOOG"], period="20y", interval="1d")
# data = yf.download(["APYX"], period="20y", interval="1d")
# data = yf.download(["APYX"], period="20y", interval="1d")
# tickers = ["APYX", "PCG", "ICL"]
# tickers = ["AACI", "AAIC", "AAME"]
tickers = pd.read_csv("./micro_tickers")['0']
for ticker in tickers:#get_sp500_tickers():
    print(f"start processing {ticker}...")
    data = yf.Ticker(ticker).history(period="60d", interval="15m")
    data = data.reset_index()

    add_model_features(data)
    # calc_naive_momentum_SSO(data)
    calc_open_positions_SSO(data)
    # calc_open_positions_PFE(data)
    calc_positions(data)
    calc_pl(data, transaction_cost=0.005, order_quantity=1)
    sharpe_res = round(calc_sharpe(data), 2)

    plt.plot(data['pl_accumulate'], label=f"model P&L, sharpe = {sharpe_res}")
    # plt.plot(data['naive_momentum_pl_accumulate'], label="naive momentum accumulated P&L")
    plt.plot(data['Close'] - data.at[0, 'Close'], label="Stock price")
    plt.title(ticker)
    plt.legend(loc='upper left')
    plt.show()
    data.to_csv("./results.csv")

# print(data.groupby(['position']).count())
# print(data.loc[data['position'] == -1])
# print(data.columns)
# print(data)

#
