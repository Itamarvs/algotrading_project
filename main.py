import itertools
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split

import data_tools
from finance_tools import sharpe_ratio

# set train & test data for each ticker
train = {}
test = {}
for ticker in data_tools.tickers:
    data = yf.Ticker(ticker).history(period="60d", interval="15m",
                                     actions=False)
    data = data.reset_index()
    ticker_train, ticker_test = train_test_split(data, test_size=0.25,
                                                 shuffle=False)
    train[ticker] = ticker_train
    test[ticker] = ticker_test


def add_SSO(data, lbw=14, smoothing_factor=3):
    lbw_high = f'lbw_{lbw}_high'
    lbw_low = f'lbw_{lbw}_low'
    data[lbw_high] = data['High'].rolling(lbw).max()
    data[lbw_low] = data['Low'].rolling(lbw).min()
    data[f'SSO'] = (data['Close'] - data[lbw_low]) / (data[lbw_high] - data[lbw_low])
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
        for j in range(0, pfe_period - 1):
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
    true_range = ranges.max(axis=1)
    data[f'atr_lbw_{atr_lbw}'] = true_range.rolling(atr_lbw).sum() / atr_lbw


def add_model_features(data,
                       SSO_lbw=14,
                       SSO_smoothing_factor=3,
                       volume_trigger_lbw=28,
                       volume_trigger_factor=1.75,
                       volume_trigger_duration=10,
                       rolling_price_lbw=14,
                       atr_lbw=14):
    add_volume_vs_lbw_volume_mean(data, volume_trigger_lbw)
    add_volume_trigger(data, volume_trigger_factor)
    add_volume_trigger_holds(data, volume_trigger_duration)
    add_SSO(data, SSO_lbw, SSO_smoothing_factor)
    add_rolling_price_feature(data, rolling_price_lbw)
    add_ATR(data, atr_lbw)


def calc_open_positions_SSO(data, close_rolling_price_diff_factor=1, SSO_boundaries=0.2):
    data.loc[0, 'open_position'] = 0
    data.loc[0, 'position'] = 0
    for i in data.index:
        open_position = 0
        if np.isfinite(data.at[i, 'volume_trigger_holds']) \
                and int(data.at[i, 'volume_trigger_holds']) == 1:
            if data.loc[i, 'close_vs_rolling_price'] >= close_rolling_price_diff_factor:  # close above rolling price
                j = i
                high_sso_since_vol_trigger = True
                while j >= 0 and high_sso_since_vol_trigger \
                        and data.loc[j, 'volume_trigger_holds'] == 1 \
                        and data.loc[j, 'position'] == 1:
                    if data.loc[j, 'smooth_SSO'] < 1 - SSO_boundaries:
                        high_sso_since_vol_trigger = False
                    if int(data.loc[j, 'volume_trigger']) == 1:  # TODO: should remove?
                        break
                    j = j - 1
                if high_sso_since_vol_trigger:
                    open_position = 1
            else:  # close under rolling price
                j = i
                low_sso_since_vol_trigger = True
                while j >= 0 and low_sso_since_vol_trigger \
                        and data.loc[j, 'volume_trigger_holds'] == 1 \
                        and data.loc[j, 'position'] == -1:
                    if data.loc[j, 'smooth_SSO'] > SSO_boundaries:
                        low_sso_since_vol_trigger = False
                    if int(data.loc[j, 'volume_trigger']) == 1:  # TODO: should remove?
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
    start_index = data.index[0]
    data.loc[start_index, 'position'] = 0
    for i in data.index:
        if i > start_index:
            position = np.sign(data.loc[i - 1, 'position'] + data.loc[i, 'open_position'])
            if position == 1:
                if data.loc[i, 'smooth_SSO'] < 1 - SSO_boundaries:
                    position = 0
            if position == -1:
                if data.loc[i, 'smooth_SSO'] > SSO_boundaries:
                    position = 0
            data.loc[i, 'position'] = position


def calc_pl_returns(data, transaction_cost, slippage_rate, order_quantity=1):
    data['prev_position'] = data['position'].shift()
    data['prev_close'] = data['Close'].shift()
    data['pl'] = data.apply(lambda row:
                            row_pl(row, transaction_cost, slippage_rate, order_quantity),
                            axis=1)
    data['pl_accumulate'] = data['pl'].cumsum()
    data.loc[data['enter_position_price'] == 0, 'returns'] = 0
    data.loc[data['enter_position_price'] != 0, 'returns'] = data['pl'] / data['enter_position_price']
    data['returns_accumulate'] = data['returns'].dropna().cumsum()


def row_pl(row, transaction_cost, slippage_rate, order_quantity, is_absolute_transaction_cost=False):
    if math.isnan(row.prev_close) or math.isnan(row.Close):
        return 0

    pl_enter_position = \
        (row['Close'] - row['Open']) * order_quantity * row['position']
    pl_exit_position = \
        (row['Open'] - row["prev_close"]) * order_quantity * row["prev_position"]
    pl_keep_in_position = \
        (row['Close'] - row["prev_close"]) * order_quantity * row['position']

    if is_absolute_transaction_cost:
        commission_cost = transaction_cost
    else:
        commission_cost = transaction_cost * order_quantity * row['Open']

    slippage = slippage_rate * abs(row['Close'] - row['Open'])

    if abs(row['position']) == 1 and row.prev_position == 0:
        return pl_enter_position - commission_cost - slippage
    elif row['position'] == 0 and abs(row.prev_position) == 1:
        return pl_exit_position - commission_cost - slippage
    elif abs(row['position']) == 1 and abs(row.prev_position) == 1 and \
            np.sign(row['position']) != np.sign(row.prev_position):
        return pl_enter_position + pl_exit_position - 2 * commission_cost - slippage
    else:
        return pl_keep_in_position


def scale_prices(data):
    mean = data['Close'].mean()
    data['scaled_open'] = data['Open'] / mean
    data['scaled_close'] = data['Close'] / mean
    data['returns'] = data['Close'].pct_change()


def calc_enter_position_price(data, order_quantity,
                              enter_position_price='enter_position_price',
                              position='position'):
    start_ind = data.index[0]
    data.loc[start_ind, enter_position_price] = 0
    for i in data.index:
        enter_position_price_val = 0
        if i > start_ind:
            if data.loc[i - 1, position] == 0 and abs(data.loc[i, position]) == 1:
                enter_position_price_val = data.loc[i, 'Open'] * order_quantity
            elif data.loc[i - 1, position] == 1 and data.loc[i, position] == -1:
                enter_position_price_val = data.loc[i, 'Open'] * order_quantity
            elif data.loc[i - 1, position] == -1 and data.loc[i, position] == 1:
                enter_position_price_val = data.loc[i, 'Open'] * order_quantity

            elif data.loc[i - 1, position] == 1 and data.loc[i, position] == 1:
                enter_position_price_val = data.loc[i - 1, enter_position_price]
            elif data.loc[i - 1, position] == -1 and data.loc[i, position] == -1:
                enter_position_price_val = data.loc[i - 1, enter_position_price]
            elif abs(data.loc[i - 1, position]) == 1 and data.loc[i, position] == 0:
                enter_position_price_val = data.loc[i - 1, enter_position_price]

            elif data.loc[i - 1, position] == 0 and data.loc[i, position] == 0:
                enter_position_price_val = 0

            data.loc[i, enter_position_price] = enter_position_price_val


def run_model(tickers,
              momentum_lbw, momentum_th, SSO_smoothing_factor,
              volume_trigger_lbw, volume_trigger_duration, volume_power, rolling_price_lbw,
              transaction_cost=0.0035, slippage_rate=0.25, order_quantity=1, is_test=False, show_plots=False):
    sharpes = []
    returns = []
    total_returns = pd.DataFrame(columns=sorted([*data_tools.tickers, *list(map(lambda tck: f"{tck}_time", tickers))]))

    for ticker in tickers:
        data_to_run = test[ticker] if is_test else train[ticker]
        add_model_features(data_to_run,
                           SSO_lbw=momentum_lbw,
                           volume_trigger_lbw=volume_trigger_lbw,
                           volume_trigger_duration=volume_trigger_duration,
                           volume_trigger_factor=volume_power,
                           rolling_price_lbw=rolling_price_lbw,
                           SSO_smoothing_factor=SSO_smoothing_factor)

        calc_model(momentum_th, order_quantity, data_to_run, transaction_cost, slippage_rate)

        sharpe_res = round(sharpe_ratio(data_to_run['returns'].dropna()), 3)
        returns_res = round(data_to_run['returns_accumulate'].dropna().iloc[-1], 3)
        sharpes.append(sharpe_res)
        returns.append(returns_res)
        total_returns[ticker] = data_to_run['returns']
        total_returns[f"{ticker}_time"] = data_to_run['Datetime']

        if show_plots:
            plot_ticker_results(data_to_run, ticker, sharpe_res)

        data_to_run['sharpe'] = sharpe_res
        data_to_run['returns'] = returns_res

    # total_returns.to_csv("./total_returns.csv")

    return np.mean(sharpes), \
           np.std(sharpes), \
           np.mean(returns), \


def calc_model(momentum_th, order_quantity, data, transaction_cost, slippage_rate):
    calc_open_positions_SSO(data, SSO_boundaries=momentum_th)
    calc_positions(data, momentum_th)
    calc_enter_position_price(data, order_quantity)
    calc_pl_returns(data, transaction_cost=transaction_cost, slippage_rate=slippage_rate, order_quantity=order_quantity)


def plot_ticker_results(data, ticker, sharpe):
    data['color'] = 'None'
    data.loc[(data['position'] == 1), 'color'] = 'green'
    data.loc[(data['position'] == -1), 'color'] = 'red'

    fig, axs = plt.subplots(2, figsize=(16, 12))
    fig.suptitle(f'{ticker} - Model performance')
    xs = data.index
    margin = 0.2

    axs[0].set_ylim((data['volume_vs_lbw_volume_mean'] / data['volume_vs_lbw_volume_mean'].max() + data[
        'Close'].min()).min() - margin,
                    max(data['Close']))
    axs[0].bar(xs, data['volume_vs_lbw_volume_mean'] / data['volume_vs_lbw_volume_mean'].max() + data[
        'Close'].min() - margin,
               color=['c' if x == 1 else 'lightgrey' for x in data['volume_trigger']], zorder=1)

    axs[0].plot(xs, data['Close'], color='black', linewidth=0.5, zorder=1, label='Stock Price')

    axs[0].plot(xs, [data.loc[i, 'Close'] if data.loc[i, 'position'] == 1 else None for i in data.index],
                color='green', linewidth=0.7, zorder=2, label='Long Trade')
    axs[0].plot(xs, [data.loc[i, 'Close'] if data.loc[i, 'position'] == -1 else None for i in data.index],
                color='red', linewidth=0.7, zorder=2, label='Short Trade')
    axs[0].scatter(xs, data['Close'],
                   c=data['color'], s=1, zorder=2)

    axs[0].title.set_text("Stock Price & Trades")
    axs[0].legend(loc='upper right')

    axs[1].plot(xs, data['returns_accumulate'])
    axs[1].title.set_text(f"Model Returns\nSharpe Ratio: {sharpe}")
    plt.subplots_adjust(hspace=0.35)
    plt.show()


def print_curr_params():
    print(f"momentum_lbw: {momentum_lbw},"
          f" momentum_th: {momentum_th},"
          f" volume_trigger_lbw: {volume_trigger_lbw},"
          f" volume_trigger_duration: {volume_trigger_duration},"
          f" volume_power: {volume_power}, "
          f" rolling_price_lbw: {rolling_price_lbw},"
          f" SSO_smoothing_factor: {SSO_smoothing_factor}")


def print_results():
    print(f"avg sharpe = {round(params_sharpe, 3)},"
          f" std between sharpes = {round(params_sharpe_std, 3)},"
          f" avg returns = {round(params_returns, 3)}")


def store_results():
    results_data = {
        'momentum_lbw': momentum_lbw,
        'momentum_th': momentum_th,
        'volume_trigger_lbw': volume_trigger_lbw,
        'volume_trigger_duration': volume_trigger_duration,
        'volume_power': volume_power,
        'sharpe': params_sharpe,
        'sharpe_std': params_sharpe_std,
        'returns': params_returns,
        'rolling_price_lbw': rolling_price_lbw,
        'SSO_smoothing_factor': SSO_smoothing_factor,
    }
    results = pd.Series(results_data)
    total_results.loc[
        f'{momentum_lbw}_{momentum_th}_{volume_trigger_lbw}_{volume_trigger_duration}_{volume_power}_{rolling_price_lbw}_{SSO_smoothing_factor}'] \
        = results


total_results = pd.DataFrame(columns=
                             ['momentum_lbw', 'momentum_th', 'SSO_smoothing_factor',
                              'volume_trigger_lbw', 'volume_trigger_duration', 'volume_power',
                              'sharpe', 'sharpe_std',
                              'returns', 'rolling_price_lbw'])

tickers = data_tools.tickers

momentum_lbws = range(1, 20, 2)
momentum_ths = np.arange(0.05, 0.45, 0.07)
volume_trigger_lbws = [50]
volume_trigger_durations = range(1, 31, 8)
volume_powers = [1, 2, 3, 4, 5]
rolling_price_lbws = [10]
SSO_smoothing_factors = [1, 2, 3, 4]

def train_model():
    global momentum_lbw, momentum_th, SSO_smoothing_factor, \
        volume_trigger_lbw, volume_trigger_duration, volume_power, \
        rolling_price_lbw, \
        params_sharpe, params_sharpe_std, params_returns

    for (momentum_lbw, momentum_th, SSO_smoothing_factor,
         volume_trigger_lbw, volume_trigger_duration, volume_power,
         rolling_price_lbw) in itertools.product(momentum_lbws, momentum_ths, SSO_smoothing_factors,
                                                 volume_trigger_lbws, volume_trigger_durations, volume_powers,
                                                 rolling_price_lbws):
        print_curr_params()
        (params_sharpe, params_sharpe_std, params_returns) = \
            run_model(tickers,
                      momentum_lbw, momentum_th, SSO_smoothing_factor,
                      volume_trigger_lbw, volume_trigger_duration, volume_power, rolling_price_lbw,
                      transaction_cost=0.0035, order_quantity=1,
                      is_test=False, show_plots=False)
        print_results()
        store_results()
        if random.randint(1, 30) == 2:
            total_results.to_csv("./results_test/summary.csv")

train_model()
