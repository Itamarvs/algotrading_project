import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split

import data_tools
from finance_tools import sharpe_ratio


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
    # true_range = np.max(ranges, axis=1)
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
    # add_pfe(data, SSO_lbw)
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


def calc_pl_returns(data, transaction_cost=0.0035, order_quantity=1):
    data['prev_position'] = data['position'].shift()
    data['prev_close'] = data['Close'].shift()
    data['pl'] = data.apply(lambda row:
                            row_pl(row, transaction_cost, order_quantity),
                            axis=1)
    data['pl_accumulate'] = data['pl'].cumsum()
    data.loc[data['enter_position_price'] == 0, 'returns'] = 0
    data.loc[data['enter_position_price'] != 0, 'returns'] = data['pl'] / data['enter_position_price']
    data['returns_accumulate'] = data['returns'].dropna().cumsum()


def row_pl(row, transaction_cost, order_quantity, slippage_rate, is_absolute_transaction_cost=False):
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


def calc_sharpe(data, returns='returns'):
    sharpe_res = sharpe_ratio(data[returns].dropna())
    return sharpe_res


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
              volume_trigger_lbw, volume_trigger_duration, volume_factor, rolling_price_lbw,
              transaction_cost=0.01, order_quantity=1):
    sharpes = []
    returns = []
    for ticker in tickers:
        data = yf.Ticker(ticker).history(period="60d", interval="15m")
        data = data.reset_index()
        train, test = train_test_split(data, test_size=0.25, shuffle=False)
        data_to_run = train
        add_model_features(data_to_run,
                           SSO_lbw=momentum_lbw,
                           volume_trigger_lbw=volume_trigger_lbw,
                           volume_trigger_duration=volume_trigger_duration,
                           volume_trigger_factor=volume_factor,
                           rolling_price_lbw=rolling_price_lbw,
                           SSO_smoothing_factor=SSO_smoothing_factor)

        calc_model(momentum_th, order_quantity, data_to_run, transaction_cost)

        sharpe_res = round(calc_sharpe(data_to_run), 3)
        returns_res = round(data_to_run['returns_accumulate'].dropna().iloc[-1], 3)
        sharpes.append(sharpe_res)
        returns.append(returns_res)

        plot_ticker_results(data_to_run, sharpe_res)

        data_to_run['sharpe'] = sharpe_res
        data_to_run['returns'] = returns_res
        data_to_run.to_csv(
            # f"./results_test/{ticker}_{momentum_lbw}_{momentum_th}_{volume_trigger_lbw}_{volume_trigger_duration}_{volume_factor}.csv")
            f"./results_test/{ticker}.csv")

    return np.mean(sharpes), \
           np.std(sharpes), \
           np.mean(returns)


def calc_model(momentum_th, order_quantity, data, transaction_cost):
    calc_open_positions_SSO(data, SSO_boundaries=momentum_th)
    calc_positions(data, momentum_th)
    calc_enter_position_price(data, order_quantity)
    calc_pl_returns(data, transaction_cost=transaction_cost, order_quantity=order_quantity)


def plot_ticker_results(data, ticker):
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
    axs[1].title.set_text("Model Returns")
    plt.subplots_adjust(hspace=0.35)
    plt.show()


def print_curr_params():
    print(f"momentum_lbw: {momentum_lbw},"
          f" momentum_th: {momentum_th},"
          f" volume_trigger_lbw: {volume_trigger_lbw},"
          f" volume_trigger_duration: {volume_trigger_duration},"
          f" volume_factor: {volume_factor}, "
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
        'volume_factor': volume_factor,
        'sharpe': params_sharpe,
        'sharpe_std': params_sharpe_std,
        'returns': params_returns,
        'rolling_price_lbw': rolling_price_lbw,
        'SSO_smoothing_factor': SSO_smoothing_factor,
    }
    results = pd.Series(results_data)
    total_results.loc[
        f'{momentum_lbw}_{momentum_th}_{volume_trigger_lbw}_{volume_trigger_duration}_{volume_factor}_{rolling_price_lbw}_{SSO_smoothing_factor}'] \
        = results


total_results = pd.DataFrame(columns=
                             ['momentum_lbw', 'momentum_th', 'SSO_smoothing_factor',
                              'volume_trigger_lbw', 'volume_trigger_duration', 'volume_factor',
                              'sharpe', 'sharpe_std',
                              'returns', 'rolling_price_lbw'])

tickers = data_tools.bio_tickers

# momentum_lbws = range(1, 12, 1)
# momentum_lbws = [2, 3, 5, 7]
# momentum_lbws = [1, 2, 3, 4, 5, 7, 10]
momentum_lbws = [8]
# momentum_ths = np.arange(0.15, 0.35, 0.05)
# momentum_ths = [0.15, 0.25, 0.4]
momentum_ths = [0.25]
# momentum_ths = np.arange(0.24, 0.28, 0.01)
# volume_trigger_lbws = range(4, 15, 2)
volume_trigger_lbws = [50]
# volume_trigger_lbws = [25]
# volume_trigger_durations = range(3, 50, 5)
volume_trigger_durations = [5]
# volume_trigger_durations = [11]

# volume_factors = [1.2, 1.5, 1.7]
# volume_factors = np.arange(1.5, 4, 0.1)
# volume_factors = np.arange(1.3, 2, 0.1)
volume_factors = [1.6]

rolling_price_lbws = [10]

# SSO_smoothing_factors = [1, 2, 3, 4, 5]
SSO_smoothing_factors = [1]


def train_model():
    global momentum_lbw, momentum_th, SSO_smoothing_factor, \
        volume_trigger_lbw, volume_trigger_duration, volume_factor, \
        rolling_price_lbw, \
        params_sharpe, params_sharpe_std, params_returns

    for (momentum_lbw, momentum_th, SSO_smoothing_factor,
         volume_trigger_lbw, volume_trigger_duration, volume_factor,
         rolling_price_lbw) in itertools.product(momentum_lbws, momentum_ths, SSO_smoothing_factors,
                                                 volume_trigger_lbws, volume_trigger_durations, volume_factors,
                                                 rolling_price_lbws):
        print_curr_params()
        (params_sharpe, params_sharpe_std, params_returns) = \
            run_model(tickers,
                      momentum_lbw, momentum_th, SSO_smoothing_factor,
                      volume_trigger_lbw, volume_trigger_duration, volume_factor, rolling_price_lbw,
                      transaction_cost=0.0035, order_quantity=1)
        print_results()
        store_results()


train_model()
total_results.to_csv("./results_test/summary.csv")
