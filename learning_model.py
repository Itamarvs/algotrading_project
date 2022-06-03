import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import itertools


def label_data(data):
    volatility = np.std(data['returns'])
    # print(volatility)
    conditions = [
        (data['returns'] >= volatility),
        (data['returns'] <= -1 * volatility),
        (data['returns'] > -1 * volatility) & (data['scaled_close'] < volatility)]
    choices = [1, -1, 0]
    data['label'] = np.select(conditions, choices, default=0)
    # print(data['label'].head(40))
    # print(data['label'].value_counts())
    # print(data['returns'][data['label'] == 1])


def label_data_with_lfw(data, lfw, volatility_factor=1):
    # lbw_volatility = data['returns'].rolling(lbw).std()
    # print(lbw_volatility)
    volatility = np.std(data['returns'])
    print(volatility)
    data[f'mean_return_lfw_{lfw}'] = data['returns'].iloc[::-1].rolling(lfw).mean().iloc[::-1]
    conditions = [
        (data[f'mean_return_lfw_{lfw}'] >= volatility / volatility_factor),
        (data[f'mean_return_lfw_{lfw}'] <= -1 * volatility / volatility_factor),
        (data[f'mean_return_lfw_{lfw}'] > -1 * volatility / volatility_factor) &
        (data[f'mean_return_lfw_{lfw}'] < volatility / volatility_factor)]
    choices = [1, -1, 0]
    data['label'] = np.select(conditions, choices, default=0)
    print(data['label'].value_counts())

# def define_positions(data, )