import math


def sharpe(vector):
    sum_diff = 0
    sumSq = 0
    count = len(vector) - 1

    for i in range(2, count):
        diff = vector[i] - vector[i - 1]
        sum_diff = sum_diff + diff
        sumSq = sumSq + (diff * diff)

    numer = 16 * sum_diff / count
    denom = (((count * sumSq) - (sum_diff * sum_diff)) / (count * (count - 1))) ** 0.5
    if denom == 0:
        sharpe_res = math.nan
    else:
        sharpe_res = numer / denom
    return sharpe_res


def sharpe_ratio(data, risk_free_rate=0.0):
    # Calculate Average Daily Return
    mean_daily_return = sum(data) / len(data)
    # Calculate Standard Deviation
    s = std_dev(data)
    # Calculate Daily Sharpe Ratio
    daily_sharpe_ratio = 0
    if s != 0:
        daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
    # Annualize Daily Sharpe Ratio
    sharpe_ratio = 252 ** (1 / 2) * daily_sharpe_ratio
    return sharpe_ratio


def std_dev(data):
    # Get number of observations
    n = len(data)
    # Calculate mean
    mean = sum(data) / n
    # Calculate deviations from the mean
    deviations = sum([(x - mean)**2 for x in data])
    # Calculate Variance & Standard Deviation
    variance = deviations / (n - 1)
    s = variance**(1/2)
    return s


def pfe(vector):
    count = len(vector)
    top = ((vector[0] - vector[count - 1]) ** 2 + (count - 1) ** 2) ** 0.5
    bottom = 0
    for i in range(count - 1):
        bottom = bottom + (((vector[i] - vector[i + 1]) ** 2 + 1) ** 0.5)
    if vector[0] > vector[count - 1]:
        top = -1 * top
    pfe_res = top / bottom
    return pfe_res
