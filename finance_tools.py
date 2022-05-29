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
