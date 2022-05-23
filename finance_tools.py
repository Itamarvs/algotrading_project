def sharpe(vector):
    sum = 0
    sumSq = 0
    count = len(vector) - 1

    for i in range(count):
        diff = vector(i) - vector(i - 1)
        sum = sum + diff
        sumSq = sumSq + (diff * diff)

    numer = 16 * sum / count
    denom = (((count * sumSq) - (sum * sum)) / (count * (count - 1))) ** 0.5
    sharpe_res = numer / denom
    return sharpe_res


def pfe(vector):
    count = len(vector)
    top = ((vector(1) - vector(count)) ** 2 + (count - 1) ** 2) ** 0.5
    bottom = 0
    for i in range(count - 1):
        bottom = bottom + (((vector.i - vector(i + 1)) ** 2 + 1) ** 0.5)
    if vector(1) > vector(count):
        top = -1 * top
    pfe_res = top / bottom
    return pfe_res