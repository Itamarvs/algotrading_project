from datetime import datetime, timedelta

import yfinance as yf

data = yf.download(["AMZN"], period="20y", interval="1d")
print(type(data))


dates = []
for timestamp in list(data.index):
    date = timestamp.to_pydatetime()
    dates.append(date)

print(data.columns)


def volume_trigger(date, diff_factor=1.5, lbw=20):
    vol = data['Volume'][date.isoformat()]
    prev_day = date - timedelta(1)
    lbw_vol_data = data['Volume'][(prev_day - timedelta(lbw)).isoformat(): prev_day.isoformat()]
    is_trigger = vol > lbw_vol_data.mean() * diff_factor
    return is_trigger

date = datetime(2008, 5, 21)
volume_trigger(date)
