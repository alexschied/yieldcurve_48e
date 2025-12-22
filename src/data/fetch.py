import pandas as pd

def fetch_series_monthly_mean(fred, series_id, start, end):
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s = pd.Series(s)
    s.index = pd.to_datetime(s.index)
    return s.resample("M").mean()
