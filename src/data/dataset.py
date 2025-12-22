import pandas as pd
from .fred_client import fred_client
from .fetch import fetch_series_monthly_mean
from .transforms import month_begin_index, add_regime_dummies

def build_dataset(start, end):
    fred = fred_client()

    yield_ids = {
        "DGS3MO": "y_3m", "DGS6MO": "y_6m", "DGS1": "y_1y",
        "DGS2": "y_2y", "DGS3": "y_3y", "DGS5": "y_5y",
        "DGS7": "y_7y", "DGS10": "y_10y", "DGS30": "y_30y"
    }

    macro_ids = {
        "UNRATE": "unrate", "INDPRO": "indpro",
        "UMCSENT": "umcsent", "CPIAUCSL": "cpi"
    }

    frames = []

    for sid, name in yield_ids.items():
        s = fetch_series_monthly_mean(fred, sid, start, end)
        s.name = name
        frames.append(s)

    rec = fetch_series_monthly_mean(fred, "USREC", start, end)
    rec = (rec > 0.5).astype(int)
    rec.name = "recession"
    frames.append(rec)

    for sid, name in macro_ids.items():
        s = fetch_series_monthly_mean(fred, sid, start, end)
        s.name = name
        frames.append(s)

    df = pd.concat(frames, axis=1).sort_index()
    df = month_begin_index(df)

    df["infl_yoy"] = 100 * (df["cpi"] / df["cpi"].shift(12) - 1)
    df["indpro_yoy"] = 100 * (df["indpro"] / df["indpro"].shift(12) - 1)
    df["spread_10y_3m"] = df["y_10y"] - df["y_3m"]
    df["spread_10y_2y"] = df["y_10y"] - df["y_2y"]

    return add_regime_dummies(df)
