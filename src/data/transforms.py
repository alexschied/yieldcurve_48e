import pandas as pd

def month_begin_index(df):
    df = df.copy()
    df.index = (df.index + pd.offsets.MonthBegin(0)).normalize()
    return df

def add_regime_dummies(df):
    df = df.copy()
    idx = df.index

    def in_range(start, end):
        return ((idx >= start) & (idx <= end)).astype(int)

    df["D_GFC"] = in_range("2007-12-01", "2009-06-01")
    df["D_COVID"] = in_range("2020-03-01", "2020-05-01")
    df["D_ZLB_QE"] = in_range("2008-12-01", "2015-12-01")
    return df
