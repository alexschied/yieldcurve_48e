def holdout_split(df, frac):
    cut = int(len(df) * (1 - frac))
    return df.iloc[:cut], df.iloc[cut:]
