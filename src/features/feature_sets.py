def make_feature_sets(df, pc_df):
    fsets = {}

    fsets["spreads"] = df[["spread_10y_3m", "spread_10y_2y", "target"]].dropna()
    fsets["yield_pca"] = pc_df.join(df["target"]).dropna()

    return fsets
