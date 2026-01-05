from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def make_feature_sets(df, pc_df):
    fsets = {}

    fsets["spreads"] = df[["spread_10y_3m", "spread_10y_2y", "target"]].dropna()
    fsets["yield_pca"] = pc_df.join(df["target"]).dropna()

    return fsets

def compute_pca(df, yield_cols, n_components=3):
    X = df[yield_cols].dropna()
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(Xs)

    pc_df = pd.DataFrame(
        pcs, index=X.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=yield_cols,
        columns=pc_df.columns
    )

    evr = pd.Series(
        pca.explained_variance_ratio_,
        index=pc_df.columns
    )

    return pc_df, loadings, evr

def rolling_pca_loadings(df: pd.DataFrame, yield_cols: List[str],
                         window_months: int = 120, step: int = 1,
                         random_state: int | None = None):


    Xfull = df[yield_cols].dropna()
    pc_loadings = {1: [], 2: [], 3: []}
    times = []

    for end_i in range(window_months, len(Xfull) + 1, step):
        w = Xfull.iloc[end_i - window_months:end_i]
        t = w.index[-1]

        Xs = StandardScaler().fit_transform(w.values)
        pca = PCA(n_components=3, random_state=random_state).fit(Xs)
        L = pca.components_.T

        if L[:, 0].mean() < 0: L[:, 0] *= -1
        if L[-1, 1] < 0: L[:, 1] *= -1
        if L[-1, 2] < 0: L[:, 2] *= -1

        for k in [1, 2, 3]:
            pc_loadings[k].append(pd.Series(L[:, k - 1], index=yield_cols))
        times.append(t)

    return {
        f"PC{k}": pd.DataFrame(pc_loadings[k], index=pd.DatetimeIndex(times))
        for k in [1, 2, 3]
    }
