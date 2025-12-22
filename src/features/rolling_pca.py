import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def rolling_pca_loadings(df, yield_cols, window):
    X = df[yield_cols].dropna()
    out = []

    for i in range(window, len(X)):
        w = X.iloc[i-window:i]
        Xs = StandardScaler().fit_transform(w)
        pca = PCA(n_components=3).fit(Xs)
        out.append(pd.Series(pca.components_[0], index=yield_cols, name=w.index[-1]))

    return pd.DataFrame(out)
