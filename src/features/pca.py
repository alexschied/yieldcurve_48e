import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
